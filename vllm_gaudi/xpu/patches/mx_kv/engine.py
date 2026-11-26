# SPDX-License-Identifier: Apache-2.0
"""engine.py — MXFP4/MXFP8/MXK8V4 KV-cache attention backend (P6.47,
vllm_gaudi/xpu/patches/mx_kv/).

ENGINE INTEGRATION of the from-scratch MX (OCP microscaling) KV lane
(leg B). ``--kv-cache-dtype {mxfp4, mxfp8, mxk8v4}`` routes here through
``ArcXpuPlatform.get_attn_backend_cls`` (platform.py), which also stamps
``MxAttentionBackend.active_kv_dtype`` at selection time — the one
process-wide format selector (spec customization needs the format
before any spec carries it).

DESIGN: subclass, don't fork. MxAttentionBackend/MxAttentionImpl extend
the wheel's TurboQuant pair so the whole TQ plumbing rides along —
uint8 paged allocation via ``customize_spec``'s state_content_bytes,
page-size unification, TurboQuantMetadata (slot_mapping/block_table/
reorder), XPU-graph capture, and the P6.19 tq_spec_fix hook installed on
the parent classes (patches/tq_spec_fix.py): the spec-verify
capture-forcing stays active because ``get_builder_cls`` returns the
(patched) TurboQuantMetadataBuilder, and the split-row continuation law
is re-implemented 1:1 in the MX ``_prefill_attention`` below (leg-A
kernels + the engine-proven dispatch shape).

WHAT CHANGES VS TQ (every override below is one of these):
- slot geometry: MxConfig maps the dtype to mx_layout byte math
  (mxfp8 264 B, mxk8v4 200 B, mxfp4 136 B at D=128) — no Hadamard, no
  centroids,
  no per-row fp16 scale+zero: E8M0 group scales at 1 B / 32 elements.
- store: mx_store (triton_mx_store) — fused quantize+pack+scatter.
- decode: mx_decode (triton_mx_decode) — paged flash decode with
  inline MX dequant.
- continuation prefill: mx_dequant_cache (mx_dequant) — bulk dequant
  of the cached prefix, then flash-attn (no inverse rotation: MX
  quantizes in the native space).

FORMAT DISCOVERY LAW: every kernel infers the format from the cache
slot width (mx_layout law), so an MxConfig mismatch against the actual
allocated cache is impossible by construction.

Provenance: from-scratch in-house lane (P6.47, 2026-11-11);
integration surface follows the wheel's TurboQuant backend
(v1/attention/backends/turboquant_attn.py — survey
results/2026-11-11-p647-mxkv-research/wheelref/) and the fork's
tq_spec_fix split-row law (P6.19, results/2026-11-04-p619-tqverify/).
"""

from __future__ import annotations

import math
import os
import threading
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends import turboquant_attn as _tq
from vllm.v1.attention.backends.turboquant_attn import (
    TurboQuantAttentionBackend,
    TurboQuantAttentionImpl,
    TurboQuantMetadata,
    TurboQuantMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

try:
    # P-C2 (2026-11-25): the fa_utils shim (vllm._xpu_ops
    # flash_attn_varlen_func on XPU) accepts window_size and threads
    # the pair to the wheel interface. Windowless engines never touch
    # this import — the fallback keeps every pre-P-C2 posture intact.
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func as _fa_varlen,
    )
    _HAS_FA_VARLEN = True
except ImportError:  # noqa: BLE001 — module must import on any posture
    _fa_varlen = None
    _HAS_FA_VARLEN = False

from .mx_dequant import mx_dequant_cache
from .mx_layout import offsets as _mx_offsets
from .mx_layout import slot_bytes
from .mx_quant import decode_kv as _ref_decode_kv
from .native_decode import mx_decode_native
from .triton_mx_decode import mx_decode, mx_decode_rows, mx_decode_rows_dot
from .triton_mx_store import mx_store

# vllm-namespace logger (platform.py law): serve-time dictConfig
# attaches its handler only to the "vllm" logger — a vllm_gaudi.*
# logger's INFO records are silently dropped in serve processes (the
# first trace run produced zero lines exactly this way).
logger = init_logger("vllm.platforms.arc.mxkv")

MX_KV_DTYPES = ("mxfp4", "mxfp8", "mxk8v4")
# Element-class analogues used ONLY to run the parent __init__ (which
# builds a TurboQuantConfig): same packed-uint8 geometry family, never
# read by an MX code path after the swap below.
MX_TQ_INIT_ALIAS = {
    "mxfp8": "turboquant_k8v4",
    "mxfp4": "turboquant_4bit_nc",
    "mxk8v4": "turboquant_k8v4",  # e4m3 K + e2m1 V: the TRUE analogue
}

# tq_spec_fix laws mirrored here (patches/tq_spec_fix.py): verify rows
# dispatch as SEPARATE single-row decodes — the W1-proven-clean call
# shape under XPU-graph replay.
_SPLIT_MAX_Q = 8
_CONTINUATION_DECODE_THRESHOLD = 128

# Path telemetry (telemetry-capture-points directive): ARC_MXKV_TRACE=1
# logs every attention call's shape + dispatch path for the first
# _TRACE_BUDGET calls — the spec-verify corruption hunt needs to see
# WHICH impl instance (target vs MTP draft) runs which shapes, and the
# budget must cover the BATTERY phase too (600 burned on warmup alone:
# trace3/trace4, 2026-11-12 — battery-phase xcheck evidence requires
# ~50k; debug-run only, never on the clean path).
_TRACE = os.environ.get("ARC_MXKV_TRACE", "") == "1"
_TRACE_BUDGET = 200000
_trace_calls = 0

# v4 verify-row sharing (A11 lever 2, 2026-11-12): ONE walk per request
# serving all its MTP verify rows (shared K/V dequant — the ALU the A10
# attribution showed binding at oversubscription). REFUTED as a timing
# win by the p159 probe (r166: 0.20-0.49x mxfp4, 0.72-0.88x mxfp8,
# 0.58-0.79x mxk8v4 at G8 — row-sharing serializes the dequant and
# loses program parallelism); reachable only via ARC_MXKV_V4_VERIFY=1
# for A/B. _V4_R_TILE must equal the kernel's group width (spec=3 ->
# k+1 = 4).
_V4_VERIFY = os.environ.get("ARC_MXKV_V4_VERIFY", "") == "1"
_V4_R_TILE = 4
# v5 (A11 lever 3): the same row-group dispatch with the XMX tl.dot
# kernel (score path as fp16 MMA — the ctrl lane's economy). Takes
# precedence over v4 when both are set (the p160 A/B face).
# DEFAULT ON for mxfp8 + mxk8v4 (A13 probe, r167 2026-11-13): v5 beats
# the v2 batch at every cell — mxfp8 1.21-1.90x (G8@32K 43.9 vs
# 83.5 ms), mxk8v4 1.33-2.86x, correctness <=1.2e-4 (fp16-operand
# class). mxfp4 stays on v2 (Phase 6 B1, r188 2026-11-19): A13's
# "0.94-1.05x wash" predated K3 — with the e2m1 fast-unpack the v5
# KERNEL runs 7.618 vs the v2 batch's ~8.2-9.2 ms (r179 k3cycleA18) —
# but the e2e A/B REFUTED the blanket flip: m4qv5 regresses n1 at every
# cell (+8..14 ms — the v2 batch grid (R x Hq x splits) out-parallelizes
# v5's (G x HK x splits) at G=1; v5's plan-build bt_groups tax does not
# amortize) while WINNING n8 long-ctx (32K ss 1978->1703, aggregate p50
# 1551->696).
# G-ADAPTIVE mxfp4 DISPATCH (Phase 7 closing rung, 2026-11-19): dispatch
# by the split-batch request count at plan build — v2 below the knee,
# v5 at/above it (ARC_MXKV_V5_G_MIN, default 8 = the n8 face; the B1
# bank's knee sits between the 8K and 16K observed-concurrency
# steady-state cells). Capture-lawful: the count is a host-known shape
# at plan-build time and the plan cache freezes the choice per shape.
_V5_G_MIN = int(os.environ.get("ARC_MXKV_V5_G_MIN", "8") or 8)
# ARC_MXKV_VERIFY_DECODER={v2,v4,v5} overrides the per-format default
# per boot (verify-batch path only — ARC_MXKV_DECODER stays the
# certified v1|v2|v3 knob for the per-row kernel and must not see
# these values: mx_decode's launcher maps any non-v2 to v3); the
# legacy opt-in flags below win over everything.
_V5_DOT = os.environ.get("ARC_MXKV_V5_DOT", "") == "1"
_DECODER_OVERRIDE = os.environ.get("ARC_MXKV_VERIFY_DECODER", "")
_V5_DEFAULT = ("mxfp8", "mxk8v4")
# arc7 (Phase 8): the MX-NATIVE vendor-class decode kernel (CUTLASS/Xe2,
# wheel-side external_patches/vllm-xpu-kernels/arc7_mx_attn/ — the vendor
# structure ported to MX formats, dequant fused into the load path).
# DORMANT until certified: Triton v2/v5 stay the certified default; the
# env routes the verify-batch path through native_decode.mx_decode_native
# (the arc7mx wheel's torch op). The engine refuses with a warning when
# the GQA group exceeds the q16 policy bucket (plan-build-time, so the
# warning rides shape builds only). Validation ladder: r198 offline
# correctness -> r199 T1 perf -> r200 fastcycle -> r201 battery -> cert26.
_NATIVE = os.environ.get("ARC_MXKV_NATIVE", "") == "1"
# r224a in-vivo A/B (diagnostic, default off): ARC_MXKV_AB=1 runs the
# certified Triton v2 decoder ON THE SAME (q_rows, kv_cache, bt_rows,
# sl_rows) the native call consumed and banks the divergence + the
# call-site payload (pool window WITH its real shape/strides — the
# r220c offline gate's contiguous synthetic pool is exactly what it
# cannot see). EAGER boots only: the verify path runs inside captured
# XPU graphs in steady state, and D2H/.cpu()/torch.save under capture
# is illegal (pageable-H2D law) — the block self-guards on an active
# capture and stays silent, so a graphs-on boot is unaffected. Payload
# budget ARC_MXKV_AB_N (default 3) bounds disk and stall (~30 MB each,
# saved from a daemon thread per the dump-probe law).
_AB = os.environ.get("ARC_MXKV_AB", "") == "1"
_AB_N = int(os.environ.get("ARC_MXKV_AB_N", "3") or 3)
# r224a-2: minimum max-seqlen for a capture. The first leg's budget was
# consumed entirely by the eager DUMMY executions (rows=152 then rows=4
# rounds, sl<=101) before any real generation decode ran — MINSL skips
# them so the budget lands on real traffic (essay-class steps, the
# refusal's failing class). The sl read is a D2H sync, legal in this
# eager-only diagnostic boot (item-d2h law).
_AB_MINSL = int(os.environ.get("ARC_MXKV_AB_MINSL", "0") or 0)
_ab_n = 0
# r212a/r212c CTX-ADAPTIVE DISPATCH (geometry law, 2026-11-21): the
# native kernel's FIXED barrier/walk term loses to the certified Triton
# lanes at small WALKED token budgets while its slope is far shallower.
# r212c recalibrated at the REAL engine geometry (head_dim 256, page 64,
# G 6 — the r208-r212 E5b numbers were D=128/BLOCK=16/G=4, priced off
# the config's GDN linear head dim): D=256 doubles the Triton slopes
# (bytes/token) while native's walk/barrier-dominated cost grows far
# slower, so EVERY format crosses between Σ 4K-16K walked (mxfp8
# ~11.6K vs v5 — moved from ~214K, mxfp4 ~11.6K vs v2 / earlier vs v5,
# mxk8v4 ~6.6K vs v5); at 32K ctx native is 1.5-3.2x faster than the
# best lane. Tuples below = crossing + fill margin, per LANE (the G
# split mirrors _V5_G_MIN, the same knee the g-adaptive chain uses);
# the estimator is the bucket-cap (see the plan-build site — r212a
# capture-freeze law). ARC_MXKV_NATIVE_MINLEN="fmt:N" (both lanes) or
# "fmt:N1:N2" (n1-lane:n8-lane) retunes per boot.
_NATIVE_MINLEN: dict[str, tuple[int, int]] = {
    "mxfp8": (16384, 16384),
    "mxfp4": (12288, 8192),
    "mxk8v4": (8192, 8192),
}
for _spec in filter(
    None, os.environ.get("ARC_MXKV_NATIVE_MINLEN", "").split(",")
):
    _name, _, _vals = _spec.partition(":")
    _nums = [v for v in _vals.split(":") if v.strip().isdigit()]
    if _name and _nums:
        _NATIVE_MINLEN[_name.strip()] = (
            int(_nums[0]),
            int(_nums[-1]),
        )
# one-time dispatch markers (the posture-line pattern): fmt+decoder
# pairs already logged by this process — keeps every serve log carrying
# the verify-decoder engagement line without per-step spam.
_DECODER_LOGGED: set[tuple[str, str]] = set()

# Ground-truth dump hook (debug-run-only, default off): with
# ARC_MXKV_DUMP=1, the first attention call after `touch /tmp/mxkv_dump`
# saves the low pool region + metadata to /tmp/mxkv_dump.<pid>.pt. The
# corruption hunt byte-diffs SPEC vs EAGER runs of the same temp-0
# prompt: committed tokens match until divergence, so any cache-byte
# difference before it is a wrong store (self-consistency xchecks are
# blind to wrong-layer/wrong-token bytes — both sides read the cache).
# LAYER PIN (r152): impls are built in layer order, so ord N = the Nth
# attention impl; every layer owns a SEPARATE pool tensor at the SAME
# slot ids, so payloads from different impls are incomparable (r151's
# 97%-differ "pair" was two layers). Dumps are gated to one ordinal
# (ARC_MXKV_DUMP_ORD, default 0 = first impl) BEFORE the sentinel is
# consumed, so each touch banks exactly that layer's next forward.
_DUMP = os.environ.get("ARC_MXKV_DUMP", "") == "1"
_DUMP_SENTINEL = "/tmp/mxkv_dump"
_DUMP_ORD = int(os.environ.get("ARC_MXKV_DUMP_ORD", "0"))
# ARC_MXKV_DUMP_FILTER=p1: consume sentinels ONLY for battery-prompt-1
# shaped forwards — r152's captures were all eaten by warmup/other-request
# forwards (97/6656/7488/8197 classes run BEFORE p1; the async scheduler
# interleaves them with p1's steps). With the filter the pre-placed
# sentinel survives warmup untouched and banks p1's prefill (mql==sl==14)
# then each touch banks a p1 verify/decode step (mql 1|2, sl<=79).
_DUMP_FILTER = os.environ.get("ARC_MXKV_DUMP_FILTER", "")
_dump_n = 0
_impl_n = 0
_dump_threads: dict[int, threading.Thread] = {}


def _dump_shape_wanted(mql: int, sl0: int) -> bool:
    # "p1": the battery's p1 request (prefill 14/14, verifies 1|2 at
    # sl 15..79 — excludes the warmup's 97/99/8199-class forwards).
    # "mq<N>": ANY forward with exactly N query rows (trace22: a custom
    # P+suffix prefill — token-count known at launch, unique in the
    # boot; warmup shapes 97/6656/8197 never collide with a small N).
    if _DUMP_FILTER != "p1" and not _DUMP_FILTER.startswith("mq"):
        return True
    if _DUMP_FILTER.startswith("mq"):
        return mql == int(_DUMP_FILTER[2:])
    return (mql == 14 and sl0 == 14) or (mql in (1, 2) and 14 < sl0 < 80)


def _maybe_dump(kv_cache: torch.Tensor, attn_metadata: Any, impl) -> None:
    global _dump_n
    if not _DUMP:
        return
    if os.path.exists(_DUMP_SENTINEL):
        # What does the NEXT forward look like while a touch awaits
        # consumption? (r152 trace16: 20 touches, one payload, zero
        # filter/save failures — the consumption path is invisible
        # without this line. Bounded: layer 0 runs first per forward,
        # so a pending sentinel logs ~1 line per touch.)
        try:
            _sl0 = int(attn_metadata.seq_lens[:1].cpu().item())
        except Exception:  # noqa: BLE001 — diagnostics must never kill serving
            _sl0 = -1
        logger.info(
            "MXKVDIAG pending ord=%d mql=%s sl0=%d",
            getattr(impl, "_mxkv_ord", -1),
            attn_metadata.max_query_len,
            _sl0,
        )
    if getattr(impl, "_mxkv_ord", -1) != _DUMP_ORD:
        return  # only the pinned layer dumps — cross-layer pools differ
    try:
        if _DUMP_FILTER:
            sl0 = int(attn_metadata.seq_lens[:1].cpu().item())
            if not _dump_shape_wanted(int(attn_metadata.max_query_len), sl0):
                return
    except Exception as exc:  # noqa: BLE001 — debug instrument, never kill serving
        # Never silently drop the series: bank the payload anyway and
        # name the fault (r152 trace15 lost every verify capture to a
        # swallowed exception — 1 payload per boot, no diagnostic line).
        logger.error("MXKV dump filter failed (dumping anyway): %r", exc)
    try:
        os.rename(_DUMP_SENTINEL, _DUMP_SENTINEL + ".done")
    except OSError:
        return  # already taken by another impl this phase
    _dump_n += 1
    try:
        # Inline cost = one D2H of blocks 0..208 (~30 MB): covers both
        # lanes' battery-prompt-1 allocations (spec ~91, eager ~169).
        # The multi-second torch.save runs in a daemon thread — a sync
        # save inside the attention call stalls the forward so hard the
        # touch fan banks ~1 payload per boot (r152 trace14: prefill
        # only). Backgrounding keeps each dump ~30 ms.
        payload = {
            "blocks": kv_cache[0:208].contiguous().cpu(),
            "bt": attn_metadata.block_table[:4, :32].cpu(),
            "seq_lens": attn_metadata.seq_lens[:8].cpu(),
            "smap": attn_metadata.slot_mapping[-32:].cpu(),
            "mql": attn_metadata.max_query_len,
            "msl": attn_metadata.max_seq_len,
            "impl": id(impl),
            "ord": getattr(impl, "_mxkv_ord", -1),
        }
        path = f"{_DUMP_SENTINEL}.{os.getpid()}.{_dump_n}.pt"
        th = threading.Thread(target=torch.save, args=(payload, path), daemon=True)
        _dump_threads[_dump_n] = th
        th.start()
    except Exception as exc:  # noqa: BLE001 — debug instrument, always log
        logger.error("MXKV dump failed: %s", exc)


def _mx_ab(
    impl: Any,
    q_rows: torch.Tensor,
    kv_cache: torch.Tensor,
    bt_rows: torch.Tensor,
    sl_rows: torch.Tensor,
    out_native: torch.Tensor,
) -> None:
    """r224a: certified-Triton A/B against the native call, in vivo.

    Runs ``mx_decode`` (the certified v2 lane the refused format falls
    back to) on the IDENTICAL inputs the native kernel just consumed,
    banks the divergence + the call-site payload (pool window WITH its
    real shape/strides — the r220c contiguous synthetic pool is the
    blind spot this closes), and logs one verdict line. Bounded by
    ``_AB_N``; thread-save per the dump-probe law. Eager only — the
    caller guards on an active capture.
    """
    global _ab_n
    if _AB_MINSL and int(sl_rows.max().item()) < _AB_MINSL:
        return
    _ab_n += 1
    n = _ab_n
    ref = mx_decode(q_rows, kv_cache, bt_rows, sl_rows, impl.scale)
    d = (out_native.float() - ref.float()).abs()
    # r224a-2: bank the blocks row 0 actually walks (first leg's fixed
    # [0:48] window held dummy rounds' blocks 0-47 while the captured
    # rows lived in 221+/442+ — unaddressable bytes). Capped at 96
    # blocks = 6144 tokens: covers essay-class ctx, bounds the payload.
    _nb = min(int(bt_rows.shape[1]), 96)
    _win_idx = bt_rows[0, :_nb].to(torch.long)
    payload = {
        "fmt": impl.mx_config.fmt,
        "scale": float(impl.scale),
        "rows": int(q_rows.shape[0]),
        "q": q_rows[:8].detach().cpu(),
        "bt": bt_rows[:8, :48].detach().cpu(),
        "sl": sl_rows[:8].detach().cpu(),
        "kv_shape": tuple(kv_cache.shape),
        "kv_stride": tuple(kv_cache.stride()),
        "kv_dtype": str(kv_cache.dtype),
        "kv_dim": kv_cache.dim(),
        "kv_off": int(kv_cache.storage_offset()),
        "bt_head": _win_idx.detach().cpu(),
        "kv_win": kv_cache.index_select(0, _win_idx).contiguous().cpu(),
        "out_native": out_native[:8].detach().cpu(),
        "out_triton": ref[:8].detach().cpu(),
        "dmax": float(d.max().item()),
        "dmean": float(d.mean().item()),
        "nan_native": int(torch.isnan(out_native).sum().item()),
        "nan_triton": int(torch.isnan(ref).sum().item()),
    }
    path = f"/tmp/mxkv_ab.{os.getpid()}.{n}.pt"
    _dump_threads[n] = threading.Thread(
        target=torch.save, args=(payload, path), daemon=True
    )
    _dump_threads[n].start()
    logger.info(
        "MXKV AB n=%d fmt=%s rows=%d dmax=%.3e dmean=%.3e nan=%d/%d "
        "kv_shape=%s kv_stride=%s off=%d saved=%s",
        n,
        payload["fmt"],
        payload["rows"],
        payload["dmax"],
        payload["dmean"],
        payload["nan_native"],
        payload["nan_triton"],
        payload["kv_shape"],
        payload["kv_stride"],
        payload["kv_off"],
        path,
    )


def _ab_ok() -> bool:
    """Fire the A/B only when it cannot interact with graph capture."""
    if not (_AB and _ab_n < _AB_N):
        return False
    cap = getattr(torch.xpu, "is_current_stream_capturing", None)
    if cap is None:
        # No capture API on this torch: only an explicitly graph-off
        # boot may run the (D2H-touching) instrument.
        return os.environ.get("VLLM_XPU_ENABLE_XPU_GRAPH", "1") != "1"
    try:
        return not bool(cap())
    except Exception:  # noqa: BLE001 — instrument never kills serving
        return False


def _trace(tag: str, *args: object) -> None:
    global _trace_calls
    if not _TRACE or _trace_calls >= _TRACE_BUDGET:
        return
    _trace_calls += 1
    logger.info("MXKVTRACE %s %s", tag, " ".join(str(a) for a in args))


@dataclass(frozen=True)
class MxConfig:
    """Slot geometry for one MX kv-cache dtype — the Mx analogue of the
    wheel's TurboQuantConfig (all byte math delegates to mx_layout, the
    single source of truth)."""

    fmt: str
    head_dim: int

    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> MxConfig:
        if cache_dtype not in MX_KV_DTYPES:
            raise ValueError(
                f"Unknown MX kv-cache dtype: {cache_dtype!r}. Valid: {MX_KV_DTYPES}"
            )
        return MxConfig(fmt=cache_dtype, head_dim=head_dim)

    @property
    def slot_size(self) -> int:
        return slot_bytes(self.head_dim, self.fmt)

    @property
    def slot_size_aligned(self) -> int:
        # Even slots keep effective_head_size = slot // 2 integral (the
        # wheel's TQ law); both MX formats are even at every legal D.
        s = self.slot_size
        return s + (s % 2)

    # ---- TQ-compat surface (inherited paths may read these; honest MX
    # geometry, never semantically used on an MX code path) ----
    @property
    def key_fp8(self) -> bool:
        return self.fmt in ("mxfp8", "mxk8v4")

    @property
    def key_mse_bits(self) -> int:
        return 8 if self.fmt in ("mxfp8", "mxk8v4") else 4

    @property
    def effective_value_quant_bits(self) -> int:
        return 4 if self.fmt in ("mxfp4", "mxk8v4") else 8

    @property
    def norm_correction(self) -> bool:
        return False

    @property
    def key_packed_size(self) -> int:
        return self.slot_size // 2

    @property
    def value_packed_size(self) -> int:
        return self.slot_size - self.slot_size // 2

    @property
    def n_centroids(self) -> int:
        return 1

    @property
    def centroid_bits(self) -> int:
        return 1


class MxAttentionBackend(TurboQuantAttentionBackend):
    """MX KV cache backend: TQ plumbing, MX kernels and slot geometry."""

    # Stamped by ArcXpuPlatform.get_attn_backend_cls at selection time
    # (process-wide one-format law; the spec layer carries no dtype
    # string, so the resolved backend class is the format authority).
    active_kv_dtype: ClassVar[str | None] = None

    supported_kv_cache_dtypes: ClassVar[list[str]] = [*MX_KV_DTYPES]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """MX packs K+V (plus E8M0 scale groups) into one slot per head."""
        if spec.state_content_bytes is not None:
            return spec
        if cls.active_kv_dtype is None:
            raise RuntimeError(
                "MxAttentionBackend.customize_spec called before platform "
                "routing stamped active_kv_dtype (ArcXpuPlatform."
                "get_attn_backend_cls must route the MX dtypes)."
            )
        mx = MxConfig.from_cache_dtype(cls.active_kv_dtype, spec.head_size)
        return replace(spec, state_content_bytes=mx.slot_size_aligned)

    @staticmethod
    def get_name() -> str:
        # MX is its own backend family (2026-11-11) —
        # AttentionBackendEnum carries a true MXKV member pointing at
        # THIS class: injected at platform register time
        # (platform._install_mxkv_enum_member — the registry is always
        # imported before the platform's hooks install), with the
        # source transform (platform._add_mxkv_backend_member) covering
        # the late-import case. Enum-native resolution paths
        # (["MXKV"], get_class()) land on the real MX backend.
        return "MXKV"

    @staticmethod
    def get_impl_cls() -> type[MxAttentionImpl]:
        return MxAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TurboQuantMetadataBuilder]:
        # The wheel's builder, WITH the P6.19 tq_spec_fix capture-forcing
        # patch installed at module exec (hook fires before this module
        # imports it): spec-verify graphs never bake the raw-KV flash
        # fast path (#TQ-blind-verify law, applies dtype-wide).
        return TurboQuantMetadataBuilder

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        if kv_cache_dtype is None:
            return False
        return kv_cache_dtype.startswith("mxfp")


class MxAttentionImpl(TurboQuantAttentionImpl):
    """MX attention implementation: TQ dispatch, MX kernels.

    Inherited unchanged: ``forward`` (the tq_spec_fix-instrumented
    prefill/decode/mixed dispatcher), ``do_kv_cache_update`` (calls
    ``_store_kv``), ``_flash_attn_varlen``, the FlyDSL-off SoA stubs.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        if kv_cache_dtype not in MX_KV_DTYPES:
            raise ValueError(
                f"MxAttentionImpl requires an MX dtype, got {kv_cache_dtype!r}"
            )
        # Parent init on the element-class analogue preset builds every
        # piece of inherited plumbing (fa detection, split constants,
        # config caches); the format-bearing state swaps to MxConfig
        # right after. The alias never reaches an MX code path.
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=MX_TQ_INIT_ALIAS[kv_cache_dtype],
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            **kwargs,
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.mx_config = MxConfig.from_cache_dtype(kv_cache_dtype, head_size)
        # Layer ordinal for the dump instrument (see _maybe_dump): impls
        # are constructed in layer order, so this tags every impl with its
        # layer identity without reaching into model internals.
        global _impl_n
        self._mxkv_ord = _impl_n
        _impl_n += 1
        # Inherited paths that still read tq_config see honest MX
        # geometry via the MxConfig compat surface.
        self.tq_config = self.mx_config
        self._mse_bytes = self.mx_config.key_packed_size
        self._val_data_bytes = self.mx_config.value_packed_size
        self._n_centroids = 1
        logger.info(
            "MXKV impl: fmt=%s head=%d slot=%d B (capacity %.2fx fp16 KV) "
            "dump=%s ord=%d",
            self.mx_config.fmt,
            head_size,
            self.mx_config.slot_size_aligned,
            4 * head_size / self.mx_config.slot_size_aligned,
            _DUMP,
            self._mxkv_ord,
        )
        # P-C2 (2026-11-25): sliding-window impls (gemma-4-31b's 50
        # sliding layers, W=1024) boot ONLY on the native decode lane
        # — the Triton decode lanes are window-blind, and the native
        # masked kernel twin is compiled at D=256 (page 16/64) in the
        # qgroup<=8 bucket. Refuse at INIT (boot-loud), never at the
        # first decode. Windowless impls (qwen target; gemma
        # full_attention layers) are unaffected.
        if sliding_window:
            if not _NATIVE:
                raise ValueError(
                    "MxAttentionImpl with sliding_window="
                    f"{sliding_window} requires the native decode lane "
                    "(set ARC_MXKV_NATIVE=1): the Triton decode lanes "
                    "carry no window surface and would attend the full "
                    "context at ctx > W — refusing rather than serving "
                    "window-blind"
                )
            if head_size != 256:
                raise ValueError(
                    "MxAttentionImpl with sliding_window="
                    f"{sliding_window}: the native masked kernel twin is "
                    f"compiled at D=256 only (this impl is D={head_size}; "
                    "instantiate the masked twin for this head class "
                    "before serving a windowed model on it)"
                )
            _g = num_heads // (num_kv_heads or num_heads)
            if _g > 8:
                raise ValueError(
                    "MxAttentionImpl with sliding_window="
                    f"{sliding_window}: the native masked kernel twin is "
                    f"compiled at the qgroup<=8 bucket only (this impl "
                    f"carries G={_g})"
                )
            logger.info(
                "MXKV P-C2 windowed impl: fmt=%s head=%d W=%d (native "
                "masked decode; prefill rides the flash window pair "
                "(W-1, 0))",
                self.mx_config.fmt,
                head_size,
                sliding_window,
            )

    def _flash_attn_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """P-C2 (2026-11-25): window-correct prefill for sliding impls.

        The inherited TQ method is window-blind (causal only) — a
        gemma-4-31b 40K prefill on a sliding layer would attend the
        whole context instead of the last W tokens per row. The shim
        threads the flash pair (W-1, 0) into the wheel interface
        (real_window_size -> the stock cutlass Local tuples); the
        right-aligned varlen rows keep continuation chunks correct
        (query row i rides position k_len - q_len + i). Windowless
        impls fall through byte-identical to the inherited path
        (qwen target, gemma full_attention layers).
        """
        if self.sliding_window:
            if not _tq._HAS_FLASH_ATTN or not _HAS_FA_VARLEN:
                raise NotImplementedError(
                    "MXKV sliding_window="
                    f"{self.sliding_window} prefill requires the flash "
                    "varlen lane with window_size support"
                )
            return _fa_varlen(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                window_size=(self.sliding_window - 1, 0),
            )
        return super()._flash_attn_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

    # ------------------------------------------------------------------ #
    # Buffers: no rotation, no centroids — sentinels for the inherited   #
    # forward's passthrough reads, pre-allocated BEFORE capture (graph   #
    # law: no lazy allocation can land inside a replay).                 #
    # ------------------------------------------------------------------ #
    def _ensure_on_device(self, layer, device):
        if not hasattr(layer, "_tq_cached"):
            d = self.head_size
            eye = torch.eye(d, device=device, dtype=torch.float32)
            layer._tq_PiT = eye
            layer._tq_Pi = eye
            layer._tq_Pi_half = torch.eye(d, device=device, dtype=torch.float16)
            layer._tq_centroids = torch.zeros(1, d, device=device, dtype=torch.float32)
            layer._tq_midpoints = torch.zeros(1, device=device, dtype=torch.float32)
            layer._tq_cached = True
        max_len = self._max_model_len
        ac = getattr(self, "_arange_cache", None)
        if ac is None or ac.device != torch.device(device) or ac.shape[0] < max_len + 2:
            self._arange_cache = torch.arange(
                0, max_len + 2, device=device, dtype=torch.int32
            )
        cu = getattr(self, "_cu_2", None)
        if cu is None or cu.device != torch.device(device):
            self._cu_2 = torch.zeros(2, device=device, dtype=torch.int32)

    # ------------------------------------------------------------------ #
    # Store: fused quantize+pack+scatter (triton_mx_store)               #
    # ------------------------------------------------------------------ #
    def _store_kv(
        self,
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot)
        slot_mapping: torch.Tensor,
        layer: Any,
    ):
        # slot_mapping is the linear token position in the paged pool —
        # the same law the TQ store kernel applies (blk = slot //
        # BLOCK_SIZE). Negative slots (warmup dummy batches map every
        # row to -1) are skipped INSIDE the kernel — the upstream store
        # law; passing them through unguarded was an OOB write before
        # the pool base (P6.47 spec-corruption root cause, 2026-11-12).
        # Pure device arithmetic: graph-safe.
        block_size = kv_cache.shape[1]
        if _TRACE:
            # One layout line per impl (telemetry: the engine-side pool
            # layout law — shape/stride decide whether shape[1] is block
            # size, i.e. whether the inherited transpose produced the
            # raw-like view the kernel strides assume). smap = the actual
            # slot ids (small batches log in full — verify steps are
            # k+1 rows; the corruption hunt compares them against the
            # bt-walk slot ids logged by xcheck3).
            _smap = (
                slot_mapping[-8:].tolist() if slot_mapping.numel() else []
            )
            _trace(
                "store", f"cache={tuple(kv_cache.shape)}",
                f"stride={kv_cache.stride()}", f"slots={slot_mapping.shape[0]}",
                f"smap={_smap}",
                f"smax={int(slot_mapping.max()) if slot_mapping.numel() else -1}",
                f"bs_from_shape1={block_size}", f"id={id(self)}",
            )
        tok_block = torch.div(slot_mapping, block_size, rounding_mode="floor")
        tok_pos = slot_mapping - tok_block * block_size
        mx_store(key, value, kv_cache, tok_block.to(torch.int32), tok_pos.to(torch.int32))

    # ------------------------------------------------------------------ #
    # Decode: paged flash decode with inline MX dequant                  #
    # (triton_mx_decode)                                                 #
    # ------------------------------------------------------------------ #
    def _decode_attention(
        self,
        query: torch.Tensor,  # (B, Hq, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: torch.nn.Module | None = None,
    ) -> torch.Tensor:
        # Pi/centroids/PiT are TQ passthroughs — MX quantizes in the
        # native space, nothing to rotate or renormalize.
        _trace(
            "dec", f"B={query.shape[0]}", f"msl={attn_metadata.max_seq_len}",
            f"id={id(self)}",
        )
        if _DUMP:
            _maybe_dump(kv_cache, attn_metadata, self)
        # q passes natively (fp16/bf16): Triton promotes the q*K products
        # to fp32 in-kernel — bit-equal to the old .float() upcast (A9
        # dtype face) and two fewer cast kernels per layer. out follows
        # q's dtype via empty_like, so the .to() below is a no-op guard.
        if self.sliding_window:
            # P-C2 (2026-11-25): windowed solo decode rides the native
            # lane — the Triton decode kernel carries no window surface
            # and would attend the full context at ctx > W. Same row
            # contract as the verify path: one row per request, its
            # causal limit is the live seq_len (device tensor; the
            # bridge casts/cu-fills device-side — capture-safe).
            if not _NATIVE:
                raise NotImplementedError(
                    "MXKV sliding_window="
                    f"{self.sliding_window} decode requires "
                    "ARC_MXKV_NATIVE=1 (the init gate normally refuses "
                    "this posture at boot)"
                )
            out = mx_decode_native(
                query,
                kv_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                self.scale,
                attn_metadata.max_seq_len,
                self.mx_config.fmt,
                sliding_window=self.sliding_window,
            )
            return out.to(query.dtype)
        out = mx_decode(
            query,
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self.scale,
        )
        return out.to(query.dtype)

    # ------------------------------------------------------------------ #
    # Prefill: raw-KV flash for first chunks (inherited), MX decode or   #
    # bulk-dequant for continuations — the tq_spec_fix dispatch law      #
    # mirrored 1:1 (patches/tq_spec_fix.py):                             #
    #   1 < q_len <= 8   split rows -> ONE batched mx_decode per step    #
    #                    (_split_verify_decode, LIMIT-LAW per-row        #
    #                    limits; debug builds keep per-row splits +      #
    #                    the xcheck3 oracle)                             #
    #   8 < q_len <= 128 multi-row synthetic decode (eager chunked       #
    #                    prefill only — never capture-shaped)           #
    #   q_len > 128      bulk dequant + flash (mx_dequant_cache)         #
    # ------------------------------------------------------------------ #
    def _prefill_attention(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        key: torch.Tensor,  # (N, Hk, D)
        value: torch.Tensor,  # (N, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot)
        attn_metadata: TurboQuantMetadata,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        PiT: torch.Tensor | None = None,
        layer: Any = None,
    ) -> torch.Tensor:
        n_tok, hq, d = query.shape
        _trace(
            "pre", f"n_tok={n_tok}",
            f"mql={attn_metadata.max_query_len}",
            f"msl={attn_metadata.max_seq_len}",
            f"id={id(self)}",
        )
        if _DUMP:
            _maybe_dump(kv_cache, attn_metadata, self)

        # First-chunk fast path: every request's K/V is in the batch —
        # raw flash-attn, format-free (the MX cache was written by
        # do_kv_cache_update but is not read here).
        if _tq._HAS_FLASH_ATTN and attn_metadata.max_query_len == attn_metadata.max_seq_len:
            _trace("path", "batch-firstchunk", f"n_tok={n_tok}")
            return self._flash_attn_varlen(
                q=query,
                k=key,
                v=value,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_query_len,
            )

        hk = key.shape[1]
        use_gqa = hk < hq
        query_start_loc = attn_metadata.query_start_loc
        num_reqs = query_start_loc.shape[0] - 1

        output = torch.zeros(n_tok, hq, d, device=query.device, dtype=query.dtype)

        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        _max_seq = attn_metadata.max_seq_len
        _ac: torch.Tensor = self._arange_cache
        if _ac.shape[0] <= _max_seq:
            self._arange_cache = torch.arange(
                0, _max_seq + 1, device=query.device, dtype=attn_metadata.seq_lens.dtype
            )
            _ac = self._arange_cache

        # Split-branch requests (clean path) — resolved in ONE batched
        # mx_decode after the loop (see _split_verify_decode).
        split_rows: list[tuple[int, int, int]] = []

        for i in range(num_reqs):
            q_start = qsl[i]
            q_end = qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]  # (q_len, Hq, D)
            k_seq = key[q_start:q_end]  # (q_len, Hk, D)
            v_seq = value[q_start:q_end]  # (q_len, Hk, D)

            if q_len == seq_len:
                # First-chunk prefill: all K/V in the current batch.
                _trace("path", f"i={i}", "firstchunk", f"q={q_len}", f"sl={seq_len}")
                if _tq._HAS_FLASH_ATTN:
                    self._cu_2[1:2] = q_len
                    out = self._flash_attn_varlen(
                        q=q_seq,
                        k=k_seq,
                        v=v_seq,
                        cu_seqlens_q=self._cu_2,
                        cu_seqlens_k=self._cu_2,
                        max_seqlen_q=q_len,
                        max_seqlen_k=q_len,
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        q_seq.transpose(0, 1).contiguous(),
                        k_seq.transpose(0, 1).contiguous(),
                        v_seq.transpose(0, 1).contiguous(),
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=use_gqa,
                    ).transpose(0, 1)
                output[q_start:q_end] = out.to(query.dtype)
                continue

            # Continuation chunk: cached prefix lives in the MX cache.
            cached_len = seq_len - q_len
            bt_row = attn_metadata.block_table[i : i + 1]
            if 1 < q_len <= _SPLIT_MAX_Q:
                # Split-row dispatch (tq_spec_fix call shape). The clean
                # path defers to the post-loop _split_verify_decode — ONE
                # mx_decode for ALL split rows in the step (MTP verify is
                # the per-step hot shape: spec=3 makes every decode step a
                # q_len=k+1 batch, and per-row launches cost ~a kernel per
                # row per request per layer — ~2k kernels/step at agg8,
                # the dominant decode-gap term, A10 telemetry directive
                # 2026-11-12). Debug builds keep the W1-proven per-row
                # call shape the xcheck3 oracle below validates against.
                #
                # LIMIT LAW (P6.47 corruption fix, 2026-11-12), unchanged
                # by the batching: derive each row's causal limit from ITS
                # OWN slot via the block table, NOT from seq_lens. MTP
                # verify batches carry the next-draft row (nat = q+1) and
                # the wheel's seq_lens counts the allocator extent
                # INCLUDING that row (rig-proven trace4-8: step-2+
                # verifies' seq_lens run one ahead of the verify rows'
                # true positions, so the seq_lens-derived limits leak the
                # yet-unverified draft token's K/V into every verify row —
                # chunk_pair 0->1 + stale last slot, xcheck3). The
                # engine's slot assignment is the ground truth (v1
                # [5829,5830]=pos 5,6; v2 [5831,5832]=pos 7,8 —
                # store+cache bit-clean). Slot->position inversion is pure
                # recorded device arithmetic off live buffers
                # (slot_mapping/block_table refresh at replay — same law
                # as the seq_lens reads it replaces).
                if _TRACE:
                    _trace("path", f"i={i}", "split", f"q={q_len}", f"sl={seq_len}")
                    sm = attn_metadata.slot_mapping[q_start:q_end]
                    sm_blk = torch.div(
                        sm, kv_cache.shape[1], rounding_mode="floor"
                    )
                    sm_pos = sm - sm_blk * kv_cache.shape[1]
                    # bt walk inversion: the row's block's index in
                    # bt_row is its position // block_size (blocks
                    # appear once per row).
                    bt_match = bt_row[0].unsqueeze(0) == sm_blk.unsqueeze(1)
                    true_pos = (
                        bt_match.to(torch.int32).argmax(dim=1).to(sm.dtype)
                        * kv_cache.shape[1]
                        + sm_pos
                    )
                    sl_rows = (true_pos + 1)[:q_len]
                    for r in range(q_len):
                        out_r = mx_decode(
                            q_seq[r : r + 1],
                            kv_cache,
                            bt_row,
                            sl_rows[r : r + 1],
                            self.scale,
                        )
                        output[q_start + r : q_start + r + 1] = out_r.to(query.dtype)
                else:
                    split_rows.append((i, q_start, q_len))
                if _TRACE:
                    # Three-way cross-check (debug D2H sync is acceptable
                    # under TRACE; never on the clean path): split rows vs
                    # bulk dequant+flash vs the byte-law reference decode
                    # (mx_quant.decode_kv on raw cache bytes) — ALL heads,
                    # every row. Two agree = one path's addressing is
                    # wrong; the reference names it.
                    ref, k_full_, v_full_ = self._continuation_prefill(
                        layer,
                        q_seq,
                        k_seq,
                        v_seq,
                        kv_cache,
                        bt_row,
                        cached_len,
                        seq_len,
                        Pi,
                        centroids,
                        return_kv=True,
                    )
                    d_ = q_seq.shape[-1]
                    hk_ = kv_cache.shape[2]
                    hq_ = q_seq.shape[1]
                    fmt_ = next(
                        f for f in MX_KV_DTYPES
                        if slot_bytes(d_, f) == kv_cache.shape[-1]
                    )
                    vo = _mx_offsets(d_, fmt_)["v"]
                    bs_ = kv_cache.shape[1]
                    tp_list = true_pos.tolist()
                    # ALL-HEADS byte-law reference (P6.47 2026-11-12): the
                    # original oracle hardcoded head 0 — heads > 0 were
                    # never validated against content, and the split-vs-
                    # bulk cross-check cannot say WHICH path is wrong.
                    # GQA: query head h reads KV head h // (Hq // Hk).
                    refs = []
                    kr_last = None
                    kvref: dict[int, tuple] = {}
                    for r in range(q_len):
                        limit = tp_list[r] + 1
                        t = torch.arange(limit, device=q_seq.device)
                        blks = bt_row[0, torch.div(t, bs_, rounding_mode="floor")].long()
                        poss = (t % bs_).long()
                        sb_ = kv_cache[blks, poss]  # (limit, Hk, slot)
                        row_refs = []
                        for h in range(hq_):
                            kvh = h // max(hq_ // hk_, 1)
                            kr, vr = _ref_decode_kv(
                                sb_[:, kvh, :vo], sb_[:, kvh, vo:], fmt_, d_
                            )
                            if kvh == 0:
                                # Sub-check 2 baseline: KV head 0's K rows
                                # (trace20's kr_last-as-last-head bug made
                                # prefix_dequant_vs_ref compare head 1 vs
                                # head 0 — 20-25 max-abs noise; keep the
                                # head-0 rows the check reads).
                                kr_last = kr  # r = q_len-1 covers 0..seq_len-1
                            if r == q_len - 1:
                                # Longest walk per kv head (trace24
                                # per-head prefix check).
                                kvref[kvh] = (kr, vr)
                            s = torch.softmax(
                                q_seq[r, h, :].float() @ kr.T * self.scale, dim=-1
                            )
                            row_refs.append(s @ vr)
                        refs.append(torch.stack(row_refs))  # (Hq, D)
                    refall = torch.stack(refs)  # (q_len, Hq, D)
                    ref0 = refall[:, 0, :]
                    # Sub-check 1 (store consistency): the cache's copy of
                    # the CURRENT chunk at the rows' TRUE slots vs the raw
                    # key/value this forward received — must differ only by
                    # MX quant noise (diagonal pairing).
                    cbytes = kv_cache[sm_blk[:q_len].long(), sm_pos[:q_len].long(), 0, :]
                    ck, _ = _ref_decode_kv(cbytes[:, :vo], cbytes[:, vo:], fmt_, d_)
                    kk_ = k_seq[:, 0, :].float()
                    chunk_diff = float((ck - kk_).abs().max())
                    # PER-KV-HEAD store SNR (trace24): every prior store
                    # check hardcoded head 0. Relative-L2 of stored-vs-raw
                    # per kv head, K and V — the real-tensor noise floor of
                    # the MX store (e4m3 theory ~6% worst, ~2-3% RMS) and a
                    # per-head store defect in one number.
                    snr_ = []
                    for kvh_ in range(hk_):
                        cb_ = kv_cache[sm_blk[:q_len].long(), sm_pos[:q_len].long(), kvh_, :]
                        ck_, cv_ = _ref_decode_kv(cb_[:, :vo], cb_[:, vo:], fmt_, d_)
                        rk_ = k_seq[:, kvh_, :].float()
                        rv_ = v_seq[:, kvh_, :].float()
                        snr_.append(
                            f"h{kvh_}K{float((ck_ - rk_).norm() / rk_.norm().clamp_min(1e-6)):.3f}"
                            f"V{float((cv_ - rv_).norm() / rv_.norm().clamp_min(1e-6)):.3f}"
                        )
                    # Row-level match matrix: which raw row does each cache
                    # row hold? (0->1,1->0 = swapped slot order; 0->none =
                    # stale/unwritten bytes — also log max|cache row| so a
                    # zero row (never written) is unmistakable.)
                    pair_ = []
                    for ci in range(q_len):
                        diffs = [
                            float((ck[ci] - kk_[j]).abs().max()) for j in range(q_len)
                        ]
                        bj = min(range(q_len), key=diffs.__getitem__)
                        pair_.append(f"{ci}->{bj}:{diffs[bj]:.3f}")
                    chunk_maxabs = float(ck.abs().max())
                    walk_slots = sm[:q_len].tolist()
                    # Sub-check 2 (dequant kernel): mx_dequant_cache prefix
                    # rows vs the reference decode of the same rows.
                    alloc_ = math.ceil(cached_len / bs_) * bs_
                    pbk = torch.empty(1, hk_, alloc_, d_, dtype=torch.float16, device=q_seq.device)
                    pbv = torch.empty(1, hk_, alloc_, d_, dtype=torch.float16, device=q_seq.device)
                    mx_dequant_cache(kv_cache, bt_row, pbk, pbv, cached_len)
                    prefix_diff = float(
                        (pbk[0, 0, :cached_len].float() - kr_last[:cached_len]).abs().max()
                    )
                    # Per-kv-head prefix dequant check (trace24): head 0
                    # was the only validated head; each kv head's buffer
                    # rows vs the reference walk of the same head.
                    prefix_h = []
                    for kvh_ in range(hk_):
                        if kvh_ not in kvref:
                            continue
                        kr_, vr_ = kvref[kvh_]
                        n_ = min(cached_len, kr_.shape[0], vr_.shape[0])
                        prefix_h.append(
                            f"h{kvh_}K{float((pbk[0, kvh_, :n_].float() - kr_[:n_]).abs().max()):.3f}"
                            f"V{float((pbv[0, kvh_, :n_].float() - vr_[:n_]).abs().max()):.3f}"
                        )
                    split0 = output[q_start:q_end, 0, :].float()
                    bulk0 = ref[:, 0, :].float()
                    # ALL-HEADS audit (P6.47 2026-11-12): the original
                    # check compared head 0 only — a head-indexing defect
                    # in the batch-1 split decode (heads > 0) would be
                    # invisible while corrupting every verify output.
                    # Per-head max |split - bulk| over ALL heads; report
                    # the worst head (both paths read the same pool, so
                    # any per-head disagreement names the broken path).
                    sh = output[q_start:q_end].float()  # (q, Hq, D)
                    bh = ref.float()
                    per_head = (sh - bh).abs().amax(dim=(0, 2))  # (Hq,)
                    worst_head = int(per_head.argmax())
                    # Per-head vs the byte-law oracle: splitref_* names the
                    # SHIPPING path's deviation per head (a split-only
                    # deviation at h>0 = the read-path bug); bulkref_*
                    # separates a bulk-path defect (warmup multi-request
                    # verifies showed bulk off at head 0 in trace19).
                    sp_h = (sh - refall).abs().amax(dim=(0, 2))  # (Hq,)
                    bk_h = (bh - refall).abs().amax(dim=(0, 2))  # (Hq,)
                    # Flash-vs-SDPA discriminator over the SAME assembled
                    # K/V (trace20: bulk deviates ~1 at the MEDIAN head —
                    # too big for the raw-tail explanation; if SDPA over
                    # k_full_/v_full_ matches the oracle while flash over
                    # the same tensors does not, the XPU flash varlen
                    # causal handling for cu_q != cu_k is the defect).
                    q_pos_ = torch.arange(q_len, device=q_seq.device).unsqueeze(1) + cached_len
                    k_pos_ = torch.arange(seq_len, device=q_seq.device).unsqueeze(0)
                    sdpa_ = F.scaled_dot_product_attention(
                        q_seq.transpose(0, 1).unsqueeze(0).float(),
                        k_full_.transpose(0, 1).unsqueeze(0).float(),
                        v_full_.transpose(0, 1).unsqueeze(0).float(),
                        attn_mask=(k_pos_ <= q_pos_),
                        scale=self.scale,
                        enable_gqa=(hk_ < hq_),
                    )[0].transpose(0, 1)
                    sd_h = (sdpa_ - refall).abs().amax(dim=(0, 2))  # (Hq,)
                    fs_h = (bh - sdpa_).abs().amax(dim=(0, 2))  # flash vs sdpa, same inputs
                    md_sl = attn_metadata.seq_lens[i].item()
                    md_sl_cpu = (
                        attn_metadata.seq_lens_cpu[i].item()
                        if attn_metadata.seq_lens_cpu is not None else -1
                    )
                    md_smap = attn_metadata.slot_mapping[-6:].tolist()
                    _trace(
                        "xcheck3",
                        f"q={q_len}", f"sl={seq_len}", f"cl={cached_len}", f"hk={hk_}", f"fmt={fmt_}",
                        f"md_sl={md_sl}", f"md_slcpu={md_sl_cpu}",
                        f"nat={attn_metadata.num_actual_tokens}",
                        f"md_smap={md_smap}",
                        f"bt0={bt_row[0, :2].tolist()}",
                        f"split_vs_ref={float((split0 - ref0).abs().max()):.6f}",
                        f"bulk_vs_ref={float((bulk0 - ref0).abs().max()):.6f}",
                        f"allheads_worst={float(per_head.max()):.6f}",
                        f"allheads_h={worst_head}",
                        f"allheads_med={float(per_head.median()):.6f}",
                        f"splitref_worst={float(sp_h.max()):.6f}",
                        f"splitref_h={int(sp_h.argmax())}",
                        f"splitref_med={float(sp_h.median()):.6f}",
                        f"bulkref_worst={float(bk_h.max()):.6f}",
                        f"bulkref_h={int(bk_h.argmax())}",
                        f"bulkref_med={float(bk_h.median()):.6f}",
                        f"sdparef_worst={float(sd_h.max()):.6f}",
                        f"sdparef_med={float(sd_h.median()):.6f}",
                        f"flash_vs_sdpa={float(fs_h.max()):.6f}",
                        f"chunk_cache_vs_raw={chunk_diff:.6f}",
                        f"chunk_snr={';'.join(snr_)}",
                        f"prefix_heads={';'.join(prefix_h)}",
                        f"chunk_pair={';'.join(pair_)}",
                        f"chunk_maxabs={chunk_maxabs:.3f}",
                        f"walk_slots={walk_slots}",
                        f"prefix_dequant_vs_ref={prefix_diff:.6f}",
                        f"id={id(self)}",
                    )
            elif q_len <= _CONTINUATION_DECODE_THRESHOLD:
                _trace("path", f"i={i}", "synth", f"q={q_len}", f"sl={seq_len}")
                # Multi-row synthetic decode (eager chunked prefill:
                # q_len in (8, 128] is never a verify capture shape —
                # those are q_len = k+1 <= 8 and take the split path).
                # Host ints are correct in eager mode.
                synth_seq_lens = (
                    attn_metadata.seq_lens[i] - q_len
                ) + _ac[1 : q_len + 1]
                synth_bt = bt_row.expand(q_len, -1)
                if self.sliding_window:
                    # P-C2: windowed synthetic decode rides native
                    # (eager-only path; the Triton kernel is
                    # window-blind). Rows share one block table; the
                    # per-row synth limits carry each row's true
                    # position — the window band rides them in the
                    # kernel.
                    out = mx_decode_native(
                        q_seq,
                        kv_cache,
                        synth_bt,
                        synth_seq_lens,
                        self.scale,
                        attn_metadata.max_seq_len,
                        self.mx_config.fmt,
                        sliding_window=self.sliding_window,
                    )
                else:
                    out = mx_decode(
                        q_seq,
                        kv_cache,
                        synth_bt,
                        synth_seq_lens.to(attn_metadata.seq_lens.dtype),
                        self.scale,
                    )
                output[q_start:q_end] = out.to(query.dtype)
            else:
                _trace("path", f"i={i}", "bulk", f"q={q_len}", f"sl={seq_len}")
                # Large continuation: dequant the cached prefix once,
                # flash over cached+chunk (the stock TQ structure, MX
                # dequant, no inverse rotation).
                out = self._continuation_prefill(
                    layer,
                    q_seq,
                    k_seq,
                    v_seq,
                    kv_cache,
                    bt_row,
                    cached_len,
                    seq_len,
                    Pi,
                    centroids,
                )
                output[q_start:q_end] = out.to(query.dtype)

        if split_rows:
            self._split_verify_decode(
                query, output, kv_cache, attn_metadata, split_rows
            )

        return output

    def _split_verify_decode(
        self,
        query: torch.Tensor,  # (N, Hq, D)
        output: torch.Tensor,  # (N, Hq, D) — split rows written in place
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot)
        attn_metadata: TurboQuantMetadata,
        split_rows: list[tuple[int, int, int]],
    ) -> None:
        """ONE mx_decode for every split-row request in the step.

        The MTP verify hot shape (spec=3 -> q_len=k+1 rows per request,
        every decode step) previously ran a per-request slot-inversion
        cascade plus one mx_decode per row, per layer — the dominant
        decode-gap term (~2k kernels/step at concurrency 8 vs ~80 here,
        A10 telemetry directive 2026-11-12). Same LIMIT LAW math as the
        per-row path: each row's causal limit comes from ITS OWN slot
        via the block-table inversion, computed once per step
        (layer-invariant) and cached on self keyed to the metadata's
        identity + slot_mapping version, and the recorded device ops
        re-read the live slot_mapping/block_table at XPU-graph replay
        (the same refresh law as the seq_lens reads the inversion
        replaces).
        """
        plan = None
        smeta = attn_metadata.slot_mapping
        # One-step cache on self (the metadata may be pooled/reused by
        # the wheel, so attaching to it could serve a STALE plan — that
        # would poison every limit in the step). Key on metadata identity
        # plus the slot_mapping buffer identity/version plus the split
        # shape (a pooled-metadata re-CAPTURE with the same buffer and
        # unbumped version must not reuse a differently-shaped plan), and
        # hold the metadata reference so its id cannot recycle under the
        # cache.
        key = (
            id(attn_metadata),
            smeta.data_ptr(),
            smeta._version,
            tuple(split_rows),
        )
        cached = getattr(self, "_mx_split_cache", None)
        if cached is not None and cached[0] == key:
            plan = cached[1]
        if plan is None:
            bt = attn_metadata.block_table
            stride = kv_cache.shape[1]
            # CAPTURE LAW: this runs inside XPU-graph capture (the plan-
            # build kernels are what replay refreshes the limits with).
            # Every op here must be a device op — torch.tensor(<py list>)
            # is a pageable H2D memcpy whose event wait is illegal under
            # capture ("wait method cannot be used for an event associated
            # with a command graph", m8 cert25y boot 2026-11-12). Build the
            # request-index the same way as idx: device fills whose scalars
            # ride as kernel args, no memcpy.
            idx_parts = []
            req_parts = []
            for i, qs, ql in split_rows:
                idx_parts.append(torch.arange(qs, qs + ql, device=bt.device))
                req_parts.append(
                    torch.full((ql,), i, device=bt.device, dtype=torch.long)
                )
            idx = torch.cat(idx_parts)
            req_idx = torch.cat(req_parts)
            sm = smeta.index_select(0, idx)
            bt_rows = bt.index_select(0, req_idx)
            sm_blk = torch.div(sm, stride, rounding_mode="floor")
            sm_pos = sm - sm_blk * stride
            # bt walk inversion (each row against its OWN request's
            # blocks): the slot's block index in bt = position //
            # block_size (blocks appear once per row).
            blk_idx = (
                (bt_rows == sm_blk.unsqueeze(1))
                .to(torch.int32)
                .argmax(dim=1)
                .to(sm.dtype)
            )
            sl_rows = blk_idx * stride + sm_pos + 1
            # v4/v5 (A11 levers 2-3): when every split request carries the
            # MTP verify shape (q_len = R_TILE rows), ONE walk per request
            # can serve all its rows — group block-table rows, same
            # per-row limits. Device ops only (capture law above). The
            # decoder choice rides the plan (per-format default at module
            # top; env overrides win). Logged once per process per format
            # — every leg's serve log carries the engagement marker the
            # A/B banks cite (the posture-line pattern).
            _g = query.shape[1] // kv_cache.shape[2]
            if _NATIVE and _g > 16:
                logger.warning(
                    "MXKV native refused: G=%d exceeds the q16 policy "
                    "bucket — the Triton lanes serve this shape",
                    _g,
                )
            # r212a: walked-token budget vs the per-FORMAT, per-LANE E5b
            # crossing — below it the native kernel's fixed barrier/walk
            # term dominates and the certified Triton lane is faster.
            # The lane split rides the same G knee as the g-adaptive
            # chain below. Estimator: rows x per-request KV CAPACITY
            # (bt width x block stride), NOT metadata.max_seq_len —
            # decode replays never re-run this python (the branch
            # freezes per captured graph), and the capture dummy
            # dilutes max_seq_len across the bucket's requests, while
            # bt.shape[1]*stride tracks the BUCKET's token class: the
            # runtime step maps to the same bucket by sum(num_blocks),
            # so the frozen side is the replayed side (r212a fastcycle
            # law, 2026-11-21: the max_seq_len branch froze triton on
            # every graph via the rows=148/max_seq_len=9 dummy). Eager
            # steps get rows x max per-req blocks — an upper bound of
            # the true walked sum (native engages at-or-before the
            # crossing; E5b boundary margins absorb it).
            _minpair = _NATIVE_MINLEN.get(self.mx_config.fmt)
            _minlen = (
                _minpair[1]
                if (
                    _minpair
                    and len(split_rows) >= _V5_G_MIN
                )
                else (_minpair[0] if _minpair else None)
            )
            _walked_cap = idx.shape[0] * bt.shape[1] * stride
            _native_ctx_ok = (
                _minlen is not None and _walked_cap >= _minlen
            )
            decoder = (
                "native"
                if _NATIVE
                and _g <= 16
                # P-C2: windowed impls ride native at ANY walk length —
                # the Triton lanes are window-blind and the E5b
                # crossing is a perf choice correctness forbids.
                and (_native_ctx_ok or self.sliding_window)
                else _DECODER_OVERRIDE or (
                    "v5"
                    if _V5_DOT
                    else (
                        "v4"
                        if _V4_VERIFY
                        else (
                            "v5"
                            if self.mx_config.fmt in _V5_DEFAULT
                            else (
                                # G-adaptive mxfp4 (B1 verdict): v2 below the
                                # knee (n1 — the v2 batch grid wins at G=1),
                                # v5 at/above it (n8 long-ctx — 32K ss -14%).
                                "v5"
                                if (
                                    self.mx_config.fmt == "mxfp4"
                                    and len(split_rows) >= _V5_G_MIN
                                )
                                else "v2"
                            )
                        )
                    )
                )
            )
            # P-C2 (2026-11-25): the Triton decode lanes carry no
            # window surface — a sliding impl that lands on one would
            # serve window-blind (gemma sliding layers at ctx > W
            # silently attend the whole context). Refuse loudly; the
            # native kernel owns every windowed decode. Fires at
            # plan-build time, so graph capture dies here too —
            # nothing replays a window-blind plan.
            if self.sliding_window and decoder != "native":
                raise NotImplementedError(
                    "MXKV sliding_window="
                    f"{self.sliding_window} requires the native decode "
                    "lane (ARC_MXKV_NATIVE=1 and the format's E5b "
                    "crossing reached); the Triton lanes are "
                    "window-blind — refusing rather than serving "
                    "unmasked"
                )
            # r212a posture line (once per fmt per SIDE per process —
            # the engagement marker A/B banks cite; the cap value
            # identifies the bucket class that froze each side).
            _dside = "native" if _native_ctx_ok else "triton"
            if (
                _NATIVE
                and _g <= 16
                and (self.mx_config.fmt, f"native-dispatch-{_dside}")
                not in _DECODER_LOGGED
            ):
                _DECODER_LOGGED.add(
                    (self.mx_config.fmt, f"native-dispatch-{_dside}"))
                logger.info(
                    "MXKV native dispatch fmt=%s rows=%d walked_cap=%d "
                    "minlen=%d -> %s (E5b crossover)",
                    self.mx_config.fmt,
                    idx.shape[0],
                    _walked_cap,
                    _minlen or -1,
                    _dside,
                )
            _dkey = (self.mx_config.fmt, decoder)
            if _dkey not in _DECODER_LOGGED:
                _DECODER_LOGGED.add(_dkey)
                logger.info(
                    "MXKV verify decoder=%s fmt=%s source=%s",
                    decoder,
                    self.mx_config.fmt,
                    (
                        "env-override"
                        if _DECODER_OVERRIDE
                        else (
                            "native-env"
                            if decoder == "native"
                            else (
                                "env-flag"
                                if (_V5_DOT or _V4_VERIFY)
                                else (
                                    "g-adaptive"
                                    if self.mx_config.fmt == "mxfp4"
                                    else "format-default"
                                )
                            )
                        )
                    ),
                )
            bt_groups = None
            if decoder not in ("v2", "native") and all(
                ql == _V4_R_TILE for _, _, ql in split_rows
            ):
                bt_groups = bt.index_select(
                    0,
                    torch.arange(
                        len(split_rows), device=bt.device, dtype=torch.long
                    ),
                )
            plan = (idx, bt_rows, sl_rows, bt_groups, decoder)
            self._mx_split_cache = (key, plan, attn_metadata)
        idx, bt_rows, sl_rows, bt_groups, decoder = plan
        _trace("path", "batch", "splitbatch", f"R={idx.shape[0]}")
        q_rows = query.index_select(0, idx)
        if decoder == "native":
            out = mx_decode_native(
                q_rows,
                kv_cache,
                bt_rows,
                sl_rows,
                self.scale,
                attn_metadata.max_seq_len,
                self.mx_config.fmt,
                # P-C2: the impl's window width rides the masked twin
                # ((W-1, 0) inside the bridge); None on windowless
                # impls (qwen target, gemma full_attention layers).
                sliding_window=self.sliding_window,
            )
            if _ab_ok() and not self.sliding_window:
                # P-C2: the A/B's Triton ref is window-blind — a
                # windowed impl would false-alarm on the unmasked
                # reference (the offline window oracle owns the
                # masked-lane correctness verdict).
                try:
                    _mx_ab(self, q_rows, kv_cache, bt_rows, sl_rows, out)
                except Exception as exc:  # noqa: BLE001 — never kill serving
                    logger.error("MXKV AB failed: %r", exc)
        elif bt_groups is not None:
            if decoder == "v5":
                out = mx_decode_rows_dot(
                    q_rows,
                    kv_cache,
                    bt_groups,
                    sl_rows,
                    self.scale,
                    r_tile=_V4_R_TILE,
                )
            else:
                out = mx_decode_rows(
                    q_rows,
                    kv_cache,
                    bt_groups,
                    sl_rows,
                    self.scale,
                    r_tile=_V4_R_TILE,
                )
        else:
            out = mx_decode(
                q_rows,
                kv_cache,
                bt_rows,
                sl_rows,
                self.scale,
            )
        output.index_copy_(0, idx, out.to(query.dtype))

    def _continuation_prefill(
        self,
        layer: Any,
        query: torch.Tensor,  # (q_len, Hq, D)
        key_chunk: torch.Tensor,  # (q_len, Hk, D)
        val_chunk: torch.Tensor,  # (q_len, Hk, D)
        kv_cache: torch.Tensor,  # (num_blocks, block_size, Hk, slot)
        block_table: torch.Tensor,  # (1, max_num_blocks)
        cached_len: int,
        seq_len: int,
        Pi: torch.Tensor,
        centroids: torch.Tensor,
        return_kv: bool = False,
    ) -> torch.Tensor:
        """Dequant the cached MX prefix, concatenate the live chunk, flash."""
        q_len, hq, d = query.shape
        hk = key_chunk.shape[1]
        device = query.device
        block_size = kv_cache.shape[1]

        # Block-padded dequant buffers from the shared workspace (the
        # stock TQ law: one set across layers, pre-sized for capture).
        alloc_len = math.ceil(cached_len / block_size) * block_size
        buf_shape = (1, hk, alloc_len, d)
        k_buf, v_buf = current_workspace_manager().get_simultaneous(
            (buf_shape, torch.float16),
            (buf_shape, torch.float16),
        )
        mx_dequant_cache(kv_cache, block_table, k_buf, v_buf, cached_len)

        # MX quantizes in the native space — no inverse rotation.
        k_cached_trim = k_buf[0, :, :cached_len, :].transpose(0, 1)
        v_cached_trim = v_buf[0, :, :cached_len, :].transpose(0, 1)

        qdtype = query.dtype
        k_full = torch.empty(seq_len, hk, d, dtype=qdtype, device=device)
        v_full = torch.empty(seq_len, hk, d, dtype=qdtype, device=device)
        k_full[:cached_len] = k_cached_trim.to(qdtype)
        k_full[cached_len:] = key_chunk
        v_full[:cached_len] = v_cached_trim.to(qdtype)
        v_full[cached_len:] = val_chunk

        # q_len queries over seq_len K/V, causal at cached_len offset.
        cu_q = self._cu_2
        cu_q[1:2] = q_len
        if not hasattr(self, "_cu_2_k"):
            self._cu_2_k = torch.zeros(2, device=device, dtype=torch.int32)
        self._cu_2_k[1:2] = seq_len
        out: torch.Tensor
        if _tq._HAS_FLASH_ATTN:
            out = self._flash_attn_varlen(
                q=query,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=self._cu_2_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        else:
            # SDPA fallback with the causal-at-offset mask (stock TQ shape).
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos
            if self.sliding_window:
                # P-C2: window band on the fallback — allowed keys ride
                # [pos - W + 1, pos] (the same (W-1, 0) pair as the
                # flash lane).
                mask = mask & (k_pos >= q_pos - (self.sliding_window - 1))
            out = F.scaled_dot_product_attention(
                query.transpose(0, 1).unsqueeze(0),
                k_full.transpose(0, 1).unsqueeze(0),
                v_full.transpose(0, 1).unsqueeze(0),
                attn_mask=mask,
                scale=self.scale,
                enable_gqa=(hk < hq),
            )
            out = out[0].transpose(0, 1)
        if return_kv:
            # Trace-only: hand the assembled K/V back so the caller can
            # run an exact-semantics reference over the SAME inputs the
            # flash kernel saw (xcheck3 flash-vs-sdpa discriminator).
            return out, k_full, v_full
        return out
