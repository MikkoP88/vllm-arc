# SPDX-License-Identifier: Apache-2.0
"""W13-AR quantized-transport TP2 all-reduce (p621; line-D donor port).

Provenance: iexperimental W13H staging, rig ``/root/build/w13h/ar_quant.py``
(md5 ``b744f090369a53272b9733c425c9af0a``), banked verbatim at
``results/2026-11-05-p621-arquant/donor/`` with its installer
(``patch_ar_quant_xpu``) and RowParallelLinear call-site hook. Port deltas
(evidence-cited in the plan P6.21 scope review): dtype gate widened to
{bf16, fp16} — our p44 W8A16 activations are fp16 (r67b external-id proof);
capture probe resolved defensively (absent API => lane inert, stock AR);
env names moved to the VLLM_ARC_ namespace (r45g law); wiring reshaped to
this tree's in-place patch architecture (_install() + platform
registration, wrap-and-rebind — the donor's RPL replacement file is not
needed because register() precedes model import and the rebind loop covers
early binders).

Mechanism (W13D law, donor header): on this SYS-topology pair the oneCCL
all-reduce already runs at the P2P link ceiling (9.5-9.7 GB/s measured,
~= the 9.54 GB/s raw copy rate), so no re-wiring of the collective can
beat it, and oneCCL has no fp8 all-reduce op. The only byte-cut left is
precision of the transported payload. all_gather is a COPY collective --
dtype-agnostic, it moves bytes. Each rank keeps its own partial exact
(bf16/fp16) and ships the PEER a per-row-scaled e4m3 copy (half the AR
bytes); every rank reconstructs the sum locally:

    out = x_local (exact) + dequant(q_peer)   [fp32 accumulate]

Only the peer's partial is quantized -- half the noise of quantizing both
sides. Expected error class: e4m3 with per-row dynamic scale, ~1e-3..3e-2
relative on the peer term only. NOT byte-exact by construction; gates
assert semantic equivalence and determinism WITHIN a posture.

Sizing gate: prefill-class tensors only (numel >= VLLM_ARC_AR_QUANT_MIN_ELEMS,
default 2M elements). Decode/verify ARs (<= 256 rows x hidden) stay on the
plain collective -- they are latency-bound. This path must never run inside
an XPU graph capture (it allocates and calls the process group).

Fail-soft: any error on first use sticks the lane off for the process and
forwards to the ORIGINAL all-reduce (_install stashes it on
vllm.distributed as _arc_ar_quant_orig before wrapping, so the fallback
never re-enters the wrapper).

Target (r67b, rank0 device time at U32k): the TP2 allreduce family is
30.4 % (ring 2.25 s + ReduceCopy 2.26 s + barrier 0.15 s, c10d-verified)
-- second-largest slice after the already-optimal W8A16 GEMM (39.7 %,
P6.7 rung C closed negative).

Arming: ``VLLM_ARC_AR_QUANT=1`` (default OFF until the r86 A/B evidence;
adoption flips the default with ``=0`` kill switch, the L3/cert9
precedent). Engagement is greppable: "ar_quant: quantized-transport
all-reduce active" at first qualifying payload.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc")

_installed = False

_ARMED = os.environ.get("VLLM_ARC_AR_QUANT", "0") == "1"
_MIN_ELEMS = int(os.environ.get("VLLM_ARC_AR_QUANT_MIN_ELEMS",
                                str(2 * 1024 * 1024)))
_FAILED = False
_LOGGED = False
_ORIG = None

# Graph-capture probe. Conservative when unavailable: treat as capturing so
# the lane stays on stock AR everywhere rather than ever diverting a capture.
_CAPTURE_PROBE = getattr(torch.xpu, "is_current_stream_capturing", None)

# Per-size gather buffers, cached so a steady prefill chunk rate does not
# churn the allocator (rows are MNBT-stable between drains).
_BUFS: dict[tuple[int, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _capturing() -> bool:
    if _CAPTURE_PROBE is None:
        return True
    try:
        return bool(_CAPTURE_PROBE())
    except Exception:  # noqa: BLE001 - never let the probe itself divert
        return True


def _orig_all_reduce(x: torch.Tensor) -> torch.Tensor:
    global _ORIG
    if _ORIG is None:
        import vllm.distributed as _vd
        _ORIG = getattr(_vd, "_arc_ar_quant_orig", None)
        if _ORIG is None:
            _ORIG = _vd.tensor_model_parallel_all_reduce
    return _ORIG(x)


def _rank_pair() -> tuple[bool, int, int, dist.ProcessGroup | None]:
    try:
        from vllm.distributed import (get_tensor_model_parallel_rank,
                                      get_tensor_model_parallel_world_size,
                                      get_tp_group)
        if get_tensor_model_parallel_world_size() != 2:
            return False, 0, 0, None
        grp = get_tp_group().device_group
        return (True, get_tensor_model_parallel_rank(),
                1 - get_tensor_model_parallel_rank(), grp)
    except Exception:  # noqa: BLE001 - not under TP2, plain AR is correct
        return False, 0, 0, None


def ar_quant_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Drop-in for tensor_model_parallel_all_reduce on TP2 XPU prefill."""
    global _FAILED, _LOGGED
    if _FAILED or not _ARMED:
        return _orig_all_reduce(x)
    ok, rank, peer, grp = _rank_pair()
    if (not ok or x.device.type != "xpu"
            or x.dtype not in (torch.bfloat16, torch.float16)
            or x.numel() < _MIN_ELEMS or _capturing()):
        return _orig_all_reduce(x)
    try:
        rows, hidden = x.reshape(-1, x.shape[-1]).shape
        f = x.reshape(rows, hidden)
        amax = f.abs().float().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = amax / 448.0
        q = (f.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        key = (rows, hidden, x.dtype)
        g = _BUFS.get(key)
        if g is None:
            g = (torch.empty(2, rows, hidden, dtype=torch.uint8,
                             device=x.device),
                 torch.empty(2, rows, dtype=torch.float32, device=x.device))
            _BUFS[key] = g
        g_q, g_s = g
        dist.all_gather_into_tensor(g_q.view(torch.uint8),
                                    q.view(torch.uint8), group=grp)
        dist.all_gather_into_tensor(g_s, scale.reshape(-1), group=grp)
        dq = g_q[peer].view(torch.float8_e4m3fn).float() \
            * g_s[peer].unsqueeze(1)
        out = (f.float() + dq).to(x.dtype)
        if not _LOGGED:
            _LOGGED = True
            logger.warning(
                "ar_quant: quantized-transport all-reduce active "
                "(peer e4m3 per-row, local exact; %dx%d payloads)",
                rows, hidden)
        return out.reshape(x.shape)
    except Exception as e:  # noqa: BLE001 - sticky fallback, never kill
        _FAILED = True
        logger.warning(
            "ar_quant: disabled after failure (%s: %s); plain all-reduce "
            "from now on", type(e).__name__, e)
        return _orig_all_reduce(x)


def _install() -> None:
    """Wrap vllm.distributed.tensor_model_parallel_all_reduce (armed only).

    The original is stashed as vllm.distributed._arc_ar_quant_orig for the
    lane's fallbacks, and rebound in every module that had bound it by
    from-import before this patch ran (later importers bind the wrapped
    name directly). Inert unless VLLM_ARC_AR_QUANT=1.
    """
    global _installed
    if _installed or not _ARMED:
        return
    import vllm.distributed as _vd

    _orig = _vd.tensor_model_parallel_all_reduce
    if getattr(_orig, "_arc_ar_quant", False):
        _installed = True
        return
    setattr(_vd, "_arc_ar_quant_orig", _orig)

    def _arq(x):
        return ar_quant_all_reduce(x)

    _arq._arc_ar_quant = True  # noqa: B010
    _vd.tensor_model_parallel_all_reduce = _arq
    for _m in list(sys.modules.values()):
        if _m is None:
            continue
        if getattr(_m, "tensor_model_parallel_all_reduce", None) is _orig:
            setattr(_m, "tensor_model_parallel_all_reduce", _arq)
    _installed = True
    if _CAPTURE_PROBE is None:
        logger.warning(
            "ar_quant: armed (VLLM_ARC_AR_QUANT=1) but "
            "torch.xpu.is_current_stream_capturing is unavailable on this "
            "build -- lane inert, plain all-reduce everywhere")
    else:
        logger.warning(
            "ar_quant: quantized-transport all-reduce patch installed "
            "(VLLM_ARC_AR_QUANT=1, W13H donor port, p621)")
