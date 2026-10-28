# p33 draft-chain structure probe (r26).
#
# Ground truth needed before the chain design: the r25 split measured the
# draft phase (MTPSpeculator.propose) at 14.1-14.6 ms steady-state with
# micro-compute (k4 = 1 draft prefill + 3 one-token decode steps through a
# single dense mtp layer). Code reading (config.json: mtp.layers.0 has
# self_attn only, no linear_attn) says the draft attn group is dense ->
# FlashAttention builder implements supports_draft_decode_metadata_update
# -> upstream use_fused_multi_step_decode=True -> propose = prefill FULL
# replay + ONE fused FULL replay = 2 replays, not the 4 the r25 note
# assumed. If that holds, the tail is ~2x per-replay fixed cost and the
# p33 lever is chaining step 0 into the fused graph (2 replays -> 1).
#
# This probe instruments, per propose (steady decode only; capture-time
# calls are guarded out by the current-row check):
#   - prefill replay duration (SpeculatorCudaGraphManager.run_fullgraph on
#     the speculator's prefill manager) + call count
#   - decode/fused replay duration + call count
#   - GPU-timeline gap prefill_end -> decode_start (CPU metadata build +
#     prepare_decode_inputs between the replays)
#   - CPU wall of _fused/_multi_step_decode and of _build_uniform_attn_metadata
#   - eager fallback counts (_prefill, _generate_draft)
#   - one-time structure log: use_fused flag, draft attn groups/backends/
#     supports flags, draft model layer types, captured sizes of both
#     speculator graph managers
#
# XPU law: events need enable_timing=True (default returns elapsed 0).
# Timer law: +3.8% wall distortion — probe legs only; battery gates are
# correctness gates (logits unaffected by event recording).
#
# Loaded TWICE via sitecustomize meta-path hooks (speculator module and
# autoregressive.cudagraph_utils module) under a fixed module name so the
# two installs share state; each entry installs only what its import
# context can safely import (partial-module circular-import law: the
# cudagraph_utils hook fires while the speculator module is still
# mid-exec, so it must not import speculator names).

import os
import time
from collections import deque

import torch

_N = 96  # flush window (matches r25 ring)

# ---- shared state ---------------------------------------------------------
_ROWS = deque(maxlen=_N * 4)
_CUR = {"row": None}
_MGR_KIND = {}  # id(manager) -> "prefill" | "decode"
_LOGGED = {"structure": False, "sizes": False}


def _log(msg: str) -> None:
    print(f"[arc-p33] {msg}", flush=True)


def _ev():
    return torch.xpu.Event(enable_timing=True)


def _p50(xs):
    if not xs:
        return -1.0
    xs = sorted(xs)
    return xs[len(xs) // 2]


class _Row:
    __slots__ = (
        "e_in", "e_out", "pin", "dec", "gap", "fused_cpu", "meta_cpu",
        "n_prefill_eager", "n_gen_eager", "n_reqs",
    )

    def __init__(self):
        self.e_in = None
        self.e_out = None
        self.pin = []   # [[e0,e1], ...] prefill-manager fullgraph replays
        self.dec = []   # [[e0,e1], ...] decode-manager fullgraph replays
        self.gap = []   # [[prefill_e1, dec_e0], ...]
        self.fused_cpu = 0.0
        self.meta_cpu = 0.0
        self.n_prefill_eager = 0
        self.n_gen_eager = 0
        self.n_reqs = 0


def _flush() -> None:
    torch.xpu.synchronize()
    rows = [r for r in _ROWS if r.e_in is not None and r.e_out is not None]
    if not rows:
        return
    prop, prefd, decd, gapd, fcpu, mcpu = [], [], [], [], [], []
    n_pin = n_dec = n_pe = n_ge = 0
    for r in rows:
        prop.append(r.e_in.elapsed_time(r.e_out))
        for pair in r.pin:
            prefd.append(pair[0].elapsed_time(pair[1]))
        for pair in r.dec:
            decd.append(pair[0].elapsed_time(pair[1]))
        for pair in r.gap:
            gapd.append(pair[0].elapsed_time(pair[1]))
        if r.fused_cpu > 0:
            fcpu.append(r.fused_cpu * 1000.0)
        if r.meta_cpu > 0:
            mcpu.append(r.meta_cpu * 1000.0)
        n_pin += len(r.pin)
        n_dec += len(r.dec)
        n_pe += r.n_prefill_eager
        n_ge += r.n_gen_eager
    n = len(rows)
    _log(
        f"draft structure n={n} | propose p50={_p50(prop):.2f}ms | "
        f"prefill.rep p50={_p50(prefd):.2f}ms x{n_pin / n:.2f} | "
        f"decode.rep p50={_p50(decd):.2f}ms x{n_dec / n:.2f} | "
        f"gap p50={_p50(gapd):.2f}ms | fused_cpu p50={_p50(fcpu):.2f}ms | "
        f"meta_cpu p50={_p50(mcpu):.2f}ms | "
        f"prefill.eager={n_pe} gen.eager={n_ge}"
    )


def _close_row(row) -> None:
    if row is None or row.e_in is None:
        return
    row.e_out = _ev()
    row.e_out.record()
    _ROWS.append(row)
    if len(_ROWS) % _N == 0:
        _flush()


# ---- entry: speculator-module context -------------------------------------
_SPEC_INSTALLED = False


def install_for_speculator() -> bool:
    global _SPEC_INSTALLED
    if _SPEC_INSTALLED:
        return True
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        AutoRegressiveSpeculator,
    )
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

    _orig_propose = AutoRegressiveSpeculator.propose
    _orig_fused = AutoRegressiveSpeculator._fused_multi_step_decode
    _orig_multi = AutoRegressiveSpeculator._multi_step_decode
    _orig_meta = AutoRegressiveSpeculator._build_uniform_attn_metadata
    _orig_prefill = AutoRegressiveSpeculator._prefill
    _orig_gen = AutoRegressiveSpeculator._generate_draft

    def propose(self, *a, **kw):
        if _CUR["row"] is not None:  # nested propose: don't corrupt state
            return _orig_propose(self, *a, **kw)
        row = _Row()
        _CUR["row"] = row
        row.n_reqs = a[0].num_reqs if a else -1
        row.e_in = _ev()
        row.e_in.record()
        try:
            return _orig_propose(self, *a, **kw)
        finally:
            _close_row(row)
            _CUR["row"] = None

    def fused(self, *a, **kw):
        row = _CUR["row"]
        if row is None:
            return _orig_fused(self, *a, **kw)
        t0 = time.perf_counter()
        try:
            return _orig_fused(self, *a, **kw)
        finally:
            row.fused_cpu += time.perf_counter() - t0

    def multi(self, *a, **kw):
        row = _CUR["row"]
        if row is None:
            return _orig_multi(self, *a, **kw)
        t0 = time.perf_counter()
        try:
            return _orig_multi(self, *a, **kw)
        finally:
            row.fused_cpu += time.perf_counter() - t0

    def meta(self, *a, **kw):
        row = _CUR["row"]
        if row is None:
            return _orig_meta(self, *a, **kw)
        t0 = time.perf_counter()
        try:
            return _orig_meta(self, *a, **kw)
        finally:
            row.meta_cpu += time.perf_counter() - t0

    def prefill(self, *a, **kw):
        row = _CUR["row"]
        if row is not None:
            row.n_prefill_eager += 1
        return _orig_prefill(self, *a, **kw)

    def gen(self, *a, **kw):
        row = _CUR["row"]
        if row is not None:
            row.n_gen_eager += 1
        return _orig_gen(self, *a, **kw)

    AutoRegressiveSpeculator.propose = propose
    AutoRegressiveSpeculator._fused_multi_step_decode = fused
    AutoRegressiveSpeculator._multi_step_decode = multi
    AutoRegressiveSpeculator._build_uniform_attn_metadata = meta
    AutoRegressiveSpeculator._prefill = prefill
    AutoRegressiveSpeculator._generate_draft = gen

    # ---- one-time structure facts ----
    _orig_cfg = AutoRegressiveSpeculator._configure_fused_multi_step_decode
    _orig_init_cg = AutoRegressiveSpeculator.init_cudagraph_manager
    _orig_capture = AutoRegressiveSpeculator.capture
    _orig_load = DraftModelSpeculator.load_model

    def cfg(self):
        _orig_cfg(self)
        if _LOGGED["structure"]:
            return
        _LOGGED["structure"] = True
        groups = []
        for gs in getattr(self, "attn_groups", []) or []:
            for g in gs:
                groups.append(
                    (
                        getattr(getattr(g, "backend", None), "__name__", "?"),
                        len(getattr(g, "layer_names", []) or []),
                        bool(
                            getattr(
                                g, "supports_draft_decode_metadata_update", False
                            )
                        ),
                    )
                )
        _log(
            f"structure: use_fused={self.use_fused_multi_step_decode} "
            f"advance_pos={self.advance_draft_positions} "
            f"k={self.num_speculative_steps} "
            f"share_topk={getattr(self, 'share_mtp_topk_indices', None)} "
            f"draft_groups={groups}"
        )

    def init_cg(self, cudagraph_mode):
        _orig_init_cg(self, cudagraph_mode)
        _MGR_KIND[id(self.prefill_cudagraph_manager)] = "prefill"
        _MGR_KIND[id(self.decode_cudagraph_manager)] = "decode"
        _log(f"managers: mode_in={cudagraph_mode} registered (pid={os.getpid()})")

    def capture(self):
        _orig_capture(self)
        if not _LOGGED["sizes"]:
            _LOGGED["sizes"] = True
            try:
                pf = self.prefill_cudagraph_manager.captured_token_counts()
                dc = self.decode_cudagraph_manager.captured_token_counts()
                _log(f"captured sizes: prefill={pf} decode={dc}")
            except Exception as e:  # noqa: BLE001
                _log(f"captured sizes FAILED: {e!r}")

    def load_model(self, target_model):
        _orig_load(self, target_model)
        try:
            m = self.model
            kids = [
                f"{n}:{type(c).__name__}" for n, c in list(m.named_children())[:6]
            ]
            inner = getattr(m, "model", None)
            ikids = (
                [
                    f"{n}:{type(c).__name__}"
                    for n, c in list(inner.named_children())[:8]
                ]
                if inner is not None
                else []
            )
            _log(f"draft model: {type(m).__name__} children={kids} inner={ikids}")
        except Exception as e:  # noqa: BLE001
            _log(f"draft model log FAILED: {e!r}")

    AutoRegressiveSpeculator._configure_fused_multi_step_decode = cfg
    AutoRegressiveSpeculator.init_cudagraph_manager = init_cg
    AutoRegressiveSpeculator.capture = capture
    DraftModelSpeculator.load_model = load_model

    _SPEC_INSTALLED = True
    _log(f"speculator probe installed (pid={os.getpid()})")
    return True


# ---- entry: cudagraph_utils-module context --------------------------------
_MGR_INSTALLED = False


def install_for_cudagraph_utils() -> bool:
    global _MGR_INSTALLED
    if _MGR_INSTALLED:
        return True
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        SpeculatorCudaGraphManager,
    )

    _orig_run = SpeculatorCudaGraphManager.run_fullgraph

    def run_fullgraph(self, desc):
        row = _CUR["row"]
        if row is None:
            return _orig_run(self, desc)
        kind = _MGR_KIND.get(id(self), "other")
        e0 = _ev()
        e0.record()
        try:
            return _orig_run(self, desc)
        finally:
            e1 = _ev()
            e1.record()
            if kind == "prefill":
                row.pin.append([e0, e1])
            elif kind == "decode":
                row.dec.append([e0, e1])
                if row.pin:
                    row.gap.append([row.pin[-1][1], e0])

    SpeculatorCudaGraphManager.run_fullgraph = run_fullgraph
    _MGR_INSTALLED = True
    _log(f"manager probe installed (pid={os.getpid()})")
    return True
