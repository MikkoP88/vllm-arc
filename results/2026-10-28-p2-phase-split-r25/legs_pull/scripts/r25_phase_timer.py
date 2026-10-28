"""r25 phase timer (v2 runner) — DIAGNOSTIC ONLY (not merged; evidence leg).

vLLM 0.30 runs Model Runner V2 here (XPUModelRunnerV2 = vllm.v1.worker.gpu
.model_runner.GPUModelRunner; no V1-fallback warning in the boot log). The
V2 async step is SPLIT: execute_model prepares inputs + launches the target
forward (replay) and stashes execute_model_state; the continuation
sample_tokens (async output path) runs sample+verify, the MTP drafter
(speculator.propose, its own workspace lane), and postprocess/copies.

Wraps each phase with XPU event pairs; logs per-phase GPU-stream elapsed
p50s every 96 steps. elapsed between in-order-stream markers includes
upstream waits — the point: locate the compile-independent ~63 ms c1 /
~77 ms c8 step wall (r23/r24 laws). Upstream ships the same pattern
(StepTimingCollector, gpu/async_utils.py:48) but only wires it to
adaptive-verification capture profiling — this leg wires it to steady state.

Phases: forward=execute_model | tail=sample_tokens | sample=Runner.sample
(sample+verify) | draft=speculator.propose | tailres=tail-sample-draft
(postprocess, async copies, bookkeeping).
"""
from __future__ import annotations

import statistics
from collections import deque

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc.r25")

_N = 96
_installed = False
_spec_wrapped = False


class _Timer:
    # cur[phase] = list of [e0, e1] pairs — a phase may run more than once
    # per step; log-time sums per step. Cross-phase stream gaps are paired
    # into pseudo-phases: fgap = forward_end->tail_start (the async-output
    # handoff wait — r22a's 58 ms sync lives here), bgap = tail_end->
    # next forward_start (scheduler/launch idle before the next step).
    def __init__(self) -> None:
        self.ring: deque[dict] = deque(maxlen=_N)
        self.cur: dict | None = None
        self.count = 0
        self._fw_end: object | None = None
        self._tail_end: object | None = None

    def start(self, phase: str):
        if self.cur is None:
            self.cur = {}
        e = torch.xpu.Event(enable_timing=True)
        e.record()
        if phase == "forward" and self._tail_end is not None:
            self.cur.setdefault("bgap", []).append([self._tail_end, e])
        if phase == "tail" and self._fw_end is not None:
            self.cur.setdefault("fgap", []).append([self._fw_end, e])
        self.cur.setdefault(phase, []).append([e, None])

    def end(self, phase: str):
        c = self.cur
        if c is not None and c.get(phase) and c[phase][-1][1] is None:
            e = torch.xpu.Event(enable_timing=True)
            e.record()
            c[phase][-1][1] = e
            if phase == "forward":
                self._fw_end = e
            elif phase == "tail":
                self._tail_end = e

    def close_step(self):
        if self.cur:
            self.ring.append(self.cur)
            self.count += 1
            if self.count % _N == 0:
                self._log()
        self.cur = None

    def _log(self):
        torch.xpu.synchronize()
        # per-step per-phase totals (sum over the step's call pairs)
        per_step: list[dict[str, float]] = []
        for step in self.ring:
            tot: dict[str, float] = {}
            for ph, pairs in step.items():
                s = 0.0
                for e0, e1 in pairs:
                    if e1 is None:
                        continue
                    try:
                        s += e0.elapsed_time(e1)
                    except Exception:  # noqa: BLE001
                        pass
                if s > 0:
                    tot[ph] = s
            per_step.append(tot)
        inner = ("forward", "sample", "draft")
        parts = []
        for ph in ("tail",) + inner + ("fgap", "bgap"):
            v = sorted(t[ph] for t in per_step if ph in t)
            if not v:
                continue
            parts.append(f"{ph} p50={statistics.median(v):.1f} "
                         f"p90={v[int(len(v) * 0.9)]:.1f} max={v[-1]:.1f}")
        # tail residual: tail minus sample/draft inside it (postprocess,
        # async copies, bookkeeping) — only meaningful when tail measured
        res = []
        for t in per_step:
            if "tail" in t:
                r = t["tail"] - sum(t.get(p, 0.0) for p in ("sample", "draft"))
                if r > -1.0:
                    res.append(max(r, 0.0))
        if res:
            res.sort()
            parts.append(f"tailres p50={statistics.median(res):.1f} "
                         f"max={res[-1]:.1f}")
        logger.info("[arc-r25] phase split n=%d | %s", self.count,
                    " | ".join(parts))


TIMER = _Timer()


def _wrap_method(cls, name, phase):
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_arc_r25", False):
        return False

    def wrapped(self, *args, **kwargs):
        TIMER.start(phase)
        try:
            return orig(self, *args, **kwargs)
        finally:
            TIMER.end(phase)

    wrapped._arc_r25 = True
    setattr(cls, name, wrapped)
    return True


def _wrap_speculator() -> bool:
    global _spec_wrapped
    if _spec_wrapped:
        return True
    try:
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import (
            MTPSpeculator,
        )
        ok = _wrap_method(MTPSpeculator, "propose", "draft")
        _spec_wrapped = ok
        return ok
    except Exception:  # noqa: BLE001 — not built yet / circular import
        return False


def _install() -> bool:
    global _installed
    if _installed:
        return True
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    _wrap_method(GPUModelRunner, "execute_model", "forward")
    _wrap_method(GPUModelRunner, "sample_tokens", "tail")
    _wrap_method(GPUModelRunner, "sample", "sample")
    _wrap_speculator()

    orig_tail = GPUModelRunner.sample_tokens

    def sample_tokens(self, *args, **kwargs):
        try:
            if not _spec_wrapped:
                _wrap_speculator()
            return orig_tail(self, *args, **kwargs)
        finally:
            TIMER.close_step()

    sample_tokens._arc_r25 = True
    GPUModelRunner.sample_tokens = sample_tokens
    _installed = True
    logger.info("[arc-r25] v2 phase timer installed (forward/tail/sample/"
                "draft; logs every %d steps)", _N)
    return True
