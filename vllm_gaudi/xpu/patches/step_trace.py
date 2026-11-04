# SPDX-License-Identifier: Apache-2.0
"""step_trace.py — p620 fork patch (vllm_gaudi/xpu/patches/).

Engine-side per-step host timeline (plan P6.20 lane L2). Diagnostic
class: silent-off by default (the ARC_SNAP_PROBE law — documented in
docs/envs.md diagnostics, deliberately NOT in the envs.py registry),
zero cost when off (nothing is wrapped, not even an env read past the
install gate).

Why this exists (plan P6.20 scoping, 2026-11-04): the banked step
decompositions cover decode faces (p10/r19-r22a: ~80 ms steady step,
~50 ms CPU-serialized in the GDN metadata build — since fixed) and
device-side slices (r67b torch-profiler: AR 30.4 % / FA 20.3 % of
rank0 device time at U32k), but no instrument decomposes the PREFILL
step's host path end-to-end — scheduler -> state update -> input
prep -> forward submission -> sample -> draft -> the LMCache
wait_for_save tail (flush + barrier + D2H gathers + store) -> output
handoff. P6.20's S1 finding puts a device-wide barrier + store work
inside every step's wait_for_save
(lmcint/vllm_v1_adapter.py::_lmc_flush_pending), so the flush segment
is measured explicitly (pending-store count included) alongside the
step total.

Design (r30/r35 wrap pattern — class-level monkey wraps, re-entrant,
never changes behavior):
  * every segment records wall time around the ORIGINAL call and
    appends one JSONL line  {"ts", "rank", "step", "seg", "dt_ms",
    "n": ...}  to ARC_STEP_TRACE_FILE (default
    /tmp/arc_step_trace_rank{rank}.jsonl);
  * "step" is a module counter bumped on each sample_tokens entry —
    the per-step boundary every other segment nests inside;
  * files are line-buffered appends; ~10 lines/step at <=20 steps/s
    is negligible I/O; off = no file, no wraps;
  * wraps are defensive (missing attr logs once and skips); the
    LMCache adapter class is wrapped LAZILY at the first traced step —
    by then lmcache is fully imported and the ENGINE'S adapter module
    (the vendored lmcache_integration.vllm_v1_adapter our lmcint kit
    source-overrides) can be entered safely (importing it at
    platform-register time would re-enter a partially initialized
    lmcache — the r42 circular-import death the platform loader guards
    against).

Segments wrapped (CORE classes — the Arc engine lineage; the
vllm_gaudi hpu_* runner needs habana_frameworks and never loads on the
Arc image, r80 lesson):
  schedule                      core Scheduler / AsyncScheduler .schedule
  update_states                 GPUModelRunner._update_states
  prep_inputs                   GPUModelRunner._prepare_inputs (unified)
  execute_model                 GPUModelRunner.execute_model (submit wall)
  sample_tokens                 GPUModelRunner.sample_tokens (STEP)
  draft                         GPUModelRunner.propose_draft_token_ids
  wfs                           GPUModelRunner.finalize_kv_connector (the
                                spec-decode-deferred wait_for_save tail)
  wfs_flush                     adapter _lmc_flush_pending (+n pending)
  wfs_body                      adapter _lmc_flush_body (arc-lmc-store
                                worker cost under p620 L3 async store)
  wfs_total                     adapter wait_for_save
  get_finished                  adapter get_finished

Provenance: vllm-arc P6.20 (evidence tree
results/2026-11-04-p620-prefill-opt/); wrap idiom = the r30 cap
sitecustomize pattern productized in vllm_gaudi/xpu/platform.py
(_XPU_PATCH_MODULES), segment decomposition style = p10
(r19-r22a step decomposition).
"""

import functools
import json
import os
import time

_TRACE = bool(os.environ.get("ARC_STEP_TRACE"))
_TRACE_FILE = os.environ.get(
    "ARC_STEP_TRACE_FILE"
) or "/tmp/arc_step_trace_r{}_p{}.jsonl"  # rank + pid: APIServer, EngineCore
# and each TP worker each get their own file (RANK is unset in v1 workers,
# so pid is the differentiator; r80 lesson — one shared name interleaved
# every process into a single JSONL).

_step = [0]
_fh = None
_rank = None
_missing: set[str] = set()
_lmc_wrapped = [False]


def _emit(seg: str, dt_ms: float, n=None) -> None:
    global _fh
    if _fh is None:
        try:
            _fh = open(
                _TRACE_FILE.format(_rank, os.getpid()), "a", buffering=1)
        except OSError:
            return  # never block serving on trace I/O
    rec = {"ts": round(time.perf_counter(), 6), "rank": _rank,
           "step": _step[0], "seg": seg, "dt_ms": round(dt_ms, 3)}
    if n is not None:
        rec["n"] = n
    _fh.write(json.dumps(rec) + "\n")


def _wrap(cls, name: str, seg: str, count_attr=None, bump=False):
    """Wrap ``cls.name``; when ``count_attr`` is given, read it off the
    bound self BEFORE the call (pending-store count for the flush).
    ``bump`` marks the step-boundary segment (counter increment)."""
    key = f"{cls.__name__}.{name}"
    orig = getattr(cls, name, None)
    if orig is None:
        if key not in _missing:
            _missing.add(key)
        return
    if getattr(orig, "_arc_step_traced", False):
        return

    @functools.wraps(orig)
    def traced(self, *args, **kwargs):
        global _step
        if bump:
            _step[0] += 1
        n = None
        if count_attr is not None:
            pend = getattr(self, count_attr, None)
            n = len(pend) if pend is not None else 0
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            _emit(seg, (time.perf_counter() - t0) * 1e3, n)

    traced._arc_step_traced = True  # noqa: SLF001 — idempotence mark
    setattr(cls, name, traced)


def _wrap_v2(cls, name: str, seg: str, count_attr=None, bump=False):
    """V2-runner variant of _wrap: on the step-boundary segment also fire
    the lazy lmcint wrap (the V1 hook rode the maybe_wait_for_kv_save
    static, which the V2 path never calls — r81 lesson)."""
    key = f"{cls.__name__}.{name}"
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_arc_step_traced", False):
        if orig is None and key not in _missing:
            _missing.add(key)
        return

    import functools as _ft  # local: keep the V2 helper self-contained

    @_ft.wraps(orig)
    def traced(self, *args, **kwargs):
        if bump:
            _step[0] += 1
        n = None
        if count_attr is not None:
            pend = getattr(self, count_attr, None)
            n = len(pend) if pend is not None else 0
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            if bump:
                _lmc_lazy_wrap()
            _emit(seg, (time.perf_counter() - t0) * 1e3, n)

    traced._arc_step_traced = True
    setattr(cls, name, traced)


def _wrap_static(cls, name: str, seg: str):
    """Wrap a @staticmethod preserving its no-self contract."""
    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_arc_step_traced", False):
        return

    @functools.wraps(orig)
    def traced(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            _lmc_lazy_wrap()
            _emit(seg, (time.perf_counter() - t0) * 1e3)

    traced._arc_step_traced = True
    setattr(cls, name, staticmethod(traced))


def _wrap_maybe_async(cls, name: str, seg: str):
    """Wrap ``cls.name`` whether it is a plain def or an async def —
    ``AsyncScheduler.schedule`` is async; the dt must cover the AWAITED
    body, not coroutine creation."""
    import inspect

    orig = getattr(cls, name, None)
    if orig is None or getattr(orig, "_arc_step_traced", False):
        return
    if not inspect.iscoroutinefunction(orig):
        _wrap(cls, name, seg)
        return

    @functools.wraps(orig)
    async def traced(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return await orig(self, *args, **kwargs)
        finally:
            _emit(seg, (time.perf_counter() - t0) * 1e3)

    traced._arc_step_traced = True
    setattr(cls, name, traced)


def _lmc_lazy_wrap():
    """First-wfs hook: wrap the LMCache adapter class. Safe only here
    (post-boot, lmcache fully imported — see module docstring).

    The class is taken from the ENGINE'S module path: the lmcint kit is a
    source-override kit (lmcache_hma_connector._SrcOverrideFinder exec's
    our file's source into the vendored
    vllm...lmcache_integration.vllm_v1_adapter namespace), so the engine's
    LMCacheConnectorV1Impl is bound there — importing the vllm_gaudi patch
    module directly would define a second, engine-invisible class (r82
    lesson: wraps landed, wfs_flush never fired)."""
    if _lmc_wrapped[0]:
        return
    _lmc_wrapped[0] = True
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter import (
            LMCacheConnectorV1Impl,
        )
        assert hasattr(LMCacheConnectorV1Impl, "_lmc_flush_pending"), (
            "stock lmcache_integration shape — no arc flush body")
    except Exception:  # noqa: BLE001 — non-LMCache / stock-shape boots
        return
    _wrap(LMCacheConnectorV1Impl, "_lmc_flush_pending", "wfs_flush",
          count_attr="_lmc_pending_stores")
    # arc p620 L3: the body runs on the arc-lmc-store worker when async
    # store is on — its dt is the WORKER's cost (engine cost is the
    # wfs_flush handoff above).
    _wrap(LMCacheConnectorV1Impl, "_lmc_flush_body", "wfs_body")
    _wrap(LMCacheConnectorV1Impl, "wait_for_save", "wfs_total")
    _wrap(LMCacheConnectorV1Impl, "get_finished", "get_finished")
    import logging

    logging.getLogger("vllm.platforms.arc").info(
        "step_trace: LMCache adapter wrapped (module %s)",
        LMCacheConnectorV1Impl.__module__)


def _install(site: str = "serve") -> None:  # noqa: ARG001 — site idiom
    if not _TRACE:
        return
    global _rank
    _rank = os.environ.get("RANK", "0")
    import logging
    _log = logging.getLogger("vllm.platforms.arc")

    # CORE classes — the Arc engine runs core's runner + scheduler lineage
    # (the vllm_gaudi hpu_* modules need habana_frameworks, absent on the
    # Arc image; the r80 lesson). THE ENGINE POSTURE IS V2: XPUWorker
    # instantiates XPUModelRunnerV2 = subclass of
    # vllm.v1.worker.gpu.model_runner.GPUModelRunner (a DIFFERENT class
    # from v1/worker/gpu_model_runner.py's same-named V1 runner — the r81
    # lesson: V1-only wraps emitted nothing in the workers). Wrap BOTH;
    # each wrap idles where its class is not used.
    from vllm.v1.worker.gpu.model_runner import (
        GPUModelRunner as GPUModelRunnerV2,
    )

    _wrap_v2(GPUModelRunnerV2, "sample_tokens", "sample_tokens", bump=True)
    _wrap_v2(GPUModelRunnerV2, "prepare_inputs", "prep_inputs")
    _wrap_v2(GPUModelRunnerV2, "execute_model", "execute_model")
    _wrap_v2(GPUModelRunnerV2, "propose_draft_token_ids", "draft")
    # V2's per-step wfs tail: ActiveKVConnector.post_forward -> adapter
    # wait_for_save (gpu/kv_connector.py:111) — the S1/S2 site.
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    _wrap_v2(ActiveKVConnector, "post_forward", "wfs")

    # V1 runner (non-V2 postures; idle here but kept for posture drift).
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    _wrap(GPUModelRunner, "sample_tokens", "sample_tokens", bump=True)
    _wrap(GPUModelRunner, "_update_states", "update_states")
    _wrap(GPUModelRunner, "_prepare_inputs", "prep_inputs")
    _wrap(GPUModelRunner, "execute_model", "execute_model")
    _wrap(GPUModelRunner, "propose_draft_token_ids", "draft")
    # The per-step wfs tail: spec decode defers the kv-connector finalize
    # (wait_for_save + clear metadata) into sample_tokens — the S1/S2 site.
    _wrap(GPUModelRunner, "finalize_kv_connector", "wfs")

    # Core scheduler lineage — whichever runs emits; the other wrap idles.
    try:
        from vllm.v1.core.sched.scheduler import Scheduler

        _wrap_maybe_async(Scheduler, "schedule", "schedule")
        try:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler

            _wrap_maybe_async(AsyncScheduler, "schedule", "schedule")
        except ImportError:
            pass
    except ImportError:
        pass

    _log.info(
        "ArcXpuPlatform: step_trace INSTALLED (ARC_STEP_TRACE=1, rank %s, "
        "pid %d, file %s)", _rank, os.getpid(),
        _TRACE_FILE.format(_rank, os.getpid()))
