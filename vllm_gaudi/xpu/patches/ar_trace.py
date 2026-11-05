# SPDX-License-Identifier: Apache-2.0
"""ar_trace.py — p622 fork patch (vllm_gaudi/xpu/patches/).

Per-call collective census: the missing message-size telemetry for the
TP2 allreduce family (plan P6.22 lane). r67b sized the slice from the
device side (~30.4 % of rank0 device time at U32k: ring_write_multi +
ReduceCopy + barrier, x792) and p71 proved the transport clean (direct
P2P, 10.23 GB/s both directions, every collective on the device-kernel
`topo sycl` path) — but no instrument ever recorded WHAT SIZES the
collectives carry (the p67 note: "own capture leg needed: message
sizes"). Without sizes the algorithmic question — oneCCL 2-phase ring
vs a one-shot peer-write + fused local add at the P2P floor — cannot
be answered; this patch is that capture leg's in-tree instrument.

Diagnostic class: silent-off by default (the ARC_SNAP_PROBE /
ARC_STEP_TRACE law — documented in docs/envs.md diagnostics,
deliberately NOT in the envs.py registry), zero cost when off (nothing
is wrapped, not even an env read past the install gate).

Design (r30/r35/step_trace wrap pattern — attribute rebind at the
lowest version-stable choke point, re-entrant, never changes behavior):
  * wraps ``torch.distributed.{all_reduce, all_gather_into_tensor,
    reduce_scatter_tensor, barrier}`` — the functions stock vllm calls
    via module attribute (``torch.distributed.all_reduce(...)``), so
    the rebind catches every caller in every platform-hooked process
    (engine + both TP workers); a defensive sweep also rebinds
    by-value references already imported into vllm's communication
    modules at install time;
  * each call appends one JSONL line {"ts", "rank", "pid", "op",
    "numel", "dtype", "bytes", "shape", "caller", "dt_ms"} to
    ARC_AR_TRACE_FILE (default /tmp/arc_ar_trace_r{rank}_p{pid}.jsonl
    — rank + pid: every process gets its own file, the r80 lesson);
  * "dt_ms" is the HOST enqueue wall (c10d XPU collectives enqueue
    async; device time per call comes from the torch-profiler trace
    external-id linkage, merged offline by call ORDER);
  * "caller" = first stack frame outside torch.* (module:function) —
    separates RowParallelLinear / lmcache / mamba paths without
    recording full stacks;
  * files are line-buffered appends; a U32k face costs O(1000) lines —
    negligible I/O; off = no file, no wraps.

Provenance: vllm-arc P6.22 (evidence tree results/2026-11-05-p622-*/);
wrap idiom = the step_trace patch (p620) productizing the r30 cap
sitecustomize pattern in vllm_gaudi/xpu/platform.py
(_XPU_PATCH_MODULES).
"""

import functools
import json
import os
import sys
import time

_TRACE = bool(os.environ.get("ARC_AR_TRACE"))
_TRACE_FILE = os.environ.get(
    "ARC_AR_TRACE_FILE"
) or "/tmp/arc_ar_trace_r{}_p{}.jsonl"  # rank + pid (r80 lesson)

_fh = None
_rank = None

# Collective entry points to census. all_gather (legacy tuple form) is
# wrapped too when present — c10d deprecated it in newer torch but the
# wrap is defensive and simply idles if absent.
_OPS = ("all_reduce", "all_gather_into_tensor", "reduce_scatter_tensor",
        "barrier", "all_gather")


def _emit(op: str, tensor, dt_ms: float, caller: str) -> None:
    global _fh
    if _fh is None:
        try:
            _fh = open(
                _TRACE_FILE.format(_rank, os.getpid()), "a", buffering=1)
        except OSError:
            return  # never block serving on trace I/O
    rec = {"ts": round(time.perf_counter(), 6), "rank": _rank,
           "pid": os.getpid(), "op": op, "caller": caller,
           "dt_ms": round(dt_ms, 3)}
    if tensor is not None:
        try:
            rec["numel"] = tensor.numel()
            rec["dtype"] = str(tensor.dtype).replace("torch.", "")
            rec["bytes"] = tensor.numel() * tensor.element_size()
            rec["shape"] = list(tensor.shape)
        except Exception:  # noqa: BLE001 — malformed arg must not kill AR
            rec["numel"] = None
    _fh.write(json.dumps(rec) + "\n")


def _caller(depth: int) -> str:
    """First frame outside torch.* and outside this module — the calling
    module:function. ``depth=2`` from ``traced``: frame 1 is traced
    itself (this module), frame 2 the real caller."""
    f = sys._getframe(depth)
    hops = 0
    while f is not None and hops < 64:
        mod = f.f_globals.get("__name__", "?")
        if (mod != "torch" and not mod.startswith("torch.")
                and mod != __name__):
            return f"{mod}:{f.f_code.co_name}"
        f = f.f_back
        hops += 1
    return "?"


def _wrap(module, name: str) -> None:
    """Rebind ``module.name`` to a census wrapper (idempotent)."""
    orig = getattr(module, name, None)
    if orig is None:
        return
    if getattr(orig, "_arc_ar_traced", False):
        return

    @functools.wraps(orig)
    def traced(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            tensor = None
            if args and hasattr(args[0], "numel"):
                tensor = args[0]
            _emit(name, tensor, (time.perf_counter() - t0) * 1e3,
                  _caller(2))

    traced._arc_ar_traced = True  # noqa: SLF001 — idempotence mark
    setattr(module, name, traced)


def _rebind_imported_refs() -> None:
    """Modules that did ``from torch.distributed import all_reduce``
    BEFORE our install hold by-value refs that bypass the module
    attribute rebind. Rebind those too — only modules already in
    sys.modules (import-order-defensive; attribute-style callers need
    nothing)."""
    import torch.distributed as dist

    for modname in ("vllm.distributed.communication_op",
                    "vllm.distributed.parallel_state"):
        mod = sys.modules.get(modname)
        if mod is None:
            continue
        for name in _OPS:
            wrapped = getattr(dist, name, None)
            if wrapped is not None and getattr(
                    wrapped, "_arc_ar_traced", False):
                setattr(mod, name, wrapped)


def _install(site: str = "serve") -> None:  # noqa: ARG001 — site idiom
    if not _TRACE:
        return
    global _rank
    import logging

    _rank = os.environ.get("RANK", "0")
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            try:
                _rank = dist.get_rank()
            except Exception:  # noqa: BLE001 — pre-init groups
                pass
    except ImportError:
        logging.getLogger("vllm.platforms.arc").warning(
            "ar_trace: torch.distributed unavailable at %s — skipped", site)
        return

    for name in _OPS:
        _wrap(dist, name)
    _rebind_imported_refs()
    logging.getLogger("vllm.platforms.arc").info(
        "ArcXpuPlatform: ar_trace INSTALLED (ARC_AR_TRACE=1, rank %s, "
        "pid %d, file %s)", _rank, os.getpid(),
        _TRACE_FILE.format(_rank, os.getpid()))
