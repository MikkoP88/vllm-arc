#!/usr/bin/env python3
"""patch_p135a.py -- P6.35 leg A: append ARC_P135 telemetry wrappers to the
cert22 engine sources (probe image only; never ships).

Laws applied (banked, results/2026-10-31-p66-r66b-step-decomposition):
1. child processes never deliver print() to serve_full.log -> file-append
   O_APPEND line-atomic (/root/p135_steps.log, shared fs across ranks);
2. async scheduling -> live engine loop is step_with_batch_queue (EngineCore);
3. live runner = vllm/v1/worker/gpu/model_runner.py GPUModelRunner (V2),
   workers inherit gpu_worker.Worker.execute_model/sample_tokens (xpu_worker
   adds no overrides);
4. walls are HOST brackets only (XPU enqueue is async; p66b law #5) -- device
   truth comes from the XPU profiler windows armed by /root/PROF_ON;
5. no .item()/sync added on any device tensor (D2H-sync hot-path trap law).

Anchors verified against cert22 wheel (0.30.0): EngineCore.core.py:105,
Scheduler.schedule sched/scheduler.py:561, GPUModelRunner model_runner.py:187
(execute_model:1631 sample_tokens:1997 sample:1512), MambaHybridModelState
mamba_hybrid.py:77 (preprocess_state:183), CudaGraphManager
cudagraph_utils.py:141 (run_fullgraph:529 run_pw_graph:550), Worker
gpu_worker.py:381 (sample_tokens:1348 execute_model:1355).
"""
import ast
import os
import sys

SRC = "/root/build/p635/probe_src"
OUT = "/root/build/p635/probe_out"

LOGFN = "/root/p135_steps.log"

COMMON = '''
def _p135_log(line):
    try:
        with open("%s", "a") as _f:
            _f.write(line + "\\n")
    except Exception:
        pass
''' % LOGFN

BLOCK_CORE = (
    "\n\n# ---- ARC_P135 probe telemetry (P6.35 decode-step capture; probe image only)\n"
    "import time as _p135_t\n" + COMMON +
    """
_p135_orig_step = EngineCore.step_with_batch_queue


def _p135_step(self):
    _t0 = _p135_t.perf_counter()
    _r = _p135_orig_step(self)
    _p135_log("STEP %.3f" % ((_p135_t.perf_counter() - _t0) * 1e3))
    return _r


EngineCore.step_with_batch_queue = _p135_step
""")

BLOCK_SCHED = (
    "\n\n# ---- ARC_P135 probe telemetry (probe image only)\n"
    "import time as _p135_t\n" + COMMON +
    """
_p135_orig_sched = Scheduler.schedule


def _p135_sched(self, *a, **kw):
    _t0 = _p135_t.perf_counter()
    _r = _p135_orig_sched(self, *a, **kw)
    try:
        _p135_log("SCHED %.3f ntok=%d nreq=%d" % (
            (_p135_t.perf_counter() - _t0) * 1e3,
            getattr(_r, "total_num_scheduled_tokens", -1),
            len(getattr(_r, "num_scheduled_tokens", {}) or {})))
    except Exception:
        _p135_log("SCHED %.3f" % ((_p135_t.perf_counter() - _t0) * 1e3))
    return _r


Scheduler.schedule = _p135_sched
""")

BLOCK_RUNNER = (
    "\n\n# ---- ARC_P135 probe telemetry (probe image only)\n"
    "import time as _p135_t\n" + COMMON +
    """
def _p135_wall(cls, name, tag, rank_arg=False):
    _orig = getattr(cls, name, None)
    if _orig is None:
        return

    def _w(self, *a, **kw):
        _t0 = _p135_t.perf_counter()
        _r = _orig(self, *a, **kw)
        try:
            _p135_log("%s r%s %.3f" % (tag, getattr(self, "rank", "?"),
                                       (_p135_t.perf_counter() - _t0) * 1e3))
        except Exception:
            pass
        return _r

    setattr(cls, name, _w)


_p135_orig_exec = GPUModelRunner.execute_model


def _p135_exec(self, scheduler_output, *a, **kw):
    _t0 = _p135_t.perf_counter()
    _r = _p135_orig_exec(self, scheduler_output, *a, **kw)
    try:
        _p135_log("EXEC r%s %.3f ntok=%d nreq=%d" % (
            getattr(self, "rank", "?"),
            (_p135_t.perf_counter() - _t0) * 1e3,
            getattr(scheduler_output, "total_num_scheduled_tokens", -1),
            len(getattr(scheduler_output, "num_scheduled_tokens", {}) or {})))
    except Exception:
        pass
    return _r


GPUModelRunner.execute_model = _p135_exec
_p135_wall(GPUModelRunner, "prepare_inputs", "PINP")
_p135_wall(GPUModelRunner, "prepare_attn", "PATT")
_p135_wall(GPUModelRunner, "sample_tokens", "SMPL")
_p135_wall(GPUModelRunner, "sample", "SAMP")
""")

BLOCK_MAMBA = (
    "\n\n# ---- ARC_P135 probe telemetry (probe image only)\n"
    "import time as _p135_t\n" + COMMON +
    """
def _p135_wall(cls, name, tag):
    _orig = getattr(cls, name, None)
    if _orig is None:
        return

    def _w(self, *a, **kw):
        _t0 = _p135_t.perf_counter()
        _r = _orig(self, *a, **kw)
        _p135_log("%s %.3f" % (tag, (_p135_t.perf_counter() - _t0) * 1e3))
        return _r

    setattr(cls, name, _w)


_p135_wall(MambaHybridModelState, "preprocess_state", "PMAM")
_p135_wall(MambaHybridModelState, "prepare_attn", "META")
""")

BLOCK_CG = (
    "\n\n# ---- ARC_P135 probe telemetry (probe image only)\n"
    "import time as _p135_t\n" + COMMON +
    """
def _p135_wall(cls, name, tag):
    _orig = getattr(cls, name, None)
    if _orig is None:
        return

    def _w(self, *a, **kw):
        _t0 = _p135_t.perf_counter()
        _r = _orig(self, *a, **kw)
        _p135_log("%s %.3f" % (tag, (_p135_t.perf_counter() - _t0) * 1e3))
        return _r

    setattr(cls, name, _w)


_p135_wall(CudaGraphManager, "run_fullgraph", "RFG")
_p135_wall(CudaGraphManager, "run_pw_graph", "RPW")
""")

BLOCK_WORKER = (
    "\n\n# ---- ARC_P135 probe telemetry (probe image only)\n"
    "import os as _p135_os\n"
    "import time as _p135_t\n" + COMMON +
    """
_P135_PROF = {"prof": None, "left": 0}


def _p135_arm(self):
    # /root/PROF_ON containing N arms an N-execute_model XPU profiler window
    # (both TP ranks arm independently; each exports its own trace).
    if _P135_PROF["prof"] is None and _P135_PROF["left"] == 0:
        try:
            if _p135_os.path.exists("/root/PROF_ON"):
                _n = int((open("/root/PROF_ON").read().strip() or "24"))
                _p135_os.remove("/root/PROF_ON")
                from torch.profiler import profile, ProfilerActivity
                _p = profile(activities=[ProfilerActivity.CPU,
                                         ProfilerActivity.XPU])
                _p.start()
                _P135_PROF.update(prof=_p, left=_n)
                _p135_log("PROF_START r%s n=%d" % (self.rank, _n))
        except Exception as _e:
            _p135_log("PROF_ARM_ERR %s" % _e)
            _P135_PROF.update(prof=None, left=0)


def _p135_maybe_stop(self):
    if _P135_PROF["prof"] is not None:
        _P135_PROF["left"] -= 1
        if _P135_PROF["left"] <= 0:
            try:
                _P135_PROF["prof"].stop()
                _p = "/root/p135_prof_rank%s.json" % self.rank
                _P135_PROF["prof"].export_chrome_trace(_p)
                _p135_log("PROF_DONE r%s -> %s" % (self.rank, _p))
            except Exception as _e:
                _p135_log("PROF_STOP_ERR %s" % _e)
            _P135_PROF.update(prof=None, left=0)


_p135_orig_wexec = Worker.execute_model


def _p135_wexec(self, scheduler_output, *a, **kw):
    _p135_arm(self)
    _t0 = _p135_t.perf_counter()
    _r = _p135_orig_wexec(self, scheduler_output, *a, **kw)
    try:
        _p135_log("WEX r%s %.3f ntok=%d" % (
            self.rank, (_p135_t.perf_counter() - _t0) * 1e3,
            getattr(scheduler_output, "total_num_scheduled_tokens", -1)))
    except Exception:
        pass
    _p135_maybe_stop(self)
    return _r


Worker.execute_model = _p135_wexec

_p135_orig_wsamp = Worker.sample_tokens


def _p135_wsamp(self, *a, **kw):
    _t0 = _p135_t.perf_counter()
    _r = _p135_orig_wsamp(self, *a, **kw)
    _p135_log("WSM r%s %.3f" % (self.rank,
                                (_p135_t.perf_counter() - _t0) * 1e3))
    return _r


Worker.sample_tokens = _p135_wsamp
""")

FILES = {
    "core.py": ("class EngineCore:", BLOCK_CORE),
    "scheduler.py": ("class Scheduler(", BLOCK_SCHED),
    "model_runner.py": ("class GPUModelRunner(", BLOCK_RUNNER),
    "mamba_hybrid.py": ("class MambaHybridModelState(", BLOCK_MAMBA),
    # base class only (no paren anchor): ModelCudaGraphManager subclasses it
    # and routes through super().run_fullgraph -> wrap fires exactly once.
    "cudagraph_utils.py": ("class CudaGraphManager:", BLOCK_CG),
    "gpu_worker.py": ("class Worker(WorkerBase):", BLOCK_WORKER),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (anchor, block) in FILES.items():
        src = os.path.join(SRC, name)
        with open(src, encoding="utf-8") as f:
            txt = f.read()
        n = txt.count(anchor)
        assert n == 1, "anchor %r in %s: %d occurrences (want 1)" % (anchor, name, n)
        assert "ARC_P135" not in txt, "%s already patched" % name
        out = txt + block
        ast.parse(out)  # syntax gate before it ever reaches the image
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(out)
        print("PATCHED %s (+%d B)" % (name, len(block)))
    print("P135_PATCH_OK")


if __name__ == "__main__":
    sys.exit(main())
