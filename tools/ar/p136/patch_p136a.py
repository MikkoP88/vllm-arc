#!/usr/bin/env python3
"""patch_p136a.py -- P6.36 leg A: prefill-window decomposition probe.

Appends telemetry wrappers to the cert23 engine sources (probe image only,
never ships). Target: the U8k/U32k TTFT window -- metrics proved the time
lives INSIDE the engine prefill window (request_prefill_time 1.5-5 s vs
~1.0 s device EXEC; queue sub-ms), so this kit brackets the window per
rank with EPOCH timestamps (same node -> cross-process/cross-rank
comparable) to name the chunk-boundary host cost and the rank skew:

  ARR   Scheduler.add_request        epoch, req_id, num_tokens
  SCHE  Scheduler.schedule           epoch, running/waiting, sched tokens
  ESTEP EngineCore.step_with_batch_queue  epoch start/end (loop cadence)
  WEXE  Worker.execute_model         epoch start/end, sched tokens, RANK
  REXE  GPUModelRunner.execute_model epoch start/end, RANK (inner wall)

Laws applied (p66b/p135a lineage): O_APPEND line-atomic file log shared
across ranks; host walls only (XPU enqueue async); no .item()/sync added;
anchors asserted against the cert23 wheel (= cert22 for these files).

Log: /root/p136_prefill.log ; lines 'E <epoch> <pid> <rank> <TAG> ...'.
"""
import ast
import os
import sys

SRC = os.environ.get("P136_SRC", "/root/build/p636/probe_src")
OUT = os.environ.get("P136_OUT", "/root/build/p636/probe_out")

LOGFN = "/root/p136_prefill.log"

COMMON = '''
def _p136_log(line):
    try:
        with open("%s", "a") as _f:
            _f.write(line + "\\n")
    except Exception:
        pass
''' % LOGFN

HDR = '''
def _p136_line(tag, rest=""):
    import os as _o
    _p136_log("E %.6f %d %s %s %s" % (
        _p136_t.time(), _o.getpid(), _o.environ.get("RANK", "-"), tag, rest))
'''


def _block(body):
    return (
        "\n\n# ---- ARC_P136 prefill-window probe (P6.36 leg A; probe image only)\n"
        "import time as _p136_t\n" + COMMON + HDR + body
    )


BLOCK_SCHED = _block("""
_p136_orig_add_request = Scheduler.add_request


def _p136_add_request(self, request, *a, **kw):
    _r = _p136_orig_add_request(self, request, *a, **kw)
    try:
        _p136_line("ARR", "req=%s ntok=%d" % (
            getattr(request, "request_id", "?"),
            len(getattr(request, "prompt_token_ids", []) or [])))
    except Exception:
        pass
    return _r


Scheduler.add_request = _p136_add_request

_p136_orig_schedule = Scheduler.schedule


def _p136_schedule(self, *a, **kw):
    _t0 = _p136_t.perf_counter()
    _r = _p136_orig_schedule(self, *a, **kw)
    try:
        _p136_line("SCHE", "ms=%.3f run=%d wait=%d tot=%d" % (
            (_p136_t.perf_counter() - _t0) * 1e3,
            len(getattr(self, "running", None) or []),
            len(getattr(self, "waiting", None) or []),
            getattr(_r, "total_scheduled_tokens", -1)))
    except Exception as _e:
        _p136_line("SCHE", "ms=%.3f err=%r" % (
            (_p136_t.perf_counter() - _t0) * 1e3, _e))
    return _r


Scheduler.schedule = _p136_schedule
""")

BLOCK_CORE = _block("""
_p136_orig_step = EngineCore.step_with_batch_queue


def _p136_step(self):
    _t0 = _p136_t.perf_counter()
    _r = _p136_orig_step(self)
    _p136_line("ESTEP", "ms=%.3f" % ((_p136_t.perf_counter() - _t0) * 1e3))
    return _r


EngineCore.step_with_batch_queue = _p136_step
""")

BLOCK_WORKER = _block("""
_p136_orig_wexe = Worker.execute_model


def _p136_wexe(self, scheduler_output, *a, **kw):
    _t0 = _p136_t.perf_counter()
    _tot = getattr(scheduler_output, "total_scheduled_tokens", -1)
    _p136_line("WEXE", "b tot=%d" % _tot)
    _r = _p136_orig_wexe(self, scheduler_output, *a, **kw)
    _p136_line("WEXE", "e tot=%d ms=%.3f" % (
        _tot, (_p136_t.perf_counter() - _t0) * 1e3))
    return _r


Worker.execute_model = _p136_wexe
""")

BLOCK_RUNNER = _block("""
_p136_orig_rexe = GPUModelRunner.execute_model


def _p136_rexe(self, scheduler_output, *a, **kw):
    _t0 = _p136_t.perf_counter()
    _tot = getattr(scheduler_output, "total_scheduled_tokens", -1)
    _p136_line("REXE", "b tot=%d" % _tot)
    _r = _p136_orig_rexe(self, scheduler_output, *a, **kw)
    _p136_line("REXE", "e tot=%d ms=%.3f" % (
        _tot, (_p136_t.perf_counter() - _t0) * 1e3))
    return _r


GPUModelRunner.execute_model = _p136_rexe
""")

ANCHORS = {
    "core.py": ("class EngineCore:", BLOCK_CORE),
    "scheduler.py": ("class Scheduler(", BLOCK_SCHED),
    "model_runner.py": ("class GPUModelRunner(", BLOCK_RUNNER),
    "gpu_worker.py": ("class Worker(WorkerBase):", BLOCK_WORKER),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    fails = []
    for fname, (anchor, block) in ANCHORS.items():
        src_path = os.path.join(SRC, fname)
        with open(src_path, encoding="utf-8") as f:
            text = f.read()
        if anchor not in text:
            fails.append("ANCHOR_MISSING %s :: %r" % (fname, anchor))
            continue
        if "ARC_P136" in text:
            fails.append("ALREADY_PATCHED %s" % fname)
            continue
        idx = text.index(anchor)
        patched = text + block  # module-level append; anchor verifies class
        ast.parse(patched)  # syntax gate before writing
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
            f.write(patched)
        print("PATCHED %s (anchor @%d)" % (fname, idx))
    if fails:
        for f in fails:
            print("FAIL " + f)
        sys.exit(1)
    print("P136_PATCH_OK %d files" % len(ANCHORS))


if __name__ == "__main__":
    main()
