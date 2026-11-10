#!/usr/bin/env python3
"""patch_p135k.py -- P6.35 leg K: AsyncOutput waiter-mode A/B.

Leg J pinned the step gate: a ~40 ms zeEventHostSynchronize (one per decode
step, thread != engine thread) = AsyncOutput.get_output's copy_event sleep
(vllm/v1/worker/gpu/async_utils.py). Device idles 29 of 40.5 ms; the same
trace shows zeEventQueryStatus (~3.5 us/call) and stream sync
(zeCommandListHostSynchronize ~1 ms) are fast paths.

Leg K varies ONLY the wait primitive (baked at patch time, MODE arg):
  event  = stock sleep (control = leg J numbers, no image needed)
  busy   = poll copy_event.query() until complete
  stream = copy_stream.synchronize() instead of the event

Walls added: CEVT (the wait itself, + mode + tid), GETO (whole get_output),
EVTS (copy ctor). Build: python3 patch_p135k.py probe_src/async_utils.py
<mode> -> probe_out/async_utils.py. Probe scratch only; never ships.
"""
import ast
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/root/build/p635/probe_src/async_utils.py"
MODE = sys.argv[2] if len(sys.argv) > 2 else "busy"
assert MODE in ("busy", "stream", "event"), MODE
OUT = sys.argv[3] if len(sys.argv) > 3 else "/root/build/p635/probe_out/async_utils.py"

src = open(SRC, encoding="utf-8").read()


def rep(old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, "anchor x%d (want %d): %r" % (c, n, old[:70])
    src = src.replace(old, new)


# 1. helpers + baked mode before AsyncOutput
rep("class AsyncOutput(AsyncModelRunnerOutput):",
    '''import time as _p135_time
import threading as _p135_th
_P135_MODE = "%s"


def _p135_log(tag, wall_ms, extra=""):
    try:
        with open("/root/p135_steps.log", "a") as _f:
            _f.write("%%.3f %%s %%.3f %%s\\n" %% (_p135_time.time(), tag, wall_ms, extra))
    except Exception:
        pass


def _p135_cevt(mode, self):
    _t0 = _p135_time.perf_counter()
    if mode == "stream":
        self._p135_copy_stream.synchronize()
    elif mode == "busy":
        while not self.copy_event.query():
            pass
    else:
        self.copy_event.synchronize()
    _p135_log("CEVT", (_p135_time.perf_counter() - _t0) * 1e3,
              "mode=%%s tid=%%d" %% (mode, _p135_th.get_ident() %% 1000000))


class AsyncOutput(AsyncModelRunnerOutput):''' % MODE)

# 2. keep the copy stream reference (both classes)
rep("        self.copy_event = torch.cuda.Event(blocking=True)",
    "        self.copy_event = torch.cuda.Event(blocking=True)\n"
    "        self._p135_copy_stream = copy_stream", 2)

# 3. EVTS marker at copy-ctor end (both classes)
rep("            self.copy_event.record(copy_stream)",
    "            self.copy_event.record(copy_stream)\n"
    '        _p135_log("EVTS", 0.0, "tid=%d" % (_p135_th.get_ident() % 1000000))', 2)

# 4. AsyncOutput.get_output: wall + mode wait
rep("    def get_output(self) -> ModelRunnerOutput:\n"
    "        self.copy_event.synchronize()",
    "    def get_output(self) -> ModelRunnerOutput:\n"
    "        _p135_g0 = _p135_time.perf_counter()\n"
    "        _p135_cevt(_P135_MODE, self)")

# 5. AsyncPoolingOutput.get_output: wall + mode wait
rep("        self.copy_event.synchronize()\n"
    "        self.model_runner_output.pooler_output = pooler_output",
    "        _p135_g0 = _p135_time.perf_counter()\n"
    "        _p135_cevt(_P135_MODE, self)\n"
    "        self.model_runner_output.pooler_output = pooler_output")

# 6. GETO wall at both returns
rep("        return self.model_runner_output",
    '        _p135_log("GETO", (_p135_time.perf_counter() - _p135_g0) * 1e3,\n'
    '                  "tid=%d" % (_p135_th.get_ident() % 1000000))\n'
    "        return self.model_runner_output", 2)

ast.parse(src)
open(OUT, "w", encoding="utf-8").write(src)
print("P135K_PATCHED mode=%s GETO/CEVT/EVTS walls + waiter=%s" % (MODE, MODE))
