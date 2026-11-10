#!/usr/bin/env python3
"""patch_p135c.py -- P6.35 leg C: fine-grained decode brackets + TTFT path
timestamps on cert22 sources (probe image only; never ships).

Extends leg A (patch_p135a):
- every telemetry line gains an epoch-ts prefix (client/server clock align
  for TTFT decomposition);
- new brackets: SpecDecodeBaseProposer.{propose,_sample_draft_tokens,
  set_inputs_first_pass,build_model_inputs_first_pass,
  build_per_group_and_layer_attn_metadata,_determine_batch_execution_and_
  padding} across every proposer class in the module (leaf-wrap loop),
  DraftModelSpeculator.{propose,observe_verification},
  DraftTokensHandler.set_draft_tokens,
  GPUModelRunner.postprocess_sampled, EngineCore.add_request.

Leg-A findings these brackets split: decode step ~40.5 ms at 27.7% device
busy; one ~29.3 ms device-idle gap per step between the verify prep tail
and the logits/sampler burst (= sample_tokens python: sampler + draft
propose + postprocess), plus ~5.8/2.9 ms gaps in scheduler/input-prep.
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
        import time as _p135_tt
        with open("__LOGFN__", "a") as _f:
            _f.write("%.3f %s\\n" % (_p135_tt.time(), line))
    except Exception:
        pass
'''.replace("__LOGFN__", LOGFN)

HDR = "\n\n# ---- ARC_P135C probe telemetry (P6.35 leg C; probe image only)\n"

BLOCK_CORE = HDR + "import time as _p135_t\n" + COMMON + """
_p135_orig_step = EngineCore.step_with_batch_queue


def _p135_step(self):
    _t0 = _p135_t.perf_counter()
    _r = _p135_orig_step(self)
    _p135_log("STEP %.3f" % ((_p135_t.perf_counter() - _t0) * 1e3))
    return _r


EngineCore.step_with_batch_queue = _p135_step

if hasattr(EngineCore, "add_request"):
    _p135_orig_addreq = EngineCore.add_request

    def _p135_addreq(self, request, *a, **kw):
        _t0 = _p135_t.perf_counter()
        _r = _p135_orig_addreq(self, request, *a, **kw)
        try:
            _p135_log("ADDR %.3f id=%s ntok=%d" % (
                (_p135_t.perf_counter() - _t0) * 1e3,
                getattr(request, "request_id", "?"),
                len(getattr(request, "prompt_token_ids", []) or [])))
        except Exception:
            pass
        return _r

    EngineCore.add_request = _p135_addreq
"""

BLOCK_SCHED = HDR + "import time as _p135_t\n" + COMMON + """
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
"""

RUNNER_WALLS = """
def _p135_wall(cls, name, tag):
    _orig = getattr(cls, name, None)
    if _orig is None or getattr(_orig, "_p135_w", False):
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

    _w._p135_w = True
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
_p135_wall(GPUModelRunner, "postprocess_sampled", "PSTP")
"""

BLOCK_RUNNER = (HDR + "import time as _p135_t\n" + COMMON + RUNNER_WALLS)

BLOCK_MAMBA = (HDR + "import time as _p135_t\n" + COMMON + """
def _p135_wall(cls, name, tag):
    _orig = getattr(cls, name, None)
    if _orig is None or getattr(_orig, "_p135_w", False):
        return

    def _w(self, *a, **kw):
        _t0 = _p135_t.perf_counter()
        _r = _orig(self, *a, **kw)
        _p135_log("%s %.3f" % (tag, (_p135_t.perf_counter() - _t0) * 1e3))
        return _r

    _w._p135_w = True
    setattr(cls, name, _w)


_p135_wall(MambaHybridModelState, "preprocess_state", "PMAM")
_p135_wall(MambaHybridModelState, "prepare_attn", "META")
""")

BLOCK_CG = (HDR + "import time as _p135_t\n" + COMMON + """
def _p135_wall(cls, name, tag):
    _orig = getattr(cls, name, None)
    if _orig is None or getattr(_orig, "_p135_w", False):
        return

    def _w(self, *a, **kw):
        _t0 = _p135_t.perf_counter()
        _r = _orig(self, *a, **kw)
        _p135_log("%s %.3f" % (tag, (_p135_t.perf_counter() - _t0) * 1e3))
        return _r

    _w._p135_w = True
    setattr(cls, name, _w)


_p135_wall(CudaGraphManager, "run_fullgraph", "RFG")
_p135_wall(CudaGraphManager, "run_pw_graph", "RPW")
""")

BLOCK_WORKER = (HDR + "import os as _p135_os\nimport time as _p135_t\n" + COMMON + """
_P135_PROF = {"prof": None, "left": 0}


def _p135_arm(self):
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

BLOCK_PROPOSER = (HDR + "import time as _p135_t\n" + COMMON + """
def _p135_pwrap(name, tag):
    # wrap on every proposer class in THIS module (leaf classes included);
    # the _p135_w flag prevents double-wrap through inheritance.
    for _nm in list(globals()):
        _c = globals()[_nm]
        if not (isinstance(_c, type)
                and issubclass(_c, SpecDecodeBaseProposer)):
            continue
        _f = getattr(_c, name, None)
        if _f is None or getattr(_f, "_p135_w", False):
            continue

        def _mk(_orig, _tag):
            def _w(self, *a, **kw):
                _t0 = _p135_t.perf_counter()
                _r = _orig(self, *a, **kw)
                _p135_log("%s %.3f" % (_tag,
                                       (_p135_t.perf_counter() - _t0) * 1e3))
                return _r
            _w._p135_w = True
            return _w

        setattr(_c, name, _mk(_f, tag))


_p135_pwrap("propose", "PROP")
_p135_pwrap("_sample_draft_tokens", "SDRT")
_p135_pwrap("set_inputs_first_pass", "SIFP")
_p135_pwrap("build_model_inputs_first_pass", "BMIF")
_p135_pwrap("build_per_group_and_layer_attn_metadata", "BPAM")
_p135_pwrap("_determine_batch_execution_and_padding", "DDET")
""")

BLOCK_SPECULATOR = (HDR + "import time as _p135_t\n" + COMMON + """
def _p135_wall(cls, name, tag):
    _orig = getattr(cls, name, None)
    if _orig is None or getattr(_orig, "_p135_w", False):
        return

    def _w(self, *a, **kw):
        _t0 = _p135_t.perf_counter()
        _r = _orig(self, *a, **kw)
        _p135_log("%s %.3f" % (tag, (_p135_t.perf_counter() - _t0) * 1e3))
        return _r

    _w._p135_w = True
    setattr(cls, name, _w)


_p135_wall(DraftModelSpeculator, "propose", "SPRP")
_p135_wall(DraftModelSpeculator, "observe_verification", "OBSV")
""")

BLOCK_SDU = (HDR + "import time as _p135_t\n" + COMMON + """
def _p135_wall(cls, name, tag):
    _orig = getattr(cls, name, None)
    if _orig is None or getattr(_orig, "_p135_w", False):
        return

    def _w(self, *a, **kw):
        _t0 = _p135_t.perf_counter()
        _r = _orig(self, *a, **kw)
        _p135_log("%s %.3f" % (tag, (_p135_t.perf_counter() - _t0) * 1e3))
        return _r

    _w._p135_w = True
    setattr(cls, name, _w)


_p135_wall(DraftTokensHandler, "set_draft_tokens", "SDTK")
""")

FILES = {
    "core.py": ("class EngineCore:", BLOCK_CORE),
    "scheduler.py": ("class Scheduler(", BLOCK_SCHED),
    "model_runner.py": ("class GPUModelRunner(", BLOCK_RUNNER),
    "mamba_hybrid.py": ("class MambaHybridModelState(", BLOCK_MAMBA),
    "cudagraph_utils.py": ("class CudaGraphManager:", BLOCK_CG),
    "gpu_worker.py": ("class Worker(WorkerBase):", BLOCK_WORKER),
    "llm_base_proposer.py": ("class SpecDecodeBaseProposer", BLOCK_PROPOSER),
    "speculator.py": ("class DraftModelSpeculator(", BLOCK_SPECULATOR),
    "sd_utils.py": ("class DraftTokensHandler:", BLOCK_SDU),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (anchor, block) in FILES.items():
        with open(os.path.join(SRC, name), encoding="utf-8") as f:
            txt = f.read()
        n = txt.count(anchor)
        assert n == 1, "anchor %r in %s: %d occurrences (want 1)" % (anchor, name, n)
        assert "ARC_P135" not in txt, "%s already patched" % name
        out = txt + block
        ast.parse(out)
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(out)
        print("PATCHED %s (+%d B)" % (name, len(block)))
    print("P135C_PATCH_OK")


if __name__ == "__main__":
    sys.exit(main())
