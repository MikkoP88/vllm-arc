#!/usr/bin/env python3
# patch_p66f.py — P6.6 leg-2 probe patcher v6 = THE LIVE RUNNER.
# v5 census settled it: "Using V2 Model Runner" (xpu_worker.py:169,
# VLLM_USE_V2_MODEL_RUNNER) — the live runner is
# vllm/v1/worker/gpu/model_runner.py::GPUModelRunner(LoRAModelRunnerMixin)
# (the V2 file), NOT vllm/v1/worker/gpu_model_runner.py (v1, dead) and NOT
# vllm_gaudi's hpu_model_runner.py (dead under the arc plugin).
# v6 keeps the validated-live engine-side subs (core.py STEP/SCHED/WAIT),
# the gpu_worker WRK wrappers + import markers, and instruments the V2
# runner inline:
#   execute_model -> ARC_P66_EXEC (state/prep | fwd(eager vs replay) |
#                    post | total, cg mode, allreduce count+ms)
#   sample_tokens -> ARC_P66_SAMPLE (module-tail wrapper)
# All lines -> /root/p66_steps.log. Every anchor must match EXACTLY once.
import pathlib
import sys

SRC = pathlib.Path("/root/build/p66/probe_src")
OUT = pathlib.Path("/root/build/p66/probe_out")
OUT.mkdir(exist_ok=True)

fails = []


def patch(path, subs, tail=""):
    p = SRC / path
    src = p.read_text()
    for old, new in subs:
        n = src.count(old)
        if n != 1:
            fails.append(f"{path}: anchor x{n}: {old[:70]!r}")
            continue
        src = src.replace(old, new)
    (OUT / p.name).write_text(src + tail)
    return src


CORE = "vllm/v1/engine/core.py"
patch(CORE, [
    (
        """        # Step the engine core.
        outputs, model_executed = self.step_fn()""",
        """        # Step the engine core.
        _p66_t0 = time.perf_counter()
        outputs, model_executed = self.step_fn()
        open('/root/p66_steps.log', 'a').write(
            f'ARC_P66_STEP pid={__import__("os").getpid()} wall={(time.perf_counter()-_p66_t0)*1e3:.1f} '
            f'out={0 if outputs is None else len(outputs.items())} exec={int(model_executed)}\\n')""",
    ),
    (
        """        batch_queue = self.batch_queue
        assert batch_queue is not None""",
        """        _p66_ts = time.perf_counter()
        batch_queue = self.batch_queue
        assert batch_queue is not None""",
    ),
    (
        """        if self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())""",
        """        if self.scheduler.has_requests():
            _p66_s0 = time.perf_counter()
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            _p66_sched_ms = (time.perf_counter() - _p66_s0) * 1e3
            open('/root/p66_steps.log', 'a').write(
                f'ARC_P66_SCHED pid={__import__("os").getpid()} tok={scheduler_output.total_num_scheduled_tokens} '
                f'sched_ms={_p66_sched_ms:.1f} ts={(time.perf_counter()-_p66_ts)*1e3:.1f}\\n')""",
    ),
    (
        """            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()""",
        """            _p66_w0 = time.perf_counter()
            model_output = future.result()
            _p66_wait_ms = (time.perf_counter() - _p66_w0) * 1e3
            open('/root/p66_steps.log', 'a').write(
                f'ARC_P66_WAIT pid={__import__("os").getpid()} tok={scheduler_output.total_num_scheduled_tokens} '
                f'wait_ms={_p66_wait_ms:.1f} ts={(time.perf_counter()-_p66_ts)*1e3:.1f}\\n')
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()""",
    ),
], tail="""

# --- p66 probe: import marker ---
open('/root/p66_steps.log', 'a').write(
    f'ARC_P66_IMPORT_CORE pid={__import__("os").getpid()} file={__file__}\\n')
""")

# THE LIVE RUNNER (V2): vllm/v1/worker/gpu/model_runner.py
RUNNER = "vllm/v1/worker/gpu/model_runner.py"
patch(RUNNER, [
    # module-level allreduce counter before the runner class
    (
        """class GPUModelRunner(LoRAModelRunnerMixin):""",
        """# --- p66 probe (probe-only, never ships) ---
_p66_ar = {"n": 0, "ms": 0.0}
try:
    import torch.distributed as _p66_dist

    for _p66_name in ("all_reduce", "_all_reduce_base"):
        _p66_orig = getattr(_p66_dist, _p66_name, None)
        if _p66_orig is None:
            continue

        def _p66_wrap(_orig=_p66_orig):
            def _w(*a, **kw):
                _t = __import__("time").perf_counter()
                _r = _orig(*a, **kw)
                _p66_ar["n"] += 1
                _p66_ar["ms"] += (__import__("time").perf_counter() - _t) * 1e3
                return _r
            return _w
        setattr(_p66_dist, _p66_name, _p66_wrap())
except Exception:
    pass


class GPUModelRunner(LoRAModelRunnerMixin):""",
    ),
    # execute_model entry clock (after the request-state block head)
    (
        """        if not dummy_run:
            # Update the request states.
            self.update_pp_decode_requests()""",
        """        _p66_t0 = __import__("time").perf_counter()
        _p66s = {"t0": _p66_t0, "st": _p66_t0, "sd": _p66_t0, "sp0": _p66_t0,
                 "td0": _p66_t0, "td1": _p66_t0}
        _p66_ar["n"], _p66_ar["ms"] = 0, 0.0
        if not dummy_run:
            # Update the request states.
            self.update_pp_decode_requests()""",
    ),
    # s1 end: request-state block done (apply_staged_writes is its last stmt)
    (
        """            self.block_tables.apply_staged_writes()""",
        """            self.block_tables.apply_staged_writes()
            _p66s["st"] = __import__("time").perf_counter()""",
    ),
    # s2 end: gather_batch_req_state + dispatch done
    (
        """        if batch_desc.num_tokens == 0:""",
        """        _p66s["sd"] = __import__("time").perf_counter()
        if batch_desc.num_tokens == 0:""",
    ),
    # s3 start: prepare_inputs (common-case block)
    (
        """            input_batch = self.prepare_inputs(""",
        """            _p66s["sp0"] = __import__("time").perf_counter()
            input_batch = self.prepare_inputs(""",
    ),
    # s4 span: attention-metadata build [attn_metadata=None .. input_ids line]
    (
        """        attn_metadata = None""",
        """        _p66s["td0"] = __import__("time").perf_counter()
        attn_metadata = None""",
    ),
    (
        """        input_ids = input_batch.input_ids""",
        """        _p66s["td1"] = __import__("time").perf_counter()
        input_ids = input_batch.input_ids""",
    ),
    # end-of-prep clock: right before the forward dispatch
    (
        """        # Run model.
        if batch_desc.cg_mode == CUDAGraphMode.FULL:""",
        """        # Run model.
        _p66_t1 = __import__("time").perf_counter()
        if batch_desc.cg_mode == CUDAGraphMode.FULL:""",
    ),
    # forward end clock
    (
        """        self.kv_connector.finish_forward()""",
        """        _p66_t2 = __import__("time").perf_counter()
        self.kv_connector.finish_forward()""",
    ),
    # EXEC write before the V2 state stash
    (
        """        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,""",
        """        _p66_t3 = __import__("time").perf_counter()
        open('/root/p66_steps.log', 'a').write(
            f'ARC_P66_EXEC pid={__import__("os").getpid()} tok={num_toks} dm={int(dummy_run)} '
            f'cg={batch_desc.cg_mode.name} '
            f'prep_ms={(_p66_t1-_p66_t0)*1e3:.1f} '
            f'st_ms={(_p66s["st"]-_p66s["t0"])*1e3:.1f} '
            f'disp_ms={(_p66s["sd"]-_p66s["st"])*1e3:.1f} '
            f'pin_ms={(_p66s["sp0"]-_p66s["sd"])*1e3:.1f} '
            f'meta_ms={(_p66s["td1"]-_p66s["td0"])*1e3:.1f} '
            f'res_ms={(_p66_t1-_p66s["td1"])*1e3:.1f} '
            f'fwd_ms={(_p66_t2-_p66_t1)*1e3:.1f} '
            f'post_ms={(_p66_t3-_p66_t2)*1e3:.1f} total_ms={(_p66_t3-_p66_t0)*1e3:.1f} '
            f'ar_n={_p66_ar["n"]} ar_ms={_p66_ar["ms"]:.1f}\\n')
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch,""",
    ),
], tail="""

# --- p66 probe: import marker + sample_tokens wrapper ---
open('/root/p66_steps.log', 'a').write(
    f'ARC_P66_IMPORT_RUNNER pid={__import__("os").getpid()} file={__file__}\\n')


def _p66_make_sample_wrapper(_orig):
    def _p66_sample_tokens(self, *args, **kwargs):
        _t0 = __import__("time").perf_counter()
        _r = _orig(self, *args, **kwargs)
        open('/root/p66_steps.log', 'a').write(
            f'ARC_P66_SAMPLE pid={__import__("os").getpid()} '
            f'sample_ms={(__import__("time").perf_counter()-_t0)*1e3:.1f}\\n')
        return _r
    return _p66_sample_tokens


GPUModelRunner.sample_tokens = _p66_make_sample_wrapper(GPUModelRunner.sample_tokens)
""")

WORKER = "vllm/v1/worker/gpu_worker.py"
patch(WORKER, [], tail="""

# --- p66 probe: import marker + entry wrappers ---
open('/root/p66_steps.log', 'a').write(
    f'ARC_P66_IMPORT_WORKER pid={__import__("os").getpid()} file={__file__}\\n')


def _p66_mkwrap(_orig, _tag):
    def _w(self, *args, **kwargs):
        _t0 = __import__("time").perf_counter()
        _r = _orig(self, *args, **kwargs)
        open('/root/p66_steps.log', 'a').write(
            f'ARC_P66_{_tag} pid={__import__("os").getpid()} '
            f'ms={(__import__("time").perf_counter()-_t0)*1e3:.1f}\\n')
        return _r
    return _w


Worker.execute_model = _p66_mkwrap(Worker.execute_model, 'WRK_EXEC')
Worker.sample_tokens = _p66_mkwrap(Worker.sample_tokens, 'WRK_SAMPLE')
""")

XWORKER = "vllm/v1/worker/xpu_worker.py"
patch(XWORKER, [], tail="""

# --- p66 probe: import marker ---
open('/root/p66_steps.log', 'a').write(
    f'ARC_P66_IMPORT_XWORKER pid={__import__("os").getpid()} file={__file__}\\n')
""")

if fails:
    print("PATCH FAILED:")
    for f in fails:
        print(" -", f)
    sys.exit(1)

import py_compile  # noqa: E402

for f in sorted(OUT.glob("*.py")):
    py_compile.compile(str(f), doraise=True)
    print(f"PATCH_OK {f.name}")
print("PATCHER: DONE")
