#!/usr/bin/env python3
# patch_p66e.py — P6.6 leg-2 probe patcher v5 = v4 + WHERE-ARE-THE-WORKERS
# diagnostics. Symptom to explain: engine core (pid 698) writes
# ARC_P66_STEP/SCHED/WAIT, but NO worker process ever writes EXEC/SAMPLE,
# although serving works and workers provably import GPUModelRunner
# (fork hygiene logs "runner-entry hygiene installed (pid 930...)").
# Hypotheses to partition with ONE boot:
#   H1 workers import a DIFFERENT gpu_model_runner (vendored / finder)
#      -> marker lines absent from worker pids, or different __file__
#   H2 a fork patch REPLACES GPUModelRunner.execute_model wholesale
#      (never calls orig) -> WRK lines appear (Worker.execute_model runs)
#         but EXEC lines don't
#   H3 worker file writes fail/silent                        -> no WRK either
# Adds: per-module IMPORT markers with __file__ + pid (core.py,
#       gpu_model_runner.py, gpu_worker.py, xpu_worker.py), module-tail
#       wrappers around Worker.execute_model/sample_tokens (WRK lines)
#       and GPUModelRunner.execute_model (EXEC2 line), keeping every v4
#       sub unchanged. All lines -> /root/p66_steps.log.
# Every anchor must match EXACTLY once or the patcher aborts.
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

# --- p66 probe v5: import marker ---
open('/root/p66_steps.log', 'a').write(
    f'ARC_P66_IMPORT_CORE pid={__import__("os").getpid()} file={__file__}\\n')
""")

RUNNER = "vllm/v1/worker/gpu_model_runner.py"
patch(RUNNER, [
    (
        """class GPUModelRunner(""",
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


class GPUModelRunner(""",
    ),
    (
        """        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )""",
        """        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )
        _p66_t0 = __import__("time").perf_counter()
        _p66_ar["n"], _p66_ar["ms"] = 0, 0.0""",
    ),
    (
        """            self.eplb_state.prepare_forward(
                self.model_config,
                num_tokens_unpadded,
                ubatch_slices_padded,
            )
        with (
            set_forward_context(""",
        """            self.eplb_state.prepare_forward(
                self.model_config,
                num_tokens_unpadded,
                ubatch_slices_padded,
            )
        _p66_t1 = __import__("time").perf_counter()
        with (
            set_forward_context(""",
    ),
    (
        """            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )""",
        """            _p66_f0 = __import__("time").perf_counter()
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )
            _p66_f1 = __import__("time").perf_counter()""",
    ),
    (
        """        with record_function_or_nullcontext("gpu_model_runner: postprocess"):""",
        """        _p66_t2 = __import__("time").perf_counter()
        with record_function_or_nullcontext("gpu_model_runner: postprocess"):""",
    ),
    (
        """        self.execute_model_state = ExecuteModelState(""",
        """        _p66_t3 = __import__("time").perf_counter()
        open('/root/p66_steps.log', 'a').write(
            f'ARC_P66_EXEC pid={__import__("os").getpid()} tok={num_scheduled_tokens} '
            f'prep_ms={(_p66_t1-_p66_t0)*1e3:.1f} fwd_ms={(_p66_f1-_p66_f0)*1e3:.1f} '
            f'post_ms={(_p66_t3-_p66_f1)*1e3:.1f} total_ms={(_p66_t3-_p66_t0)*1e3:.1f} '
            f'ar_n={_p66_ar["n"]} ar_ms={_p66_ar["ms"]:.1f}\\n')
        self.execute_model_state = ExecuteModelState(""",
    ),
], tail="""

# --- p66 probe v5: import marker + class-attr wrappers ---
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


def _p66_make_exec_wrapper(_orig):
    def _p66_execute_model(self, *args, **kwargs):
        _t0 = __import__("time").perf_counter()
        _r = _orig(self, *args, **kwargs)
        open('/root/p66_steps.log', 'a').write(
            f'ARC_P66_EXEC2 pid={__import__("os").getpid()} '
            f'exec_ms={(__import__("time").perf_counter()-_t0)*1e3:.1f}\\n')
        return _r
    return _p66_execute_model


GPUModelRunner.execute_model = _p66_make_exec_wrapper(GPUModelRunner.execute_model)
""")

WORKER = "vllm/v1/worker/gpu_worker.py"
patch(WORKER, [], tail="""

# --- p66 probe v5: import marker + entry wrappers ---
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

# --- p66 probe v5: import marker ---
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
