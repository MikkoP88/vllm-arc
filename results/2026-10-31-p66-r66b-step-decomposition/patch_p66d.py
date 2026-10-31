#!/usr/bin/env python3
# patch_p66d.py — P6.6 leg-2 probe patcher v4 (r48j pattern: probe-only,
# never ships). v3 lesson: the arc plugin runs the STOCK vllm XPU stack —
# ArcXpuPlatform extends vllm.platforms.xpu.XPUPlatform, so the live runner
# is vllm/v1/worker/gpu_model_runner.py::GPUModelRunner (execute_model =
# full forward + ExecuteModelState stash; sample_tokens = sampling tail).
# vllm_gaudi's hpu_model_runner.py is DEAD CODE under the arc plugin.
#
#   vllm/v1/engine/core.py              (v3 subs, unchanged, validated live)
#     -> ARC_P66_STEP / ARC_P66_SCHED / ARC_P66_WAIT   (file writes)
#   vllm/v1/worker/gpu_model_runner.py  execute_model
#     -> ARC_P66_EXEC (prep / fwd / post / total + allreduce count+ms)
#     -> ARC_P66_SAMPLE (appended wrapper around sample_tokens)
# All lines -> /root/p66_steps.log (append; O_APPEND line-atomic).
# Every anchor must match EXACTLY once or the patcher aborts.
# Caveat recorded: fwd wall is CPU-side enqueue unless a later sync lands
# inside it; postprocess absorbs the device wait. Cross-check vs STEP/WAIT.
import pathlib
import sys

SRC = pathlib.Path("/root/build/p66/probe_src")
OUT = pathlib.Path("/root/build/p66/probe_out")
OUT.mkdir(exist_ok=True)

fails = []


def patch(path, subs):
    p = SRC / path
    src = p.read_text()
    for old, new in subs:
        n = src.count(old)
        if n != 1:
            fails.append(f"{path}: anchor x{n}: {old[:70]!r}")
            continue
        src = src.replace(old, new)
    (OUT / p.name).write_text(src)
    return src


CORE = "vllm/v1/engine/core.py"
patch(CORE, [
    # engine-step total wall (wraps step_fn whatever it is)
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
    # entry clock inside the live batch-queue step
    (
        """        batch_queue = self.batch_queue
        assert batch_queue is not None""",
        """        _p66_ts = time.perf_counter()
        batch_queue = self.batch_queue
        assert batch_queue is not None""",
    ),
    # schedule bracket
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
    # result-wait bracket (the phase-B block on the popped batch)
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
])

RUNNER = "vllm/v1/worker/gpu_model_runner.py"
patch(RUNNER, [
    # module-level allreduce counter before the runner class
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
    # execute_model entry clock
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
    # end-of-prep clock (before the forward context opens)
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
    # forward bracket
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
    # postprocess start clock
    (
        """        with record_function_or_nullcontext("gpu_model_runner: postprocess"):""",
        """        _p66_t2 = __import__("time").perf_counter()
        with record_function_or_nullcontext("gpu_model_runner: postprocess"):""",
    ),
    # EXEC write just before the state stash
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
])

# appended sample_tokens wrapper (module tail; runs at import, BEFORE the
# fork's graph_safe_output hygiene wraps the attribute — they compose).
p = SRC / RUNNER
src = (OUT / p.name).read_text()
src += '''

# --- p66 probe (probe-only, never ships): sample_tokens wall ---
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
'''
(OUT / p.name).write_text(src)

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
