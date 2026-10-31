#!/usr/bin/env python3
# patch_p66c.py — P6.6 leg-2 probe patcher v3 (r48j pattern: probe-only,
# never ships). v2 lesson: async scheduling => batch_queue enabled =>
# step_fn == step_with_batch_queue (step() is dead code), and child-process
# print()s never reach serve_full.log => v3 instruments the LIVE path and
# writes every line to /root/p66_steps.log (append; O_APPEND line-atomic).
#
#   vllm/v1/engine/core.py            _process_engine_step -> ARC_P66_STEP
#                                     step_with_batch_queue -> ARC_P66_SCHED
#                                                                ARC_P66_WAIT
#                                     (early-return iterations -> STEP early=1)
#   vllm_gaudi/v1/worker/hpu_model_runner.py sample_tokens
#     -> ARC_P66_CHUNK (per prefill chunk fwd wall) + ARC_P66_RUN
#        (prep / prefill-block / tail / total + allreduce count+ms)
# Every anchor must match EXACTLY once or the patcher aborts.
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

RUNNER = "vllm_gaudi/v1/worker/hpu_model_runner.py"
patch(RUNNER, [
    # module-level allreduce counter
    (
        """class HpuModelAdapter(torch.nn.Module, HpuKVConnectorModelRunnerMixin):""",
        """# --- p66 probe (probe-only, never ships) ---
_p66_ar = {"n": 0, "ms": 0.0}
_p66_orig_all_reduce = None
try:
    import torch.distributed as _p66_dist

    _p66_orig_all_reduce = _p66_dist.all_reduce

    def _p66_all_reduce(*a, **kw):
        _t = __import__("time").perf_counter()
        _r = _p66_orig_all_reduce(*a, **kw)
        _p66_ar["n"] += 1
        _p66_ar["ms"] += (__import__("time").perf_counter() - _t) * 1e3
        return _r

    _p66_dist.all_reduce = _p66_all_reduce
except Exception:
    pass


class HpuModelAdapter(torch.nn.Module, HpuKVConnectorModelRunnerMixin):""",
    ),
    # entry clock + per-step allreduce reset
    (
        """        scheduler_output = self.scheduler_output
        warmup_mode = self.warmup_mode
        self.scheduler_output = None
        self.warmup_mode = False""",
        """        scheduler_output = self.scheduler_output
        warmup_mode = self.warmup_mode
        self.scheduler_output = None
        self.warmup_mode = False
        _p66 = {"t0": __import__("time").perf_counter(), "fwd": 0.0, "nch": 0,
                "toks": 0, "wm": int(bool(warmup_mode))}
        _p66_ar["n"], _p66_ar["ms"] = 0, 0.0""",
    ),
    # prep bracket
    (
        """        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            prefill_input_data, decode_input_data = self._prepare_inputs(scheduler_output, num_prefills, num_decodes,
                                                                         warmup_mode)""",
        """        _p66["tp0"] = __import__("time").perf_counter()
        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            prefill_input_data, decode_input_data = self._prepare_inputs(scheduler_output, num_prefills, num_decodes,
                                                                         warmup_mode)
        _p66["tp1"] = __import__("time").perf_counter()""",
    ),
    # prefill block start
    (
        """        ######################### PREFILLS #########################
        if num_prefills > 0:""",
        """        ######################### PREFILLS #########################
        _p66["tf0"] = __import__("time").perf_counter()
        if num_prefills > 0:""",
    ),
    # per-chunk forward bracket (inside the prefill loop)
    (
        """                htorch.core.mark_step()
                non_flattened_hidden_states, aux_hidden_states, \\
                    sample_hidden_states, logits_device = \\
                    self._execute_model_generic(""",
        """                htorch.core.mark_step()
                _p66_f0 = __import__("time").perf_counter()
                non_flattened_hidden_states, aux_hidden_states, \\
                    sample_hidden_states, logits_device = \\
                    self._execute_model_generic(""",
    ),
    (
        """                        warmup_mode=warmup_mode)
                htorch.core.mark_step()
                non_flattened_hidden_states_prefills.append(non_flattened_hidden_states)""",
        """                        warmup_mode=warmup_mode)
                _p66_f1 = __import__("time").perf_counter()
                _p66["fwd"] += _p66_f1 - _p66_f0
                _p66["nch"] += 1
                _p66["toks"] += token_ids.shape[-1]
                open('/root/p66_steps.log', 'a').write(
                    f'ARC_P66_CHUNK pid={__import__("os").getpid()} r={getattr(self, "tp_rank", -1)} wm={_p66["wm"]} '
                    f'idx={idx} tokens={token_ids.shape[-1]} fwd_ms={(_p66_f1-_p66_f0)*1e3:.1f}\\n')
                htorch.core.mark_step()
                non_flattened_hidden_states_prefills.append(non_flattened_hidden_states)""",
    ),
    # tail start
    (
        """        if self.use_structured_output:
            # Scheduler places cached before prompt
            logits_combined = logits_decode + logits_prompt""",
        """        _p66["tf1"] = __import__("time").perf_counter()
        if self.use_structured_output:
            # Scheduler places cached before prompt
            logits_combined = logits_decode + logits_prompt""",
    ),
])

# final RUN line: anchored after the logprobs build, past all heavy work
# (both return paths — async + sync — sit after this point).
p = SRC / RUNNER
src = (OUT / p.name).read_text()
anchor = ("        max_req_index = max(self.input_batch.req_id_to_index.values())\n"
          "        logprobs = self._build_logprobs_output(logprobs_segments, max_req_index + 1)")
ret_pos = src.find(anchor)
if ret_pos < 0:
    fails.append(f"{RUNNER}: logprobs-anchor not found")
else:
    ins = ret_pos + len(anchor)
    src = src[:ins] + (
        "\n        _p66['t9'] = __import__('time').perf_counter()"
        "\n        open('/root/p66_steps.log', 'a').write("
        "\n            f\"ARC_P66_RUN pid={__import__('os').getpid()} r={getattr(self, 'tp_rank', -1)} wm={_p66['wm']} \""
        "\n                  f\"npf={num_prefills} ndec={num_decodes} \""
        "\n                  f\"prep_ms={(_p66.get('tp1', _p66['t0'])-_p66['t0'])*1e3:.1f} \""
        "\n                  f\"pfblk_ms={(_p66.get('tf1', _p66.get('tp1', _p66['t0']))-_p66.get('tf0', _p66.get('tp1', _p66['t0'])))*1e3:.1f} \""
        "\n                  f\"fwd_ms={_p66['fwd']*1e3:.1f} nch={_p66['nch']} toks={_p66['toks']} \""
        "\n                  f\"tail_ms={(_p66['t9']-_p66.get('tf1', _p66.get('tp1', _p66['t0'])))*1e3:.1f} \""
        "\n                  f\"total_ms={(_p66['t9']-_p66['t0'])*1e3:.1f} \""
        "\n                  f\"ar_n={_p66_ar['n']} ar_ms={_p66_ar['ms']:.1f}\\n\")"
    ) + src[ins:]
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
