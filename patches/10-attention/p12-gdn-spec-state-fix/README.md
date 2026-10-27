# p12 — GDN spec-state + XPU graph-safe output copies (defect-b family)

## What (part 1 — graph-safe output copies)

`vllm_gaudi/xpu/patches/graph_safe_output.py` patches
`vllm.v1.worker.gpu.async_utils.{AsyncOutput,AsyncPoolingOutput}` in place:
upstream's async D2H copies + blocking event stay, then a full
`torch.xpu.synchronize()` barrier is taken and the (possibly command-graph
associated) event is replaced by a no-op. Inside an active capture the patch
defers to upstream (device sync is illegal mid-capture), and a `get_output`
wrapper self-heals if such a capture-time event ever raises at wait time:
device-sync barrier once, neutralize the event, retry — instead of killing
the engine.

## Why

0.30's async output path kills the engine on the FIRST served step with XPU
graphs enabled: L0 command graphs associate events recorded while a graph is
open, and `copy_event.synchronize()` then raises
`wait method cannot be used for an event associated with a command graph`
(repro: results/2026-10-27-p2r1-graph-only — boot healthy, first
/v1/completions 200, then Worker_TP0 dies). Same error family as plan
Appendix C defect (b); fix direction "graph-safe copy" per the appendix.
No donor fix exists: donor trees are 0.26-based and predate the async path
(checked radiance + iexperimental 2026-10-27); core 0.30 has no env/config
off-switch for the async path (AsyncOutput is unconditional in
gpu_model_runner.py:2082).

## Install points (per-process, idempotent, ALL visible in serve logs)

plugin `register()` (earliest — EXPECTED to fail mid platform-resolution:
importing `vllm.v1.worker.gpu` there is circular; logged as warning),
`ArcXpuPlatform.check_and_update_config` (APIServer + EngineCore), and
`ArcXpuPlatform.check_if_supports_dtype` (every WorkerProc via
`XPUWorker.init_device` — the guaranteed post-import moment). `_install`
logs its pid on success; the wrapper logs first-3 apply/defer/probe-error
lines; `get_output` logs each self-heal. p2r2 lesson: silent excepts made
not-installed indistinguishable from deferred — every branch now speaks.

## Scope

Part 2 (GDN builder persistent-buffer spec_state_indices port from the 0.26
Intel arc, `src_v126_gdn_attn.py` reference) lands after graph-only serving
is green — it is required for MTP legs, not for graph-only legs. Core 0.30's
GDNAttentionMetadataBuilder already ships the persistent-buffer machinery
pre-sized by `decode_cudagraph_max_bs`; part 2 may shrink to verification.

## Validation

- repro (pre-fix): results/2026-10-27-p2r1-graph-only/serve_death.log
- silent-fail leg (post-fix v1): results/2026-10-27-p2r2-graph-safe/ — same
  death; root cause of the silent non-install never observable (fixed by
  the visibility law above)
- post-fix leg v2: results/2026-10-27-p2r3-graph-safe/ (target:
  FULL_DECODE_ONLY graph posture survives first request + k1k cells
  complete + install/apply lines present in Worker logs)
