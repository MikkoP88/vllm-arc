# p12 — GDN spec-state + XPU graph-safe output copies (defect-b family)

## What (part 1 — graph-safe output copies)

`vllm_gaudi/xpu/patches/graph_safe_output.py` patches
`vllm.v1.worker.gpu.async_utils.{AsyncOutput,AsyncPoolingOutput}` in place:
upstream's async D2H copies + blocking event stay, then a full
`torch.xpu.synchronize()` barrier is taken and the (possibly command-graph
associated) event is replaced by a no-op. Inside an active capture the patch
defers to upstream (device sync is illegal mid-capture; capture-time handles
synchronize fine — boot captures completed 27/27 in the repro).

## Why

0.30's async output path kills the engine on the FIRST served step with XPU
graphs enabled: L0 command graphs associate events recorded while a graph is
open, and `copy_event.synchronize()` then raises
`wait method cannot be used for an event associated with a command graph`
(repro: results/2026-10-27-p2r1-graph-only — boot healthy, first
/v1/completions 200, then Worker_TP0 dies). Same error family as plan
Appendix C defect (b); fix direction "graph-safe copy" per the appendix.

## Install points (per-process, idempotent)

plugin `register()` (best-effort early), `ArcXpuPlatform.
check_and_update_config` (APIServer + EngineCore), and
`ArcXpuPlatform.check_if_supports_dtype` (every WorkerProc via
`XPUWorker.init_device` — the guaranteed post-import moment).

## Scope

Part 2 (GDN builder persistent-buffer spec_state_indices port from the 0.26
Intel arc, `src_v126_gdn_attn.py` reference) lands after graph-only serving
is green — it is required for MTP legs, not for graph-only legs.

## Validation

- repro (pre-fix): results/2026-10-27-p2r1-graph-only/serve_death.log
- post-fix leg: results/2026-10-27-p2r2-graph-safe/ (target: FULL_DECODE_ONLY
  graph posture survives first request + k1k cells complete)
