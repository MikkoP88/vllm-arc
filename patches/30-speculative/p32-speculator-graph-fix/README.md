# p32 — speculator FULL-capture coercion (defect-a: PIECEWISE boot death)

## What

`vllm_gaudi/xpu/patches/speculator_full_capture.py` wraps
`vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils.
SpeculatorCudaGraphManager.__init__` and coerces any PIECEWISE-family
cudagraph mode (`PIECEWISE`, `FULL_AND_PIECEWISE`) to `FULL` before the
manager builds. Installed from the same ArcXpuPlatform hooks as
graph_safe_output (config time + per-worker init_device; idempotent;
visible in serve logs under `vllm.platforms.arc`).

## Why (plan Appendix C defect (a))

v1's default mode is FULL_AND_PIECEWISE — a spec leg that doesn't pin a
mode routes the speculator's PREFILL manager into piecewise capture:
graph segments around eager breaks (`BreakableCUDAGraphCapture.add_eager`
— compute_logits + gumbel sampling run between segments) and offloader
copy-stream joins recorded INSIDE the capture context. On XPU L0 command
graphs that path performs device-sync-class work inside an open graph →
boot crash (observed on radiance v0.30.0 / qwen3.8-27b-fp8, Leg B). The
eager-break route is also the wrong-logits family (0.26 acceptance
collapse, plan falsified-path #1) — piecewise has no value on this stack
even where it boots.

All three manager instantiation sites (autoregressive prefill :129,
decode :143, multi_module_mtp :106) construct through
`SpeculatorCudaGraphManager`, so one choke point covers every speculator
family (MTP included). The autoregressive decode manager already
downgrades itself to FULL_DECODE_ONLY/NONE — untouched.

Fix direction = Appendix C first arm ("force the FULL capture path for
the speculator"). Known iteration risk, decided by leg evidence only:
FULL draft-prefill graphs capture gumbel sampling inside one graph; if
XPU RNG-in-graph fails at the P2 spec leg, the coercion target moves to
NONE (eager draft prefill) — never without that leg.

## Validation plan (P2 spec legs, after p12 graph-green + p13 wheel)

1. boot: spec leg with default mode (no cudagraph_mode pin) — expect the
   coercion log line + NO piecewise capture path + boot survives.
2. compile×spec A/B acceptance law (falsified-path #1 REVAL on 0.30):
   acceptance with compile on vs TORCH_COMPILE_DISABLE=1 — gate stays
   regardless (acceptance ≥ 0.85).
3. MTP k4 first-request survival + golden-logit spot check (also closes
   p12 part-2 verification: spec_state_indices geometry).
