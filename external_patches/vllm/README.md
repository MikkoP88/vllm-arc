# vllm — B70 prefill launch-config table for the triton unified-attention launcher

## What

`patch_arc_xpu_attn_launcher.py` inserts a per-platform prefill
launch-config table into the vllm wheel's
`vllm/v1/attention/ops/triton_unified_attention.py` (stock wheel file —
no source rebuild, staged pins stay pristine). Three insertions, mark
`[ARC-P628-XPU-TUNE]`, backup `.arcxpubak`, idempotent, `--audit` mode:

1. module table + band selector after `float8_info`
   (`_ARC_XPU_PREFILL_CONFIGS` keyed by padded head size →
   (≤64, 65–4095, ≥4096) bands → `(BLOCK_M, TILE, warps, stages)` or None)
2. launcher branch after the `tuned_large_head` BLOCK_M block and BEFORE
   `total_num_q_blocks` (the 2D grid upper bound shrinks with BLOCK_Q),
   guarded `current_platform.is_xpu()` + `num_queries_per_kv <= 16` +
   `max_seqlen_q > 64` + `not is_batch_invariant` + `q.element_size() <= 2`
3. tile override after the `tuned_large_head` TILE block:
   `TILE_SIZE_PREFILL = max(32, tile)` (fp8-KV legality, PR#60418 guard;
   the stock `use_td` clamp still applies downstream)

Decode / 3D split-KV path untouched (band guard keeps every
decode-shaped call on stock configs). Only launch parameters change —
kernel source and numerics untouched (r108k numeric dumps allclose(1e-2)
vs stock).

B70 table (rig sweep r108k, 2026-11-06):

| padded head | 65–4095 band | ≥4096 band |
|---|---|---|
| 256 | (64, 64, 8, None) | (64, 64, 8, None) |
| 512 | (64, 32, 8, None) | (64, 32, 8, None) |

≤64 band = None (stock) for both. `num_stages=None` = triton default,
exactly what the sweep measured. TILE=64 at head 512 is infeasible on
B70 (all T64@512 sweep points die pre-print — SM-side memory wall, the
same LDS boundary class vllm#52585 documented for TILE=128 on gfx1100).

## Why

P6.28 gemma-4 serving "froze" on any real prompt. Root cause (r108j
bare-kernel discriminator, banked micro_disc/): NOT a deadlock — the 2D
prefill kernel runs at ~0.3% of achievable peak (1.1 TFLOP in 8.265 s
steady vs 43.2 TFLOPS fp16 matmul reference on the same device) at the
stock launch config (BLOCK_M=16 / TILE=32 / 4 warps). Gemma-4's 60-layer
5:1 mix of head-512 full / head-256 sliding prefill burns ~20 min device
time per 32k-token face → the 300 s `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`
rpc kill reads as a hang. Exonerated along the way: tensor-parallel
deadlock, fp8-KV path, device asymmetry, block_table ordering, tile
pruning (already present in the wheel's helpers).

Headline sweep numbers (per-rank GQA 16Q:8KV, fp8-KV, seqlen 8192):
head-512 ntok=1024: stock 2.002 s → (64, 32, w8) **0.428 s = 4.67×**
(warps=8 is load-bearing: M64w4 = 2.02× only); head-256 sliding ntok=1024:
stock 0.0461 s → (64, 64, w8) **0.0222 s = 2.08×**.

## Provenance / upstream lineage

The fix pattern is upstream's own, applied to the platform upstream
doesn't tune yet:

- **vLLM PR#60418** `[ROCm][RDNA][Perf] Add tuned configs for TRITON_ATTN
  kernel` (merged 2026-11-03, 4d2fa5637): module-level per-platform
  config tables in this exact launcher, keyed by padded head size ×
  max_seqlen_q band, prefill branch before grid computation, decode
  branch after 2D/3D choice, "only launch parameters change".
- **vLLM#52585** (gfx1100, 2026-08-17): the same defect class — stock
  BLOCK_M=16 under-tiles prefill ~2×; tuned BLOCK_M/warps = 1.1–1.3×
  TTFT bitwise-identical; TILE=128 blows the LDS budget.
- **vLLM#38887**: Gemma-4 slow prefill on NVIDIA — still open, no fix;
  this is the first landed fix in this defect line, on XPU.
- **intel-xpu-backend-for-triton#8090**: Intel's offline-autotune
  direction — swept configs into static dicts + heuristics (citing
  #60418), explicitly NOT `@triton.autotune` (Triton maintainers: not
  production-intended).
- **vllm.ai Triton backend deep dive** (2026-03-04, IBM/Red Hat/AMD — the
  kernel's authors): "autotuning selects appropriate block sizes for each
  platform"; KV-side tiles are page-size-constrained so "optimization
  focuses on the query side" — i.e. a per-platform Q-block launch-config
  table IS the designed tuning surface of this kernel.

Scope law (plan §3/App I): generic shared-component fix (whole launcher
file, platform-gated, model-agnostic config table), no donor code —
pattern is upstream vLLM's, values are our own rig measurements.

## Validation (§10 gates before cert12 pin)

1. patcher: `--audit` PATCHABLE → patch → `ALREADY_PATCHED` on re-run →
   `py_compile` clean (local, pristine wheel file extracted from cert11).
2. rig in-container: patched launcher imports, MARK present, table
   entries readable; bare-kernel full-shape (ntok=8192) timing matches
   sweep winners (grid now shrinks with BLOCK_Q — same or better).
3. engine G2: both gemma legs re-measured vs the fp8 incumbent row
   (17.715 s e2e / 39.025 s 32k prefill face @ 53.5 ms decode); decode
   unchanged (kernel/config path untouched for max_seqlen_q ≤ 64).
4. numeric sanity: r108k output dumps allclose(1e-2) vs stock at both
   head sizes.

Evidence: `results/2026-11-06-p628-gemma4/` (r108j micro_disc/, r108k
tile_sweep/, G2 legs, write-up).
