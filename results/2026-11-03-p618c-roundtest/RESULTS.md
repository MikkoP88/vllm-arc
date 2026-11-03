# P6.18c Class-B numeric gate — TurboQuant SoA store→decode roundtrip (XPU)

**Date**: 2026-11-03 (evening) · **Image**: `vllm-arc:v0.30.0-b70-cert6`
(--rm container, GPUs otherwise idle) · **Script**:
`p618c_tq_roundtrip.py` (banked here; rig copy `/root/build/p618/`)

## Question

p618b proved the degenerate-generation class (Class B) lives in the TQ
forward-compute path (cold-included, LMC-independent, fp8 baseline
clean). Is that an IMPLEMENTATION defect of the live SoA Triton pair
(`vllm/v1/attention/ops/turboquant_soa/triton_turboquant_store` →
`triton_turboquant_decode_attention`, the "FlyDSL unavailable → SoA
Triton decode" fallback pinned by the cert6 boot logs), or the quality
ceiling of 4-bit-class V quantization for this model?

## Method

Per preset (k8v4 / 4bit_nc / k3v4_nc / 3bit_nc — the four
certified-menu configs from `vllm/config/cache.py` +
`turboquant/config.py`), with the engine's own deterministic tables
(Sylvester Hadamard `Pi`, Lloyd-Max centroids `get_centroids(D,
mse_bits)`, midpoints) and the model's per-GPU TP2 geometry
(head_dim=256, Hq=12, Hk=2, bf16 KV; two sequences sharing a prefix,
lens 192/100, pages [0,1,2,3]/[0,1], block_size 64):

1. **DECODE GATE (primary, self-coherent)** — kernel decode output vs
   torch attention computed from the K/V dequantized out of the
   kernel's OWN stored bytes (the decode-kernel addressing formula,
   re-implemented in torch). Quantization information loss cancels;
   any store/decode layout or arithmetic defect lands here at full
   magnitude.
2. **STORE SPEC GATE** — kernel cache bytes vs a torch reference
   quantizer mirroring every rounding op (fp16 scale/zero + fp16
   K-norm storage, round-half-up index cast, little-endian nibble
   packing, searchsorted-right bucketize). Tolerances cover legal
   rounding-path drift only (≤1.5 V quant steps; ≤1 adjacent MSE
   bucket; fp8-K within one e4m3 mantissa step; K-norm at fp16
   resolution).
3. **DETERMINISM** — second store pass byte-identical; second decode
   byte-identical.

## Harness iteration (two bugs found in MY reference, not the kernels)

- v1 run: uniform FAIL pattern (k_bit_mismatch=29 on ALL presets incl.
  fp8-K; MSE V catastrophic) — diagnosed as (a) my V scale/zero reads
  at SoA indices 0/1 (correct only for the fp8-key layout; MSE presets
  put k_norm first) and (b) a bit-exact K compare against torch's
  bf16→e4m3 cast (different rounding path than Triton's).
- v2 run: V clean everywhere; K still garbage + NaN — diagnosed as my
  `unpack_k` computing the slot stride without the V bytes (wrong data
  offset for every K read; fp16 metadata reinterpreted as e4m3 → NaN
  patterns).
- v3 run (banked): all gates green. The two intermediate runs are
  recorded here as evidence the failures were harness artifacts with
  exact structural explanations, not kernel behavior.

## Result (v3, 2026-11-03 ~21:40)

| preset | store det | decode det | decode rel_err | V step err | K err | verdict |
|---|---|---|---|---|---|---|
| turboquant_k8v4 | True | True | 0.000006 | 1.000 (tie) | 0.0000 (value-exact vs torch fp8 cast) | **PASS** |
| turboquant_4bit_nc | True | True | 0.000000 | 1.000 (tie) | 0.0000 | **PASS** |
| turboquant_k3v4_nc | True | True | 0.000000 | 0.000 | 0.0000 | **PASS** |
| turboquant_3bit_nc | True | True | 0.000000 | 0.000 | 0.0000 | **PASS** |

**P618C ROUNDTEST: PASS.**

## Verdict

The TurboQuant SoA store→decode pair is **numerically correct on XPU**:
layout-coherent, spec-conformant, deterministic, fp8-K cast exact.
Class B is therefore NOT an implementation defect — it is the quality
ceiling of 4-bit-class V quantization (16 affine levels per token-head;
8 levels on 3bit) for this model, corroborated by upstream's own
preset table (+2.71 % PPL @ 4bit_nc, +10.63 % @ k3v4, +20.59 % @ 3bit,
measured on gpt-oss; qwen3.8-27b-fp8 is beyond what those presets can
carry — p618b: degenerate REP-LOOP attractors on all four presets,
cold-included, while fp8_e4m3 runs clean).

**Disposition (plan p618c lane 2)**: deprecate `turboquant_*`
kv-cache dtypes on this fork — refuse them at the platform layer
(`vllm_gaudi/xpu`), loud error naming the certified dtype; new quant
kernels exist in NO admitted donor (scope law §3/App I) and are not
certifiable for this model anyway. fp8_e4m3 default untouched. Fix
lands in cert7 (real artifact-level refusal, not a serve-script
guard).
