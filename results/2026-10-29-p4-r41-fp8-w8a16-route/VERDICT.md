# r41 — p44 fix leg: block-fp8 reroute to fp8_gemm_w8a16 — SUPERIORITY BARS BEATEN (solo +8.3%, agg#2 +9.6–12.2%, TTFT parity −5% at all three depths)

One change (E-LAW): `XPUFp8BlockScaledMMKernel.apply_input_quant=False` +
`apply_block_scaled_mm -> _xpu_C.fp8_gemm_w8a16(A_fp16, B.t(), Bs.t(), …)`
for fp16/bf16 activations (stock W8A8 fallback kept). Product form =
`vllm_gaudi/xpu/patches/fp8_block_w8a16.py` (registered in
`_XPU_PATCH_MODULES`); leg form = the same file + one sitecustomize hook
(`arc_p44_patch`) in the arc-test cap — one source, two callers. Zero new
binaries. Root cause + microbench: r40
(results/2026-10-29-p4-r40-forward-gemm-capture).

## Faces (TIMERS-OFF clean boot, first-traffic battery, D-certified tools
md5-pinned unchanged: run7n e5b8d594…, longctx 3854f26f…, prefix bfeaa4c4…)

| face | r37 baseline | champion bar | r41 | verdict |
|---|---|---|---|---|
| solo200 tps | 59.07–59.13 | **79.74** | **86.31/86.33/86.64** | **BAR BEATEN +8.3%** (+46% vs baseline) |
| agg4x256#1 | 196.06–209.20 | 254.33 | **272.58 / 278.12** | bar#1 beaten +7–9% |
| agg4x256#2 (warm) | 209.20 | **274.41** | **300.60 / 308.07** | **BAR BEATEN +9.6–12.2%** |
| solo dt | 3.38–3.39 s | 2.51 | **2.31–2.32 s** | faster than champion |
| TTFT t1 8k | 5726 ms | 3944 | **3729** | parity bar beaten |
| TTFT t1 16k | 12074 ms | 8083 | **7788** | parity bar beaten |
| TTFT t1 32k | 25288 ms | 17447 | **16630** | parity bar beaten |
| prefill rate | 1.27–1.39k | 1836 (D) | **~2.1k tok/s** (7950/3.729s) | beats D's matrix face |
| warm t2 | 1.14–1.85 s | 0.587–0.741 band | 1.108 / 0.922 / 1.426 s | r36 posture class preserved |

Step arithmetic (from run7n + counters): solo emitted ≈ 4.14, steps ≈ 48.3
→ **c1 step ≈ 47–48 ms** (r37 63–70, D 43) = −27…−34% vs baseline, −10%
residual vs D. Agg#2: emitted 4.31–4.87 → c4 step ≈ 53–58 ms (D ≈ 46):
our emitted advantage (4.3–4.9 vs D 3.46) more than covers the residual.
Bars clear with the CURRENT tail intact — exactly the r39/r40 projection.

## Gates
- survival: 832-token flip payload FIRST traffic HTTP 200 (6.60 s), small 200.
- spec gate: pos1 **0.8919 ≥ 0.85 target MET** (hard 0.5); profile
  [0.892/0.762/0.632/0.562], agg 0.712, mean len 4.36 (r38:
  [0.910/0.806/0.682/0.632] agg 0.7575) — slight per-pos decay from the
  numerics change, swamped by the step-time win; still an open P4
  convergence item (below).
- determinism: 3/3 fresh seeded prompts byte-stable.
- **integrity: 5/5** cold-vs-re-hit sha byte-identical (LMCache D-law).
- golden vs banked ref: 3/5 DRIFT — **expected**: w8a16 removes activation
  quant (numerics strictly better per r40 relerr), outputs legitimately
  shift on near-ties (drifted cells differ by EOS timing/length; 256/1024
  byte-identical). **golden_ref re-bank owed at P5 certification** with the
  p44-baked image; not a regression.
- prefix bench: cold med 2.51 s / rehit 0.99 / churn 112@2.72 / rehit-after
  1.17 — posture class preserved.
- patch visibility: "fp8 block linear reroute ACTIVE" logged in every
  process (API server + both TP workers); kill-switch
  VLLM_ARC_FP8_W8A16_DISABLE=1 for A/B.

## Numerics note
w8a16 consumes the original fp16 activation (no per-token-group quant) —
r40 measured relerr 4–5e-4 vs the fp32 dequant-block reference on every
shape; the change REMOVES a quantization error, it does not add one. The
acceptance decay (above) is a draft/target argmax-tie redistribution, not
degradation — tracked as the existing optional diagnostic.

## Owed next (P4 convergence → P5)
1. Bake p44 into the image (arc_build src update + build tag
   v0.30.0-b70-p44) — fold with any remaining P4 legs; banking directive.
2. golden_ref re-bank + full 18-cell matrix on the certified image.
3. k5 candidate re-decision on the new posture (re-canonicalizes sha
   batteries; k5's +6.7/7.8% was measured on the OLD forward).
4. Optional residual lanes (post-bars, owner-gated): acceptance-decay
   diagnostic (drafter-quant override one-boot), c4 adder shape, ESIMD
   M≤2 port (~0.1 ms/step), k1k matrix cells re-run.

## Rig end-state
arc-test stopped after leg; radiance-test (W14) restored, HEALTH=200
(poll 3). No arc image left running. Leg artifacts:
rig /root/build/legs/arc-p4-r41/, local results dir (this file's dir).
