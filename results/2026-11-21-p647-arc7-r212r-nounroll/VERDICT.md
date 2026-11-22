# r212r — no-unroll probe: a PERFECT null; code size and loop form are innocent

**Date**: 2026-11-21 · **Leg**: r212r (arc7 Phase-8 iteration) · **Stack**:
r212r wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`6ca9f43efe43f078a6b346b0d2a410f4`; r212q tree + `p212r_patch.py` —
`CUTE_UNROLL` dropped from both e2m1 fast loops, identical u32 body
under a runtime loop; TU preflight clean, ninja-resume build
`r212r_build1.log` 17:33–17:49 UTC) + `vllm-arc:v0.30.0-b70-cert25zc` ·
**Run**: `r212r_leg1.log` 17:50–17:52 UTC (one cell: m0 × 32768, both
formats)

## Verdict

**The no-unroll runtime loop reproduces the unrolled time EXACTLY:
m4 m0 717.7 vs r212q's 717.7 (−0.0%). A ~15-instruction loop body
times identically to the ~400-instruction unrolled body — unrolled
CODE SIZE (icache) and LOOP FORM are both innocent, and the machine
is not latency/serialization-bound in this loop (a runtime loop
should be strictly worse if chains mattered). The ~337 µs assembly
pool is set by the per-element work itself, in a form both loop
shapes share: the scalar int ops and/or the 2-byte memcpy at the
fragment store.**

### The numbers (mxfp4/mxfp8 32K)

| cell | r212r | baseline | delta |
|---|---|---|---|
| m4 m0 (no-unroll u32) | **717.7** | 717.7 (r212q) | **−0.0%** |
| m8 m0 (byte-identity oracle) | 1313.7 | 1320.5 (r212q) | −0.5% ✓ |
| surviving assembly (m4 − tail 380.7) | 337.0 | 337.0 | 0.0 |
| canaries | 0 / 0 | — | ✓ |

### Disposition

- **PATCH STAYS** (neutral, strictly smaller code; `wheel.r212rbak`
  = 6ca9f43e preserved).
- Disasm route attempted twice (1.5 GB `libattn_kernels_xe_2.so` with
  fatbin-embedded device images; `.tgtimg` descriptor extraction
  fought back) and PARKED — the proper sycl-offload dump is a future
  dedicated leg if needed.
- **Next (r212s)**: INT-OP PRICING LADDER — probe rungs 12 (light
  tail + 4 dependent int ops/element) and 13 (+8) bracket the
  m0−m11 op delta (~8 ops = 337 µs) on the SAME wheel with m11/m0
  rerun. `slope = (m13−m12)/4` prices one scalar int op directly:
  ops share ≫ residual → algebraic op reduction (r212t); residual ≫
  ops share → the memcpy/store pool dominates → fragment
  store-packing lane.

## Artifacts

`offline/p212r_m0_32768.json`, `r212r_build1.log`, `r212r_tu.log`,
`r212r_leg1.log`. Rig: `/root/build/p199/out/p212r_m0_32768.json`,
`/root/build/r212r_{build1,tu,leg1}.log`. Tools: `p212r_patch.py`,
`r212r_nounroll.sh`.
