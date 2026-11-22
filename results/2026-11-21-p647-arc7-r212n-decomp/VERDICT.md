# r212n — fresh full decomposition (incl. probe 9) on the converged r212m3 wheel

**Date**: 2026-11-21 · **Leg**: r212n (arc7 Phase-8, deep-analysis leg on
the shipping binary) · **Stack**: r212m3 wheel (md5
`e50f1c3aa38729931c0d6ccc9c57a370`, pinned), `vllm-arc:v0.30.0-b70-
cert25zc`, harness `p204_graphtime.py` at engine geometry (D=256 /
HREP=6 / BLOCK=64, graph 32×50, SPLITS=auto) · **Run**:
`r212n_leg1.log` 15:07–15:19 UTC (6 cells: probes {0,2,3,6,8,9} × 32768)

## Verdict

**Same-binary reproduction is exact (m0 749.5/1320.7 vs banked
749.3/1319.7 = +0.0/+0.1%; every lane within 1.6%) — the banked r212m3
table stands. Probe 9 on this wheel prices the post-direct-bf16 decode
residual at ≥370.5 µs (49% of the mxfp4 call) and exposes a +104 µs
e2m1-skeleton floor growth (246.6 vs the r212k-era 143; mxfp8's floor
stayed at 171.6 = the r212k value exactly, so the growth is
e2m1-path-specific, not a wheel effect).**

### The table (32K)

| lane | mxfp4 | mxfp8 |
|---|---|---|
| full (m0) | 749.5 | 1320.7 |
| K chain | 178.2 (−0.0%) | 390.6 (−1.6%) |
| V gather | 196.7 (−0.1%) | 334.2 (−1.3%) |
| V stage | 16.7 | 364.9 |
| floor (c8) | **246.6** | 171.6 |
| **m9** | **379.0** | **588.6** |
| decode residual (m0−m9) | **370.5** | 732.1 |
| walk+softmax+P pool (m9−c8) | 132.4 | 417.0 |

### Three findings that steer the next iteration

1. **The decode tail is STILL the #1 pool after direct-bf16.** The
   residual is ≥370.5 µs (a lower bound — m3's probe-9 tail is heavier
   than r212m's was: it re-derives `e8m0_to_scale(kc[r])` per element
   because the trimmed sc vector only materializes when
   `!fast_band || probe_val`, so part of the old tail cost moved INTO
   the probe cell). Cross-check: K+Vg lanes (375 µs total) ≈ the
   residual — the lanes are nearly pure decode ALU, matching r212k's
   per-element-bound triangulation.
2. **The scalar-fragment-store tax hypothesis.** ALU arithmetic says
   the direct-bf16 tail (~10 ops/element × ~50M elements/call) should
   cost ~36–140 µs at SIMD/scalar mixed throughput — the measured
   ≥370 µs implies the per-element `dst(i)` fragment writes (scalar
   register inserts per element) dominate, not the arithmetic.
   Next lever candidate: output-side packing (assemble 4×bf16 into a
   u64 where the dst fragment tv places a WI's elements contiguously —
   distinct from the CLOSED input nibble-pairing, which failed on the
   cross-WI phase law).
3. **The m4 floor's +104 µs growth is e2m1-specific.** m8's floor
   reproduced r212k's 171.0 exactly (171.6) while m4's moved 143 →
   246.6 across the m-series. The prep ALU itself is too small to
   explain it (~30 ops/group × 786K groups ≈ 2–3 µs); the suspect is
   code-footprint/register-allocation shift in the skeleton from
   carrying the band+branch machinery — layout-class, not
   arithmetic-class. Pricing instrument queued (probe 10: kc/band →
   const) but expected ~0 on the ALU count; the source read comes
   first.

### Disposition

- Next: read the `dequant_dtile`/`dequant_vslm` walk + dst fragment tv
  layouts on the rig to assess output packing; then the r212o
  iteration (probe 10 for the prep and/or the packed-store rewrite).
- The BW-floor north star: mxfp4's walk bytes ≈ HALF of fp8's — at
  pure BW, m4 decode should be FASTER than fp8; the entire 1.72× e2e
  gap is walk inefficiency + floor, all recoverable in principle.

## Artifacts

`offline/p212n_m{0,2,3,6,8,9}_32768.json`, `r212n_leg1.log`. Rig:
`/root/build/p199/out/p212n_*`, `/root/build/p199/logs/r212n_*`,
`/root/build/r212n_leg1.log`. Tool: `tools/ar/p147_mxkv/r212n_probedec.sh`.
