# r213 — the e2e truth face on the converged r212aa wheel: mxfp4 best-ever at every ctx, mxfp8 mildly regressed, and BOTH probe signs inverted (A/A-confirmed)

**Date**: 2026-11-22 · **Leg**: r213 + r213b A/A confirmation (arc7
Phase-8 e2e) · **Stack**: r212aa wheel
(`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5 `9a01790b...`; template-split
shipping body, zero probe branches) + `vllm-arc:v0.30.0-b70-cert25zc`
+ the mx_kv overlay bridge (native dispatch, engine.py 1169 posture
lines) · **Run**: `r213_e2e.log` 01:31–01:45 UTC (two boots, faces
c8192/c16384/c32768 n1 per format) + `r213b_e2e.log` 01:48–02:01 (the
same wheel, fresh boots, same faces — the A/A)

## Verdict

**GATE_PASS both legs, zero errors, zero tracebacks, 156-157 native-call
telemetry lines per format, dispatch posture `-> native` on the E5b
crossover. A/A spread 0.0-0.1% on all six faces — every sign below is
real, not boot dice.**

| fmt | ctx | r213 | r213b | mean | r212m4 bank | dev | vs fp8 bar |
|---|---|---|---|---|---|---|---|
| m4qa | 8192 | 48.5 | 48.5 | 48.5 | 48.9 | −0.9% | 1.20x |
| m4qa | 16384 | 55.9 | 55.9 | 55.9 | 56.7 | −1.5% | 1.36x |
| m4qa | 32768 | 70.2 | 70.3 | 70.2 | 73.1 | **−3.9%** | 1.65x |
| m8qa | 8192 | 51.6 | 51.6 | 51.6 | 50.5 | +2.2% | 1.28x |
| m8qa | 16384 | 63.2 | 63.2 | 63.2 | 60.3 | +4.8% | 1.54x |
| m8qa | 32768 | 84.8 | 84.8 | 84.8 | 81.7 | **+3.8%** | 2.00x |

(fp8 bar 40.4/41.1/42.5 = ctrlA24 cert25zc; ITL p50 ms/token, n1,
MTP spec 3, graph on.)

**mxfp4 is the best e2e MX state ever banked** — improved at every ctx,
monotone with decode share, and now leads mxfp8 by 17% at 32K (70.2 vs
84.8; at the r212m4 bank the lead was 11%). mxfp8 regressed mildly and
consistently (+2.2/+4.8/+3.8%).

## The finding: probe/e2e sign inversion

The offline probe cells predicted the OPPOSITE direction on BOTH lanes:
the m0 mxfp4 cell regressed +29.6% (749.5 → 971.6, the "attractor tax"
that drove r212y-z-aa-ab) while e2e mxfp4 IMPROVED −3.9%; the m8 mxfp8
oracle hit a new best −9.6% (1320.7 → 1193.7) while e2e mxfp8 REGRESSED
+3.8%. Two boots each, spreads ≤0.1% — this is not noise.

**Reading**: the p204 offline harness replays the decode kernel in its
own captured shape; the engine's plan-build buckets capture different
shapes, and post-template-split each instantiation's codegen basin is
shape-dependent (the r212aa per-instantiation asymmetry law, now
observed across shapes at engine scale). The probe harness keeps its
valid scope — byte-correctness canaries, same-shape lane attribution
(k_lane structure), same-shape A/B — but **cross-edit deltas and even
signs do not transfer to e2e**. The whole r212y-ab "attractor tax"
narrative was probe-shape-scoped: the engine never paid it.

## Consequences

1. **e2e faces are the arbiter for cross-edit deltas from here.** The
   fast-cycle instrument for the remaining gap is boot-and-face (two
   boots ≈ 15 min on this image), not the offline cell.
2. **The remaining pool to the goal** (fp8-level decode): m4 at 32K is
   70.2 vs 42.5 = 27.7 ms/tok of gap (1.65x); m8 is 2.00x. The r212n
   lane attribution (K 178 / Vg 197 / floor 247 / other ~111 µs of the
   749-class call) still names WHERE the time goes, but its magnitudes
   are probe-shape-scoped and must be re-priced against e2e deltas.
3. **The queued SWAR decode-ALU item stays queued** — but its payoff
   estimate (~120-150 µs, probe-derived) is now an upper bound of
   unknown transfer; it needs an e2e face to certify any real gain.
4. mxfp8's +4% e2e regression rides the e4m3 engine-shape basin of the
   r212o-r212aa edit series; not urgent (m4 dominates as the KV
   compression lane), but it inverts the "m8 new best" probe reading —
   the probe oracle's 1193.7 is not an engine truth.

**Reverted**: nothing — the wheel under test is the converged shipping
state (r212aa), measured as-is. No tree changes were made this leg.
