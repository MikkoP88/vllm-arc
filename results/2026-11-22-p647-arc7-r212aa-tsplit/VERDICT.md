# r212aa — template-split: MXFP8 LANE ESCAPES AND SETS A NEW BEST (1193.7, −9.6%); mxfp4 partial (971.6); instrumented body preserved dead-on

**Date**: 2026-11-22 · **Leg**: r212aa (arc7 Phase-8 instrument) ·
**Stack**: r212aa wheel
(`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`9a01790b5748cb4d7e1eac9c099a997e`; r212y tree + `p212aa_patch.py` —
`operator()` becomes a thin dispatcher on `params.vlane_probe == 0`;
the whole mainloop body moves to
`body<has_large_surface, kInstr, QVCoord>` with every probe read
(kp/vp sever gates, dequant probe args, stage_on, barrier gate,
const-P, dbg lane) folded under `kInstr`, so the SHIPPING body
(kInstr=false) compiles with **zero probe branches**; TU clean, 9
pre-existing warnings only, `r212aa_tu.log`) +
`vllm-arc:v0.30.0-b70-cert25zc` · **Run**: `r212aa_leg1.log`
00:10–00:16 UTC (three same-wheel cells: m0 / m11 / m17 × 32768)

## Verdict

**The e4m3 (MXFP8) shipping lane escaped the attractor and improved
past the best-known layout: m8 = 1193.7 = −9.6% vs the r212q-era
1320.5 reference — the first oracle below 1320 ever measured, canary
0.0.** The five-wheel +34% basin era is over for mxfp8: the
zero-probe-branch structure (no runtime sever checks in GEMM 1,
stage_on folded true, unconditional barrier, no dbg check, constant
dequant_reg args) is not just clean — it is a genuinely better e4m3
layout by ~127 µs. The queued "~+450 µs e4m3 attractor regression"
item closes as a WIN in the other direction.

**The e2m1 (MXFP4) shipping lane recovered partially: m0 = 971.6**
(r212y attractor 1006.6, −35; r212u-era best 718.0, +254). The two
lanes are separate template instantiations and took different fresh
basins from the same edit — the per-instantiation asymmetry law
(r212z) again. The remaining +254 rides the K walk (see k_lane below).

**The instrumented body kept its layout dead-on**: m11 = 414.2 (r212y:
414.0) and k_lane = +609.3 (band 595–606, 5th confirmation) — expected,
as body<true> is textually the old code. Consequence worth recording:
the m11 floor itself carries the r212u→r212v-era +34 (r212u-era floor
was ~380), so "band ~414" is an attractor-era floor; cross-era floor
comparisons must account for that.

| cell | r212aa | r212y | note |
|---|---|---|---|
| m0 (SHIPPING body, K+V real) | 971.6 | 1006.6 | r212u-era 718.0 |
| m11 (K const + V light, instr) | 414.2 | 414.0 | dead-on |
| m17 (K real + V light, instr) | 1023.5 | 1020.3 | dead-on |
| k_lane (m17−m11) | +609.3 | +606.3 | band 595–606 |
| m8 m0 oracle (mxfp8) | **1193.7 (−9.6%)** | 1769.2 (+34.0%) | new best |
| m11 mxfp8 (rode along) | 549.7 | — | e4m3 const floor |

**Gate note**: the leg's `M8_STABILITY_FAIL` (rc=1) is an
asymmetric-band misfire — the ±6% oracle band was built to catch
attractor REGRESSIONS; a canary-clean byte-correct improvement on the
low side is a win, not instability. Fork text and the m8 check are
corrected post-hoc to one-sided (fail only beyond +6%; low side prints
M8_NEW_BEST). All canaries 0.0.

## The fork

**r212ab: compile-time-parity pipelined D-loop in the SHIPPING e2m1
body only.** r212z proved ~490 µs of the K lane is overlap-coverable
(decode (K,D+1) before gemm(D)) but its runtime-parity ternary demoted
the operand (region collapse). The peel removes the ternary entirely:
in body<false>, the e2m1 GEMM-1 D-loop becomes an unrolled-by-2 loop
with STATIC buffer names per textual half (even half: cur=tSrK,
lookahead→tSrK1; odd half: cur=tSrK1, lookahead→tSrK), prologue decodes
(K,0), DD=4 folds the guards. The e4m3 branch and the instrumented
body stay textually untouched (e4m3 is the new best — do not re-roll
it; body<true> keeps ablation semantics). Cells 0/11/17 + m8 oracle as
always; register pressure (2×32-reg B fragments + staging) is the
declared risk — m0 and canary gate it, m11 gates the instrumented
lane's stability. Upside: up to ~490 µs off m0 (toward ~480–550),
fresh dice besides.
