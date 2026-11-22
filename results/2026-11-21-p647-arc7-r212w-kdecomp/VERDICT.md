# r212w — k-decomposition: THE REORDER IS 100% OF THE K LANE; second consecutive m8 dice-FAIL

**Date**: 2026-11-21 · **Leg**: r212w (arc7 Phase-8 instrument) · **Stack**:
r212w wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`9060d4e75b9ca342d404250440fda213`; r212v tree + `p212w_patch.py` —
m18 = K real + reorder SEVERED (dequant_dtile runs fully, tSrK
const-filled after), m20 = K real + dtile light tail + REAL reorder;
mainloop-only; TU preflight clean, ninja-resume build
`r212w_build1.log`) + `vllm-arc:v0.30.0-b70-cert25zc` · **Run**:
`r212w_leg1.log` 20:57–21:07 UTC (five same-wheel cells:
m0 / m11 / m17 / m18 / m20 × 32768)

## Verdict

**THE REORDER IS THE K LANE'S ENTIRE COST. m18 (K real + reorder
severed) = 430.6 ≡ m11 (K const) = 430.1 (+0.5 µs): the K byte loads
(fully covered by the r201 prefetch distance), the window staging,
and the decode all cost ZERO incremental. reorder(tKrDeq, tSrK)
carries 595.1 µs = 100% of the K lane on this wheel (~338 µs on the
clean r212u dice):**

| cell | µs | reading |
|---|---|---|
| m0 (K real + V real) | 996.3 | |
| m11 (K const + V light) | 430.1 | |
| m17 (K real + V light) | 1025.7 | k_lane = 595.5 |
| **m18 (K real, reorder severed)** | **430.6** | **≡ m11: loads+staging+decode = 0** |
| m20 (K light tail + reorder) | 769.5 | reorder still costs 339 with a lighter producer |
| (c) reorder share | **595.1 (100% of k_lane)** | |

This is the exact r200a story on the V side (cross-frame reorder
classified Generic — scalar shuffles, +0.36 ms) that r200b deleted
via SLM staging; the K twin reorder was never deleted. The (a)/(b)
sub-split came out confounded (negative under overlap) — moot once
(a)+(b) ≡ 0.

**m8 FAILED AGAIN: 1770.5 (+34.1% vs r212v's +33.8%)** — two
consecutive builds regressed by the SAME +34% after the
semantically-null gate edits: this is a DETERMINISTIC codegen
attractor for the edited GEMM-1 region, not random dice. Absolute
cross-wheel numbers on these wheels are untrustworthy; same-wheel
deltas (the decomposition above) are the currency.

### The fork

**Next (r212x): DELETE THE K REORDER — the r200b twin.** Pass tSrK
(`thr_mma_qk.partition_sg_fragment_B`) directly as dequant_dtile's
dst and drop the reorder call. dtile derives its walk from
dst.tv_layout(): if tSrK's val form qualifies, the staged law runs
straight into the MMA operand; if not, the r198 scalar-walk fallback
(correct by construction, slower reads); law violation trips
assert_staged_tv_form loudly. m0's canary is the byte-level net;
m17 should collapse to m11 if the K lane dies. m8 (e4m3 untouched)
gates the dice.

## Artifacts

`offline/p212w_{m0,m11,m17,m18,m20}_32768.json`, `r212w_build1.log`,
`r212w_tu.log`, `r212w_leg1.log`. Rig: `/root/build/p199/out/p212w_*`,
`/root/build/r212w_{build1,tu,leg1}.log`. Tools: `p212w_patch.py`,
`r212w_kdecomp.sh`.
