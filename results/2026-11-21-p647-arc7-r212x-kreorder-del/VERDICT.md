# r212x — K-reorder deletion: NEUTRAL (canary-clean); r212w's "reorder = 100% of K lane" refuted as DCE contamination

**Date**: 2026-11-21 · **Leg**: r212x (arc7 Phase-8 instrument) · **Stack**:
r212x wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`ee1724bbae0764461e7b2e8f29e24f22`; r212w tree + two edits —
`p212x_patch.py`: dequant_dtile's dst `tKrDeq` -> `tSrK`
(`thr_mma_qk.partition_sg_fragment_B`) with the
`reorder(tKrDeq, tSrK)` call DELETED (the r200b twin), the probe-18
severance retired; `p212x2_guard.py`: NESTED-FORM GUARD — tSrK's tv
is rank-2 `((2,8),((2,8),(1,4)))` = the SAME walk law as the pinned
rank-3 staged form (idx2crd: dv = v0 + 2·v1 + 16·v2, row = t_b + 8·v0,
elem = t_a + 2·v1 + 16·v2; the (1)-mode is stride-0 degenerate), so
the staged body runs verbatim into the MMA operand and only the form
pin differs: rank-3 keeps `assert_staged_tv_form`, rank-2 gets a
positive shape static_assert; first attempt's dedicated MMA-walk
branch was wrong (dv>>3) and reverted — TU preflight clean
(`r212x_build2.log`; the two rank-3-pin trips archived as
`r212x_tu.log`/`r212x_tu2.log`)) + `vllm-arc:v0.30.0-b70-cert25zc` ·
**Run**: `r212x_leg1.log` 21:58–22:04 UTC (three same-wheel cells:
m0 / m11 / m17 × 32768)

## Verdict

**THE REORDER CALL WAS NOT THE COST. Deleting it changed nothing:
k_lane = m17−m11 = +600.1 vs r212w's +595.5 (same within dice), and
m0's canary = 0.0 — the staged body writes the MMA operand directly,
BYTE-CORRECT.** The r200b-twin hypothesis is closed as a negative:

| cell | µs | r212w same cell | reading |
|---|---|---|---|
| m0 (K->tSrK + V real) | 995.9 | 996.3 | unchanged |
| m11 (K const + V light) | 417.6 | 430.1 | baseline |
| m17 (K->tSrK + V light) | 1017.6 | 1025.7 | k_lane 600.1 vs 595.5 |
| m8 oracle (mxfp8) | 1775.8 (+34.5%) | 1770.5 (+34.1%) | 3rd attractor wheel |

**r212w's attribution was DCE-contaminated.** m18 (K real + reorder
severed) measured "≡ m11" — but severing the reorder left tKrDeq's
data DEAD, so the compiler eliminated the dequant body (gmem loads
included); "loads+staging+decode = +0.5 µs" was vacuous, not a
measurement. The consistent structure across r212w's m17/m18/m20 and
r212x's m17: **the ~595 µs follows tSrK LIVENESS, not the writer** —
with real K in the MMA operand, GEMM 1's DPAS serializes behind the
dequant dependency chain (loads -> staging -> decode -> tSrK
registers -> DPAS); with tSrK const-filled the chain is severed at
the source and GEMM 1 issues freely. Rough live split from r212w
same-wheel cells: full dtile incremental ~256 µs (m17−m20) +
light-live floor ~339 µs (m20−m11) — both components of the same
chain, not reorder line-items.

The m8 +34% GEMM-1 attractor persisted through a THIRD wheel (edits
no longer semantically null) — k_lane on attractor wheels is stable
(595.5/600.1) and remains the working currency; the clean-wheel pool
(r212u: 338 µs) says the chain cost may be ~260 µs smaller off the
attractor.

### The fork

**Next (r212y): attack the DEPENDENCY CHAIN, not the writer.**
Candidates in order: (1) software-pipeline the K decode across
k-tiles (double-buffered tSrK — decode (K,D+1) while GEMM consumes
(K,D)) so the chain hides under GEMM work; (2) widen the r201
prefetch distance / early staging to shave the load-latency share;
(3) attractor-escape baseline (precomputed-bool probe gates or a
fresh minimal-diff wheel from the r212u tree) if cross-wheel
absolutes become load-bearing. Also queued: the m18-class severance
pattern needs a liveness canary (opaque sink) — DCE silently converts
severance probes into vacuous floors.

## Artifacts

`offline/p212x_{m0,m11,m17}_32768.json`, `r212x_build2.log`,
`r212x_tu.log`, `r212x_tu2.log`, `r212x_leg1.log`. Rig:
`/root/build/p199/out/p212x_*`, `/root/build/r212x_{build2,tu,tu2,leg1}.log`.
Tools: `p212x_patch.py`, `p212x2_guard.py`, `r212x_kreorder_del.sh`.
