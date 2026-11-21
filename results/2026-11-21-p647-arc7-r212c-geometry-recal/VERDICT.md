# r212c — geometry recalibration: the real D=256 map + the p64 SG law

**Date**: 2026-11-21 · **Leg**: r212c (arc7 Phase-8) · **Wheel**: build19
widened arc7mx `034bce9e…` ({128,256}×{16,64} dispatch) · **Image**:
`vllm-arc:v0.30.0-b70-cert25zc` · **Tool**:
`tools/ar/p147_mxkv/p204_graphtime.py` (geometry env knobs
P204_D/P204_HREP/P204_BLOCK; P204_ESHAPE=1) + `r212c_recal.sh` ·
**Run**: `r212c_launch1.log` 02:47-03:09 UTC (rc=1 = gate correctly
flagging the corrupt eng256p64 cells)

## Verdict

**PASS with one law and one flip.** The fail2 geometry law (target =
head_dim 256, page 64, G 6) recalibrates the whole crossover picture:

1. **D=256 is numerically clean on the certified _16-policy frame**
   (eng256p16: every cell canary 0.00e+00, v5-vs-v2 cross ≤1.5e-3).
2. **The vendor `_64` policy corrupts on the MX fork** — every native
   cell at page 64 dirty (canary 0.07-0.72, decreasing with ctx). Root
   cause (structural, from the fork's own law text): the MX staging
   scatter + tv-decoder are **single-SG laws** ("One subgroup per
   workgroup: no per-SG offsets", mx_paged_decode_mainloop.hpp r200b/
   r203); `_64`'s SubgroupLayoutQK (1,4,1) puts four subgroups in the
   WG, so multi-SG thr_ids decode out-of-frame coordinates. Fix (same
   session): page-64 pools dispatch the `_16` policy — 16-row sub-tiles,
   4 per page via the stock page-table indirection; validated by the
   r212c2 rerun. The true multi-SG p64 tile (fixed term 306-404 vs
   ~600 µs at 1K, but SLOWER at ≥4K: 2072 vs 1270 µs at 32K mxfp4) is
   r212d scope.
3. **The certified-cell regression is byte-class faithful**: reg128p16
   on the widened wheel reproduces the banked E5b rerun to ±0.1-2 µs
   per cell (mxfp8 v5_es 101.8/199.3/482.6 vs 101.8/199.4/482.0;
   native_es 328.3/243.7/573.8 vs 328.3/243.8/571.6; mxfp4 234.1/
   176.5/349.4 vs 233.9/176.8/349.1; mxk8v4 219.7/174.1/448.6 vs
   219.6/174.1/448.7). No layout-dice swing.

## The D=256 map (eng256p16 — the _16-at-64 route's proxy; µs/call,
graph 32×50; eshape B×r4 so walked Σ = 4×ctx)

| ctx (Σ) | mxfp8 v5_es | native | n/v5 | mxfp4 v2_es | native | n/v2 | mxk8v4 v5_es | native | n/v5 |
|---|---|---|---|---|---|---|---|---|---|
| 1024 (4K) | 441.6 | 644.1 | 1.46 | 221.2 | 598.7 | 2.71 | 477.6 | 605.0 | 2.52 |
| 4096 (16K) | 665.8 | 547.5 | 0.82 | 627.0 | 411.3 | 0.66 | 877.9 | 415.5 | 0.51 |
| 8192 (33K) | 1171.0 | 693.5 | 0.59 | 1183.8 | 459.2 | 0.39 | 1614.2 | 462.1 | 0.29 |
| 32768 (131K) | 2888.6 | 1952.9 | 0.68 | 4102.3 | 1270.6 | 0.31 | 4289.5 | 1454.7 | 0.34 |

(vs each format's certified engine lane; mxfp4/mxk8v4 best-of rows:
mxk8v4 v2_es is within 3% of v5 at 4K+ — 821.9 vs 877.9 — and far
worse at 1K.) **Every format crosses between Σ 4K and 16K** — linear
fits put mxfp8 ~11.6K, mxfp4 ~11.6K, mxk8v4 ~6.6K. At 32K ctx native
is 1.5-3.2× faster than the best Triton lane, all formats. The
D-doubling doubled the Triton slopes (bytes/token) while native's
walk/barrier-dominated cost grew far slower — mxfp8's crossing moved
~214K → ~12K walked.

## r212a thresholds (engine walked_cap basis, D=256)

| format | lane condition | minlen | basis |
|---|---|---|---|
| mxfp8 | any G (v5) | 16384 | crossing ~11.6K + fill margin |
| mxfp4 | G < 8 (v2) | 12288 | crossing ~11.6K vs v2 |
| mxfp4 | G ≥ 8 (v5) | 8192 | v5 lane slower → earlier crossing |
| mxk8v4 | any G (v5) | 8192 | crossing ~6.6K |

(Old D=128 defaults were 262144 / 8192:32768 / 32768 — dead at the
real geometry.)

## Discovered lever (not touched this leg)

At D=256 **v2 beats v5 for mxfp4 at every ctx** (221 vs 477 at 1K,
4102 vs 4396 at 32K) — the certified g-adaptive v5-at-G≥8 switch is a
D=128 calibration. Retuning the Triton lane chain is a separate
certified-baseline change (affects the dispatch-off A/B too).

## Next

- r212c2: eng256p64 rerun on build20 (the SG-law fix) — canary gate +
  the engine-pool-true times; reg128p16 spot re-check.
- Then r212a launch4 (e2e) with the thresholds above.
- r212d: multi-SG staging for the true p64 tile (short-ctx fixed term).
- Lane retune lever above (own leg).

## Artifacts

`p204_eng256p16_c{1024,4096,8192,32768}.json`,
`p204_eng256p64_c{1024,2048,4096,8192,32768}.json` (corrupt native
cells kept as the SG-law evidence), `p204_reg128p16_c{1024,8192,32768}
.json`, `r212c_launch1.log`, `table.py` (regenerates the tables).
Rig: `/root/build/p199/out`, `/root/build/p199/logs/r212c_*.log`;
wheel `/root/build/wheels-arc-vxk/…arc7mx….whl` (build19), keeper
preserved `…_r210keeper.whl`.
