# r212 E5b — engine-shape crossover: native vs the lanes the engine rides

**Date**: 2026-11-21 · **Leg**: r212_e5b (arc7 Phase-8, E5b) ·
**Wheel**: arc7mx r210 keeper `7247e83b` · **Image**:
`vllm-arc:v0.30.0-b70-cert25zc` · **Tool**:
`tools/ar/p147_mxkv/p204_graphtime.py` (P204_ESHAPE=1: B×r4 draft
rows — the spec-k3 verify shape; v5 cell via `mx_decode_rows_dot`) +
`r212_e5b.sh`

## Verdict

**PASS with calibration flip — E5's v2-based thresholds are DEAD for
the v5-lane formats; the r212a defaults are now per-lane.** E5b adds
engine-shape cells (R=8 rows) and — critically — the V5 comparator for
mxfp8/mxk8v4 (their certified engine lane; E5 compared v2 only).
v5 dominates v2_es at every cell for those formats (e.g. mxk8v4 1K:
75 vs 224 µs), so the honest native crossings sit far higher than
E5's: **mxfp8 ~214K walked tokens (extrapolated; never at n1 — the
32K cell is still v5-favorable 1.19×), mxfp4 ~7.5K (n1, v2 lane) /
~33.8K (n8, v5 lane), mxk8v4 ~33K (v5 lane)**. mxfp4-n1 is native's
big engine win (2K+ ctx: 0.86→0.14×, 7.2× at 32K).

## First-run defect (superseded by the rerun)

Run 1 (`r212_e5b_launch.log`) called `mx_decode_rows_dot` WITHOUT
`block_size` — default 64 vs the harness pool's BLOCK=16: the v5_es
cell walked wrong strides (NaN/garbage canaries; the lone 0.0 was
same-deterministic-garbage). Fixed by passing `block_size=BLOCK` at
both call sites + adding the v5-vs-v2 eager cross-check. Rerun
(`r212_e5b_launch2.log`): all v5_es canaries 0; cross-checks
1e-4–1.9e-3 (fp16-dot class, gate ≤2e-3) — the v5 block-16 walk is
correct. All numbers below are the rerun.

## Results (device-only µs/call; graph 32×50; walked Σ = 4×ctx)

| ctx (Σ) | mxfp8 v5_es | native | n/v5 | mxfp4 v2_es | native | n/v2 | mxk8v4 v5_es | native | n/v5 |
|---|---|---|---|---|---|---|---|---|---|
| 1024 (4K) | 101.8 | 328.3 | 3.22 | 96.4 | 233.9 | 2.43 | 75.3 | 219.6 | 2.92 |
| 2048 (8K) | 107.5 | 204.2 | 1.90 | 171.4 | 146.8 | 0.86 | 90.3 | 138.8 | 1.54 |
| 4096 (16K) | 108.5 | 219.4 | 2.02 | 331.2 | 156.5 | 0.47 | 111.7 | 152.4 | 1.36 |
| 8192 (33K) | 199.4 | 243.8 | 1.22 | 675.1 | 176.8 | 0.26 | 173.1 | 174.1 | 1.01 |
| 32768 (131K) | 482.0 | 571.6 | 1.19 | 2529.7 | 349.1 | 0.14 | 456.8 | 448.7 | 0.98 |

(v2_es/v5_es shown per the lane each format rides: v5 for
mxfp8/mxk8v4, the G-adaptive v2/v5 chain for mxfp4 — n1 compares v2_es,
n8 v5_es.) native_es fixed term stays ~flat 147-328 µs across Σ
(E4-lite's barrier/walk protocol); its slope is the shallowest.

## r212a operative thresholds (walked Σ = rows × max_seq_len)

| format | lane condition | minlen | basis |
|---|---|---|---|
| mxfp8 | any G (v5) | 262144 | n/v5 crosses ~214K (extrapolated 8K→32K); engages n8-2K+ ctx |
| mxfp4 | G < 8 (v2) | 8192 | n/v2 crosses ~7.5K; native from 2K ctx at n1 |
| mxfp4 | G ≥ 8 (v5) | 32768 | n/v5 crosses ~33.8K (8K cell = 0.99 wash) |
| mxk8v4 | any G (v5) | 32768 | n/v5 crosses ~33K (8K cell 1.01, 32K 0.98) |

Env retune: `ARC_MXKV_NATIVE_MINLEN="fmt:N"` (both lanes) or
`"fmt:N1:N2"` (n1-lane:n8-lane).

## Findings

1. **E5's v2 comparison overstated native at n1 for mxfp8/mxk8v4** —
   the engine lanes there are v5, which is 1.5-3× faster than v2_es;
   the native kernel only overtakes v5 at high walked budgets (n8
   long ctx) or, for mxfp4-n1, from 2K ctx (v2 lane).
2. **mxfp4 is the dispatch's headline**: n1 2K-32K goes 0.86→0.14×
   (up to 7.2× device-side at 32K) with zero short-ctx exposure.
3. The native fixed term did NOT grow with R=8 (147-328 µs flat) —
   the Σ (walked-token) cost model holds across row counts, so the
   rows×max_seq_len estimator is sound.
4. Native's r208 "32K wins" (1.80×/2.02×/4.73× vs v2) re-rates vs the
   true lanes: mxfp4 7.2× (bigger — v2_es walks 4× the rows), mxfp8
   0.85× (v5 wins at n1-32K), mxk8v4 1.02× (wash) — the dispatch
   keeps each format on its winner.

## Next

- r212a fastcycle (engine e2e, ARC_MXKV_NATIVE=1 + per-lane
  thresholds): dispatch posture lines per face, quality gate, rates
  vs the zcA27 certified cells.
- r212b native barrier reduction (raises the native ceiling — would
  pull every crossing left; mxfp8 could become n1-viable).

## Artifacts

`p204_graphtime_es_c{1024,2048,4096,8192,32768}.json`,
`r212_e5b_c*.log`, `r212_e5b_launch.log` (run 1, superseded),
`r212_e5b_launch2.log` (rerun; rig: `/root/build/p199/out`,
`/root/build/p199/logs`).
