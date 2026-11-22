# r212j/r212j2 — plan-ABI repair and the honest split-cost curve

**Date**: 2026-11-21 · **Leg**: r212j + r212j2 (arc7 Phase-8, offline
fast-cycle) · **Stack**: build21 wheel (same sha as r212g/r212h/i) +
`vllm-arc:v0.30.0-b70-cert25zc`, harness `p204_graphtime.py` at
engine geometry (D=256 / HREP=6 / BLOCK=64, mxfp4, graph 32×50) ·
**Runs**: `r212j_launch1.log` 10:24-10:28, `r212j2_launch1.log`
10:32-10:37 UTC

## Verdict

**The forced-plan path is REHABILITATED (two-part ABI fix,
source-proven), and the honest curve REFUTES the r212i headroom
claim: the auto heuristic beats every uniform forced split — there
is no cheap split-tuning win.**

### The two ABI defects (r212j)

1. **TILE-UNIT LAW**: the MX decode kernel walks 16-row tiles at
   EVERY page (`mx_paged_decode_xe2.cpp` routes page-64 pools
   through the `_16` policy — the r212c SG law; "every admitted
   geometry walks 16-row tiles"), but the python plan builder
   mirrored the STOCK unit (`_kv_tile_from_block_size` = 16/32/64
   by page). At BLOCK=64 the plans partitioned 64-token tiles while
   the kernel reads `wl_tile_start/count` as 16-row indices —
   1/4 KV coverage. BLOCK=16 coincided (16 == 16), which is exactly
   why the r210 ladder passed and every post-widening leg failed.
   Harness fix: `kt = 16` (`P204_KVT`).
2. **COVERAGE LAW**: the work-list walk branch is UNCLAMPED — it
   walks `[wl_tile_start, +wl_tile_count)` verbatim, while the auto
   branch clamps via `windowed_k_blocks`. r212j validated the unit
   fix with `PLANLEN=cap` (capacity-lens plans) and canaries stayed
   at ref scale (0.37-0.63): the plan walked 8× past the real 32K
   context into stale/zero pages. The valid combination is
   **`kt=16 + PLANLEN=real`** — plan coverage == auto coverage,
   only the split boundaries differ.

r212j2 ran that combination: **all forced cells canary 0 at every
ctx and split count** (s1 isolates the walk — direct O write, no
reduce kernel: CLEAN; s5/s16/s64 — partials + reduce: CLEAN).

### The honest split-cost curve (r212j2)

| ctx | auto µs | s1 µs | s5 µs | s16 µs | s64 µs |
|---|---|---|---|---|---|
| 1024 | 598.4 | 1147.6 | 603.8 | 618.8 | 618.6 |
| 8192 | 422.9 | 9097.4 | 1854.0 | 706.6 | 706.7 |
| 32768 | 1040.0 | 36323.0 | 7306.2 | 2649.3 | 2649.4 |

- **Auto wins**: 1.67× vs best-forced at 8K, 2.55× at 32K, par at
  1K. The r212i "1.5-2.1× headroom" was entirely the 4× under-walk
  artifact — retracted.
- The ~600 µs per-call floor at 1K is REAL work (s5/s16 price the
  same as auto; the starved s1 walk reproduces the r212f pathology
  at plain shape: 9.1/36.3 ms at 8K/32K).
- s16 == s64 to 0.1 µs: the builder's `target_tiles_per_wg`
  arithmetic saturates the split count at 16 for this geometry.
- The auto heuristic's device-side per-row pricing is the superior
  partition; the remaining MX-vs-fp8 walk excess (~1 ms/layer, prop
  ctx) is intrinsic to the walk/dequant structure — the next lever
  is the deeper kernel iteration (walk width, dequant fusion), not
  plan tuning.

### Engine-lane note

Wiring prebuilt plans into the engine bridge is BOTH unneeded
(auto wins) and structurally incompatible as-is: plans built at
capture time see the DUMMY-diluted lens (9/13/18/5), and the wl
walk is unclamped — engine adoption would need a device-side clamp
(`end_blk <= k_block0 + windowed_k_blocks`) plus empty-split-safe
partials. Closed as a direction; the wheel-side python builder
keeps the stock mirror and should gain the MX unit if ever
revisited (`P204_KVT` documents the law).

## Artifacts

`offline/p212j_{1024,8192,32768}.json` (unit fix at PLANLEN=cap —
the coverage-law evidence), `offline/p212j2_{1024,8192,32768}.json`
(the valid combination — all-clean curve), `r212j_launch1.log`,
`r212j2_launch1.log`. Rig: `/root/build/p199/out/p212j*`, logs
`/root/build/p199/logs/`. Tools: `r212j_tileunit.sh`,
`r212j2_realplan.sh`; harness `kt=16` (`P204_KVT`) with the
source-cited comment. The r212i bank's Section 3 headroom table is
superseded by this verdict.
