# VERDICT — P6.47 Phase 7 attempt 2: deep-split re-sweep, TILE_FLOOR (r193)

**LEG**: 2026-11-19-p647-v7a-attempt2-tfloor (rig bank /root/build/p647/tflA26)
**IMAGE**: vllm-arc:v0.30.0-b70-cert25zb + in-container swap of the
TILE_FLOOR-bearing triton_mx_decode.py (r192 mechanism; image untouched)
**KNOB**: TILE_FLOOR constexpr on both split-dot kernels (commit d9479d45),
bit-frozen default 4 (the certified A14 rule); env ARC_MXKV_V5_TFLOOR
(call-time read, capture law) + existing ARC_MXKV_V5_SPLITS cap.
**HYPOTHESIS**: post-K2/K3 the serial bt->slot->gather chain is the larger
residual; deep splits (floor 1-2) REMOVE serial tiles (ride concurrency)
traded against combine cost — the sweep finds the U-curve minimum.

## Correctness — PASS

All 30 non-baseline cells d <= 1.1e-5 vs certified (4,256) — the A11
split-regrouping reassociation class exactly (2-3e-8 per combine, 1e-5
on the f16 output quantization step). All finite. TFL_PASS.

## Perf — the U-curve is FLAT at 32K, wins at <=8K (ms/call)

G1 (GOAL lane):

| cell | tf4/sp256 (cert) | tf2/sp256 | tf1/sp256 | tf1/sp1024 |
|---|---|---|---|---|
| m8 8K | 1.807 | 1.642 (-9.1%) | **1.590 (-12.0%)** | 2.611 |
| m8 32K | 4.897 | **4.849 (-1.0%)** | 4.885 | 6.040 |
| m4 8K | 2.532 | **2.203 (-13.0%)** | 2.220 | 3.185 |
| m4 32K | 7.615 | 7.636 (+0.3%) | 7.627 | 8.691 |
| mkq 8K | 2.482 | **2.216 (-10.7%)** | 2.242 | 3.220 |
| mkq 32K | 7.605 | 7.606 (+0.0%) | 7.714 | 8.711 |

G8: tf2/sp256 neutral (m8 -0.6%, m4 -0.1%, mkq +0.5% at 32K); deeper
splits monotonically worse (sp1024 +12-20% G8).

## Verdicts

1. **The 32K kernel is WORK/TRAFFIC-FLOORED, not serial-chain-floored.**
   Removing the last serial tiles (tf1 at 32K = 1024 splits x 1 tile)
   buys NOTHING (4.885 vs 4.897) while the mid-buffer round-trip tax
   grows with splits: at sp1024 the (4,24,1024,257) f32 mid is ~100 MB
   written+read vs ~69 MB of useful KV bytes — the combine ate the
   concurrency gain. This CONFIRMS r182's ablation floor from the
   concurrency side and closes the "more parallelism" lane.
2. **tf2/sp256 is the universal winner config** (single knob, no
   per-format branch): -9..-13% at the <=8K G1 cells where the MX lanes
   sit closest to parity, -1..+0.5% at 32K/G8 (noise class). Candidate
   default flip PENDING e2e fastcycle confirmation (rides the attempt-3
   legs); the certified default stays 4 until then.
3. **Attempt 2 CLOSED**: a tuning rung re-swept post-K2/K3 (modest,
   low-ctx); the GOAL-lane 32K deficit is untouched — consistent with
   the work-floor conclusion.
4. **Attempt 3 OPENED (BN re-sweep with K2/K3)**: the v6 BN=64
   refutation ("double-N tile re-climbs the A9 spill slope") predates
   the packed lanes — K2/K3 eliminated the f32 dequant materialization
   (the spill driver: a (64,256) f32 tile is 64 KB vs 32 KB packed f16),
   so BN 64/128 may now win at long ctx: half the tiles = half the
   bt-chain fixed costs + longer gather runs. NO code change needed
   (ARC_MXKV_V5_BN is a lawful existing knob) — r194 sweeps
   BN {32,64,128} x tf {2,4} at sp256.

**OUTCOME: correctness PASS (1e-5 class), tf2/sp256 banked as the
universal winner config (<=8K -9..-13%, 32K flat), 32K work-floor
CONFIRMED from the concurrency side, attempt 3 (BN re-sweep) opened.**
