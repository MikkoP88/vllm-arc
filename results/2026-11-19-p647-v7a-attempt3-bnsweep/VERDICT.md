# VERDICT — P6.47 Phase 7 attempt 3: BLOCK_N re-sweep with K2/K3 (r194)

**LEG**: 2026-11-19-p647-v7a-attempt3-bnsweep (rig bank
/root/build/p647/bnsA26; the first fire's BN=128 OutOfResources
traceback is preserved in the leg log — the re-run trimmed it)
**IMAGE**: vllm-arc:v0.30.0-b70-cert25zb + in-container swap of the
TILE_FLOOR-bearing triton_mx_decode.py (r192 mechanism)
**HYPOTHESIS**: the v6 BN=64 refutation ("double-N tile re-climbs the
A9 spill slope") predates K2/K3 — the packed lanes eliminated the f32
dequant materialization that drove the spills; with the 32K residual
work/traffic-floored (r193), per-tile fixed cost is what larger BN
amortizes.

## BN=128 — OUT ON MEASURED PHYSICS

The compiled kernel requires 147,456 B shared memory against the
131,072 B hardware limit (triton OutOfResources, first fire). The A9
spill cliff, quantized exactly. Not forceloaded at lower stages — the
SMEM deficit is structural at this tile shape.

## BN=64 — REFUTED DECISIVELY (ms/call; bn32/tf2 = the r193 winner)

| cell | bn32/tf2 | bn64 best | ratio |
|---|---|---|---|
| m8 G1 8K | 1.539 | 3.178 | 0.48x |
| m8 G1 32K | 4.166 | 11.206 | 0.37x |
| m8 G8 32K | 30.505 | 85.576 | 0.36x |
| m4 G1 8K | 2.311 | 3.344 | 0.69x |
| m4 G1 32K | 6.897 | 11.700 | 0.59x |
| m4 G8 32K | 52.263 | 90.134 | 0.58x |
| mkq G1 32K | 6.890 | 11.899 | 0.58x |
| mkq G8 32K | 52.962 | 91.301 | 0.58x |

(tf sweep within bn64 is noise — the BN term dominates.)

Correctness at bn64: all cells d <= 1.3e-5 (the regrouping class),
finite — the loss is pure occupancy: SMEM per CTA doubles -> half the
resident CTAs -> the work-floor regime (r193) punishes it directly.
The v6 BN verdict SURVIVES K2/K3: freed registers do not rescue the
double-N tile because the binder was never dequant registers at this
shape — it is resident-CTA count x per-tile work.

## CONTAINMENT CLOSED — Phase 7 plateau

All three structural attempts are now banked:

1. v7a manual raw-tile double buffering (r192): bitwise-clean,
   0.61-0.95x — no in-language prefetch on Triton/XPU (toolchain law).
2. TILE_FLOOR deep splits (r193): 32K flat — the kernel is
   work/traffic-floored, not serial-chain-floored; tf2/sp256 banks as
   the universal winner (-9..-13% at <=8K G1, neutral elsewhere).
3. BN re-sweep (r194): BN=64 1.5-2.8x WORSE (occupancy), BN=128
   SMEM-impossible — the v6 BN law stands post-K2/K3.

The two-launch split-walk structure at bn32/sp256 is at its measured
plateau: the 32K G1 residual is work x resident-CTA throughput, with
no lawful lever left inside Triton on this geometry (prefetch: none;
parallelism: combine-tax-bound; tile size: SMEM/occupancy-bound).
Ctrl FA's ~0.2 ms/layer rides a fused-attention implementation class
outside Triton's reach here.

## Disposition

- **Plateau BANKED per the plan's superiority contract**: MX ships
  1.81x (mxfp4) / 1.27x (mxk8v4) KV capacity with the measured decode
  cost curve; decode parity at n1 long-ctx is not reachable by kernel
  tuning within this structure.
- **Value capture before close: tf2/sp256 e2e confirmation (r195)** —
  the one surviving win (-9..-13% at <=8K G1). Serve ladder cells
  (4K/8K/16K/32K, n1+n8) with ARC_MXKV_V5_TFLOOR=2 on the m4q/m8q/mkq
  chains vs banked baselines; if e2e confirms, TILE_FLOOR default 2
  flips -> cert25zc -> full battery -> release. The G-adaptive m4q
  dispatch (Phase 6 designed-held) rides the same legs.

**OUTCOME: BN=64 refuted 1.5-2.8x, BN=128 SMEM-impossible, v6 BN law
stands post-K2/K3, 3-attempt containment CLOSED at the measured
plateau, tf2 e2e confirmation is the closing rung.**
