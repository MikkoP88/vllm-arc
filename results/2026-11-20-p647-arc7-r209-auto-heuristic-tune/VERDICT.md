# r209 — auto-heuristic tune (reduce knee 8x + kv_tiles/16 workload bound)

**VERDICT: GATED (split outcome — heuristic superseded by r210, wheel NOT
promoted). 1K improved (telemetry mxfp8 0.442 -> 0.387 ms), 8K REGRESSED
hard (0.232 -> 0.351 ms, 0.62x of v2), 32K held in telemetry but the probe
A/B shows the underlying posture is WORSE (64 splits vs 40: mxfp4 +10%,
mxk8v4 +13%). Both r209 deltas misfired at 8K/32K; the A/B evidence they
bought is the leg's value and directly pins r210: revert the knee to the
stock 4x (= 40 slots/head) and keep a workload bound recalibrated by the
1K sweep.**

- Wheel: build17 md5 `21562e40` (r209 heuristic in
  mx_paged_decode_xe2.cpp: splits_reduce_cap 4 -> 8 num_wg_slots/Hq,
  plus `splits <= kv_tiles/16`).
- Chain (`r209_chain.sh`, 21:25-21:31Z): synth2 x3 rc=0, faces VERDICT
  PASS (numerics envelope unmoved — heuristic changes only move split
  counts), probes 0-8 rc=0, W14 auto-restored.

## Telemetry (wall ms, B=1, auto posture; v2 ratio in parens)

| ctx | fmt | r208 (40 splits) | r209 (1K:4 / 8K:32 / 32K:64) |
|---|---|---|---|
| 1024 | mxfp8 | 0.442 (0.63x) | **0.387** (0.47x) |
| 1024 | mxfp4 | 0.340 (0.52x) | **0.292** (0.60x) |
| 1024 | mxk8v4 | 0.326 (0.54x) | **0.278** (0.63x) |
| 8192 | mxfp8 | **0.232** (0.94x) | 0.351 (0.62x) |
| 8192 | mxfp4 | **0.199** (0.96x) | 0.267 (0.71x) |
| 8192 | mxk8v4 | **0.201** (2.12x) | 0.264 (1.62x) |
| 32768 | mxfp8 | **0.548** (1.40x) | 0.552 (1.39x) |
| 32768 | mxfp4 | **0.389** (1.70x) | 0.391 (1.69x) |
| 32768 | mxk8v4 | **0.414** (3.79x) | 0.417 (3.76x) |

## Probe A/B (device ms, same geometry, ONLY splits differ)

r208 wheel = 40 splits at 8K/32K, 5 at 1K; r209 wheel = 32 at 8K,
64 at 32K, 4 at 1K (p200c_probe_0.json, both banks):

| ctx | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| 1024 (5 -> 4) | 1.507 -> 1.057 | 0.393 -> 0.322 | 0.382 -> 0.309 |
| 8192 (40 -> 32) | 0.171 -> 0.283 | 0.142 -> 0.234 | 0.140 -> 0.222 |
| 32768 (40 -> 64) | 0.427 -> 0.428 | 0.327 -> 0.359 | 0.333 -> 0.377 |

**40 splits wins at BOTH 8K and 32K on engine-like geometry.** The
telemetry B=1 face agrees at 8K (40 = 0.232 vs 32 = 0.351) — two
geometries, one optimum.

## Laws from this leg

1. **Bench-geometry split optima do not transfer.** The r207/r208 sweep
   (2 seqs, HK=2 -> 4 WGs per split-set) favored s=64 because tiny
   slot-fills want deep splits; probe (B=32) and telemetry (B=1)
   geometries both put the optimum at 40. Forcing sweeps on a geometry
   the engine never runs measures the wrong curve.
2. **The stock 4x reduce knee is the measured optimum** (40 slots/head
   at Hq=32) — the vendor knee was already tuned; r209's 8x lifting
   enabled the 32K 64-split overshoot. Reverted in r210.
3. **splits_par over-requests at small batch + short ctx** (B=1 1K: 40
   splits for 64 tiles = 1.6 tiles/split, lost 12% to the 4-split
   posture): the workload bound direction is right, the divisor is not
   16-for-all (8K's 512 tiles run best at 40 = 12.8 tiles/split ~ the
   vendor bw divisor 12). r210 pins it with a forced 1K sweep.
4. **mxfp8 short-ctx floor pathology (OPEN, next-loop target).** At
   ctx=1024 probe, mxfp8 floor modes (m2/m3/m5/m6 = 0.87-0.89 ms) sit
   3.6x above mxfp4's (0.24-0.26) with identical grids and floor-mode
   code paths — a format-specific occupancy/SLM/codegen effect at short
   ctx, not splits (5 -> 4 splits only removed 30% of it). m7 compute
   floor 0.342 vs mxfp4's 0.127 shows the same asymmetry inside the
   compute skeleton.

## Addendum (r210, same day — measurement correction)

The r210 same-wheel forced-splits sweep shows the 8K/32K deltas in the
probe A/B table above were dominated by single-sample native variance
(+/-25% at 8K; v2 stable at 0.218 across chains) — the r208/r210 wheels
measure IDENTICALLY at fixed splits (8K s32 0.4507 vs 0.4522). The
"+58-65% at 8K" attribution to the 32-split posture was noise-dominated;
by forced sweep, s32-vs-s40 at 8K differs by <1%. Law 1 (bench-geometry
optima don't transfer) and the 1K over-split finding stand as written;
the 8K/32K "regressions" do not. See the r210 bank.

## Artifacts

- `r209_chain.log` — full chain (synth2 x3, faces, telemetry, probes).
- `p200c_probe_0..8.json` — probe mode tables on wheel 21562e40.
- `p201_synth2_{mxfp8,mxfp4,mxk8v4}.json` — synth2 faces (0 mismatches).
