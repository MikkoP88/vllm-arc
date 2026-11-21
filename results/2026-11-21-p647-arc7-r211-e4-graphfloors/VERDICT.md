# r211 E4-lite — graph-replay device floors per probe mode (1K)

**Date**: 2026-11-21 · **Leg**: r211_e4 (arc7 Phase-8, E4-lite) ·
**Wheel**: arc7mx r210 keeper `7247e83b` · **Image**:
`vllm-arc:v0.30.0-b70-cert25zc` · **Tool**:
`tools/ar/p147_mxkv/p204_graphtime.py` (mode-suffixed) +
`r211_e4.sh`

## Verdict

**PASS — the 1K fixed cost is located: the STAGING-BARRIER/WALK
PROTOCOL dominates, not loads and not math.** Graph-replaying the
probe-mode floors device-only (auto posture, ctx 1024): the walk+
barrier skeleton (mode 3, everything skipped but the tile loop and
its cross-WI barriers) costs 131-209 µs — 60-64% of the full kernel —
while the compute skeleton (mode 7: const-K/V, staging barriers gone)
is 42-46 µs and loads add 89-119 µs on top (mode 0). The launch
floor (mode 8: 36-39 µs) alone equals v2's ENTIRE kernel (45-50 µs).

## Results (device-only µs/call; graph 32×50; auto posture)

| mode | meaning | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|---|
| 0 | full | 327.9 | 233.6 | 219.4 |
| 3 | floor: skip both stages (walk + barriers) | 208.7 | 135.4 | 130.8 |
| 7 | compute floor: const-K/V, skip staging | 46.2 | 41.9 | 44.2 |
| 8 | gemm floor: 7 + const-P, skip softmax | 38.7 | 36.3 | 38.6 |

Deltas (per format mxfp8/mxfp4/mxk8v4):

- **walk/barrier protocol (m3−m7): 162.5 / 93.5 / 86.6 µs** — the
  dominant term; ~13 tiles/WG × staging barriers ≈ 3-6 µs/barrier on
  this face. m3 > m7 because mode 7 skips the STAGING BARRIERS (const
  operands need no cross-WI sync) while mode 3 keeps the barrier
  protocol with loads/compute dead — the barrier tree, not the walk
  arithmetic, is what mode 3 pays.
- loads + dequant (m0−m3): 119.2 / 98.2 / 88.6 µs.
- softmax/online-update (m7−m8): 7.5 / 5.6 / 5.6 µs — trivial.
- launch+reduction floor (m8): ~36-39 µs ≈ v2's whole kernel.
- mxfp8's m3 is +73 µs over mxfp4's with the same skips: the e4m3-V
  template walks a fatter stage protocol (2× V pitch → more barrier
  waits per tile), before any bytes move.

## Findings

1. **Barrier protocol is the 1K kernel killer.** Splits can't touch
   it (E2'': flat s5→s32), loads are second, math is noise. The 8K/32K
   wins come from amortizing exactly this fixed per-tile cost over
   bandwidth-saturated walks.
2. **v2 (the certified Triton MX decoder, same uint8 MX pool) does
   the whole 1K decode in 45-50 µs device** — the engine ALREADY owns
   a superior short-ctx MX path. The superior architecture is
   ctx-adaptive dispatch inside the MX lane: Triton at short ctx,
   native CUTLASS at long ctx (crossover to be mapped device-only).
3. Kernel-side r212 levers (always-loop, native floor): merge the two
   per-tile staging barriers into one (or stage K+V together), single
   stage region at short ctx (no ping-pong), fold the reduce into
   main at short ctx (the m8 floor shows the second launch costs
   ~nothing extra, but the serialized pair still pays teardown);
   realistic target ~120-150 µs — still 3× v2, so dispatch remains
   the strategic fix for 1K.
4. Mode-8 canaries NaN = const-P epilogue leaves the poison (floor
   semantics, E1' law); cells stay valid as structural floors.

## Next

- **E5 crossover map** (p204 at ctx 1024/2048/4096/8192/32768, cells
  v2 + native_auto): the device-only crossover per format sets the
  dispatch threshold.
- **r212a**: engine-side ctx-adaptive dispatch in the mx_kv native
  bridge (short ctx → the Triton lane, long ctx → native kernel);
  needs the E5 table.
- **r212b**: native-kernel barrier reduction at short ctx (the
  always-loop improvement on the native floor).

## Artifacts

`p204_graphtime_m{0,3,7,8}.json`, `r211_e4_m{0,3,7,8}.log`,
`r211_e4_launch.log` (rig: `/root/build/p199/out`, `/root/build/p199/logs`).
