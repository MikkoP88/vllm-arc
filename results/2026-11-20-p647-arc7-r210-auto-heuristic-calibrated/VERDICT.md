# r210 — auto-heuristic calibrated (stock 4x knee + kv_tiles/12 bound)

**VERDICT: PASS — keeper wheel `7247e83b` (build18). Same kernel binaries
as the r208-certified wheel (proven by the same-geometry forced-splits
sweep: mxfp8 identical within noise at every cell, mxfp4/mxk8v4 equal or
faster) with the 1K over-split fixed: auto lands 5 splits at 1K (the
forced-sweep optimum), 40 at 8K/32K (the stock 4x knee). Telemetry 1K
0.385/0.290/0.277 ms (from r208's 0.442/0.340/0.326), 32K unchanged
0.556/0.389/0.414. Faces all PASS, numerics envelope unmoved.**

- Heuristic (`mx_paged_decode_xe2.cpp`): knee RESTORED to stock 4x
  slots/head; workload bound `splits <= max(1, kv_tiles/12)` (the vendor
  bandwidth divisor — never denser than the bw knee).
- Chain 22:00-22:07Z: synth2 x3 rc=0, faces VERDICT PASS, probes 0-8
  rc=0, W14 auto-restored (health 200 verified).

## The measurement-noise discovery (corrects r209's A/B reading)

The r209 bank attributed the 8K telemetry/probe deltas (0.232 -> 0.351,
+58-65%) to the 32-split posture. The r210 same-wheel forced sweep
refutes that attribution:

- 8K forced cells, r208 wheel vs r210 wheel (B=2 bench geometry):
  s32 0.4507 -> 0.4522, s48 0.4478 -> 0.4482, s64 0.4507 -> 0.4496
  (mxfp8; other formats equal-or-faster). s32-vs-s40 differs by <1%.
- 32K forced cells: identical within noise (s64 0.7192 -> 0.7202).

So the kernels and their split-response curves are THE SAME on both
wheels; the probe-m0 8K spread (0.171 / 0.283 / 0.233 across
r208/r209/r210 chains at the same 40-split posture) is **single-sample
variance in the native path at 8K (~+/-25%)** — the v2 side is rock
stable (0.218 across all three chains). LAW: telemetry/probe
single-sample native datapoints at 8K-class regimes must not be A/B'd
across wheels; the same-wheel forced-splits sweep is the arbiter.

Practical consequence: r209's "8K/32K regressions" were noise-dominated
(the heuristic deltas there were also ~sub-1% by forced sweep); the only
real posture effect in play was 1K, and r210 keeps that fix.

## 1K calibration (forced sweep on ONE wheel, `p210_1k_splits.json`)

s=1 is latency-bound and 1.5x WORSE than the plateau (1.545 / 0.654 /
0.640 ms) — the "skip the reduce launch" hypothesis is dead. Plateau
0.43-0.53 ms with the optimum at s=5 (mxfp4 0.435, mxk8v4 0.422,
mxfp8 flat) = exactly kv_tiles/12 at 64 tiles. Canary 9.77e-4 constant
across ALL splits incl. 1 = one-bf16-ULP at short-seq output magnitude
(larger max |out| than 32K), not reduce corruption.

## Telemetry (wall ms, B=1; v2 ratio)

| ctx | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| 1024 | 0.385 (0.47x) | 0.290 (0.60x) | 0.277 (0.63x) |
| 8192 | 0.298 (0.73x)* | 0.232 (0.82x)* | 0.231 (1.84x)* |
| 32768 | 0.556 (1.38x) | 0.389 (1.70x) | 0.414 (3.79x) |

*8K single-sample cells carry the +/-25% native variance above; the
r208-wheel datapoints for the same posture read 0.232/0.199/0.201.

## Probe m0 (device ms; postures 1K->5, 8K->40, 32K->40)

mxfp8 1.062 / 0.233 / 0.425; mxfp4 0.316 / 0.193 / 0.327; mxk8v4
0.311 / 0.191 / 0.367. 32K matches the r208 bank exactly (0.427 /
0.327 / 0.333); 8K cells are noise-bound (above); 1K improved vs the
r208 bank (0.393 -> 0.316 mxfp4) — note the r208 mxfp8@1K 1.507 was
itself an unstable cell (r209/r210 measure 1.06 both).

## Open (next loop targets, priority order)

1. **mxfp8 short-ctx floor pathology**: @1024 probe the mxfp8 floor
   modes (m2/m3/m5/m6) sit ~3x above mxfp4's with identical grids and
   floor-mode code (r210: m0 1.062 vs 0.316; bench plateau 0.53 vs
   0.44). Format-specific occupancy/SLM/codegen in the mxfp8 mainloop
   — THE 1K blocker (1K is 0.47-0.63x of v2).
2. 32K m0-vs-m8 gap (0.427 vs 0.198 = 2.2x) — walk/staging overhead.
3. Then convergence: r200-e2e fastcycle (ARC_MXKV_NATIVE=1) -> new
   image -> full battery -> cert26.

## Artifacts

- `r210_chain.log`, `p200c_probe_0..8.json`, `p201_synth2_*.json` —
  chain on wheel 7247e83b.
- `p210_1k_splits.json` + `r210_1kbench.log` — 1K forced-sweep
  calibration (driver `tools/ar/p147_mxkv/r210_1kbench.sh`).
- `p210_splits_sweep.json` + `r210_splitbench.log` — 8K/32K forced
  sweep on the r210 wheel (driver `tools/ar/p147_mxkv/r210_sweep.sh`;
  p200h now env-tunable via P200H_CTX / P200H_SPLITS).
- Wheel archived `wheels-arc-vxk/fix/...+arc7mx-r210.whl` (7247e83b).
