# r231d e2e face leg: PASS — dynamic shallow-ctx win, record parity exact (P6.47, 2026-11-24)

**Leg**: r231de — the r231d seqused-tightened split clamp on the
SERVING engine (m4q = the record lane, mxfp4 native). Harness
`tools/ar/r231d_e2e.sh` (rig md5 b2d498f6396a641d3d29c212e818b5ee),
bank `/root/build/p647/r231d_e2e/` (local `bank/`). One boot per
cell, r230 posture exactly (ARC_GRAPH=1 spec k3 fp8 weights,
ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1, qwen3.8-27b-fp8, n1 MT=192).

**Cells**: `m4q25` cert27 as-is (libattn verified 2d73ff7a…);
`m4q28a`/`m4q28b` cert27 + build28 wheel pre-install (wheel
e6d8d337e43b907504537061414128f8, in-container libattn verified
61d2f5a3… per cell). Two build28 boots = the A/A bound.

## Results (ITL p50 ms, chunks<=57 all OK, errs 0, native telemetry 156 lines/cell)

| cell | c2048 | c4096 | c8192 | c32768 | tps @2K |
|---|---|---|---|---|---|
| m4q25 (build25) | 47.0 | 47.9 | 49.4 | 58.1 | 81.9 |
| m4q28a (build28) | 43.4 | 45.2 | 48.5 | 58.0 | 88.7 |
| m4q28b (build28) | 43.4 | 45.3 | 48.6 | 58.1 | 88.7 |
| b28 ratio (A/B mean) | **0.924 (−7.6%)** | **0.945 (−5.6%)** | 0.984 (−1.6%) | 1.000 (0.0%) | +8.3% |

- **A/A spread** (two build28 boots): 0.0-0.1% on every face — the
  win is 50-75x the A/A noise floor.
- **Baseline re-discrimination**: m4q25 c8192 = 49.4, c32768 = 58.1
  — EXACTLY the banked r230 cert27 record; cross-boot dice ~0.
- c32768 (record context, blocks>=2032, true clamp no-op): 1.000x
  byte-for-value parity. c8192 (walks 7936-8128 -> clamp 124-127):
  0.984x — slightly better, consistent with the flat 64-128 regime
  of the 8K split curve.

## Gates — all PASS (R231DE_GATE: PASS, 23:26:51 UTC)

- PARITY: b28a/b28b within +5% of same-day baseline at c8192
  (48.5/48.6 <= 49.4) and c32768 (58.0/58.1 <= 58.1) — OK.
- WIN: b28 mean < baseline x 0.97 at c2048 (43.4 < 47.0, −7.6%)
  and c4096 (45.2 < 47.9, −5.6%) — OK.
- Quality: chunks 51-57 (<= 80, r216b law), errs 0, MXKV native
  telemetry present on all three cells (156 lines each).

## Verdict

**PASS.** The device-side split clamp delivers a real, A/A-clean
dynamic-serving win at shallow walked ctx (every fresh request
decodes there: −7.6% ITL @2K, +8.3% tps) with exact record parity
at 8K+ (1.000x @32K). Offline (PR #122) + e2e (this leg) both
green -> **cert28 path**: bake build28 over cert27 (same overlay,
libattn pin swap), run the cert battery, cut the release per the
cert-release sequence. Per plan: with r231d landed positive, the MX
walk line closes at cert28-or-better.

## Hygiene

- W14 restored, HEALTH 200 at 23:31:22 UTC; no arc containers left;
  builder untouched (tree carries the patch, `.r231dprebak`
  siblings).
- The cert27 image is NOT modified by this leg (wheel pre-install
  is per-container); cert27 remains the certified image until
  cert28 is cut and gated.
