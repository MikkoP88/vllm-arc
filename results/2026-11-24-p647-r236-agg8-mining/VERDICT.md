# VERDICT — P6.47 r236: cert28 agg8 re-discrimination (bank mining) — PASS

**What**: the agg8 aggregate faces re-verified on cert28 by mining
the r233 battery bank (the `r113_battery.py decode` phase ran in
every r233 leg — the agg8 rows existed on the clamp wheel but were
not surfaced in the r233/r234 close-outs). Analysis-only leg, no
new rig time, no engine change (r231c precedent). Sources:
`results/2026-11-24-p647-cert28-battery/bank/<leg>/battery.txt`
(cert28) vs `results/2026-11-23-p647-cert27-battery/` (cert27,
as re-verified in the r230 VERDICT).

## agg8 (conc8, ctok 2048, tps mean of 3 reps)

| Lane | cert27 | cert28 | delta | cert28 ÷ ctrl |
|---|---|---|---|---|
| ctrl (fp8) | 303.5 | 307.8 | +1.4% | 1.000 |
| m4q native | 261.9 | 258.4 | −1.3% | 0.839 |
| m8q native | 249.7 | 252.0 | +0.9% | 0.819 |
| mkq native | 250.6 | 249.2 | −0.6% | 0.810 |

solo (conc1, ctok 256, tps mean of 3): ctrl 60.6, m4q 51.2, m8q
47.7, mkq 46.7 — the same flat relationship.

## Reading

- **agg8 is clamp-neutral**: every lane within ±1.4% of its cert27
  value (inside the ±6% cross-boot layout-dice band;
  single-sample-cross-wheel law — these are same-harness
  cross-image deltas at 3 reps, and the spread does not clear the
  noise floor in either direction).
- **Mechanism**: the r231d clamp lives on the legacy scalar split
  path; at conc8 the decode batch rides the G-adaptive / v5 decoder
  lane (mxfp4 G≥8 → v5 by `ARC_MXKV_V5_G_MIN=8`; m8/mkq per-format
  defaults already v5 at G8 per the T1 verdict) — a different
  kernel path the clamp does not touch. The clamp's −7.6% n1 @2K
  win is a v2-lane (low-G) property by construction; it does not
  transfer to aggregate faces, and none was claimed.
- **The aggregate record stands**: m4q 0.84× ctrl at agg8 (0.86× on
  the r228 cert27 numbers with ctrl's own +1.4% boot move — the
  ratio moves with the denominator, not the lane), all three MX
  lanes at their cert27-relative positions. No aggregate-face lever
  remains priced above noise on the clamp wheel.

## Verdict

**PASS** — the agg8 record is re-discriminated on cert28 from
banked evidence; clamp-neutral as designed (the n1 shallow win is
lane-scoped); no regression, no new lever.

## Hygiene

- No new bank (sources are the already-law-8-scrubbed r233/cert27
  banks; this VERDICT cites them). No rig time; W14 untouched
  (verified 200 before and after this analysis window).
