# VERDICT — P6.47 r234: cert28 perf re-discrimination + first shallow record faces — PASS

**What**: the certified ITL record re-measured on the shipping cert28
image (the r231d clamp changes shallow-ctx decode, so the record needs
the long-ctx anchors re-discriminated AND the first banked shallow
rows). Harness `tools/ar/r234_cert28_itl.sh` (shipped md5
`1f511453bc7ff538e5b65b2358bf4ba0`), stage 3 of the gated chain
(`R232C_LAUNCH_DONE 05:10:16` 2026-11-24 UTC). Refuses: image +
manifest `cert28_fork_sha=43f84cf9...` + installed libattn/libmqa md5s.
Cells one boot each, the certified posture (ARC_GRAPH=1 spec k3 fp8
weights, wheel BAKED — no mounts, no env overrides), p139_conc n1
MT=192, per-tag faces.

## Record table (ITL ms p50, mean over clean requests)

| Cell | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|
| ctrl (fp8) | — | — | 40.4 (1.000×) | 41.1 (1.000×) | 42.5 (0.999×) |
| m4q native | 43.4 (1.000×) | 45.2 (1.001×) | 48.5 (1.000×) | 52.2 (1.001×) | 58.1 (0.999×) |
| m8q native | **44.4** (first row) | **46.5** (first row) | 50.0 (0.983×) | 55.3 (1.000×) | 64.3 (0.998×) |
| mkq native | **44.1** (first row) | **46.0** (first row) | 49.4 (0.987×) | 53.9 (1.000×) | 61.9 (1.000×) |

Anchors: r230 cert27 rows (ctrl, m8q/mkq long faces) + the r231de
build28 rows (m4q all five). Ratios in parentheses; the +5% one-sided
band (oracle-band law) — worst observed ratio **1.001**; every
anchored face within +0.1%. tps: ctrl 88.4/86.8/80.4; m4q
88.6/81.2/72.6/72.4/65.2; m8q 86.4/79.3/73.6/63.5/53.5; mkq
87.1/79.9/69.1/68.7/58.7.

- Quality: chunks 51–57 everywhere (80 law floor), 0 errors, 0
  tracebacks, all 18 faces CONC_OK.
- Native telemetry: splits=auto on all MX cells; true line count
  `grep -c` = **108 per cell** (the in-log report's `lines=156` is the
  python universal-newlines tqdm artifact — r230 precedent, recounted
  on the banked bytes).
- Gates: `R234_GATE: PASS (cert28 record re-discriminated; shallow
  rows banked)` at 05:05:45.

## Reading

- The r231de shallow win is now the **certified record**: mxfp4
  native decode holds 43.4 ms at 2K (was 47.0 on cert27) through
  58.1 ms at 32K — the full five-face record on one image.
- m8q/mkq get their first shallow rows (44.4/46.5 resp. 44.1/46.0):
  the clamp's shallow-ctx benefit is lane-uniform, and the MX lanes'
  2K ITL (43.4–44.4) now sits within 8% of the fp8 ctrl's 8K ITL.
- Long-ctx parity exact (m4q 32K 0.999×; the no-op tail of the clamp
  at capacity-priced walks, as designed).

## Hygiene

- Law-8: rig-side redaction BEFORE pull (4 serve logs; part of the
  chain-wide 376-occurrence pass, perl -pi byte-exact); post-pull
  local sweep clean. `SHA256SUMS` regenerated post-pass (53 files).
- Bank: `bank/<cell>/{boot.txt, serve.log, telemetry.txt,
  erraudit.txt, c*_n1/{metrics_before.txt, p139_conc_c*_n1.json}}`.
