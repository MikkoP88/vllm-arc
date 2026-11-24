# VERDICT — P6.47 r235: the fp8-ctrl shallow record rows — PASS

**What**: the record's ctrl (fp8) cell had long faces only (r230/r234
ran 8/16/32K), so every MX-vs-fp8 shallow comparison was
cross-context ("MX 2K within 8% of fp8 8K"). r235 banks the missing
shallow control rows: ONE ctrl boot on the shipping cert28 image,
faces c2048/c4096 (first rows, banked ungated) + c8192 as the
same-boot sanity anchor. Measurement only — no engine change, no
cert cut (r230 precedent). Harness `tools/ar/r235_ctrl_shallow.sh`
(shipped md5 `ee15a641f48fa43cd036a17335603021`), r234-verbatim
boot/group/report bytes; refuses = image + manifest
`cert28_fork_sha=43f84cf9...` + installed libattn/libmqa md5s.

## Record rows (ITL ms p50, n1 MT=192, mean over clean requests)

| Cell | 2048 | 4096 | 8192 |
|---|---|---|---|
| ctrl (fp8) | **40.1** (first row) | **40.0** (first row) | 40.4 (1.000× sanity) |

tps 92.4 / 91.0 / 87.9; chunks 52–53 (80 law floor), 0 errors,
0 tracebacks. `R235_GATE: PASS` at 05:39:24 (2026-11-24 UTC);
boot 05:27:57 → HEALTH 05:38:26 → done 05:39:24.

## Reading

- **The fp8 ctrl is FLAT at shallow context**: 40.1/40.0/40.4 ms at
  2/4/8K (the flat-ctrl law now measured through the shallow range,
  not extrapolated) — tps 92.4 @2K.
- **The true same-context shallow MX gaps** (vs the r234 cert28
  rows): mxfp4 +8.2% @2K (43.4 vs 40.1), +13.0% @4K (45.2 vs 40.0);
  mxfp8 +10.7% @2K; mxk8v4 +10.0% @2K. The cross-context "within 8%
  of fp8 8K" phrasing is retired; the record now carries exact
  same-context ratios.
- The gap profile at 2K (+8–11% across lanes, converging from the
  +19–22% pre-clamp state at r231d analysis time) is the residual
  MX shallow tax after the seqused-tightened clamp — the walk line's
  designed floor at capacity-priced walks is a long-ctx property;
  the shallow residual is the split-clamp's floor (windowed_k_blocks
  at 2K still yields 1–2 splits).

## Hygiene

- Law-8: rig-side redaction BEFORE pull (serve.log + launch.log;
  perl -pi byte-exact, strict post-pass 0); post-pull local sweep
  clean. `SHA256SUMS` over the 10-file bank.
- W14 restored by the leg's EXIT trap: `R235: W14 HEALTH 200
  (05:43:54)`; zero arc containers after.
- Bank: `bank/ctrl/{boot.txt, serve.log, erraudit.txt,
  c{2048,4096,8192}_n1/{metrics_before.txt, p139_conc_*_n1.json}}`
  + `bank/launch.log`.
