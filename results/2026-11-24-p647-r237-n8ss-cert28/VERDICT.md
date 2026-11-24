# VERDICT — P6.47 r237: cert28 n8 steady-state long-ctx faces — PASS

**What**: the last un-re-measured face class on cert28 — n8
steady-state long-context (16K/32K, MT=128, the r185-ladder cell
shape) — measured on the certified clamp wheel. One boot per lane
(ctrl / m4q / m8q / mkq), 4 boots, ~41 min, zero errors/aborts.
Harness `tools/ar/r237_n8ss_cert28.sh` (refuses: image +
`cert28_fork_sha=43f84cf9…` + installed `.so` md5 pins). Rig bank
`/root/build/p647/r237/`, law-8 redacted rig-side before pull
(strict zero), 33 files + SHA256SUMS.

## Metric honesty (what n8-ss at MT=128 actually measures)

- `itl_ms_p50` **degenerates at these faces**: tokens arrive in
  spec-decode bursts, so the median inter-chunk gap is the burst
  cadence, identical across all 8 slots of a cell (66.5 ms in every
  ctrl 16K slot; 111.7 in every m4q slot) — a reference column only,
  never a gate.
- `ss = (total_s − ttft_s)·1000/(n_stream_chunks−1)`, median across
  slots (ramp-interleaved ITL law). The per-slot dump shows TTFTs
  staggered 10→115 s (chunked-prefill serialization across 8 slots)
  while totals collapse together — at MT=128 the ss window still
  carries absorbed ramp interference, so **ss is ramp-weighted**:
  the honest face of the two, but not a pure decode metric at these
  shapes.

## Table (recomputed from the pulled bank)

| cell | ctx | ss_ms | anchor | ratio | agg_p50 (ref) | chunks | errs |
|---|---|---|---|---|---|---|---|
| ctrl | 16384 | 778.7 | – | – | 66.5 | 39.8 | 0 |
| ctrl | 32768 | 1273.9 | 1671 | 0.762 | 1350.7 | 40.2 | 0 |
| m4q | 16384 | 588.3 | – | – | 111.7 | 41.5 | 0 |
| m4q | 32768 | 1440.4 | 1703 | 0.846 | 1046.8 | 42.8 | 0 |
| m8q | 16384 | 616.8 | – | – | 133.4 | 38.2 | 0 |
| m8q | 32768 | 1506.1 | 1743 | 0.864 | 1267.5 | 42.1 | 0 |
| mkq | 16384 | 730.8 | – | – | 126.2 | 38.6 | 0 |
| mkq | 32768 | 1399.7 | 1824 | 0.767 | 1058.3 | 39.9 | 0 |

ss_ratio vs ctrl @32K: **m4q 1.131, m8q 1.182, mkq 1.099**;
@16K (first cert28 rows): m4q 0.755, m8q 0.792, mkq 0.938.

## Anchor provenance and the cross-era comparison

The anchors recompute **exactly** from the banked cert25zb-era
JSONs (`results/2026-11-14-p647-s1-acceptknee/`) with the identical
formula and chunk shape (~40 chunks → same MT=128): ctrl 1671.0,
m8q 1743.0, mkq 1823.7 (m4q 1978.2 pre-v5; 1703 = the B1 r188
v5-flip row on the same bank). All four era cells were clean-window
measurements (mkq's Nov-13 run was interference-contaminated and
preserved as `mkq_interfered`; the clean re-run landed Nov 18 as
r185b; ctrl ran clean as r186) — the interference timeline is banked
in that dir.

| lane | era (cert25zb) | cert28 r237 | Δ | within-leg ratio era → r237 |
|---|---|---|---|---|
| ctrl | 1671 | 1273.9 | −23.8% | 1.000 → 1.000 |
| m4q (v5) | 1703 | 1440.4 | −15.4% | 1.019 → 1.131 |
| m8q | 1743 | 1506.1 | −13.6% | 1.043 → 1.182 |
| mkq | 1823.7 | 1399.7 | −23.3% | 1.091 → 1.099 |

- **All four lanes below their era anchors** (one-sided regression
  band, oracle-band law) — gate PASS.
- **Cross-era attribution**: cert26/27/28 changes are MX-kernel-scoped
  (release notes: kernel fixes + wheel bytes); the ctrl −23.8% has no
  release-notes attribution and exceeds the ±6% kernel-layout-dice
  band → stack drift beyond the notes, unbounded on this ramp-weighted
  family. Absolute cross-era deltas are **recorded, not claimed as
  lever effects**; the within-leg ratios (same boot, same harness) are
  the clean statements, and they carry the same ramp-weighting caveat.
- The 16K inversion (MX lanes *below* ctrl on ss while n1 ITL says MX
  slower) is consistent with ramp-profile weighting, not a decode
  claim — recorded, not promoted.

## Native engagement + quality

`telemetry.txt` per MX lane: 110 `MXKV native call` lines (fmt=
mxfp4 / mxfp8 / mxk8v4, `splits=auto`, bt_width 4100/4108/4114).
`erraudit.txt`: ERROR=0, TRACEBACK=0 in all four lanes; chunks ≤42.8
(gate ≤80).

## Verdict

**PASS** — no anchor regression; the first cert28 n8-ss rows banked
(16K + 32K × 4 lanes). With this leg **every face class is
re-discriminated on cert28**: n1 shallow (r234/r235), agg8 (r236,
bank-mined), n8-ss long-ctx (this leg). No lever priced above noise
from this leg — the ss family is ramp-weighted at MT=128 and the
cross-era drift bounds cannot be closed within-boot. Per plan §12
the MX hunt's standing disposition holds: **idles pending NEW
evidence**.

## Hygiene

- Rig-side law-8 redaction before pull (`perl -pi`, strict
  digit-required verify = 0); `launch.log` copied into the bank and
  redacted; local strict re-verify 0 across all 33 files.
- `*.log` force-adds get per-file sweeps at `git add` time.
- W14 restored by the harness EXIT trap; health-200 verified
  post-leg before close-out.
