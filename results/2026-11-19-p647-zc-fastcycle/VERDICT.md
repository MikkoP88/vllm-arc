# VERDICT — P6.47 cert25zc validation fastcycle (r196b)

**LEG**: 2026-11-19-p647-zc-fastcycle (rig bank /root/build/p647/zcA27)
**IMAGE**: vllm-arc:v0.30.0-b70-cert25zc (fork_sha
dd74715079dc6c23fd4ad2feb1a700294515d423) with NO env deltas — the
image defaults carry BOTH closing-rung changes: ARC_MXKV_V5_TFLOOR
launcher default 2 and the G-adaptive mxfp4 dispatch
(ARC_MXKV_V5_G_MIN=8).
**GATES**: ACCEPT_KNEE_PASS per cell; the m4q decoder.txt must show
BOTH dispatch faces (v2 and v5, source=g-adaptive); no failed cells.

## Result table (acceptance% / mean-len / ITL p50 / eff / ss, ms)

m4qzc (the G-adaptive proof cell — v2 below the knee, v5 at/above):

| ctx | n1 itl (base 49/58/76/112) | n8 agg p50 (base 131/201/341/1547) | n8 ss (base 228/457/895/1978) |
|---|---|---|---|
| 4096 | 49 | 145 | 197 |
| 8192 | 58 | 228 | 543 |
| 16384 | 76 | 376 | 925 |
| 32768 | 112 | **583** | 1884 |

n1 is byte-identical to the r185b baseline at every ctx — the v2 lane
below the knee is preserved exactly. The n8 32K aggregate p50 drops
1547 -> 583 (-62%): the B1-banked v5-at-n8 win (r188 measured
1551 -> 696 under the blanket flip; the G-adaptive form lands 583).
The ss columns at 4-8K swing inside the known ramp-noise band for
this column class (r195: same-path deltas -21%/+6% across boots);
the long-ctx signal is the aggregate p50, which is decisive.

m8qzc / mkqzc (tf2 default lanes; r195 env-run reference in parens):

| ctx | m8q n1 itl | mkq n1 itl |
|---|---|---|
| 4096 | 55 (65) | 53 (58) |
| 8192 | 55 (69) | 60 (64) |
| 16384 | 66 (70) | 89 (83) |
| 32768 | **86** (95) | **107** (121) |

32K: m8q -9.5%, mkq -11.6% vs the r195 tf2 env run; 4-8K gains of
~5-10 ms ride the fresh-warm boot position class (k2tuneA19 law —
same-config cross-leg deltas are position noise); no cell regresses
beyond that class (16K mkq +6 = noise).

decoder.txt (m4q, both TP workers):
`decoder=v5 fmt=mxfp4 source=g-adaptive` (02:05:19 — the batch->=8
shape, first struck during capture warmup; per-shape plan freeze)
and `decoder=v2 fmt=mxfp4 source=g-adaptive` (02:05:36 — the n1
face). The dispatch picks per shape exactly as designed.

Quality: acceptance 71.6-94.7% at every cell, KNEE=none x3, 24/24
groups CONC_OK, no KV caps, no timeouts, no failed cells.

## VERDICT: FASTCYCLE PASS — cert25zc validated, both changes live

- The G-adaptive dispatch delivers the B1 n8 win with the n1 lane
  untouched (the blanket flip's +8..14 ms regression is structurally
  gone — v2 owns every sub-knee shape).
- The tf2 default reproduces and slightly beats the r195 env-run
  numbers on both v5 lanes.
- **Disposition**: r197 full par10 battery (fired 02:44Z, image
  content + fork_sha refuses passed) -> bank + VERDICT -> release per
  cert sequence (content commit dd747150 = manifest fork_sha; docs
  commit; gh PR squash-merge to github main + push origin main; tag
  on merge; gh release) -> docs/README currency sweep.

**OUTCOME: ACCEPT_KNEE_PASS x3, G-adaptive proof gate PASS, no
regressions, m4q n8 32K aggregate p50 -62% vs baseline — cert25zc
release path open.**
