# r212m2+r212m3 — the trim, the e4m3 revert, and the per-lane telemetry that caught the root issue

**Date**: 2026-11-21 · **Legs**: r212m2 (register trim) + r212m3 (e2m1-gated
direct-bf16, e4m3 reverted) — arc7 Phase-8 · **Stack**: r212m3 wheel
(`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5 `e50f1c3aa38729931c0d6ccc9c57a370`;
r212l tree + probe 9 + p212m_patch + p212m2_patch + p212m3_patch, ninja
resume builds) + `vllm-arc:v0.30.0-b70-cert25zc`, harness
`p204_graphtime.py` at engine geometry (D=256 / HREP=6 / BLOCK=64,
graph 32×50, SPLITS=auto) · **Runs**: `r212m2_val1.log` 13:49-13:56 UTC
(4 cells), `r212m3_val1.log` 14:25-14:40 UTC (8 cells: full probe
decomposition + dtype-split oracle)

## Verdict

**r212m3 is the converged offline state: mxfp4 −27% (749.3 µs at 32K)
with both e2m1 decode lanes nearly halved, mxfp8 recovered to −3% (its
r212m regression root-caused and reverted), bit-exact everywhere
(canaries 0.0 at every cell including the oracle). The comprehensive
per-lane telemetry — not hypothesis — is what caught the root issue.**

### r212m2 (the trim, null on its target)

Dropping the be/sc arrays (inline `int(code)−7`, on-demand slow-path
converts) did NOT recover the mxfp8 float path — dbf0 worsened
monotonically: 1359 (r212l) → 1750 (r212m) → 1862 (r212m2). Register
pressure was not the cause. The e2m1 lanes improved slightly
(750.5 µs, −27.0%); canaries 0.0.

### The root issue (r212m3 diagnosis)

With DIRECTBF16=0 the e4m3 tail paid a per-element
`e8m0_to_scale(code-select)` that r212l's precomputed float select
never paid, plus codegen shifts from carrying the dual machinery —
net mxfp8 fast (1537.6) was +13% over the OLD float path while the
same-wheel gain was only +17.4% of an inflated base. The e2m1 lanes
were a clean −27%. **Decision: gate direct-bf16 to e2m1 only** —
`dequant_reg` (e4m3-only: mxfp8/mxk8v4 K) reverted to the probe-wheel
text; `dequant_vslm` split by `if constexpr (VIsE4m3)` with the e4m3
branch restoring r212l's exact float-vector prep and tail;
`dequant_dtile` (e2m1-only) unchanged.

### r212m3 result (the per-lane telemetry)

Fulls vs r212l: mxfp4 399.8 / 306.8 / **749.3** (−32.7 / −27.0 /
−27.1%); mxfp8 633.9 / 525.7 / **1319.7** (−3.7 / −2.9 / −2.9%) — all
canaries 0.00e+00.

Per-lane decomposition at 32K (vs r212l slices):

| lane | mxfp4 | mxfp8 |
|---|---|---|
| K chain | 365.9 → **178.2 (−51.3%)** | 470.0 → 397.1 (−15.5% vs dice-inflated base) |
| V gather | 375.9 → **196.9 (−47.6%)** | 292.3 → 338.8 (+15.9% ≈ dice band vs r212k's 317) |
| V stage | 83.9 → 17.1 | 364.3 → 359.4 |
| floor | 139.5 → 246.9 | 171.0 → 171.5 |

Both e2m1 decode tails nearly halved — the exponent fold does its
work in both lanes simultaneously. (The mxfp4 floor's +107 µs is the
c8 cell's own layout shift; the full — the number that composes into
ITL — is −271 µs.)

Dtype-split oracle at 32K: mxfp4 dbf0 829.5 → fast 749.3 (**+9.7%**
same-wheel e2m1 gain); mxfp8 dbf0 1334.6 vs fast 1319.7 (1.1% —
byte-equivalent code path, the revert proven; the residual is noise).

### Disposition

- r212m3 wheel is the working state (backups: `*.r212lbak`,
  `*.r212mbak`, `*.build21bak` sources; `...whl.{build21bak,r210keeper,
  r212lbak,probebak,r212mbak,r212m2bak}` wheels).
- Next: e2e faces on this wheel (the convergence checkpoint — mxfp4
  offline −27% should move ITL meaningfully; mxfp8 expected flat) per
  the improvement-convergence law; then the remaining offline levers
  (mxfp4 K 178 + Vg 197 still 2.5× the BW floor) via tile width (the
  softmax/P pool) if the faces justify deeper work.
- Standing telemetry law: every wheel validation runs the full probe
  decomposition + dtype oracle (r212m3_val.sh shape), not fulls alone
  — this leg is the proof that lane attribution catches what totals
  hide.

## Artifacts

`offline/p212m3_m{0,2,3,6,8}_32768.json`, `offline/p212m3_m0_{1024,
8192}.json`, `offline/p212m3_dbf0_32768.json` (r212m3);
`offline/p212m2trim_*.json` (the r212m2 cells, renamed to avoid the
r212m bank's names); `r212m2_build1.log`, `r212m2_val1.log`,
`r212m3_build1.log`, `r212m3_val1.log`. Rig: `/root/build/p199/out/`,
`/root/build/p199/logs/r212m*`. Tools: `p212m2_patch.py`,
`p212m3_patch.py`, `r212m3_val.sh`.
