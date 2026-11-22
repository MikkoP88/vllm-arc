# r212m4 — e2e convergence checkpoint of the r212m3 wheel

**Date**: 2026-11-21 · **Leg**: r212m4 (arc7 Phase-8, e2e faces on the
r212m3 wheel — the improvement-convergence checkpoint) · **Stack**: r212m3
wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`e50f1c3aa38729931c0d6ccc9c57a370`, pinned by guard at leg start) +
`vllm-arc:v0.30.0-b70-cert25zc` + MX overlay `/root/build/p199/overlay/
mx_kv` (engine.py + native_decode.py `max_seqlen_k=_max_sk`, banked
`overlay_md5.txt`) · `ARC_MXKV_DIRECTBF16` at its default ON — the faces
measure the shipping state · **Run**: `r212m4_e2e1.log` 14:47–15:00 UTC,
two boots (m4qa then m8qa), zero errors/tracebacks, 108 native-call
telemetry lines per format, GATE_PASS

## Verdict

**The offline −27% transfers end-to-end: mxfp4 lands 48.9 / 56.7 /
73.1 ms/tok (−14.9% / −23.5% at the banked ctx; c32768 banked for the
first time) and becomes the fastest MX lane at EVERY context — ahead of
mxfp8 (50.5 / 60.3 / 81.7, −0.8/−1.8/−3.1% = the byte-equivalent float
path holding exactly flat, as the oracle predicted). Gap to the fp8 bar
(40.4/41.1/42.5 ctrlA24 cert25zc): 1.21× / 1.38× / 1.72× (mxfp4),
1.25× / 1.47× / 1.92× (mxfp8).**

### The transfer coefficient

| lane | offline (32K full, r212m3 vs r212l) | e2e face | e2e delta |
|---|---|---|---|
| mxfp4 | −27.1% | c16384 (−23.5%) | ~87% of the kernel gain reaches ITL |
| mxfp8 | −2.9% | c32768 (−3.1%) | 1:1 within noise |

mxfp8's e2e flatness is the dtype-split oracle's e2e echo: the reverted
e4m3 path is byte-equivalent, so nothing but noise could move. mxfp4's
missing ~13 points at c16384 are the non-decode parts of the step
(softmax/P+staging, launch floor) that the kernel gain does not touch.

### Per-token rates (MTP k3, acceptance ~3.5 t/chunk)

| fmt | c8192 | c16384 | c32768 |
|---|---|---|---|
| fp8 bar | ~87 | ~85 | ~82 |
| mxfp4 | ~72 | ~62 | ~48 |
| mxfp8 | ~69 | ~58 | ~43 |

### Disposition

- The lane is NOT yet at fp8 parity (owner directive) — the remaining
  pool is the walk excess measured offline: mxfp4 32K full 749.3 µs vs
  ~298 µs BW-floor estimate, with K chain 178.2 + V gather 196.9 as the
  residual per-element lanes and the ~147 µs softmax/P+staging
  tile-width pool from r212k.
- Next iteration (telemetry law, ≤30-min cycle): fresh probe
  decomposition INCLUDING probe 9 on this wheel (the r212m3 lane table
  predates the c8 floor cell's layout shift; the decode-severed floor
  needs re-pricing on the converged wheel), then the tile-width lever
  decision.
- r212m3 wheel remains the working state; e2e numbers here are the new
  MX face bars for subsequent iterations.

## Artifacts

`faces/{m4qa,m8qa}/c{8192,16384,32768}_n1/p139_conc_c*_n1.json` (1
request per face, `error: null`, p50/p90/p99 + ttft/total),
`faces/*/serve.log` + `serve_snapshot.log` (redacted — see
`../REDACTION-2026-11-21-p647-r212m4.md`), `faces/*/{telemetry,dispatch,
erraudit,boot}.txt`, `overlay_md5.txt`, `wheel_md5.txt`,
`r212m4_e2e1.log` (leg stdout + gate). Rig: `/root/build/p647/r212m4/`,
`/root/build/r212m4_e2e1.log`. Tool: `tools/ar/p147_mxkv/r212m4_e2e.sh`
(commit ce7cb252).
