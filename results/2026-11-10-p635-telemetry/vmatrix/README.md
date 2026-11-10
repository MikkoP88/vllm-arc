# P6.35-V full validation matrix (cert23) — 2026-11-10

Owner-directed full matrix on the shipping image `vllm-arc:v0.30.0-b70-cert23`
(graph-exec double-buffer default ON, released as PR #92 / tag
`rel/v0.30.0-b70-cert23`): **qwen3.8-27b × {fp8, mxfp8, mxfp4-W4A16} ×
{spec0, k3, k5}** and **gemma-4-31b × {fp8, mxfp8, mxfp4-W4A16} spec0**
(gemma has no MTP head; LMC=0 posture per gemma4-lmc). Uniform protocol
per cell: boot health, db markers, r113 decode battery (ctok=256),
`p135_face.py 8192 600` solo streaming face. Rate = (max_tokens−1)/(total−ttft)
— spec SSE streams multi-token chunks; see ../README.md probe-tax and
rate-accounting corrections.

- Driver: `tools/ar/r136_vmatrix.sh` (11 cells) + `tools/ar/r136b_makeup.sh`
  (3 make-up cells) — see Incident below.
- Rig artifacts: `/root/build/p635/vmatrix/` (+ `vmatrix_r136.log`,
  `vmatrix_r136b.log`); pulled + redacted via `tools/ar/r136_vmatrix_pull.sh`
  (perl byte-exact; SHA256SUMS.redacted covers all banked files).
- Postures byte-exact from the r135 sweep boot echoes (fp8: native-fp8 ckpt,
  GMU 0.8, FP8=1; mx8/mx4: BASE ckpt + QCFG ignore [visual/linear_attn/mtp.fc],
  GMU 0.85, FP8=0; gemma: GMU 0.8, LMC 0, RPC 1200, MAXLEN 40960). Each cell's
  `posture.txt` carries the full `ARC_SERVE_LAUNCHED` echo.

## Decode — solo U8k faces (8192-token prompt, 600 max tokens)

### qwen3.8-27b

| quant \ spec | k0 | k3 | k5 |
|---|---|---|---|
| fp8 (native ckpt) | 32.8 (ITL 30.0) | 94.5 † | **114.0** (47.6) |
| mxfp8 (online) | 28.2 (35.0) | 83.6 (45.4) | 101.8 (52.7) |
| mxfp4-W4A16 (online) | **36.1** (27.2) | 103.0 (37.2) | **128.3** (44.2, 5.93 tok/chunk) |

† fp8-k3 = cert23 gate dbon/dboff replicates (94.5/94.3) — not re-run here
(double-covered). ITL = p50 ms per step.

### gemma-4-31b (spec0)

| quant | tok/s | ITL p50 | note |
|---|---|---|---|
| fp8 (online) | 26.6 | 37.7 | **new capability** — first gemma online-fp8 boot |
| mxfp8 | 26.0 | 38.5 | matches gate gemmam8 26.0 (repeatability) |
| mxfp4-W4A16 | **37.1** | 27.0 | **+43% vs mxfp8** |

## TTFT (8192-token prompt)

qwen: fp8k5 **3.48 s** (best) · fp8k0 4.27 · mx8k0 4.47 / k3 4.18 / k5 3.92 ·
mx4 k0 4.40 / k3 4.01 / k5 5.11 (mx4 online-quant prefill slower, worst at k5)
· fp8k3 clean base 4.08–4.29 (gate). gemma: 4.06 / 4.09 / 4.16 across
fp8/mx8/mx4 — quant choice does not move gemma TTFT.

## r113 decode battery (ctok=256, wall includes its 256-tok prefill; rep2 shown)

| cell | solo tps | agg8 tps |
|---|---|---|
| qf0 | 34.2 | 230.4 |
| q8_0 | 29.3 | 199.4 |
| q4_0 | 37.9 | 247.5 |
| q8_3 | 54.9 | 269.1 |
| q8_5 | 51.2 | 233.4 |
| q4_3 | 65.2 | 302.1 |
| q4_5 | 59.3 | 241.6 |
| qf5 | 55.6 | 259.6 |
| gf8 | 28.3 | 196.5 |
| gm4 | 40.5 | 262.6 |
| gm8 | 27.7 | 192.9 |

Battery ranks below faces for mx4 cells (short-generation walls are
prefill-dominated; mx4 prefill is slower) — faces are the decode truth.

## Findings

1. **mxfp4-W4A16 wins everywhere it runs.** qwen: best column at every spec
   depth (36.1 / 103.0 / **128.3**, +36% over the clean fp8-k3 base 94.5 —
   new project best; the gate's 121.0 read had lower draft acceptance, 5.50
   vs 5.93 tok/chunk). gemma: **+43%** (37.1 vs 26.0). W4A16 weight streaming
   (≈13.5 GB/rank→≈7 GB) is model-agnostic, exactly as the weight-bytes law
   predicts.
2. **mxfp8 is a decode dud at every depth** (28.2 / 83.6 / 101.8 — below
   fp8 at k0/k3 and below mx4 everywhere): online W8 quant lands on a slower
   path than the native-fp8 checkpoint. Not recommended for any posture.
3. **gemma online fp8 boots and serves** (26.6 tok/s, ITL 37.7) — previously
   untested; parity with mxfp8 (26.0). Correctness/acceptance sanity: 0
   tracebacks in every cell; faces complete at expected chunk cadence
   (spec cells ≈ k-depth tok/chunk, spec0 = 1.00).
4. **k5 > k4 > k3 across the board** (fp8: 94.5→100.6→114–115; mx4:
   103→113→123–128); spec depth is the second decode lever and its draft
   cost stays ~3–7 ms/step.
5. Posture guidance stands, now with the gemma counterpart: qwen decode
   winner = mxfp4-W4A16 k5 (QCFG ignore visual/linear_attn/mtp.fc, GMU 0.85,
   FP8=0, BASE ckpt); gemma decode winner = mxfp4-W4A16 (GMU 0.8, LMC 0,
   RPC 1200, MAXLEN 40960). fp8 native remains the prefill/TTFT pick
   (fp8k5 TTFT 3.48 s).

## Incident (does not affect cell data)

Three cells (q4_0, q8_3, gm8) initially boot-failed on a stale-container
cascade: the 27B GPU-container `docker rm -f` can take >30 s to vanish, the
30-s verify window declared `VM_FAIL_STALE_CONTAINER`, and the NEXT cell's
serve script then tripped on the dead record. All cell DATA preceding each
stale teardown was banked before the marker (banking precedes pkill), and a
rig-side watchdog (`r136_stale_watchdog.sh`, fired 16:23:25–16:24:15Z)
completed the pending rm so follow-on cells failed fast. The three lost
cells re-ran clean in `r136b_makeup.sh` with a hardened 90-s retrying
teardown (zero stale events). Banked logs carry the full sequence.

## Hashes

`SHA256SUMS` = rig-side face hashes pre-pull; `SHA256SUMS.redacted` =
all banked files post-redaction. Host addresses byte-redacted with perl
(see ../../REDACTION-2026-11-10-p635.md for the method; vmatrix redactions
follow the same pass — originals on rig at /root/build/p635/vmatrix/).
