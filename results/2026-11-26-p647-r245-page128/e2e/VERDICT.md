# VERDICT — r245 e2e: gemma-4-31b hybrid MX-KV serving on the page-128
# admission (P-C phase close-out)

**Date**: 2026-11-26 (grid 12:07:11–13:04:09 UTC; pulled + audited same day)
**Image**: `vllm-arc:v0.30.0-b70-cert28` + **build6/r245a** wheel (md5 `95451e404b463d9d6efa4bd4a4939b26`) + P-C2 overlay (engine md5 `789b229582c257f213d51157ace31650`)
**fork_sha**: `33759ce7` (main at PR #149)
**Runner**: `tools/ar/r245_gemma_pc_e2e.sh` (hardened: n_ok>0 semantic gates, warmup-death detector, build6 sidecar refuse)
**Model**: gemma4-31b fp8, hybrid KV — D=512 full_attention layers on the page-128 native lane, 50 sliding D=256 layers on the masked twin (window=1024), maxlen 40960, graph on, ARC_LMC=0

## VERDICT: PASS — the P-C phase is e2e-closed

Every cell booted healthy and served real traffic; every scheduled face
either completed with n_ok>0 or banked an honest KV_CAP refusal. The
build5 first-request death (`block_size 128 not yet compiled`) is
closed: the death-signature call — `bt_width=320 window=-1` (the
D=512 full-attention pool at page 128) — executed CLEAN in all three
cells, on both TP ranks, with zero tracebacks across every serve log.

**Census (45 scheduled faces across 3 cells):**

| | gkv_m4 | gkv_m8 | gkv_mk |
|---|---|---|---|
| CONC_OK (n_ok>0) | 13 | 12 | 12 |
| KV_CAP (pool-honest) | 2 | 4 | 2 |
| DEAD / TIMEOUT / SKIP | 0 | 0 | 0 |
| tracebacks | 0 | 0 | 0 |
| LMCache lines | 0 | 0 | 0 |
| pc2_windowed_impl | 100 | 100 | 100 |
| kv_pool_tokens | 105,669 | 54,426 | 71,856 |

The KV_CAP rows are capacity arithmetic, not failures: the gemma hybrid
pools are small (GMU 0.8, maxlen 40960), and the deeper-ctx faces'
NEED (ctx+MT+64 per slot × n) exceeds the pool — mxfp8's 54K pool
caps 8K×n8 and all of 32K×n≥2; mxfp4's 105K pool caps only 32K×n≥4.

**Per-user throughput (p139_conc, MT=600):**

| face | mxfp4 tps/user | mxfp8 tps/user | mxk8v4 tps/user |
|---|---|---|---|
| 1024 n1 | 21.2 | 18.6 | 18.7 |
| 1024 n8 | 16.8 (agg 127.8) | 14.9 (agg 114.0) | 14.8 (agg 113.5) |
| 8192 n1 | 18.8 | 17.4 | 16.4 |
| 8192 n8 | 6.8 (agg 45.8) | KV_CAP | 5.4 (agg 37.2) |
| 32768 n1 | 12.1 | 11.0 | 9.4 |
| 32768 n1 t1.0 | 10.9 | 10.0 | 8.7 |

mxfp4-KV remains the gemma throughput leader at every depth (its
105K pool is also the largest); mxk8v4 trades ~5-10% tps for the
K-precision acceptance headroom (the acceptance-k-precision law: e2m1
K costs ~10% MTP acceptance — the mixed lane keeps e4m3 K). Perf
banking is cold-pass here; the P-G warm/cold split owns the
cert29-image battery.

**Page-128 lane telemetry (per-cell first native call, both TP ranks):**
mxfp4 `bt_width=320 stride=1088`, mxfp8 `bt_width=320 stride=2112`,
mxk8v4 `bt_width=320 stride=1600`, all `window=-1` — the D=512
full-attention pool geometry, served through build6's 512/128
dispatch branch (the offline oracle's r242-authority lane extended to
page 128). The sliding layers ran the masked twin (`window=1024`)
throughout, per the P-C2 window law.

## What this closes

- **P-C hd512 + window phase**: kernel surface (PR #147), engine
  plumbing (PR #148), page-128 admission (PR #149), and now the e2e
  authority leg — the gemma MX-KV first-decode refusal named in the
  cert29 gate ("gemma hd512 MX-KV first-decode refusal") is FIXED on
  build6, proven end-to-end.
- **The hardened-runner class**: this grid is the first e2e run of
  the semantic-gate runner (n_ok>0 per face; non-200 warmup banks
  WARMUP_DEAD). The r245 failrun (37 faces n_ok=0 behind green
  CONC_OK lines) is the counter-evidence it was built against — both
  banked.

## Evidence

- `<cell>/p139_conc_*.json` — 37 face JSONs, all n_ok>0 (audited at
  pull time: faces=37 ok=37 dead=0)
- `<cell>/serve.log` — full per-cell serve logs (redaction pass
  applied at pull); `posture.txt` per cell
- `<cell>/native_telemetry.txt` — the MXKV native-call lines
- `launch.log` — the full grid log (R245_DONE failed=[] at
  13:04:09Z)
- `../VERDICT.md` — the parent bank's r245a offline-oracle VERDICT
  (runs 1-3 gate history; the failrun this grid closes)
- `SHA256SUMS` — e2e data files (VERDICT.md convention-excluded)
