# VERDICT — r240a: the solo-face report battery on cert28 (2026-11-24)

| Field | Value |
|---|---|
| Leg | r240a — the 16-cell solo-face battery (deep-sweep campaign leg 1) |
| Image | `vllm-arc:v0.30.0-b70-cert28` (fork_sha `43f84cf9`, libattn `61d2f5a3`, libmqa `190e261e`) |
| Rig | 2x Arc Pro B70, TP2; launched 14:46:59, `R240_REPORTS_COMPLETE 2026-11-24T17:59:15Z` (3h 12m) |
| Cells | 14 ran to completion + 2 designed refusal probes (kvauto, kve5m2) |
| Faces | 67 `p138_face_*.json` solo faces (c1024..c131072 ladders + s8192 6-point t-sweeps on k3 cells + t1.0 solos) — 67 FACE_OK, 0 FACE_FAIL, 0 timeouts, 0 tracebacks |
| Hygiene | W14 restored by the EXIT trap, health 200 at 18:04:16; bank pulled with `tools/ar/r240_pull.sh` pattern, redacted (see `../REDACTION-2026-11-24-r240a.md`), SHA256SUMS over 160 files |

## Era re-discrimination — every qwen report cell at parity with its era anchor

ITL p50 ms (solo faces, 1K/8K/32K; era = the shipped docs/reports context
tables, cert23-generation bank):

| Cell | r240a | Era | Verdict |
|---|---|---|---|
| qf3 | 39.7 / 40.5 / 42.7 | 39.7 / 40.5 / 42.7 | byte-equal (5-rung p50s 39.7/40.5/42.7/45.7/51.4 all match) |
| qf0 | 33.9 / 33.3 / 31.2 | 34.0 / 33.4 / 31.2 | parity (−0.1 ms) |
| q83 | 72.6 / 80.9 / 80.4 | 73.3 / 81.1 / 80.4 | parity (−0.2 ms) |
| q80 | 28.6 / 28.6 / 27.0 | 28.7 / 28.6 / 27.0 | parity (8K/32K exact) |
| q43 | 77.4 / 102.5 / 97.8 | 103.3 (record @8K) | at record (−0.8%) |
| q40 | 36.8 / 36.8 / 34.2 | 36.8 / 36.8 / 34.2 | byte-equal on all five rungs (36.8/36.8/34.2/31.4/26.6) |

Decode (implied tps, chunk-accounting law: 1000/itl_mean ×
max_tokens/chunks): qf3 93.2@1K / 90.8@32K; q43 102.5@8K (the record);
q40 36.8@1K..26.6@128K.

**Sole outlier**: qf3-8K 66.4 tps (era 93.3; ITL p50 40.5 is byte-equal —
the mean, not the p50, took a same-boot stall). Single-sample law
(±25% band on 8K-class single samples): dispositioned, NOT a regression
signal; r240b banks the n1 8K second sample on a fresh boot.

## The KV-dtype family's first cert28 solo faces (the KV report's solo columns)

ITL p50 ms, 8K/32K, vs the r234 cert28 record (same-wheel anchors):

| Cell | r240a | r234 record | Delta |
|---|---|---|---|
| kvfp8 (ctrl) | 40.5 / 42.7 | 40.4 / 42.5 | +0.2/+0.5% |
| kvm4 (mxfp4) | 49.1 / 58.8 | 48.5 / 58.1 | +1.2/+1.2% |
| kvm8 (mxfp8) | 50.8 / 65.4 | 50.0 / 64.3 | +1.6/+1.7% |
| kvmk (mxk8v4) | 49.9 / 62.8 | 49.4 / 61.9 | +1.0/+1.5% |

All within the single-sample band — the MX KV family reproduces its
certified records. The standing MX-vs-fp8 ITL gap (~+21% @8K, ~+37%
@32K, the r231/r234 MX-ITL-gap line) is unchanged on this battery —
it is the engine record, not new drift. kvauto131 (fp16 KV servable at
maxlen 131072): 95.0@8K / 91.7@32K — fp16-vs-fp8 KV decode delta
+1.3%, noise-level.

## Gemma cells (r139-era anchors, 1K/8K/32K at maxlen 40960)

gf8 37.1/37.7/39.5 (era 37.1/37.7/39.7 — parity); gm4 26.4/26.9/28.8
(mxfp4 weights, the 26-28 ms family); gm8 37.8/38.4/40.3 (mxfp8
weights, gf8-class). Gemma reproduces its era records exactly; the
128K annex faces run in r240c.

## Refusals banked (designed probes — evidence, not failures)

- **kvauto**: qwen fp16 KV at maxlen 262144 — the fp16 capacity refusal
  (POOL_FACTS class; `kvauto131` banks the servable-at-131072 bound).
- **kve5m2**: engine-level refusal — no e5m2 KV dequant path in esimd
  FA2 on Xe2 (banked on both models now: qwen here, gemma gkv_e5m2 in
  r240b).

## Next

r240b (22 cells: 9 report + 4 TQ + 2 gemma auto + 3 qwen MX + 4 gemma
MX/refusal; the kvm/gkv grids carry the full 5-rung ladder with conc
{1,2,4,8} + t1.0 solos per the three mid-campaign owner directives) —
launched directly after this bank; then r240c (gemma 128K annex + the
gkv_m4x/gkv_mkx MX-KV annex cells), then the docs/reports regeneration
with the extended `p139_report.py` (oracle-proven: 16-file intended
diff + the new gemma KV report) and the r240 close-out PR.
