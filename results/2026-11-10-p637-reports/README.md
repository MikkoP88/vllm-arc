# P6.37 — close-out per-config report bank (r138 + r138b)

The measurement bank behind `docs/reports/` (the per-config performance
analytic reports: 2 models x 3 quantizations x {MTP k3, no MTP}, the
k3 temperature sweep 0.0–1.0, and the `--kv-cache-dtype` comparison).
Run 2026-11-10 20:04–22:58 UTC on the rig, image
`vllm-arc:v0.30.0-b70-cert23`, riding the close-out F defaults
(`--shm-size 16g`, 16 GB LMCache CPU tier) — this run is the law-6
validating leg for those defaults.

- Gate: `tools/ar/r138_reports.sh` (13 cells, rig log
  `/root/build/p637_gate.log`); supplement `tools/ar/r138b_kvauto131.sh`
  (the auto-dtype 131072 cell); faces: `tools/ar/p138/p138_face.py`
  (per-tag prompt seeds — same-ctx faces never share a prompt, so no
  LMCache prefix hits corrupt TTFT).
- Per cell: `p138_face_<cell>_c<ctx>_t<temp>.json` (client-side face:
  TTFT, total, ITL quantiles, stream-chunk count), `posture.txt` (the
  banked serve echo), `metrics.txt` (engine /metrics), `serve.log`
  (full boot+serve log), `boot.txt`, `outcome.txt` (`FACES_OK` /
  `BOOT_FAIL` — a refusal is a valid banked outcome, vmatrix law).
- Face inventory: qwen k3 cells 11 faces (5 ctx 1K/8K/32K/64K/128K +
  6 temp @ 8K), qwen k0 cells 5, gemma cells 3 (1K/8K/32K — model
  maxlen 40960), kv cells 2 (8K/32K) — 61 faces total, 0 timeouts,
  0 tracebacks in any banked serve.log.
- `kvauto` = `--kv-cache-dtype auto` (fp16 KV) refusal at the certified
  `max_model_len=262144`: one max-len request needs 8.73 GiB KV vs
  6.78 GiB available (GMU 0.8, TP2) — `kvauto131` re-measures it at
  `ARC_MAXLEN=131072`. `kve5m2` = hard engine refusal:
  `FlashAttention does not support fp8_e5m2 kv-cache on this device`
  (no e5m2 KV path in the esimd FA2 backend on Xe2).
- Host addresses byte-redacted post-pull (law 8; perl, not sed):
  see `../REDACTION-2026-11-10-p635.md` P6.37 section. Hashes:
  `SHA256SUMS`/`SHA256SUMS.b` rig-side pre-redaction;
  `SHA256SUMS.redacted` authoritative over the redacted bank.
- Reports generated from this bank by `tools/ar/p138/p138_report.py`:
  `docs/reports/README.md` + 10 per-config reports + 49 line charts.
