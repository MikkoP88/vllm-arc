# REDACTION 2026-11-11 — P6.37B concurrency bank

## P637B concurrency-sweep bank (results/2026-11-11-p637b-conc/, r139 gate)

- Same two perl substitutions over the whole bank tree, applied by
  `tools/ar/r139_pull.sh` (`10.20.3.<host>` → `10.20.3.REDACTED` — 344
  occurrences; `10.100.8.<host>` → `10.100.8.REDACTED` — 333
  occurrences; 677 total). Hits live only in the 15 per-cell
  `serve.log` files (boot `mq_connect_ip=` / Uvicorn bind / health
  probe source lines); all 272 `p139_conc_*.json` faces, outcome and
  boot files verified address-free (md5 spot-checks of four faces
  across families matched the rig bytes exactly — byte-identical,
  hence no address content).
- Hashes: rig-side `SHA256SUMS` = pre-redaction hashes; local
  `SHA256SUMS.redacted` = the full bank (346 files) hashed from the
  REDACTED bytes (authoritative). Pre-redaction originals remain on
  the rig under `/root/build/p637b/`. No measurement, timestamp,
  counter, or verdict altered.
- Bank shape: 15 cells (qf3 qf0 q83 q80 q43 q40 gf8 gm8 gm4 tq_k8v4
  tq_4bit tq_k3v4 tq_3bit gkv_auto gkv_auto16). The two `gkv_*` cells
  are banked boot refusals by design: gemma fp8 + `auto` KV dtype caps
  at ~13.3K ctx on B70 at GMU 0.8 (8.23 GiB KV needed for 40K, 6.9 GiB
  for 16K, vs 5.63 GiB available) — the engine's own
  `_check_enough_kv_cache_memory` refusal, logged in each cell's
  `serve.log`; the matrix continued per the gate design.
- `tools/leak_sweep.sh` exit 0 after the pass.
