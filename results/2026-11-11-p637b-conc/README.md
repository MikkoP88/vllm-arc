# P6.37B — concurrency-sweep bank (r139)

VERDICT: PASS — 15/15 cells banked as designed (13 measured `CONC_OK`,
2 by-design boot refusals), 272/272 faces `n_ok > 0`, 0 CONC_TIMEOUT,
0 tracebacks; W14 restored after the window.

The full concurrency matrix behind `docs/reports/` (the P6.39 per-config
reports): measured conc 1/2/3/4/8 curves plus the temperature 0.0/1.0
pair per config, the turboquant KV presets, and the gemma KV-dtype
capacity wall. Run 2026-11-11 04:06–13:47 UTC on the rig, image
`vllm-arc:v0.30.0-b70-cert23`, postures byte-exact with the r138/P6.37
close-out cells.

- Gate: `tools/ar/r139_conc.sh`; faces `tools/ar/p139/p139_conc.py`
  (per-tag per-slot prompt seeds — the prompt-seed law applies per
  REQUEST, so concurrent same-ctx slots never share a prefix; one
  common submit barrier; per-request TTFT separates queue wait from
  prefill).
- 15 cells = 9 config cells (`qf3 qf0 q83 q80 q43 q40 gf8 gm8 gm4`)
  + 4 turboquant KV cells (`tq_k8v4 tq_4bit tq_k3v4 tq_3bit` — the
  qwen-fp8 k3 posture with `ARC_KVDTYPE=turboquant_*`) + 2 gemma KV
  cells (`gkv_auto` @ maxlen 40960, `gkv_auto16` @ 16384).
- Per config cell: ctx {1K, 8K, 32K} (+128K on qwen) x conc
  {1,2,3,4,8} at t0.0 + a t1.0 solo face per ctx; feasibility from the
  booted KV pool (`need = conc*(ctx+600+64)`) — infeasible points bank
  as KV_CAP rows in the cell `conc.log` (28 rows, all at 131K
  high-conc: the measured capacity wall itself); max feasible conc runs
  when 8 does not fit.
- Per cell files: `p139_conc_<cell>_c<ctx>_n<conc>[_t<temp>].json`
  (faces), `posture.txt`, `boot.txt`, `conc.log`, `serve.log`,
  `outcome.txt` (`CONC_OK` / `BOOT_FAIL` — a refusal is a valid banked
  outcome, vmatrix law).
- The two `gkv_*` cells are banked boot refusals by design: gemma fp8 +
  `--kv-cache-dtype auto` caps at ~13.3K ctx on B70 at GMU 0.8 (the
  engine's own `_check_enough_kv_cache_memory`: 8.23 GiB KV needed for
  maxlen 40960, 6.9 GiB for 16384, vs 5.63 GiB available; the engine
  reports max 13312) — the 16384 posture refused too, so both cells
  bank the same fp16-KV wall.
- Host addresses byte-redacted post-pull (law 8; perl, not sed):
  `../REDACTION-2026-11-11-p637b-conc.md`. Hashes: `SHA256SUMS`
  rig-side pre-redaction; `SHA256SUMS.redacted` authoritative over the
  redacted bank (346 files).
- Reports generated from this bank (with the P6.37 r138 and r140 banks)
  by `tools/ar/p139/p139_report.py` → `docs/reports/`.
