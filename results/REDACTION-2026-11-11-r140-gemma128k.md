# REDACTION 2026-11-11 — r140 gemma-128K annex bank

## r140 annex bank (results/2026-11-11-r140-gemma128k/, r140 + r140c gates)

- Same two perl substitutions over the bank tree, applied by
  `tools/ar/r140_pull.sh` (original pull) and `tools/ar/r140c_pull.sh`
  (the 131K re-run merge): `10.20.3.<host>` → `10.20.3.REDACTED` and
  `10.100.8.<host>` → `10.100.8.REDACTED` — 59 occurrences total in
  the merged bank (31 rig + 28 client), confined to the three serve
  logs (`gm4x/serve.log` 29, `gm4x/serve_r140c.log` 29,
  `gm8x/serve.log` 1 — boot `mq_connect_ip=` / Uvicorn bind / health
  probe source lines). All `p139_conc_*.json` faces,
  `p140_probe_*.json` probes, outcome and boot files verified
  address-free (md5 spot-checks of all three re-run faces matched the
  rig bytes exactly).
- Hashes: rig-side `SHA256SUMS` = the original r140 pull pre-redaction;
  `SHA256SUMS.r140c` = the re-run pre-redaction; local
  `SHA256SUMS.redacted` = the full merged bank (38 files) hashed from
  the REDACTED bytes (authoritative). Pre-redaction originals remain
  on the rig under `/root/build/r140/` (plus `/root/build/r140b/` and
  `/root/build/r140c/`). No measurement, timestamp, counter, or
  verdict altered.
- Bank shape: 2 cells. `gm4x` = gemma-4-31B MXFP4 W4A16 @ 131072
  max len — the 128K annex faces 1K–131K (131K n3/n4/n8 are banked
  KV_CAP skips by design: pool 288,972 tokens vs need up to
  1,053,888). `gm8x` = the 8-bit control probe (mxfp8), whose boot
  refusal at 131072 banks the boundary (same
  `_check_enough_kv_cache_memory` capacity refusal as the r139
  `gkv_*` cells).

## The 131K HTTP-400 chain (r140 → r140b → r140c)

- The original r140 131K faces (`n1`, `n2`, `n1_t1.0`) all banked
  `HTTP Error 400: Bad Request`: the harness decoded ctx−8 token ids
  to text and the server re-tokenized — gemma's sentencepiece inflates
  the round-trip **1.026x** (measured in a throwaway cert23 container:
  131064 ids → 134479 server tokens at 131K; 32760 → 33603; 8184 →
  8404), busting `max_model_len`. qwen's BPE round-trip shrinks
  slightly and always fits — its banked 131K faces passed on the
  8-id margin. The originals' sha256s (pre-overwrite):
  `n1` 2ed868750b9230197dfb03abac83d65bf9bbfb3c09470ca26fa108458d96812d,
  `n2` 7e2ca4d80d7f3f02d41743d99b0ddb0bb1741d9810f36a11e194e45eafb0da91,
  `n1_t1.0` 72f973611f80a9a30d8204a941a38e03dbee5a04f70ae9f2ca4a6a3475193250.
- r140b re-ran the faces with a trim to `prompt_len ≤ max_model_len` —
  still 400 (banked rig-side under `/root/build/r140b/`), which
  isolated the true rule. The r140c boundary probes banked the
  server's own words (`gm4x/p140_probe_gm4x_probe_b131000.json`):
  *"This model's maximum context length is 131072 tokens. However,
  you requested 600 output tokens and your prompt contains at least
  130473 input tokens, for a total of at least 131073"* — the rule is
  **prompt_len + max_tokens ≤ max_model_len** (the 130300-budget probe
  returned 200 with a full 130K prefill + 600 tokens generated).
- The final re-run (r140c, harness v2: server-side prompt budget
  ctx − max_tokens − 64) banked all three faces with `n_ok > 0`;
  the corrected jsons overwrite the 400 originals in this bank, and
  the originals stay preserved rig-side under
  `/root/build/r140/gm4x/http400_orig/`.
- `tools/leak_sweep.sh` exit 0 after the pass.
