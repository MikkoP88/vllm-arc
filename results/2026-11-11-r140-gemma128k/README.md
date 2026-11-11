# P6.39 — gemma 128K annex bank (r140 + r140c)

VERDICT: PASS — the 128K annex is measured (gm4x: 1K–128K faces,
`n_ok > 0` on every banked face incl. the three re-run 131K faces; the
8-bit control refusal banks the mxfp8 boundary); 0 tracebacks; W14
restored after every window.

The gemma MXFP4 128K annex behind `docs/reports/` (the
`gemma-4-31b-mxfp4-128k` report): the 4-bit weight savings are what
buy the KV headroom for `max_model_len 131072` on the 31B two-head-
size-GDN geometry. Run 2026-11-11 13:56–15:47 UTC on the rig, image
`vllm-arc:v0.30.0-b70-cert23`.

- Gates: `tools/ar/r140_gemma128k.sh` (the annex sweep) and
  `tools/ar/r140c_gm4x131k.sh` (boundary probes + the 131K re-run,
  with `tools/ar/p140/p140_probe.py`); faces
  `tools/ar/p139/p139_conc.py` (harness v2: server-side prompt budget
  `ctx − max_tokens − 64`, round-trip-trimmed — see below).
- Cells: `gm4x` = MXFP4 W4A16 @ 131072 — ctx {1K, 8K, 32K, 131K} x
  conc {1..8} (t0.0) + a t1.0 solo face per ctx; KV pool 288,972
  tokens (identical across the r140/r140b/r140c boots — deterministic
  posture), so 131K n3/n4/n8 bank as KV_CAP rows by design (need up
  to 1,053,888) and the measured 131K capacity verdict is **2
  concurrent requests**. `gm8x` = the mxfp8 8-bit control, whose boot
  refusal at 131072 banks the boundary (engine
  `_check_enough_kv_cache_memory`, same family as the r139 `gkv_*`
  cells).
- The 131K HTTP-400 chain, in full: the original faces 400'd because
  gemma's sentencepiece inflates decode→encode **1.026x** (131064 ids
  → 134479 server tokens > 131072; measured in a throwaway cert23
  container — qwen's BPE round-trip shrinks slightly and always
  fits). r140b's trim to `prompt_len ≤ max_model_len` still 400'd,
  isolating the rule; the r140c probes banked the server's own words
  (`gm4x/p140_probe_gm4x_probe_b131000.json`: "…you requested 600
  output tokens and your prompt contains at least 130473 input
  tokens, for a total of at least 131073…") — the rule is
  **prompt_len + max_tokens ≤ max_model_len**. Harness v2 re-ran the
  three faces within budget; the corrected jsons overwrite the 400
  originals, which stay preserved rig-side under
  `r140/gm4x/http400_orig/` (+ the r140b generation under
  `/root/build/r140b/`).
- Headline faces (131K): solo decode 28.4 tok/s, TTFT 175.9 s (t0.0)
  / 176.5 s (t1.0); n2 aggregate 3.2 tok/s wall with TTFT mean
  264.9 s. All ≤32K gemma faces prefilled ~1.026x their nominal ctx
  under the original harness (decode metrics unaffected); the 131K
  faces prefill ≤ ctx − 664 under v2.
- Host addresses byte-redacted post-pull (law 8; perl, not sed):
  `../REDACTION-2026-11-11-r140-gemma128k.md` (59 substitutions,
  original sha256s, full 400 chain). Hashes: `SHA256SUMS` /
  `SHA256SUMS.r140c` rig-side pre-redaction; `SHA256SUMS.redacted`
  authoritative over the merged bank (38 files).
- Reports generated from this bank (with the r138 + r139 banks) by
  `tools/ar/p139/p139_report.py` → `docs/reports/`.
