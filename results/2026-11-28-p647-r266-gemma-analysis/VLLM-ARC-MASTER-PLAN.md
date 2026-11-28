
### r266 BATCH-1 COMPLETE — six-leg census banked (2026-11-28, bank `2026-11-28-p647-r266-gemma-analysis`)

Matrix ran to `R266_END 01:10:33Z leg_failures=0` (6/6 legs boot+probe
rc=0). Pull: 71/71 transfer-oracle match, redaction clean, facts.md +
SHA256SUMS written. W14 restore after the batch FAILED (840 s no-health,
rc=1) — diagnosed stalled at the mamba-padding warmup line 01:12:30
(log frozen 2.5 h, workers at wait_until_ready, EngineCore at
wait_for_ready — the W14 JIT-warmup hang class); a clean-recipe rerun
(03:45) restored health 03:50:09 + GEN_OK + rc=0. The failed-restore
log is banked (logs/w14_restore_r266.log + the evidence dir).

Census (per-leg, from the banked matrix JSONs — the ground truth):

| leg | kv | chat t0 ok | raw g3 ok/72 | nan400 | first-token | from token 2 |
|---|---|---|---|---|---|---|
| b2_g3_29 | fp8 | 48/54 (6 NEXTTURN len-artifacts) | 65 | 0 | correct | correct |
| c3_m8g_29 | mxfp8 | 0/54 | 0 | 324 | CORRECT | 'our' attractor |
| c4_m4g_29 | mxfp4 | 0/54 | 0 | 323 | CORRECT | 'our' attractor |
| c5_mkg_29 | mxk8v4 | 0/54 | 0 | 324 | CORRECT | token-2 attractor |
| w1 mxfp4-w4a16 | fp8 | 48/54 | 64 | 0 | correct | correct |
| w2 mxfp8-w8a16 | fp8 | 48/54 | 63 | 0 | correct | correct |

Attribution (facts, all banked):
1. **KV-MX lanes (c3/c4/c5) are CONTENT-RED on the lawful skin** —
   chat t0 0/54, raw g3 0/72, 323-324 x 400-nan each. Weight-MX (w1/w2
   at KV fp8) is CLEAN and byte-equivalent to b2's profile (48/54,
   lp1 -0.677..-0.744, gate answers correct) — weight-MX EXONERATED;
   the defect is confined to the KV-MX store/decode path.
2. **Onset is token-exact**: every gate cell emits token 1 CORRECT
   (Paris/def/Three/HEALTH) then the attractor from token 2; the c3
   prefill NLL cells are CLEAN (para n=156 mean 0.9131 ~ b2 0.9151;
   per-prompt first_top coherent: capital_france -> Paris -0.302 top-1)
   — prefill reads stored KV CORRECT; corruption enters exactly when
   decode first writes-then-reads its own step row.
3. **The two Paris-content gate continuations stay plausible** on
   c4/c5 (c3's gate has both HEALTH cells attractor-collapsed) — the
   stale-slot-reuse signature: a wrong row id holding ANOTHER
   REQUEST'S KV looks sane when content is similar.
4. `post_native_lines=56` on c3/c4/c5 (56 native decode telemetry
   lines per leg — 50 sliding + 10 global - 4 refused? or counted at
   worker0 only); b2/w1/w2 have ZERO native lines — the native lane
   was NOT engaged on the clean legs (KV-fp8 rides the TQ store),
   isolating the defect to ARC_MXKV_NATIVE decode/store.
5. The cert-cut precondition (Law 11) remains BLOCKED on m8g/m4g/mkg
   content; lmc posture ARC_LMC=0 unchanged.

Batch-2 (b3/b5/b4/c1/c2 + the NEW eager pair e1/e2) runs on the v5
runner + the fixed probe (772cf5c3); ship at close-out of THIS bank.
