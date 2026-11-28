# VERDICT — r266: the gemma-4 lawful-skin analysis matrix (P6.47)

**Date**: 2026-11-27/28 (batch-1 launched 23:32:57Z; END 01:10:33Z)
**Images**: `vllm-arc:v0.30.0-b70-cert29` (all six legs)
**Harness**: `tools/ar/r266_gemma_skin.sh` v4 (md5 00323de7) +
`p266_matrix_probe.py` (batch-1 ran d2a6e9fa — the per-prompt-NLL
wire-key gap; the fix 772cf5c3 ships for batch-2, PR #172), runner-
stopped W14 with EXIT-trap restore; `tools/ar/r266_pull.sh` banked
71/71 transfer-oracle match, redaction clean.

## VERDICT: TWO DEFECT CLASSES SEPARATED — the lawful skin is clean;
## the engine KV-MX path is content-broken with token-exact onset

### 1. The lawful gemma skin works end-to-end (b2, KV fp8)

chat t0 ok=48/54 (the 6 non-ok are `NEXTTURN` length artifacts —
finish=length at MAXTOK=96, never degeneration), card 52/54, REPLOOP
0 everywhere, raw g3 65/72 ok, lp1 -0.68 (t0) / -1.67 (card),
PROMPT_ORACLE match=True, ptok 216/216 (engine tokenizes every
format exactly as the probe renders). The gemma test skin is
g3-legacy-text + NO parsers (owner directive posture).

### 2. gemma4-31b is a BASE/pretrained checkpoint

`tokenizer_config.json` ships NO `chat_template`;
`<start_of_turn>`/`<end_of_turn>` do not exist in the vocab (the g3
markers fragment into ordinary pieces; official specials sit at ids
98-106 in formerly-unused slots). The official g4/g4think structure
degenerates CONFIDENTLY on this checkpoint (25-26/72 raw REPLOOP,
glp_mean -0.23/-0.27 vs clean -0.38/-0.48) — checkpoint behavior.
Every prior "gemma degenerates" RED measured under a foreign (qwen)
or official-template skin was a POSTURE artifact, not engine defects.

### 3. The KV-MX native lanes are content-RED — with the onset pinned

Census (banked matrix JSONs, the ground truth):

| leg | kv | chat t0 ok | raw g3 ok/72 | nan400 | token 1 | from token 2 |
|---|---|---|---|---|---|---|
| b2_g3_29 | fp8 | 48/54 | 65 | 0 | correct | correct |
| c3_m8g_29 | mxfp8 | 0/54 | 0 | 324 | CORRECT | 'our' attractor |
| c4_m4g_29 | mxfp4 | 0/54 | 0 | 323 | CORRECT | 'our' attractor |
| c5_mkg_29 | mxk8v4 | 0/54 | 0 | 324 | CORRECT | 'our' attractor |
| w1 mxfp4-w4a16 | fp8 | 48/54 | 64 | 0 | correct | correct |
| w2 mxfp8-w8a16 | fp8 | 48/54 | 63 | 0 | correct | correct |

- All three MX fmts corrupt IDENTICALLY → fmt-independent, structural.
- Weight-MX at KV fp8 is CLEAN (w1/w2 match b2's profile) → the defect
  is confined to the KV-MX store/decode path; weight-MX exonerated.
- The c3 prefill NLL cells are CLEAN: para n=156 mean 0.9131 (b2:
  0.9151), per-prompt first_top coherent (capital_france → `Paris`
  -0.302 top-1). Prefill reads the stored KV CORRECT.
- Onset is TOKEN-EXACT: every gate cell emits token 1 correct
  (`Paris`, `def`, `Three`, `HEALTH`) then collapses from token 2 —
  the first decode step that writes-then-reads its own new row.
- Zero DEVICE_LOST, zero tracebacks, kernel health OK on all legs —
  silent corruption (the r264 gates' text-only checks could never see
  it; the logprobs-carrying probe caught it as 400 nan-JSON).

### 4. Geometry + root-cause narrowing (banked evidence)

- Impl census (c3 serve.log): NO fold — the 10 global layers are
  NATIVE D=512 (20/120 impl lines `head=512 slot=1056`; sliding
  100/120 `head=256 slot=528`). Both classes ride ONE kernel binary
  (windowed P-C2 / windowless are window args on `mx_paged_decode`).
  Sliding: D=256 Hk=8/rank page 64 W=1024; global: D=512 Hk=2/rank
  page 128 windowless.
- Kernel MATH is offline-green at both geometries (r242 D=512p64
  V-slice fix; r245a D=512p128; r243 windowed W=1024) — but those
  harnesses hand-quantized their caches and ran no graphs. The defect
  lives in the surface they cannot see, with four open gaps:
  G1 the mxfp8 x W=1024 long-ctx widewin cell (validated mxfp4 only);
  G2 the Triton store as cache-producer at gemma geometry (never
  byte-audited at D=512 — the kv-store-separate-strides class);
  G3 engine metadata VALUES (r262 fixed shape only, no content check);
  G4 the runtime A/B skipped for windowed impls (engine.py ~1517).
- The stale-slot-reuse signature (Paris-content requests continue
  PLAUSIBLY while others attract — a wrong row id holding another
  request's KV looks sane when content is similar) points hardest at
  the XPU-graph capture/replay slot/store wiring, which no offline
  harness exercises. Batch-2's eager pair (e1/e2, ARC_GRAPH=0) is the
  designed discriminator; r267 then runs the offline NaN-canary with
  the STORE as producer at both classes, all three fmts, mxfp8 W=1024
  long ctx, real auto split plan.

### 5. Probe defect found and fixed mid-flight (PR #172)

Wire dicts key token ids as STRINGS — `pl_stats` normalized to int
keys (per-prompt NLL n=0 → fixed); selftest fixtures made wire-true;
81/81 PASS. Batch-1 legs uniformly carry the gap (para NLL n=156
remains the cross-leg signal; per-prompt nll cells read n=0).

### 6. W14 restore failure banked + recovered

The batch's EXIT-trap restore failed (840 s no-health, rc=1): the
boot stalled at the mamba-padding warmup line (log frozen 01:12:30 →
diagnosis; workers at wait_until_ready / EngineCore at wait_for_ready
— the W14 JIT-warmup hang class). Clean-recipe rerun restored health
03:50:09Z + GEN_OK + rc=0. Failed-restore evidence banked.

## Consequences

- **Law 11 (cert-cut precondition) stays BLOCKED**: m8g/m4g/mkg
  content-RED on the lawful skin; lmc posture ARC_LMC=0 (needs its
  own lawful run after the KV-MX fix).
- **Every gate request set MUST carry logprobs** — NaN → http_err is
  the free corruption detector; text-only checks are existence-green
  by construction (the r264 lesson, now proven on six legs).
- The r264 m8g/m4g GREENs were existence-green (vacuous); the mkg RED
  was the same KV-MX corruption seen through the qwen skin.
- Fix order: batch-2 (b3/b5/b4/c1/c2 + e1/e2 eager pair) → r267
  offline NaN-canary (store-as-producer) → the store/metadata fix →
  re-gate with logprobs → full battery → cert29 cut ONLY on
  `CERT_CUT_ALLOWED` (m8g/m4g/mkg + lmc all green, Law 11).
