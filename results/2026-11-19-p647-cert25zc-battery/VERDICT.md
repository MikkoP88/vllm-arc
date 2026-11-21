# VERDICT — P6.47 r197 cert25zc full-par10 battery (2026-11-19)

**PASS** — all legs under the amended gate (7fe72efc). Image
`vllm-arc:v0.30.0-b70-cert25zc` (manifest `cert25zc_fork_sha=
dd74715079dc6c23fd4ad2feb1a700294515d423`, refuse-gated at battery
start). cert25zc = cert25zb content + the Phase-7 closing rung:
TILE_FLOOR launcher default 4->2 (`ARC_MXKV_V5_TFLOOR`), G-adaptive
mxfp4 verify-decoder dispatch (`ARC_MXKV_V5_G_MIN=8`), v7 research
kernel DORMANT (`ARC_MXKV_V7=0`).

## Verdict of record

The battery instance ran with the pre-amendment script (its m8q
LMC-on expectation predates commit 7fe72efc); the final verdict
re-derives posture from the banked markers under the amended gate per
the recompute-from-banked-artifacts law — `verdict_recompute.sh`
(banked here, output in `verdict_recompute.out`): **0 fails**.

## Quality faces

| leg | mx | LMC | probe | P1 | garbage | longreturn | T1_SEED | erraudit |
|-----|----|-----|-------|----|---------|-----------|---------|----------|
| ctrl | 0 | on(4) | 3/3 | Paris | - | - | - | ERROR=0 |
| m8q | 1 | auto-off | 3/3 | Paris | PASS | PASS | PASS | ERROR=0 |
| m4q | 1 | auto-off | 3/3 | Paris | PASS | PASS | PASS | ERROR=0 |
| mkq | 1 | auto-off | 3/3 | Paris | PASS | PASS | PASS | ERROR=0 |
| m8g | 1 | off | 3/3 shape | - | PASS | banked* | PASS | ERROR=0 |
| m4g | 1 | off | 3/3 shape | - | PASS | banked* | PASS | ERROR=0 |
| mkg | 1 | off | 3/3 shape | - | PASS | banked* | PASS | ERROR=0 |
| g0 | 0 | off | 3/3 | Paris | PASS | banked* | PASS | ERROR=0 |

*gemma legs are comparative (reploop-classifier-baseline law): the
native g0 control itself fails LONGRETURN on the same REP-LOOP classes;
`GEMMA_COMPARE: PASS (no new modes / no amplification; known-class
churn banked for review)` is the gemma verdict of record
(`gemma_compare.txt`). Qwen lanes gate ABSOLUTE and are clean 3/3.

Phase-6 LMC posture held: ALL MX lanes auto-off (NOTE + zero
LMCacheConnector lines), ctrl keeps the certified LMC-on.

## Decode smoke face (256-tok solo / 2048-tok agg8, tps medians)

| leg | solo | agg8 | agg8/ctrl |
|-----|------|------|-----------|
| ctrl | 59.9 | 304.1 | 1.00 |
| m8q | 54.2 | 236.7 | 0.78 |
| m4q | 57.2 | 229.0 | 0.75 |
| mkq | 55.9 | 226.6 | 0.74 |
| g0 | 27.7 | 192.5 | - |
| m8g | 25.2 | 144.5 | 0.75 (vs g0) |
| m4g | 26.6 | 142.2 | 0.74 (vs g0) |
| mkg | 26.1 | 138.0 | 0.72 (vs g0) |

Consistent with the r196b zcA27 fastcycle ladder (banked at
`results/2026-11-19-p647-zc-fastcycle/`): the closing rung held its
fastcycle gains e2e (m4q n8 32K agg p50 1547->583 ms -62% at the
ladder face). The remaining ~0.74-0.78x agg8 gap vs ctrl is the
Triton split-dot structural deficit — the arc7 MX-native kernel
(Phase 8, in flight) is the open line on it.

## Artifacts

192 files, SHA256SUMS verified 192/192 pre-redaction; 351 host-address
sites byte-exact redacted post-verify (perl; see
`../REDACTION-2026-11-19-p647-cert25zc-battery.md`). W14 restored
(health 200 verified post-battery).

Release disposition: proceed per the cert sequence — content commit
dd747150 (manifest fork_sha, already gated), docs currency commit, PR
squash-merge to main, tag cert25zc, gh release.
