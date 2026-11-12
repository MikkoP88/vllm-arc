# m8g CORRUPTION_SIGNAL disposition — gemma4-31b think-spam is baseline

**Date:** 2026-11-12 · **Leg:** r164 g0 native-KV longprobe (cert25x image) ·
**Owner directives:** "ALL --kv-cache-dtype TYPES HAS TO BE CORRUPTION/GARBAGE FREE"
(interpreted per the baseline law: no KV-dtype-attributable corruption — measured
against the same-image native control, never vs CLEAN for a model whose baseline
degenerates).

## Verdict

**The gemma m8g universal-attractor signal is NOT MX-KV corruption.** The same-image
native fp8_e4m3 KV control (r164, container `arc-g0p`, MX backend lines = 0) shows
the same class of degeneration:

| Evidence | m8g (mxfp8) | g0 (native fp8) |
|---|---|---|
| Cross-class attractor | one hash `fc2dddcc7ce6` across all classes | one hash `873d8e07cdec` across essay/code/multi |
| Top window | think-tag loops | `<think>\n\n\n\n<think>` ×38, 44.7% span (essay/code/multi) |
| simple class | degenerate | newline spam `\n`×25 ×149, span 477% |
| P1 Paris (short gen) | FAIL | **FAIL on native** — `reasoning: "\n\n\n..."`, content null, finish=length (3/3) |
| Determinism | rehit-stable per class | cold==rehit==rehit2 per class (t0_identical) |
| Probe verdict | CORRUPTION_SIGNAL | CORRUPTION_SIGNAL |

Banked evidence (rig `/root/build/p647/cert25/g0probe/`):
`longprobe_c25xg0_full.{json,txt}`, `p1_native_{1,2,3}.json`, `serve.log`,
`markers.txt` (pool 56,128 tok), `mx_backend_lines.txt` (=0), `SHA256SUMS`.

## Reading

- gemma4-31b on this wheel/image degenerates into thinking-channel spam at temp-0
  regardless of KV dtype: the attractor hash differs per lane (KV numerics) but the
  signature — prompt-independent collapse + REP-LOOP — is identical, and the native
  lane FAILS the P1 Paris battery outright.
- The differing per-boot first-generation behavior (m8g's first long gen sane, g0's
  collapsed from cold) is numerics-dependent trajectory capture, not a dtype defect:
  both lanes converge to a prompt-independent attractor and stay rehit-stable.
- Channel law confirmed: vLLM 0.30 emits gemma's stream as `message.reasoning`
  (content null). Classifiers must read content+reasoning_content+reasoning.

## Gate consequence (r161 cert25y)

- **Qwen legs (m8q/m4q/mkq): absolute program** — P1 Paris, zero-garble, longreturn
  hash determinism (qwen native baseline is clean; m8q perfect, m4q essay-class
  drift documented as the 4-bit limit).
- **Gemma legs (m8g/m4g/mkg): shape + comparative** — 3× non-empty answers, suite-13,
  erraudit, then `gemma_compare` vs g0: no new flag modes, top-window count ≤2× g0,
  rehit hash determinism.

The gemma baseline defect itself (thinking-mode degeneration on this wheel) is a
model/serving-layer issue outside the KV-dtype lane — recorded for a future P-item;
it does not block cert25y.
