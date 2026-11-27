# r264 — the cert29 gate on the rebuilt candidate (P6.47)

**VERDICT: GATE_FAIL=1 → `CERT_CUT_BLOCKED` — the r262 stride fix held
everywhere (zero DEVICE_LOST, kernel health OK on all 13 legs, the bridge
guard engaged on every MX lane), ten of eleven legs green, but the mkg lane
(gemma + mxk8v4 KV) is RED on the LONGRETURN absolute check: cross-cycle
hash drift with REP-LOOP flags on all four classes. Per the owner's
cert-cut precondition (2026-11-27), no certificate image is cut until the
gemma MX lanes m8g/m4g/mkg AND the lmc lane are all green on a candidate.**

`R264_CERT_PRECONDITION gemma_mx+lmcache=[ m8g:GREEN m4g:GREEN mkg:RED lmc:GREEN ] rest=[ gref:GREEN g0:GREEN compare:GREEN ctrl:GREEN m8q:GREEN m4q:GREEN mkq:GREEN ] gate_fail=1 -> CERT_CUT_BLOCKED`

## What ran

`tools/ar/r264_cert29_gate.sh` (FORK_SHA pin `4dd3a1b9710f1c638f7f0b5ac1885525cf57c3bc`,
serve-script md5 pin `c2db777506e776a86426852b6bb80de4`, image
`vllm-arc:v0.30.0-b70-cert29` ID `77e08563f7c7` — the r263 rebuild from the
r262 content commit), launched 2026-11-27, bank
`/root/build/p647/cert29_r263`, pulled here by `tools/ar/r264_pull.sh`
(rig-side sums as transfer oracle: 238 entries, zero mismatch, zero missing;
address audit clean — see `../REDACTION-2026-11-27-p647-cert29-gate-r264.md`).

## The record

| leg | posture | result |
| --- | --- | --- |
| gref | gemma + mxfp8 KV, native OFF | green — the PC2 refusal face (`GREF_PC2_REFUSAL_OK refusal_lines=2`) |
| m8g | gemma + mxfp8 KV + native + window | green — LONGRETURN + T1_SEED PASS, `STRIDE_FIX_ENGAGED materialized_lines_post=20` |
| m4g | gemma + mxfp4 KV + native + window | green — same |
| mkg | gemma + mxk8v4 KV + native + window | **RED — LONGRETURN FAIL** (below) |
| g0 | gemma fp8 native-KV control | green — full ungated suite clean |
| compare | gemma comparative (mkg-vs-g0 modes) | green — `GEMMA_COMPARE: PASS (no new modes / no amplification)` |
| ctrl | qwen fp8 baseline + LMC | green |
| m8q / m4q / mkq | qwen + MX KV lanes | green (pool 1.47x / 2.75x / 1.93x) |
| lmc | qwen + mxfp4 + LMCache interop | green — G1-G5, A/A re-hit 6.047→1.894 s, zero transfer errors |

Kernel health OK on every leg and the battery window (`fault_lines=0` each);
ride-completion gates green on all four mkg longprobe files
(`class_lines=4 empty_garble=0 probe_errors=0`); W14 restored by the exit
trap (`W14_RESTORE rc=0 2026-11-27T18:58:01Z`).

## The mkg RED — facts as measured

`LEG_VERDICT[mkg]: RED failed_checks=1` — the LONGRETURN absolute check:

- All four long-return classes (essay/code/multi/simple) finished
  `length` with cross-cycle hash drift (`['5c80063c79a1','fc2dddcc7ce6']`
  chars 6154/6157 on essay-code-multi, 6155/6140 on simple) and REP-LOOP
  garble flags on every repetition.
- T1_SEED PASS: within a single temperature-1 seed the returns are
  deterministic — the drift is across sampling cycles, not within one.
- No kernel event: mkg kernel health OK, no DEVICE_LOST line, no traceback;
  the r262 stride guard stayed engaged (`STRIDE_FIX_ENGAGED[mkg]
  materialized_lines_post=20`).
- The comparative verdict vs the fp8 control under the same posture:
  `GEMMA_COMPARE: PASS (no new modes / no amplification; known-class churn
  banked for review)` — the RED is the absolute content check only.

## The posture fact this gate banked (invalid-by-directive, follow-up r266)

Every gemma leg ran under the qwen chat skin — banked in each leg's
`boot.txt`: `tmpl=__qwen_default__ rp=<qwen3> tp=<qwen3_coder>` (the serve
script's model-agnostic defaults; the P6.28 gemma template never landed in
main). Per the owner directive (2026-11-28): **gemma must never be tested
under qwen env values — only gemma's own chat template and parsers are a
valid test posture.** By that rule every gemma content verdict in this gate
— the mkg RED included, and the m8g/m4g greens too — is invalid-by-posture:
not evidence about gemma, only about gemma-under-a-foreign-skin. The RED is
therefore unattributed, not fixed-to-mkg: the r266 analysis matrix (running
2026-11-28) re-measures every KV dtype under the P6.28 gemma skin with NO
parsers, plus the weight-MX/KV-fp8 lanes, to attribute it under a lawful
posture before any fix is chosen — and the section-10 gate must re-skin its
gemma legs the same way before its Law-11 precondition line can certify
anything. Also banked: the gemma legs pass `ARC_RPC=1200`, which the serve
script never reads (it reads `ARC_RPC_TIMEOUT`) — the watchdog was the
300 s default on every gemma leg; no-op defect recorded.

## Consequence

cert29 remains a candidate image. The cut is held until the gemma MX lanes
and the lmc lane are green on a candidate — the mkg lane must be resolved
on content, not waived (owner directive 2026-11-27, Law 11).
