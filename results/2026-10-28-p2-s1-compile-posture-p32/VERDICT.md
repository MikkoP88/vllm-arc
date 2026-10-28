# VERDICT — S1/p32 compile posture leg: FAILS gates (a)(c)(d) — the 0.26 compile×spec law is re-proven on 0.30 with W14's exact capture sizes (pos1 0.4697 < 0.5 hard), golden logits DRIFT on 2/5 prompts, and the forward replay is UNCHANGED (49.3–50.2 ms c1). Recorded + fallback to mode=none product posture; NO MERGE (gate law). Compile lane is now measured dead TWICE on this stack/model.

Posture: product flags verbatim except compilation-config transplanted from
the W14 serve line: `{"cudagraph_mode":"FULL_AND_PIECEWISE",
"cudagraph_capture_sizes":[4,8,16,32,64],"max_cudagraph_capture_size":64}`
(no "mode" key → CompilationMode.VLLM_COMPILE, confirmed in boot log). p10
fastpath + r25 phase timer via /root/p25cap; defect-a speculator coercion
active (2 coerced); interlock live. S-boot healthy 17:47:18 (6.4 min — 5
FULL decode sizes + piecewise phases).

## Gate results

| gate | result | evidence |
|---|---|---|
| (b) survival, flip payload first | **PASS** | 832-tok → HTTP 200 in 8.09 s (interlock window); follow-up 200 — captured pieces never loose, defect-b structurally absent on FULL_AND_PIECEWISE (2nd confirmation, r24 was 1st) |
| (a) acceptance undegraded (p31) | **FAIL** | pos1 **0.4697** < 0.5 hard; acceptance 0.353, mean len 2.41, 330 drafts — "wrong-logits family". Product band 0.8768–0.9082 |
| (c) golden logits zero drift | **FAIL** | **2/5 prompts drifted** (len=64, len=2048; sha mismatch); len 256/1024/4096 match. NEW instrument (legs_pull/scripts/p32_golden.py): seeded token-id prompts → greedy temp-0 → sha256 of generated text; ref captured timers-off on mode=none in the same leg (phase R) |
| (d) forward ≤ 45 ms c1 (split) | **FAIL** | forward p50 **49.3–50.2 ms** c1 (n=576/672 windows), 54.9–55.0 c8 — IDENTICAL to product posture (49.0–54.5); draft 14.3–15.1, sample 2.4–2.8, fgap=bgap=0 |
| step wall (diagnostic) | worse | stepprobe c1 66.0 / c8 87.6 (timer-on; product 65.3/80.0) — compile does not shrink the step, slightly harms c8 |
| solo (diagnostic) | collapsed | 40.7 tok/s (acceptance death; product 59.1) |
| controls | green | p10 0.48–0.49 ms intact; onset 0; TP0 ≡ TP1 splits |

## Interpretation

- The compile×spec law is posture-independent on this stack+model: r24
  (capture 1..320) pos1 0.382 → S1 (W14 sizes [4,8,16,32,64]) pos1 0.4697 —
  both hard-FAIL with the wrong-logits signature. W14's clean compile+MTP
  behavior is a property of the radiance fork/flash-next model line, NOT
  reproducible by transplanting the compilation-config onto our dense model.
- Golden drift (2/5) is independent proof the compiled path computes
  different logits for some prefill shapes — not an acceptance-stat artifact.
- Forward content is compile-invariant (49–55 ms both postures) — consistent
  with r24's compile-independence law; the wall is NOT fusion-fixable via
  the inductor piecewise lane on this stack.

## Disposition

- **Compile lane: CLOSED for MTP on qwen3.8-27b-fp8 (0.30)** — two
  independent falsifications + golden drift + no step benefit. Product
  posture stays `mode=none` + FULL_DECODE_ONLY + p10.
- Branch feat/arc-p32-compile-fused: evidence-only, NOT merged (gate law).
- P2 §10 S1 row: gate verdict recorded as FAILED-with-fallback — the
  forward ≤45 ms lever moves to the remaining content lanes (p33 draft
  chain; AR-attribution).
- Golden-logits instrument promoted to standing gate tool (correctness gate
  (c) for every future posture leg; ref = results golden_ref.json).
- For the upstream-issue text (owner-gated): 0.30 XPU dense+MTP ×
  FULL_AND_PIECEWISE = acceptance collapse + greedy drift; W14 sizes do not
  rescue it.

## Artifacts (legs_pull/)

survival.txt; spec_gate.txt (pos1 0.4697 FAIL); golden_{ref,run}.json +
golden_{ref,run,compare}.txt (2/5 drift); run7n_1.txt (solo 40.7);
stepprobe.txt (66.0/87.6); split.txt (forward 49.3 unchanged); p10_stats.txt;
onset_count.txt; serve_s1_final.log; scripts/{p32_s1_serve,p32_leg,
p32_teardown}.sh + p32_golden.py. Rig restored to W14 health 200 (17:57).
