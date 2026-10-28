# VERDICT — r28 use_local_argmax_reduction A/B: the flag is CORRECTNESS-CLEAN but PERF-DEAD — propose 15.07 vs 15.18 ms (−0.11 ms, noise-level), TPOT/golden/acceptance all unchanged. p33 lane now CLOSED on complete evidence: every in-scope lever measured (chained-graph: no object; local-argmax: ~0.1 ms; compile: dead ×3). Flag NOT adopted; product posture unchanged (flag stays off); NO MERGE.

Posture: product stack VERBATIM (r26 line: mode=none + FULL_DECODE_ONLY [1..320], MTP k4,
p10 fastpath + r25 timer + p33 probe via /root/r28cap, timers on — probe leg law) with the
ONE change `speculative_config.use_local_argmax_reduction=true` (upstream flag; engaged —
serve-config echo `'use_local_argmax_reduction': True`; Qwen3_5MTP implements
LocalArgmaxMixin.get_top_tokens, validation path passed at init). Boot 19:05→healthy;
battery full.

## A/B result (vs r26, same battery, same rig)

| cell | r26 (flag off) | r28 (flag on) | verdict |
|---|---|---|---|
| propose p50 (steady c1) | 15.18 ms (5.18+9.99+0.01) | **15.06–15.07 ms** (5.13+9.91+0.01) | −0.11 ms (−0.7%) — noise-level; the vocab all-gathers (4×~1.2 MB draft + 2.5 MB verify) cost ~0.1 ms total in graph lock-step, not the r27 upper estimate ~1–2 ms |
| TPOT stepprobe c1 / c8 | 77.6 / 92.8 ms | 78.0 / 92.8 ms | unchanged |
| run7n agg / solo | 196.7–209.8 / 56.1–54.1 | 197.3–197.9 / 56.9–53.5 | unchanged |
| acceptance pos1 (hard 0.5) | 0.9122 (acc 0.7329, len 3.93) | 0.8798 (acc 0.7175, len 3.87) | normal cross-leg range (r23 0.8768); PASS |
| survival (flip payload first traffic) | 200 / 6.91 s | 200 / 6.91 s | PASS, regression cell held |
| golden (vs p32 banked ref) | 1/5 drift (len 4096) | **1/5 drift — same prompt, run shas byte-identical to r26 on ALL 5** | (i) argmax-EQUIVALENCE of the flag proven (local top-k reduction selects identical tokens); (ii) r26 ≡ r28 cross-boot EXACT — the drift is ref-side (p32 capture boot), re-confirming the r26 nondeterminism-floor finding |

## Interpretation + lane closure

r27's draft attribution stands, with the collective share now bounded TIGHTER by
measurement: draft collectives ≈ 3.5 ms total of which the removable part (vocab
all-gathers) is ~0.1 ms — the rest is the per-step ARs (o_proj/down/fc), which are
mathematically required under TP2. Combined with the lm_head floor (4×2.1 ms = 8.5 ms,
56%) and the 4 one-token layer passes (~3 ms), the draft tail is at its structural
floor ≈ 15 ms on this stack. **p33 ≤7 ms target: UNREACHABLE — closed on complete
evidence.** Remaining draft-side theoretical levers are all out of scope (lm_head
quantization change, draft-model surgery, vocab truncation).

## Evidence

`legs_pull/`: survival.txt, spec_gate.{json,txt}, golden_{run.json,run.txt,compare.txt},
run7n_1.txt, stepprobe.txt, r28_structure_recovered.txt + r28_split_recovered.txt
(cell-6 file was empty — recovered from serve_r28_final.log; includes the init
structure lines and the flag-echo line), serve_r28_final.log. `scripts/`: r28_serve.sh
(flag posture), r28_sitecustomize.py + r28_probe.py (byte-identical p33 probe),
r28_boot.sh, r28_bench.sh, r28_teardown.sh. Rig restored to W14 (teardown leg brrvks7fl).

Next per owner mission order: p50 evidence-only (expected moot — r25/r27 laws), p14
prefill-parity diagnosis (TTFT ladder 5794/11997/25764/58546 vs bars
3944/8083/17447/41565; prefill is EAGER on our posture → profiler-visible), then the
P2 close-out package (rows DONE / unreachable-with-proof).
