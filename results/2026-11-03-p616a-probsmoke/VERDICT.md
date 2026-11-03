# p616a — probabilistic draft sampling: BOOT+SMOKE (2026-11-03)

Owner directive 2026-11-03: implement the upstream-direction fix inline,
start-to-end; nothing may regress the certified configurations. This dir
banks the P6.16a smoke leg — **Fix A: upstream-config-native
`draft_sample_method="probabilistic"`** (pinned vllm 0.30.0 wheel,
`vllm/config/speculative.py:588`) served from the **cert6 image
UNCHANGED** via serve-line sed only (no code, no wheel change —
one-variable twice over the banked p615c posture: dtype k8v4 + spec
method). Direct A/B against the banked
`../2026-11-03-p615c-t10collapse/` trace (same image, same dtype, same
prompt, same instrument `collapse_trace.py` ×10).

## Result in one line

**The collapse mode is ELIMINATED** (pos0 draft-acceptance death: gone
on all 10 requests — 88–95 % vs the banked 6 %), **t0 certified face is
unchanged** (solo 88.45/88.37 vs banked band 88.3–88.6; agg#2 302.67 vs
301.97), **step cadence unchanged** (median 41–42 ms on every request =
banked 41.7 ms — the gumbel/triton draft-sampling path runs inside the
FULL draft graphs at zero measurable cost), at the price of **lower
chain acceptance at draft depth 2 in high-entropy regions** (tok/round
2.19–3.77, median ~2.5, vs greedy-healthy 3.6–3.85) — so sampled legs
run 51–85 tok/s uniformly instead of the greedy face's 26–87 with a
catastrophic tail. Aggregate EV is ~neutral (−3 % on this roll);
worst-leg wall −49 % (3.89 s vs 7.56 s); wall spread 1.17× vs 3.2×.

## Gates (pre-registered in the P6.16 plan block)

| Gate | Result | Evidence |
|---|---|---|
| boots, 0 ERROR/Traceback | **PASS** | health after 310 s; 0/0 |
| engine parsed the flag | **PASS** | `speculative_config={'method':'mtp','num_speculative_tokens':3,'draft_sample_method':'probabilistic'}` (spec_method.txt, api_utils non-default args) |
| t0 solo in Run-7n band (88.3–88.6) | **PASS** | 88.45 / 88.37; agg4x256#2 = 302.67 (banked k8v4 301.97); #1 = 245.23 cold-first (usual first-agg warmup) |
| no t1.0 leg < 55 tok/s | **FAIL as written — 51.4 / 54.1** | but zero collapse-mode legs (see below): the floor was calibrated on greedy-leg speeds; the mechanism falsifies that calibration |

## The 10-request trace (t1.0-xhigh, streaming solo, byte-exact banked body)

`tok/round = (drafts+accepted)/drafts`; per-pos conditional; tok/s =
completion_tokens/wall. req1/req5 ended by `stop` (sampled a legitimate
early end — see "distributional correctness" below; the greedy bank ran
10/10 to length).

| req | wall (s) | tok/s | drafts | accepted | tok/round | pos0 | pos1 | pos2 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.56 | — (stop @31) | 10 | 22 | 3.20 | 90 % | 78 % | 86 % |
| 2 | 2.35 | 85.1 | 53 | 147 | 3.77 | 94 % | 98 % | 98 % |
| 3 | 3.70 | 54.1 | 84 | 117 | 2.39 | 93 % | 46 % | 8 % |
| 4 | 3.29 | 60.8 | 74 | 127 | 2.72 | 93 % | 71 % | 18 % |
| 5 | 0.86 | — (stop @27) | 16 | 10 | 1.63 | 44 %* | 29 %* | 50 %* |
| 6 | 2.82 | 70.9 | 63 | 136 | 3.16 | 95 % | 88 % | 43 % |
| 7 | 2.65 | 75.5 | 60 | 141 | 3.35 | 88 % | 89 % | 87 % |
| 8 | 3.35 | 59.7 | 78 | 121 | 2.55 | 92 % | 54 % | 26 % |
| 9 | 3.64 | 54.9 | 85 | 115 | 2.35 | 93 % | 42 % | 9 % |
| 10 | 3.89 | 51.4 | 91 | 108 | 2.19 | 90 % | 27 % | 18 % |

\* req5 n=16 rounds — small-n, not diagnostic.

A/B vs the banked p615c trace (greedy drafts, same everything else):

| Face | p615c greedy drafts | p616a probabilistic |
|---|---|---|
| pos0 acceptance (every full leg) | 83–96 % healthy legs, **6 % collapsed** | **88–95 %, all legs** |
| collapse-mode legs (pos0 dead) | **3/10** (1 total @26 t/s, 2 depth-dead) | **0/10** |
| tok/round | 1.11 (total collapse) – 3.85 | 2.19 – 3.77 (median ~2.5) |
| full-leg tok/s | 26.5 – 87.3 (median ~84, tail 26.5/46.7/66.2) | 51.4 – 85.1 (median 60) |
| worst wall / median wall | 7.56 / 2.39 = **3.2×** | 3.89 / 3.32 = **1.17×** |
| tokens delivered / total wall | 2000 / 31.74 s = 63.0 tok/s | 1658 / 27.11 s = 61.2 tok/s (**−3 %**) |
| median chunk dt | 41 ms | 41–42 ms |

## Mechanism read (why it works, what it costs)

With greedy drafts, the verify step (rejection sampling on target
logits) accepts a draft w.p. ≈ p_target(argmax_draft) — near zero in
flat regions of a sampled xhigh trajectory under 4-bit-V KV noise → the
banked born-collapsed legs. With **probabilistic drafts** (draft token
sampled from the draft distribution at request temperature via the
gumbel kernels, then the probability-ratio test), acceptance ≈
Σ min(p_draft, p_target) at each position — both distributions computed
on the SAME quantized KV, so the ratio stays bounded away from zero
even where the argmax match dies. Result: pos0 death gone.

The cost is mathematical, not accidental: at depth 2 the draft is a
*sampled* token, so in flat regions the chain acceptance is the
product of per-position distribution overlaps — visibly lower than the
argmax-chain overlap the greedy bank enjoyed on its healthy legs
(pos2 88–100 % there). That is the trade: uniform ~2.2–2.6 tok/round
everywhere instead of 3.7 on lucky requests and 1.1 on unlucky ones.

**Distributional correctness (the deeper point):** greedy drafts +
sampled verify is a *biased* sampler at t≥1.0 (output over-represents
argmax-chain trajectories — that's also why the banked greedy t1.0
faces were ragged). Probabilistic drafts + the ratio test provably
sample the target distribution (lossless verify;
`rejection_sampler.py:168-207`). The 2 early-`stop` requests here are
legitimate samples of "the model ends early at t1.0", which the biased
greedy face suppressed. Comparing tok/s medians between the two
postures compares a biased-fast face against the correct one.

## Certified-configuration safety (the hard law)

- t0 face: probabilistic degenerates to argmax at temp 0 — solo and
  agg **dead in the banked band** on this leg (numbers above).
- The flag lives ONLY in the serve line; the wheel and image are
  byte-identical cert6; default (flag unset) = greedy = the certified
  posture unchanged.
- KV pool, LMCache, graphs: unchanged (KV 519,972 tok k8v4 posture;
  FULL_DECODE_ONLY; no new WARN class in the serve log).

## Disposition

Smoke objectives met (boots ✓, t0 safe ✓, mechanism engages ✓, cadence
✓, 0 errors ✓). The pre-registered absolute floor (≥55 tok/s) is
falsified as a calibration: it assumed greedy-healthy speeds carry
over, but the mechanism trades depth-acceptance for tail-elimination
— recorded in the master plan as a **gate amendment BEFORE the 6.16b
matrix runs**: the merge question is collapse-elimination + uniformity
+ t0/TTFT safety + EV, not absolute t/s vs a biased greedy face.
Proceed to P6.16b (full 5-dtype matrix on the probabilistic posture)
with amended gates.

## Provenance

Leg run solo from stable W14-up (health-gated first), one-variable ×2
posture (dtype + spec method), substitution counts 1/0 and 1/1
verified in-output, W14 restored by EXIT trap (`w14_boot.sh fullres`;
health re-verified post-leg). Harness `p616a_boot_smoke.sh` (this
dir); instrument `collapse_trace.py` (deployed
`/root/build/p65/p615/`, banked copy in the p615c dir); t0 battery
`/root/build/r70/bench_run7n_t5.py` (Run-7n exact). Serve log + json
redacted byte-exact (serve_full.log 1 rig-addr ×1 = +6;
preexisting_arc-test.json 1 hostname ×1 = +5; total +11; raw-ref grep
0 after) — `../REDACTION-2026-11-03-p616a.md`.
