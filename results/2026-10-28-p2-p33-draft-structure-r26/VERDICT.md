# VERDICT — p33 draft-structure probe leg (r26): owner premise FALSIFIED by measurement — the MTP draft is TWO back-to-back gapless graph replays and the cost is replay CONTENT, not per-replay overhead. Chained-graph lever DEAD by evidence; redirected to draft-content attribution (r27 profile). Golden instrument: cross-boot nondeterminism floor discovered on mode=none (1/5 drift vs banked ref) — protocol amendment recorded below.

Posture: product stack VERBATIM (mode=none + FULL_DECODE_ONLY spec capture [1..320],
MTP k4, VLLM_XPU_ENABLE_XPU_GRAPH=1, ARC_TRACE_FLIP=1) + meta-path probes (p10 fastpath,
r25 phase timer, NEW p33 structure probe — timers ON, distortion law applies, this leg
is a structure/probe leg not a gates leg). Boot 18:13→healthy 18:16; teardown + W14
restore verified health 200 at 18:28:38.

## The decisive structure law (steady state, c1 solo, ~40 flush windows, both TP ranks identical)

```
propose p50 15.18 ms  =  prefill FULL replay 5.18   (draft step 0: fc + full-attn layer + lm_head)
                      +  fused decode FULL replay 9.99  (draft steps 1–3 inside ONE fused graph, ≈3.33/step)
                      +  GPU gap 0.01              (exact reconciliation 5.18+9.99+0.01 = 15.18)
CPU: metadata build 0.16 ms; fused CPU wall 3.77 ms (fully overlapped with the replays)
Eager in steady c1: prefill.eager=7 gen.eager=0 (x0.98/x1.00 replay hit; eager = warmup + graph-miss sizes)
```

- The r25 model "4 tiny replays × ~3.5 ms fixed per-replay overhead" is **FALSIFIED**:
  there are exactly **2** replays, back-to-back, with 0.01 ms of GPU gap between them.
  `use_fused_multi_step_decode=True` already fused draft steps 1–3 into ONE graph
  (in-graph `compute_slot_mappings` + `update_draft_decode_metadata` per step); only
  step 0 (_prefill, varlen FA over target attn metadata) runs as a separate replay.
- A draft one-token step costs ≈3.3 ms ≈ **4.3× a target layer** (0.77 ms/layer unfused).
  The premium is draft-specific CONTENT, prime suspects (r27 to attribute): per-step
  lm_head GEMM (vocab 248320, TP-split 124160), fc ColumnParallelLinear all-gather
  (gather_output=True, 10240→5120), o_proj + MLP down-proj ARs → ~3–4 CCL collectives
  per draft step × 4 steps.

## Gates / cells (this is a probe leg; product stack unchanged)

| gate | result |
|---|---|
| survival (flip payload 832-tok as FIRST traffic) | **PASS** — 200 / 6.91 s, onset 0 |
| acceptance (a) | **PASS** — pos1 0.9122 ≥ 0.5 hard (acc 0.7329, len 3.93; 0.85 target miss recorded) |
| golden (c) vs banked p32 ref | **1/5 drift (len 4096)** — see instrument finding below; NOT a regression of this leg |
| TPOT diagnostic (timers on) | c1 77.6 / c8 92.8 / c1-again 77.5 ms — consistent with r25 law (+3.8% distortion) |
| run7n | agg 196.7 / 209.8; solo 56.1 / 54.1 tok/s (timers on) |
| r25 split (concurrent) | tail 15.0–15.7 (= propose here), forward 49.4–55.0, sample 2.4–2.8, fgap=bgap=0 — r25 law re-confirmed |
| p10 fastpath | p50 0.49 ms build, fallbacks zero-spec/mixed only — intact under probe load |

## GOLDEN INSTRUMENT FINDING (amends p32 interpretation; protocol amendment proposed)

p32 ref was captured timers-off on mode=none; p32's COMPILE run drifted 2/5 (len 64, 2048)
and **matched len 4096**. Today mode=none itself drifted **1/5 (len 4096)** vs the same
ref, running prompts strictly sequentially (one in flight → deterministic batch shapes).
Different prompts drift in different legs ⇒ **cross-boot kernel-level nondeterminism floor
on this stack** (reduction/AR scheduling order flipping near-tie argmax). Consequences:

1. p32's "golden drift = independent proof of compile-path wrong logits" is **downgraded**:
   compile drift (2/5) is not separable from the floor (1/5) at n=5. The decisive,
   reproducible compile evidence remains the acceptance collapse (0.4697 vs 0.88–0.91,
   three independent falsifications) + unchanged forward replay.
2. Standing-gate protocol amendment (recorded, applies to future legs): a sha-exact
   cross-LEG golden comparison is only meaningful same-boot; cross-leg comparisons are
   informational ≤1/5 with hard-fail at ≥2/5 replicated. Same-leg ref+run remains
   sha-exact (p32 phase-R protocol).

## p33 chained-graph lever: CLOSED by measurement (never built — evidence law)

The owner's framing (chain 4 draft replays into 1, save ~3×3.5 ms) has no object: only
ONE boundary exists and its gap is 0.01 ms. Chaining prefill+fused into one graph would
save ≤0.01 ms and add capture-complexity risk. **Lane closed.** The ≤7 ms draft target,
if reachable, lives in replay CONTENT (collectives/lm_head), attributed by r27.

## Draft structure reference (logged at init, both ranks)

- Draft model `Qwen3_5MTP`: ONE dense full-attention layer (mtp.layers.0, self_attn +
  q/k_norm, no linear_attn), `fc` ColumnParallelLinear(10240→5120, gather_output=True),
  norm trio GemmaRMSNorm, lm_head ParallelLMHead(vocab 248320), LogitsProcessor;
  mixes in LocalArgmaxMixin (get_top_tokens implemented).
- Speculator: use_fused_multi_step_decode=True, advance_draft_positions=True, k=4,
  share_mtp_topk_indices=False, draft_groups=[(FlashAttentionBackend, 1, supports=True)]
  (FA builder supports in-graph draft decode metadata update, dcp_world_size==1).
- Captured graphs: prefill manager [5,10,…,320] (multiples of k+1), decode manager
  [1..16,20,…,64] (27 FULL graphs, ≤64 at 1 tok/req).

## Next lever (r27 profile leg — evidence before implementation)

Boot product posture timers-OFF + p10 only with `--profiler-config` (r19 pattern), warm +
c1 steady-decode window (LAST on boot per aggregation-hang law): (1) attribute the
15.2 ms draft window (collectives/ccl vs lm_head GEMM vs FA vs other); (2) forward-window
AR share (owner AR-attribution ask; contingency proposal + scope check if >10 ms).
Candidate one-flag upstream lever if draft comm-dominated:
`speculative_config.use_local_argmax_reduction=true` (Qwen3_5MTP implements
LocalArgmaxMixin.get_top_tokens; validation path exists in DraftModelSpeculator).

## Evidence

`legs_pull/`: survival.txt, spec_gate.{json,txt}, golden_{run.json,run.txt,compare.txt},
run7n_1.txt, stepprobe.txt, p33_structure.txt (full probe stream), split.txt (r25),
p10_stats.txt, onset_count.txt, serve_r26_final.log.
`scripts/`: p33_probe.py, p33_sitecustomize.py, p33_serve.sh, p33_boot.sh, p33_bench.sh,
p33_teardown.sh. Rig restored to W14 health 200 (18:28:38).
