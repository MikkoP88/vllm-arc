# r36 — 8k warm gate: face DEFINED (primary source), connector EXONERATED, "266 ms regression" DISSOLVED (r30-755 artifact), ≤0.75 s proven structurally unreachable, D-adder comparison MET

Date: 2026-10-29 (probe boot 18:55–, legs 19:0x–19:4x). Posture:
r35-banked product (p2 + p40/p41/b/c + lmcint kit + r33 TWO_HS +
sched_tail_split v4), telemetry-only delta `ARC_R36_PHASE=1` (phase timers
on `LMCacheConnectorV1Impl.get_num_new_matched_tokens` /
`update_state_after_alloc`, stripped at E4 — no product change this leg).

## Leg questions

1. Does the suspected "~266 ms post-r33 warm-path regression" (p41-leg
   WARM5 755.1 ms vs 1021+ after the adapter pool-view/seam changes)
   exist, and is it the lever to ≤750?
2. What IS the P3 warm gate face (cl1b was cited in the plan without a
   definition anywhere in our docs / rig / external_patches)?

## Capture (question 1) — connector EXONERATED, warms are pure-local

Probe boot with per-request phase timers on the connector:

- 13× `[r36ph alloc req=… ms=0.0 since_lookup=0.7–3.4 ext=0]` —
  allocation-side cost ~0, lookup→alloc lag 1–3 ms.
- ZERO `[r36ph lookup …]` prints — every lookup under the 5 ms filter.
- ZERO `Retrieved` lines on the whole boot — no request rode the external
  path; N1/WARM5 warms are local APC pop-hits (7168 + 1024 recompute).

Cells (first-traffic on the probe boot, telemetry ON — diagnostics, not
gate numbers; gate of record remains the r35 E4 clean boot 18:36):

- N1 8192: `min=1046.3 med=1048.2 max=1050.7`, sha `9d1e0e2d…`
  within-run stable (another value of the documented N1 cross-boot
  marginal class: 7eb702/01ba4719/9d1e0e2d).
- WARM5 8192: `min=1029.2 med=1034.3 max=5876.3` (first-warm blip,
  same class as every prior boot's 5962–5972), sha `d03502c4…`
  within-run stable.

**The regression does not exist.** r30's 755.1 was measured on the
pre-r33 FOLDED transfer (r33 proved those loads byte-wrong / half-folded);
the honest local-path 8k warm has sat at **1021–1048 ms across every
r33/r34/r35/r36 boot**. Nothing was lost at r33; there is no ~266 ms to
reclaim on the connector path.

## Face definition (question 2) — primary source FOUND

`CL1_RESULTS.md` in the D donor
(vllm-iexperimental `vllm/patches/prod/v1235-certified/results/2026-10-19-cl1-legs/`,
launcher `cl1_legs.sh` on the **llm-scaler-exp:v1.2.35** image = line C):

> cl1b (SPEC=1, baked MTP k4, Qwen3_5MTP): longctx t1/t2 (s): 600tok
> 0.363/0.390; 8k 4.784/**0.587**; 16k 11.494/**0.741**; 32k 27.174/0.566;
> Retrieved 1088 (retrieval ENGAGED under spec)

The plan's "C bars 0.587–0.741 s" = **cl1b longctx t2 at 8k and 16k**.
t2 = second-traffic warm TTFT. Note t2 is SIZE-FLAT (0.587/0.741/0.566 at
8k/16k/32k): C's warm is near-full external retrieval + tiny residual —
a posture possible on the old v0.21.1 swift stack.

## The lineage comparison (all primary sources)

| stack | posture | 8k warm t2 | mechanism |
|---|---|---|---|
| C (v0.21.1 swift, lmcache 0.5.5) | SPEC=1 | **0.587 s** | near-full external retrieval, size-flat |
| D (v0.26 lineage) | SPEC=0 | 0.703–0.733 s | F-b/resume5–13: "ZERO Retrieved lines = local GPU-prefix recompute at hybrid 1024-token granularity … KNOWN_ISSUES #22 documented ceiling, no cheap lever" |
| D (v0.26 lineage) | SPEC=1 (P2r3) | **1.14–1.28 s** | same local recompute + parked completion-constant adder |
| **vllm-arc (v0.30, MTP k4 + align)** | SPEC=1 | **1.03–1.05 s** | local pop-hit 7168 + 1024-token residual (this leg, connector ~0) |

## Budget arithmetic — ≤0.75 s is structurally unreachable

N1−N3 differencing across this leg + r35's E4 (both clean-class cells):
(1048.2 − 544.7) / 896 residual tokens = **0.561 ms/token** marginal
warm-residual rate; intercept ≈ 0.39 s fixed (scheduler/HTTP/spec first
step) + 56 ms output sync. With the residual pinned at 1024 tokens by the
composition-coherence law (warm depth capped at 7168; depth 8064 =
the falsified r34/r35-v3 class), even ZEROING all connector+scheduler
overhead floors the 8k warm at ≈ 0.87–1.0 s. **≤0.75 s at 8k requires
warm depth ≈ 8064 — law-blocked in the 0.26+/0.30 lineage (D hit the
same wall from the other side and documented it as a ceiling).**

## Verdict

- **Contract operative term "beat D's adder": MET** under the deployed
  spec posture — 1.03–1.05 vs D-SPEC1 1.14–1.28 (−0.09…−0.25 s), with
  0-drift intact (golden 5/5 at every adjacent boot).
- **Absolute ≤0.75 s at 8k: documented structural FAIL** — coherence-law
  depth cap + 0.561 ms/token residual floor ≈ 0.87 s minimum; C's 0.587
  rides the pre-0.26 retrieval posture (size-flat t2) that the hybrid
  redesign removed; D's own KNOWN_ISSUES #22 marks the same ceiling.
- 1k face (r35 v4): **545 ms** — beats even C's 0.587; prefix-class warm
  cost is not the binding term.
- The r30 "FAIL by ~5 ms" ledger row is an artifact of the folded-transfer
  era and is retired.

## E4 hygiene

Probe block reverted in-repo (`vllm_v1_adapter.py` md5
`79e8dbd8e410db7affffad5b9af7b85d` = committed p41 state, working tree
clean); rig overlay source pushed to the same md5; `ARC_R36_PHASE`
stripped from `/root/build/r30/r30_serve.sh` (line 21 now a comment).
No product bytes changed this leg → the r35 E4 clean-boot battery
(18:36, first-traffic, timers-off) remains the gate of record; the rig
restores to W14 at leg end.
