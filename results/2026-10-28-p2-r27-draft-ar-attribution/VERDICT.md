# VERDICT — r27 draft-content + forward-AR attribution leg: BOTH owner asks answered in one pass. Draft tail is 56% lm_head GEMM weight-read (physically irreducible in scope) → p33 ≤7 ms target UNREACHABLE as stated; forward TP allreduce ≈ 0–8 ms (≤ ~8%, central ~2–5 ms) → NO >10 ms contingency triggered. Method law: XPU graph replays are INVISIBLE to the torch profiler — attribution done via eager-proxy posture with per-kernel durations; every number reconciles with the banked wall laws.

Two phases, one rig session (boots 18:45 graphed → 18:52 eager; profile LAST on each boot per aggregation-hang law; teardown + W14 restore after).

## Phase A — graphed product posture (visibility check + outside-graph set)

Product posture VERBATIM (mode=none + FULL_DECODE_ONLY [1..320], MTP k4, p10 fastpath only,
timers off) + `--profiler-config` (r19 pattern). c1 1k-ctx seeded load, 8 s window,
TPOT 64.0 ms under profiler (≈ product 63 ms law).

**Replay-invisibility law (re-proven with the exact visible set):** 378 kernel
events / 6 steps / rank — per step EXACTLY: 1 `gemm_kernel` (2.2 µs), 1
`oneccl_allreduce_pcie`, 1 `allgatherv_large_su_ring_chunking`,
`_rejection/_resample/_compute_local_logits_stats/_compute_slot_mappings`,
index/copy bookkeeping. ZERO attention, ZERO gdn, ZERO lm_head — everything
inside replays is invisible (220 KB/rank trace vs 33 MB eager). Draft-step-0
varlen FA is inside the prefill replay → invisible too (no anchors exist).
Consequence: kernel attribution of replayed content REQUIRES the eager proxy.

## Phase B — eager posture (cudagraph_mode NONE, attribution)

Same line otherwise; 15.4 s window, 51 steady steps, TP0 == TP1 to <2%.
Segmentation: lm_head-class GEMMs isolated by duration (255 = 51×5: [verify,
draft×4]); step segment = (V_k, V_{k+1}]. Durations are REAL per-kernel times
(GEMMs are skew-immune); collective durations in eager are SKEW-INFLATED
(launch-bound ranks → oneCCL kernels absorb peer-skew as spin time) — used for
CALL COUNTS and transfer-size reasoning only, never mapped 1:1 to graphed wall.

### The reconciliation (all three banks cross-validate)

| measurement | value | reconciles with |
|---|---|---|
| gemm count/step | 322 = 64 layers×5 + 5 lm_heads | model structure ✓ |
| layer-GEMM avg × 320 | ~0.15 ms → **48 ms/step** | forward wall 49–55 ms (r25) = BW floor ✓ |
| lm_head GEMM | **2.1–2.7 ms** each (635 MB fp8/rank → ~302 GB/s = tile BW ceiling) | pure weight read, fp8-confirmed ✓ |
| forward collectives | **132 calls/step** (128 layer ARs + logits all-gather + prepare) | 2 AR/layer × 64 ✓ |
| draft collectives | **17.6 calls/propose** (~4.4/draft-step: fc all-gather + o_proj AR + down AR + logits all-gather) | draft-collective hypothesis ✓ |
| draft-cluster gemms | 21.6/propose = 4 lm_heads (8.5–10.6 ms) + ~17.6 layer GEMMs (~2.5 ms) | r26 propose 15.18 ms ✓ |

### Ask 1 — draft content (graphed reconcile against r26 propose 15.18 ms)

```
lm_head GEMMs          ~8.5 ms   56%   4 × 2.1 ms weight reads — IRREDUCIBLE (fp8 already;
                                          smaller lm_head = quant/kernel change = scope law)
draft layer/fc GEMMs   ~2.5 ms   16%   4 one-token dense full-attn layer passes + fc
FA + norms + quant/ew  ~0.6 ms    4%
draft collectives      ~3.6 ms   24%   18 calls in lock-step (~0.2 ms each); eager measures
                                          7.7–10.0 ms but skew-inflated
```
**p33 ≤7 ms is UNREACHABLE: the lm_head weight-read floor alone is ~8.5 ms.**
Draft floor with ZERO collectives ≈ 11.5 ms. In-scope best lever:
`speculative_config use_local_argmax_reduction=true` (upstream flag;
Qwen3_5MTP implements LocalArgmaxMixin.get_top_tokens) removes the per-step
vocab all-gathers (4×~1.2 MB draft + 2.5 MB verify transfer) → expected
~1–2 ms off propose, ~0.3–0.5 ms off forward. Worth an A/B (r28) as an
incremental; it cannot approach ≤7 ms. Recommend target revision to ~13 ms
or closing the lane with this attribution.

### Ask 2 — forward AR share (owner contingency bar: >10 ms)

Forward wall 49–55 ms ≈ GEMM+attn content alone (48 layer-GEMM + 2.1 verify
lm_head + FA/GDN/quant/norm ~4 ms ≈ 54 ms) → lock-step collective residual
**0–8 ms, central ~2–5 ms (≤ ~8%)** across 132 calls. **NOT >10 ms → no
contingency proposal triggered.** Eager collective duration (101–105 ms/step)
is published for the record WITH the skew-inflation caveat — it is an upper
bound on nothing (launch-bound posture), the count (132) and the
content-reconciliation bound are the defensible numbers.

### Strategic law (follows, affects §10 P2 reading)

The step is at the **bandwidth wall**: ~63 ms c1 ≈ 27 GB/rank of weight
reads at ~300 GB/s + spec structure. TPOT moves only via acceptance length
(P4) or step structure, not fusion/overlap — independently re-confirms r24
(compile-independent) and r25 (fgap=bgap=0, overlap dead). p50's ≤15 ms
E2E-residue target is now expected moot (r25: tailres ~0, fgap=bgap=0,
CPU build 0.49 ms) — treat p50 as evidence-only unless close-out says
otherwise.

## Evidence

`legs_pull/`: analyze_graphed.txt + analyze_graphed_names.txt (phase A),
analyze_eager.txt (phase B full output), prof_graphed/ (both rank traces,
small), prof_eager_meta/, load_eager.log, serve_eager_final.log. Eager rank
traces (33 MB gz each) stay on the rig at
`/root/build/legs/arc-p2-r27/prof_eager/` (path recorded; analyzer rerunnable
via scripts/r27_analyze.py). `scripts/`: r27_serve.sh (graphed),
r27_serve_eager.sh, r27_sitecustomize.py (p10-only cap), r27_boot.sh
(usage: `<graphed|eager>`), r27_prof.sh (usage: `<phase> <window_s>`),
r27_analyze.py (dual-mode), r27_teardown.sh. Rig restored to W14
(teardown leg bktcsgd8f).

Windows measured 18:48 (A) / 18:54 (B) on 2026-10-28.
