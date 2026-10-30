# r39 — PARD lane CLOSED (architecture-gated, source-evidenced); D-gap RE-ATTRIBUTED to forward-pass step time (champion telemetry: D emitted ≈3.46 < ours 4.03 — acceptance EXONERATED vs the bars; D step ≈43 ms c1 vs ours 63–70)

Date: 2026-10-29 (local code/doc leg — zero rig boots, zero product bytes;
wheel sources + D champion raws pulled read-only).
Inputs: 0.30 wheel spec_decode + config (image vllm-arc:v0.30.0-b70-p41),
D 0.26 wheel (image vllm-iexperimental:v1.2.0, vllm 0.26.1.dev0+g568afb3a1),
D champion leg raws `/root/build/legs/p2r6/` (P2r6-CKPT = the contract-bar
cell, plan App. F line 821), D bank results (P2r3 VERDICT + status table).

## Finding 1 — PARD is CLOSED as a lane (not port-able; trained-model gate)

`parallel_drafting` (PARD) in 0.30 is a **model-architecture feature**, not a
scheduling mode:

- `config/speculative.py:466-470` (field docstring): "requires the
  speculative model **be trained** to support parallel drafting. Only
  compatible with EAGLE and draft model methods." Auto-ON only for
  dflash/dspark (`:1462`).
- `llm_base_proposer.py:359-383` `_init_parallel_drafting_params`: hard
  ValueError at boot unless the draft config carries a trained mask token
  (`dflash_config.mask_token_id` / `mask_token_id` / `dspark_noise_token_id`
  / `pard_token` / `ptd_token_id`).
- Mechanism: the K draft queries are packed as **mask-token queries into ONE
  forward** (`_copy_and_expand_eagle_inputs` expand `:885-933`; first-pass
  early-exit + `view(-1, num_speculative_tokens)` `:632-638`; graph capture
  does ONE forward `:1637`). Each mask query is trained to predict its own
  position. step3p5's variant = a multi-token head (`spec_step_idx` logits
  per pass, `Step3p5MTPProposer`) — also a trained architecture.
- Our qwen3.8-27b-fp8 MTP head is a **sequential single-token predictor**
  (one next-token per forward; step i's input = argmax of step i−1) with no
  mask token in the checkpoint. No wiring port changes what trained weights
  compute; training a PARD head is out of scope (scope law §3, target model
  only). **r38's "PARD port REQUIRED" scoping is RETIRED.**

## Finding 2 — the D champion cell ran at emitted ≈ 3.46, BELOW ours (4.03–4.30)

P2r6-CKPT raws (`p2r6_counts.txt`, bench window 07:24:25→07:25:05):
every telemetry window shows **pos4 ≈ 0.000–0.009** (dead position),
means 2.56–3.60, Avg Draft 53.6–64.9%. The 07:25:05 window (Accepted
throughput **80.20 tok/s** = the solo face's 79.74) pins the solo cell:
**emitted 3.46, 23.2 steps/s → step ≈ 43 ms @ c1**. The bars themselves
(`p2r6_run7n_2.out`: agg 254.33/274.41, solo 79.73/79.74) are
**second-invocation warm-state** cells (invocation 1 on the same boot:
agg 54.80/142.75 — the c4 shape warms across invocations; solo is
warm-insensitive: 79.60→79.74).

Consequences:

1. **Acceptance is exonerated vs the bars.** Our k4 profile
   [0.910/0.806/0.682/0.632], emitted 4.03 (counter window)–4.295
   (cumulative) **beats** the champion's [0.871/0.847/0.742/0.000],
   emitted 3.46. r38's "positional-decay gap vs D 0.871" was measured
   against **P2r3** (twin v20, an uncertified FAIL-verdict leg with a
   healthier-but-different profile 0.954/0.886/0.835/0.775) — not the
   certified bar cell. The acceptance-decay lane drops from "required" to
   "optional diagnostic" (drafter-quantization one-boot stays banked as a
   face-isolation tool, still arithmetic net-negative).
2. **The whole throughput gap is STEP TIME.** D ≈ 43 ms c1 / ≈ 46 ms c4
   (agg#2: 73.2 steps × (3.73 s − ~0.35 TTFT) ≈ 46 ms; c1→c4 adder +3–6 ms).
   Ours: 63–70 ms c1 (r37's 63.4 used a 0.44 s TTFT subtraction borrowed
   from the 600-tok face; the solo prompt is ~10 tok → honest band 63–70),
   83.4 ms c4 (adder +13–20). Gap **−33…−40% at c1, −45% at c4** — NOT the
   −18% r37 derived from P2r3's 83 tps @ 4.45 (that cell's step ≈ 49–54 ms;
   the certified champion is 43).
3. **Physics re-check**: plan P2 gate line already records the floor —
   **~23 ms = 27 GB fp8 ÷ TP2 ÷ 608 GB/s**. D's whole step 43 = forward
   ~25–28 + draft ~12–14 + sample ~2 (D pays the SAME sequential proposer —
   the 0.26 vs 0.30 llm_base_proposer diff is warmup-refactor noise; the
   only semantic deltas are mrope dims / xdrope removal / a local-argmax
   condition D never enabled, flag absent from every D serve script).
   Our forward 49 ms = 47% of the 608 GB/s spec; D's ≈ 25–28 ms = 72–90%.
   **r27's "302 GB/s weight-read floor" is re-classed: it is OUR stack's
   achieved BW, not the hardware floor.** (W14's "33 ms control" is
   qwen3.8-**flashnext** — different model; not a like control. Noted.)

## Re-scoped P4 ladder (replaces r38's PARD-first ordering)

The bar-reach arithmetic with OUR current acceptance (emitted 4.03):
solo 79.74 needs decode ≈ 96.7 tok/s → **step ≤ ~42 ms** (TTFT ~0.12);
agg 274.41 needs **c4 step ≤ ~46 ms**. D proves both reachable on this
hardware/model. At D-parity forward (~27), our CURRENT tail+sample
(15.2 + 2.5) gives step ≈ 44.7 — the bars clear with the tail intact.

1. **r40 (next) = E-LAW forward-decomposition capture** (timers-ON
   sacrificial boot): decompose the 49 ms verify forward — bare model vs
   in-path overlay costs (mamba_hybrid_snap_glue, mamba_lmcache_glue,
   align-mode, TWO_HS, lmcint hooks), AR share (r24's never-published
   ">10 ms → contingency lane" promise), GDN/mamba share, spec verify
   shape (5-row) vs plain, per-layer GEMM times vs r27's 0.77 ms/layer.
   PIN one dominant root cause of 49-vs-~27. One-change fix leg follows.
2. Prefill lane (1.39k vs D 1.84–2.02k in the same cells — P2r6 matrix
   k1k-c1 prefill_tps 1836) — same forward-class suspect; capture shares
   r40's decomposition.
3. Draft tail (15.2) demoted to secondary — closed-adjacent at D-parity
   forward; revisit only if r40 lands above ~30 ms.
4. k5 candidate stays banked/deferred (note: the champion effectively ran
   k3-live at k4-config; our live-pos4 k4 and k5 remain measured levers).

## Corrections issued

- r37's "D 53.6 ms step / −18%" → uncertified-cell comparison; honest
  champion gap −33…−45% (above).
- r38's "PARD port REQUIRED" → retired (Finding 1); "acceptance decay vs
  D" → re-based to the certified cell, exonerated (Finding 2).

## Rig

Untouched this leg (read-only `docker run --rm` source pulls + file reads;
radiance-test never stopped). W14 serving (HEALTH=200 per r38 restore).
