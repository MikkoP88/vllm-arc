# r38 — P4 acceptance face: pos1 PASSES (0.910/0.906); the D-gap is draft-quality DECAY (aggregate 0.7575 vs D 0.871); k5 A/B = +6.7% solo / +7.8% agg (banked candidate posture); PARD scoped as the structural tail lever

Date: 2026-10-29 (boots 20:07 k4 / 20:27 k5, spec gates + run7n on each).
Posture: product VERBATIM except `num_speculative_tokens` 4→5 on boot B
(one sed, backed up, restored). Timers off, first-traffic, health-gated.
Tool: `/root/build/arc_spec_gate.py` (p31 gate; counter-delta window on a
Run-7n-parity 4×256 load) + `bench_run7n.py` unchanged.

## A/B (steady counter windows)

| cell | k4 (product) | k5 (boot B) |
|---|---|---|
| pos profile (per-pos counters) | [0.910, 0.806, 0.682, 0.632] | [0.906, 0.807, 0.680, 0.575] (+pos5 ≈ 0.27–0.5, log window) |
| **pos1** | **0.9104** | **0.9061** |
| aggregate acceptance | 0.7575 | 0.6961 |
| emitted tokens/step | 4.03 | **4.47** |
| solo200 tps | 59.72 / 59.72 (r37 parity ✓) | **62.75 / 63.73 (+6.7%)** |
| agg4x256 | 198.24 / 199.76 | **213.72 / 215.77 (+7.8%)** |
| implied step (decode window) | ~63–64 ms | ~67–68 ms (tail +3.3 ms, emitted +0.44) |

## Findings

1. **Acceptance pos1 gate: PASS at both k** (0.906–0.910; plan target
   0.85 on pos1). The aggregate-acceptance reading (tool's alt reading,
   D champion 0.865–0.875) misses at both k (0.7575 / 0.6961).
2. **The D acceptance gap is positional DECAY, not pos1**: ours vs D's
   implied profile diverges with position (−0.04 / −0.07 / −0.14 / −0.16).
   Candidates: our drafter runs the fp8 lm_head inside FULL graph replays
   (4×2.1 ms weight reads, r27) vs D's "eager MTP head" posture; upstream
   itself warns at speculative.py:1361 (k>1 re-forwards the MTP layer,
   lower acceptance). Diagnostic one-boot exists if needed:
   `speculative_config.quantization` override (drafter fp16) — arithmetic
   says throughput-NET-NEGATIVE (+8.4 ms/step weight reads) but it
   isolates numerics-vs-structure.
3. **k5 = a measured +7% lever, banked as CANDIDATE, not adopted**: it
   re-canonicalizes every sha battery (decode composition changes) — the
   posture decision belongs at P4 convergence with all tuning faces
   measured at k4 for comparability.
4. **PARD scoped (the structural tail lever)**: 0.30 ships
   `parallel_drafting` (PARD — draft queries in ONE parallel pass) for
   step3p5/dflash ONLY; the eagle/MTP proposer has zero PARD wiring.
   Arithmetic: sequential tail 15.2 ms (2 replays) vs PARD single-pass
   ~5–6 ms, and deeper k adds cheap parallel mask queries — the D-class
   tail economics. Contract math: solo 79.74 needs BOTH emitted ≈5.0 AND
   tail ≤ ~5.6 ms → the PARD port (upstream in-tree reference
   step3p5.py; eagle proposer extension) is REQUIRED for the throughput
   bars; k5 alone tops out at ~64.

## Correction to r37's lane list

r37's VERDICT listed "p32/S1 compile posture" as a step-time lever —
STALE: r24 already closed the compile lane on 0.30 (acceptance law,
pos1 0.882→0.382; step wall compile-independent 62.6–62.9 c1). The live
step-time ladder is: PARD tail port → acceptance decay → (forward 49 ms
= measured weight-read floor at 302 GB/s, r27 — out of reach in scope).
Prefill (+39–45% TTFT) stays the separate C bake-chain reval lane (r29).

## Rig

Serve restored to k4 (backup `r30_serve.sh.r38k4bak` reapplied, grep
verified); arc-test removed; radiance-test up; host w14_serve_only.sh →
**W14 HEALTH=200**. No product bytes changed this leg (probe/A-B only).
