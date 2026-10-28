# VERDICT — r24 compile A/B (FULL_AND_PIECEWISE vs product mode=none): compile lane DEAD for MTP on 0.30 — acceptance law CONFIRMED (pos1 0.8821 → 0.382 hard-gate FAIL, wrong-logits family) AND the step wall is compile-INDEPENDENT (62.6/78.6 vs 62.9/77.4 ms) — falsifying the "unfused replay" theory; the step lives in the MTP step structure, not compile.

Posture: p12b2 image, product flags verbatim except
compilation-config `{"cudagraph_mode":"FULL_AND_PIECEWISE", capture 1..320}`
(mode default → CompilationMode.VLLM_COMPILE). p10 fastpath deployed
(same PYTHONPATH), defect-a speculator coercion active (2 coerced lines:
PIEWISE-family → FULL), warmup interlock live. Boot healthy 16:30:29
(7.5 min — piecewise capture phases; 27 decode capture sizes under compile).

## Cell 1 — survival (defect-b regression on compiled prefill): PASS

Flip payload (832 tok) as FIRST traffic → HTTP 200 in 21.15 s (interlock
window; onset 0). Under FULL_AND_PIECEWISE the compiled prefill pieces are
CAPTURED, never launched loose — the b2 wedge face does not fire. The
defect-b fix family holds across both graph postures.

## Cell 2 — compile × spec acceptance (the 0.26 law, REVAL on 0.30): LAW HOLDS

| | product (compile off) | r24 (compile on) |
|---|---|---|
| pos1 | 0.8768-0.8821 | **0.382** (< 0.5 hard FAIL) |
| acceptance | 0.698-0.703 | 0.2825 |
| mean length | 3.79-3.81 | 2.13 |
| engine log window | len 4.19-4.36 | len 1.95, draft acc 23.7% |

Gate's own verdict: "wrong-logits family (D r1 mojibake signature)" —
the plan's falsified-path #1 (0.26: inductor piecewise × spec-decode
wrong logits, TORCH_COMPILE_DISABLE recovered 22.5 → 79.7 solo) is
re-proven on the 0.30 stack verbatim. Compile ON with MTP is not shippable
here. P4 row "compile on/off A/B reported on 0.30": CLOSED EARLY by this
leg.

## Cell 3 — step wall is compile-independent (secondary falsification)

Stepprobe on the compile-ON boot: c1 62.6-62.9 ms, c8 78.6 ms — IDENTICAL
to compile-off product (62.9 / 77.4). The r23 inference "compile=none
unfused replay ~1 ms/layer" is FALSIFIED: compiling the decode path does
not shorten the step on this stack. Solo on compile-ON: 7.26 tok/s
(acceptance death — output mojibake, not a step-wall effect).

⇒ The ~63 ms c1 / ~77 ms c8 step is NOT the replay kernel content. It
lives in compile-independent structure: MTP step composition (draft graph
replays + verify/sample eager tail + output copies), stream dependencies,
or fixed GPU-side serialization. Champion D's ~50 ms step (0.26, compile
OFF, twin v21 kit) bounds what this hardware does — next lever is a
GPU-side phase split of the step (r25: event-timestamped replay/draft/
verify/copy phases), then either the tail-overlap lane (p50) or the
kernel-kit delta (twin v21 vs radiance-inherited) by what the split shows.

## Disposition

- Compile lane: CLOSED for MTP (acceptance law; both cells measured).
  Product posture stays mode=none + FULL_DECODE_ONLY + p10 fastpath.
- P2 TPOT c8 gate: still OPEN at steady 77.4 / matrix 110.75-182.23.
  Lever re-aimed: phase decomposition of the compile-independent step.
- W14-style FULL_AND_PIECEWISE remains a recorded clean-booting posture
  (survival PASS) — relevant only if acceptance were ever fixed upstream
  (noted for the upstream-issue text, owner-gated).

## Artifacts (legs_pull/)

survival.txt; spec_gate.txt (REVAL numbers + wrong-logits verdict);
stepprobe.txt (compile-ON step walls); solo_compileon.txt (7.26 tok/s);
r24_serve/boot/bench.sh. serve_r24_final.log on the rig
(/root/build/legs/arc-p2-r24/). Rig restored to W14, health 200.
