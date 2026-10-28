# VERDICT — P2 CLOSE-OUT (§10 row, assessed 2026-10-28 after r29): the attention+graphs
# phase closes on the mode=none product posture with 5 gate rows DONE, 3 rows
# UNREACHABLE/MISSED-WITH-PROOF (each falsified by direct measurement on this stack, all
# evidence banked), p11 CLOSED void-on-dense, p50 CLOSED-as-evidence-complete, p14
# DIAGNOSED with an in-scope follow-up lane. Product posture MERGES to main (squash of
# the linear chain feat/arc-p33-draft-chain = p12+p12b+p32+p30+p31+p10 code + r18–r29
# evidence), per owner instruction "p12b → main at close" — the linear chain makes the
# tip the complete carrier.

Product posture at close: upstream v0.30 + platform patches (P1, on main) + graph
stream hygiene + first-request warmup interlock (p12b, defects a/b) + speculator
FULL-capture coercion (p32, defect-a boot) + MTP k4 wiring (p30) + acceptance gate
instrument (p31) + GDN metadata fastpath (p10) — served flags: mode=none,
cudagraph_mode FULL_DECODE_ONLY captures [1..320], MTP k4, fp8, kv fp8_e4m3,
prefix caching on. Every leg gated: survival flip-payload + acceptance pos1.

## §10 P2 gate row, item by item

| gate item | disposition | evidence |
|---|---|---|
| defects (a)+(b) fixed, held as regression cells | **DONE** — interlock (r17, ca75f1d8) + flip-payload cell (r18); every leg since r18: 832-tok flip as FIRST traffic → 200/200, onset 0 (r23/r25/r26/r28/r29) | survival cells in each leg dir |
| acceptance telemetry live (p31, every leg gated) | **DONE** — pos1 ≥0.5 hard; product-posture range 0.877–0.912 across r23–r29, never under hard bar; 0.85 target miss recorded → P4 lane | spec_gate.json per leg |
| CPU serial killed (p10 ≤5 ms → 0.49 ms) | **DONE** — 10× under budget (d69b2035; r23 bank) | p10 stats cells |
| S1 compile posture ACCEPTED + forward replay ≤45 ms c1 | **CLOSED-FAILED-WITH-PROOF** — FULL_AND_PIECEWISE at exact W14 capture sizes: pos1 0.4697 <0.5 hard (wrong-logits; compile×spec law on 0.30+dense, posture-independent), golden drift 2/5, forward replay UNCHANGED 49.3–50.2 ms. Floor proof (r27): 320 layer-GEMMs ×0.15 ms ≈48 ms = weight-read BW; W14's 33 ms control is a radiance-fork/flash-next property, not transplantable via compilation-config. Falsified ×3 (0.26, r24 ec76d819, p32 94089974) | p32 + r27 legs |
| draft tail ≤7 ms (p33) | **CLOSED-UNREACHABLE-WITH-PROOF** — r26 (32f84657): fused multi-step decode ALREADY chains draft steps 1–3 in ONE graph (2 replays total, gap 0.01 ms — no chain object); r27: lm_head 8.5 ms = 56% (4×635 MB fp8/rank weight reads, tile-BW ceiling) + layers ~2.5 + FA/norm ~0.6 + required TP2 ARs ~3.4 → structural floor ≈15 ms; r28 (a10be70a): the one flag lever (use_local_argmax_reduction) correctness-clean, saves 0.11 ms. Propose measured 15.06–15.18 | r26/r27/r28 |
| step decomposition published | **DONE ×2** — r25 (c07be534): forward 49.0–54.5 (75%) + tail 16.6–17.4 (sample+verify 2.4–2.8 + draft 14.1–14.6), fgap=bgap=0, tailres≈0; r26: propose 15.18 = prefill replay 5.18 + fused decode replay 9.99 + gap 0.01 | r25/r26 |
| AR share of forward published (>10 ms → contingency) | **DONE, NO CONTINGENCY TRIGGERED** — 132 collective calls/step (128 layer ARs + logits AG + prepare); lock-step residual 0–8 ms, central 2–5 (≤8%); eager oneCCL durations skew-inflated (published with caveat — counts/sizes only); r28 tightened draft-side removable share to ~0.1 ms | r27 (72ea42ef) |
| TPOT k1k-c8 ≤60 ms as measured, timers-off | **MISSED-WITH-PROOF — 77.9 ms** (c1 63.4–63.5); the step is at the BW wall (r27: ~27 GB/rank weight reads @~300 GB/s + spec structure; 63.4 ≡ r22a's pre-timer 64.4). The 60 ms bar predates the BW-wall law; TPOT moves only via acceptance length (P4) or step structure. **Distortion-law amendment**: timers-ON stepprobe faces (r25–r28: 77.6/92.8) carried ~+14 ms — gate faces must be timers-off; r29 is the reference cell | r29 (09dcdec5) |
| p11 QSA-ESIMD | CLOSED void-on-dense — no call site on target (48 GDN + 16 full-attn layers, no indexer); kernel + toolchain BANKED §6, not wired | r19–r21b |
| p50 E2E residue (≤15 ms) | **CLOSED-AS-EVIDENCE-COMPLETE, no leg spent** — fgap=bgap=0 + tailres≈0 + step 100% accounted (r25/r26): no overlap residue exists to harvest; r27's BW-wall law independently moots the target | r25/r26/r27 cited |
| p14 prefill parity (bars 3944/8083/17447/41565) | **DIAGNOSED, OPEN as §12-14 reval** — r29: ladder 5469/11286/24207/54782 (+39/+40/+39/+32%); no prefill-relevant flag delta vs D boot line (line-by-line diff); 8k window 97% GPU-busy: GEMM 64.5% (520 fp8_gemm @6.65 ms ≈63.5 TFLOPS/rank eff) + AR 22.2% (265 @4.505 ms, 84 MB @~19 GB/s eff) + FA 4.5% + GDN 3.4%; our GEMM+FA+GDN alone (3905 ms) ≈ D's whole 8k TTFT → D engine runs same math materially faster; the two pools = the C bake-chain reval lane (arstage/stall/wedge/v60/v63/v64-TTFTFIX, plan §12-14). Order: AR-staging first, GEMM bakes second; scope-law check per port | r29 |

## The three laws P2 established (carry into P3–P5)

1. **BW-wall law** — decode step ≈ weight-read bandwidth floor; TPOT responds only to
   acceptance length (P4) or step structure; structural levers measured dead (compile
   ×3, overlap, chained draft, local-argmax, kernel replacement).
2. **Profiler-visibility law** — XPU graph replays are invisible to torch profiler;
   attribution via eager-proxy; eager oneCCL durations are skew-inflated spin
   (counts/sizes only); prefill is eager on this posture → directly attributable (r29).
3. **Golden nondeterminism floor** — same-leg ref+run sha-exact stands; cross-leg drift
   ≤1/5 informational / ≥2/5 replicated hard (r26/r28 two-for proof).

Plus the operational laws: timers-off for all gate faces (stepprobe-face +14 ms
amendment, r29); profile legs LAST; STOP radiance before arc legs, restore W14 after.

## Merge disposition (executed with this close-out)

- **feat/arc-p33-draft-chain → main (squash)**: the linear chain carries ALL validated
  P2 product code (p12b stream hygiene + interlock; p32 speculator coercion; p30 MTP
  wiring; p31 gate; p10 fastpath) + the full r18–r29 evidence chain. Defect-level
  gates all PASS; the missed rows are proven-unreachable performance targets, not code
  defects — exactly the owner's "p12b → main at close" instruction (the chain is
  linear, the tip is the complete carrier).
- Evidence-only legs live inside that squash as results/ dirs (no posture changes).
- p14 reval + P4 open on fresh branches per the one-branch-per-item law.

## Phase scorecard vs the superiority contract (as of close)

solo 53.5–56.9 / agg 196.7–209.8 (r28/r29 era) vs contract solo ≥79.74 / agg ≥274.41 —
the remaining distance is now precisely attributed: acceptance-length × step-wall
(P4 tuning), prefill rate (p14 §12-14 reval), cache reuse (P3). No unexplained residue.

Next per §12: **P3** (40-cache + lmcache p40–p43 + e01–e04), then P4 (speculative —
the one live TPOT/solo lever under the BW-wall law), P5 hardening + cert.
