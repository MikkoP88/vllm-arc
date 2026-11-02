# p615 stall analysis — bystander-gap taxonomy from banked cert6 timelines (2026-11-02)

**Owner directive:** "do realy deep analyze and research is anyway to
minimize all gap p95 U32k/U64k or fixes … if is possible minimize also
fp8 stall." This is the analysis phase (master plan P6.15) — zero new
rig traffic; it decomposes the ALREADY-BANKED cert6 release-gate
timelines + serve logs.

## Method

`rig/stall_decomposition.txt` (generated in-container from the banked
jsons + logs): per face, all bystander inter-chunk gaps; spikes > 2×
baseline-p95 clustered at 300 ms; each cluster's stream set; and the
LMCache big-class chunk-store timestamps overlaid on the same send
anchor.

## Finding 1 — a bystander gap = exactly ONE chunked-prefill engine step

Every spike cluster is **engine-wide** (all 6 bystander streams stall
synchronously — it is a step boundary, not per-stream jitter), and
each cluster's END aligns to within ~50 ms with a chunk-completion
store line:

- TQ4 default (U64k): stalls 2652/3105/3221/3339/3444/3522/3654/3780/
  3917 + 1376/1689 ms; 6656-class stores at +6.01/+9.23/+12.56/+16.01/
  +19.53/+23.18/+26.96/+30.88/+32.26 s. Stall sum 33.7 s == TTFT
  33.98 s. Bystanders receive tokens ONLY in the ~0.2–0.5 s decode
  windows between chunk steps — **no decode piggyback into prefill
  steps on this stack**.
- TQ4 default (U32k): 6 stalls (2721→1318 ms) summing 15.7 s == TTFT
  15.85 s; stores at +5.99/+9.21/+12.54/+14.50 s.
- fp8 defer+slice face (thr2048): 22 (U32k) / 29 (U64k) synchronized
  stalls of 733–1442 ms, growing slowly with position — small-chunk
  regime.

**Corollary — the lever is chunk GRANULARITY, not the dtype:**
stall ≈ tokens-per-chunk / prefill-rate. Chunk size = MNBT floored to
the KV-page multiple (fp8 page 832 → 7488-token chunks at MNBT 8192;
TQ page 1664 → 6656).

## Finding 2 — the historical "fp8 baseline" was a thr2048-profile number

Every previously banked fp8 gap figure (p95 1299/1569 ms — the TQ4 A/B
comparator, rung3, cert5/cert6 slice faces) was measured with
`--long-prefill-token-threshold 2048` ACTIVE (the p65 long-context
profile face) → ~1664–2048-token chunks → ~1.0–1.4 s stalls. **True
default-posture fp8 (7488-token chunks) has never had its bystander
gaps measured.** Predicted from the taxonomy: ~6–8 s p95 per stall at
U32k — i.e. WORSE than TQ's 3.2–3.9 s (TQ's narrower KV reads make
each big chunk ~2× faster). The cert6-docs statement "TQ gaps ~2.5×
vs fp8" is a **profile-mismatch artifact**; per equal posture TQ
improves the stalls. Leg B1 measures the true fp8-default baseline.

## Finding 3 — stores are NOT the stall driver

Chunk stores fire between chunk steps (at completion, ms-scale); spike
windows align with chunk compute. The defer/slice knobs address the
END-stall store tail (M1), not the per-chunk bystander stall —
deprioritized for this lane.

## Lever matrix (leg plan B1–B5)

| Leg | Face | Expected chunk | Expected stall p95 | Expected TTFT cost |
|---|---|---|---|---|
| B1 | fp8 default (MNBT 8192) | 7488 | ~6–8 s (first-ever) | — (baseline) |
| B2 | TQ4 + thr2048 | ~1664–2048 | ~0.8–1.2 s | +24/+27 % over TQ4 (r66a fp8 shape) |
| B3 | TQ4 + MNBT 4096 | 3328 | ~1.6–1.9 s | ~+10–15 % (large-M chunks kept) |
| B4 | fp8 + MNBT 4096 | 3328 | ~3–4 s | ~+10–15 % |
| B5 | TQ4 + MNBT 2048 | 1664 | ~0.8–0.95 s | larger; only if B3 trend holds |

Pre-registered disposition rule (master plan P6.15): a face becomes
the dtype's recommended "low-stall profile" iff gap p95 ≤ ~1.5 s AND
TTFT ≤ fp8-default family AND integrity 5/5. Certified default
postures never change.

No addresses in the artifact (scanned clean).

## CLOSE-OUT — legs B1–B5 run 2026-11-02 (all on cert6, same p65
## timeline harness, integrity 5/5 + det 3/3, 0 ERROR / 0 Traceback)

| Face | Chunk | TTFT U32k / U64k | gap p95 U32k / U64k | KV capacity |
|---|---|---|---|---|
| fp8 default (B1) | 7488 | 17.25 / 39.77 s | 3986 / 5295 ms | 405,491 (1.53×) |
| fp8 + MNBT 4096 (B4) | 3328 | 19.49 / 42.83 s | 2030 / 2587 ms | 433,210 (1.65×) |
| fp8 + thr2048 (historical profile, pre-P6.15 record) | 1664 | +24/+27 % (r66a) | ~1273–1304 / ~1574–1604 ms | 405,491 (1.53×) |
| TQ4 default (cert6 gate) | 6656 | 15.85 / 33.98 s | 3227 / 3780 ms | 757,641 (2.89×) |
| TQ4 + MNBT 4096 (B3) | 3328 | 17.59 / 37.03 s | 2003 / 2049 ms | 809,161 (3.09×) |
| TQ4 + thr2048 (B2) | 1664 | 21.60 / 47.78 s | 1159 / 1329 ms | 757,641 (2.89×) |
| TQ4 + MNBT 2048 (B5) | 1664 | 22.14 / 45.38 s | 1325 / 1286 ms | 821,283 (3.13×) |

**Rule outcome:** no face satisfies both pre-registered clauses
(gaps p95 ≤ ~1.5 s AND TTFT ≤ fp8-default family). The dtype-page
quantization (TQ 1664 / fp8 832) leaves no chunk size between B3's
stalls and B2's TTFT, and B5 proved the small-chunk TTFT cost is
chunk-size-inherent (prefix re-reads), not knob overhead — so the
stall↔TTFT tradeoff is a structural Pareto frontier, not a defect.
Disposition = documented menu, nothing adopted, certified defaults
unchanged:

- **Stall-sensitive mixed fleet:** `turboquant_4bit_nc` + MNBT 4096
  (B3 — stalls −38/−46 %, capacity +6.8 %, TTFT ≤ fp8-default on U64k
  and at parity U32k); sub-1.5 s stalls need 1664-token chunks
  (B5's MNBT 2048 dominates B2's thr2048: same face + largest TQ4
  pool 3.13×) at +14–40 % TTFT.
- **fp8-fixed fleet:** MNBT 4096 (B4 — stalls −49/−51 %, +6.8 %
  capacity, +8–13 % TTFT); thr2048 remains the historical extreme.
- **Max TTFT/capacity fleet:** certified defaults as-is — note TQ4
  default ALREADY beats fp8 default on bystander stalls (3.23/3.78 vs
  3.99/5.30 s), TTFT (−8/−15 %), and capacity (2.89× vs 1.53×); the
  fp8 default's standing rests on sampled throughput + the quality
  face, not on stalls.
- **Per-leg detail:** `results/2026-11-02-p615-b{1..5}-*/VERDICT.md`;
  redactions `results/REDACTION-2026-11-02-p615.md` (all byte-exact).
