# p615 B5 — TQ4 + MNBT 2048: the discriminator — small-chunk TTFT cost is chunk-size-inherent (2026-11-02)

One-variable leg off the cert6 TQ4 face: `turboquant_4bit_nc`,
`ARC_MNBT=2048` → 1664-token chunks (1 TQ page — the finest possible
under TQ), **no thr**. Chunks confirmed **1664-class × 378** store
lines + 10× 8320 aggregate tails. Battery clean: flip 6.7 s,
integrity 5/5 + determinism 3/3, 0 ERROR / 0 Traceback.

## Result (`rig/p65_sum.txt`)

| Metric | TQ4 default | TQ4 + thr2048 (B2) | **TQ4 + MNBT 2048 (B5)** |
|---|---|---|---|
| U32k / U64k TTFT | 15.85 / 33.98 s | 21.60 / 47.78 s | **22.14 / 45.38 s** |
| W_U32k_start p50 / p95 / max | 2729 / 3227 / 3329 ms | 1017 / 1159 / 1863 ms | **1015 / 1325 / 1902 ms** |
| W_U64k_start p50 / p95 / max | 3188 / 3780 / 3918 ms | 1096 / 1329 / 1472 ms | **1058 / 1286 / 1327 ms** |
| stalls (>2×b95) U32k / U64k | 6 / 11 | 102 / 190 | 114 / 148 |
| KV capacity | 757,641 (2.89×) | 757,641 (2.89×) | **821,283 (3.13×)** |

B5 = B2's stall face (p95 1.3 s, both under the 1.5 s bar, −59/−66 %
vs TQ4 default) at B2's TTFT cost (+40/+34 %), reached by the OPPOSITE
knob (pure MNBT cap, no thr) — and it buys the **largest TQ4 KV pool
measured** (+8.4 % over default, vs B3's +6.8 %).

## The discriminator result

B2 and B5 land on the same chunk size (1664) by different mechanisms
and pay the same TTFT cost → the +35–40 % small-chunk TTFT penalty is
**inherent to 1664-token chunking** (each chunk step re-reads the full
prefix KV; ~380–400 chunk steps per long prompt vs ~114 at default),
NOT thr-mechanism overhead. Combined with the dtype-page quantization
(TQ page 1664, fp8 page 832 → chunk ladders 1664/3328/6656 and
1664/3328/7488), there is no knob between "B3 stalls" and "B2 TTFT":
the stall↔TTFT tradeoff is a Pareto frontier, not a fixable defect.

## Disposition vs the pre-registered rule

- Gaps clause: **PASS** (1325/1286 ms). TTFT clause: **FAIL**
  (+28/+14 % vs the fp8-default family 17.25/39.77 s). Integrity: PASS.
- Same split as B2 → per the rule, **no face qualifies as the strict
  "low-stall profile"; the outcome is the documented Pareto menu**
  (close-out in `results/2026-11-02-p615-stall-analysis/` + docs).
- If a 1.3 s-stall face is wanted anyway, B5 dominates B2 (same stalls
  and TTFT, +8.4 % KV capacity instead of +0) — MNBT 2048, not thr2048.

Redaction: serve_full.log +97 (8×rig +48, 7×node +49),
preexisting_arc-test.json +5 (1×host) — arithmetic-exact, raw-ref
grep 0 — `results/REDACTION-2026-11-02-p615.md`. W14 restored by trap.
