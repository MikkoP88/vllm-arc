# p615 B3 — TQ4 + MNBT 4096: stall halving at ~10 % TTFT cost, +6.8 % capacity (2026-11-02)

One-variable leg off the cert6 TQ4 face: `turboquant_4bit_nc`,
`ARC_MNBT=4096` (chunk floor 1664×2 = 3328-token chunks, confirmed by
**3328-class × 76** store lines), no thr. Battery clean: flip 8.0 s,
integrity 5/5 + determinism 3/3, 0 ERROR / 0 Traceback.

## Result (`rig/p65_sum.txt`)

| Metric | TQ4 default (cert6) | **TQ4 + MNBT 4096 (B3)** | Δ |
|---|---|---|---|
| KV capacity | 757,641 tok (2.89×) | **809,161 tok (3.09×)** | **+6.8 %** (smaller prefill workspace) |
| U32k / U64k TTFT | 15.85 / 33.98 s | 17.59 / 37.03 s | +11 % / +9 % |
| W_U32k_start p95 / max | 3227 / 3329 ms | **2003 / 2107 ms** | **−38 %** |
| W_U64k_start p95 / max | 3780 / 3918 ms | **2049 / 2125 ms** | **−46 %** |
| end windows p95 | 58 / 54 ms | 49 / 45 ms | clean |

Note the U64k window p50 = 1704 ms: at 3328-token chunks the ~19
chunks run near-back-to-back, so a bystander's TYPICAL interval during
the window is one chunk step — the stall count is what halves, and the
worst stall ~halves. TTFT stays better than fp8-true-default on U64k
(37.03 vs 39.77 s) and at parity on U32k (17.59 vs 17.25 s).

## Reading

MNBT-4096 under TQ4 is a strict improvement over TQ4-default for
mixed fleets (stalls −38…−46 %, capacity +6.8 %) at a ~10 % long-
prompt TTFT cost that still leaves U64k faster than the fp8 default.
It misses the pre-registered "low-stall profile" bar (p95 ≤ ~1.5 s) —
B2 (thr2048) tests the finer chunk granularity.

Redaction: serve_full.log +34, preexisting_arc-test.json +5 (both
arithmetic-exact, raw-ref grep 0) — `results/REDACTION-2026-11-02-p615.md`.
W14 restored by trap.
