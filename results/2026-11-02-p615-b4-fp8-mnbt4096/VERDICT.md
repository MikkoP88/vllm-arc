# p615 B4 — fp8 + MNBT 4096: stalls halved, capacity +6.8 %, TTFT +13/+8 % (2026-11-02)

The owner's fp8 ask ("if is possible minimize also fp8 stall"). One-variable
leg off the fp8 default posture: `fp8_e4m3`, `ARC_MNBT=4096` (chunk floor
4×832 = 3328-token chunks, confirmed by **3328-class × 74** store lines;
832×268 are the bystander decode-chunk stores), no thr. Battery clean:
flip 6.8 s, integrity 5/5 + determinism 3/3, 0 ERROR / 0 Traceback.

## Result (`rig/p65_sum.txt`)

| Metric | fp8 default (B1) | **fp8 + MNBT 4096 (B4)** | Δ |
|---|---|---|---|
| KV capacity | 405,491 tok (1.53×) | **433,210 tok (1.65×)** | **+6.8 %** (smaller prefill workspace) |
| U32k / U64k TTFT | 17.25 / 39.77 s | 19.49 / 42.83 s | +13 % / +8 % |
| W_U32k_start p50 / p95 / max | 3986 / 4299 ms | **2025 / 2030 / 2088 ms** | **−49 %** |
| W_U64k_start p50 / p95 / max | 5295 / 5623 ms | **2587 / 2587 / 2642 ms** | **−51 %** |
| stalls (>2×b95) U32k / U64k | 36 / 60 | 72 / 126 | more, shorter |
| end windows p95 | 59–65 ms | 59 / 64 ms | clean |

W_base shows 24 sub-second stalls (p95 566 ms) — the fp8 small-chunk
regime's per-step overhead shows up as elevated baseline jitter, unlike
the TQ faces; noted, not disqualifying (all end windows clean, 0/0
errors, integrity green).

## Disposition vs the pre-registered rule

- Gaps clause: **FAIL** — p95 2030/2587 ms vs the ~1.5 s bar (same
  miss magnitude as B3; the fp8 832-token page makes 3328 the finest
  "large" chunk, and fp8 chunk steps are ~2× slower than TQ's at equal
  tokens).
- TTFT clause: **FAIL** — 19.49/42.83 s vs family 17.25/39.77 s
  (+13/+8 %).
- Integrity clause: PASS.

Reading: fp8's balanced corner mirrors B3's shape — roughly half the
stall at ~10 % TTFT plus a real +6.8 % capacity gain — but at 2.6×
worse absolute stalls than the TQ faces and 1.65× vs 2.89×/3.09×
capacity. **For stall-sensitive fleets the TQ4 dtype dominates every
fp8 chunking face** (TQ4 default alone beats B4 on both axes), so the
fp8 disposition is: MNBT 4096 is the fp8 stall mitigation if the dtype
is fixed; otherwise TQ4 faces are strictly better. fp8 + thr2048 (the
historical p65 profile, ~1273–1604 ms p95 at +24/+27 % TTFT per r66a)
remains the fp8 low-stall extreme already on record.

Redaction: serve_full.log +125 (8×rig +48, 11×node +77),
preexisting_arc-test.json +5 (1×host) — arithmetic-exact, raw-ref
grep 0 — `results/REDACTION-2026-11-02-p615.md`. W14 restored by trap.
