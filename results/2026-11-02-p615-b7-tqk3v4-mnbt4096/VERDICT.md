# p615 B7 — turboquant_k3v4_nc + MNBT 4096: max-capacity NC preset's balanced stall face (2026-11-02)

All-presets completion set (owner directive: comprehensive analysis across
every turboquant type). One-variable leg off the k3v4_nc default face:
`turboquant_k3v4_nc`, `ARC_MNBT=4096`, no thr. **k3v4_nc's KV page is 1856**
→ chunks floor to 2×1856 = **3712-token** (confirmed **3712-class × 58**
store lines; 1856×158 short-prompt stores). Battery clean: flip 8.6 s,
integrity 5/5 + determinism 3/3, 0 ERROR / 0 Traceback.

## Result (`rig/p65_sum.txt`)

| Metric | k3v4_nc default (P6.14) | fp8 default (B1) | **k3v4_nc + MNBT 4096 (B7)** |
|---|---|---|---|
| KV capacity | 853,220 tok (3.25×) | 405,491 (1.53×) | **911,660 tok (3.48×, +6.8 %)** |
| U32k / U64k TTFT | 15.48 / 33.78 s | 17.25 / 39.77 s | **16.99 / 36.90 s** (−1.5 % / −7.2 % vs family — under fp8-default on BOTH faces) |
| W_U32k_start p95 / max | 3736 / 3736 ms | 3986 / 4299 ms | **2011 / 2208 ms** (−46.2 % vs k3v4 def) |
| W_U64k_start p95 / max | 4186 / 4360 ms | 5295 / 5623 ms | **2309 / 2415 ms** (−44.8 % vs k3v4 def) |
| stalls (>2×b95) U32k / U64k | — | 36 / 60 | 39 / 94 |
| end windows p95 | — | 59–65 ms | 44.5 / 58.7 ms (clean) |

Vs the certified fp8 baseline: gaps **−49.6 % / −56.4 %**, TTFT under
family on both faces, capacity 3.48× — the strongest "balanced" outcome of
the completion set (its 3712-token chunks cost slightly more per-chunk time
than 4bit's 3328 but the NC prefill is fast).

## Disposition vs the pre-registered rule

- Gaps clause: **FAIL (narrow)** — 2011/2309 ms vs ~1.5 s (the recurring
  ~2 s floor of large-chunk faces).
- TTFT clause: **PASS** — under the fp8-default family on both faces.
- Integrity clause: PASS.

k3v4_nc + MNBT 4096 = the max-capacity low-stall face (3.48× KV with
fp8-default TTFT and ~2.0–2.3 s stalls); its sampled-throughput cost is the
known TQ NC face (t1.0 xhigh −53 %, owner-call).

Redaction: serve_full.log +125 (8×rig +48, 11×node +77),
preexisting_arc-test.json +5 (1×host) — arithmetic-exact, raw-ref grep 0 —
`results/REDACTION-2026-11-02-p615.md`. W14 restored by trap (health 200
verified before the B8 launch).
