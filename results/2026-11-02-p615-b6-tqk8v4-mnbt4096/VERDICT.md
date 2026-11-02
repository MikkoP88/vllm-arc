# p615 B6 — turboquant_k8v4 + MNBT 4096: quality-safest preset's balanced stall face (2026-11-02)

Owner follow-up ("result has to be comprehensive performance analyzes … why
comprehensive analyze missing all turboquant types?"): P6.15's knob legs had
sampled only `4bit_nc`; B6–B8 complete the dtype × knob matrix with the
balanced face (MNBT 4096) for every remaining preset. One-variable leg off
the k8v4 default face: `turboquant_k8v4`, `ARC_MNBT=4096`, no thr.
**k8v4's KV page is 1152** (8-bit keys) → chunks floor to 3×1152 =
**3456-token** (confirmed **3456-class × 72** store lines; 1152×208 are the
short-prompt/bystander stores; 2304×4 tails). Battery clean: flip 8.5 s,
integrity 5/5 + determinism 3/3, 0 ERROR / 0 Traceback.

## Result (`rig/p65_sum.txt`)

| Metric | k8v4 default (P6.14) | fp8 default (B1) | **k8v4 + MNBT 4096 (B6)** |
|---|---|---|---|
| KV capacity | 526,445 tok (2.01×) | 405,491 (1.53×) | **562,045 tok (2.14×, +6.8 %)** |
| U32k / U64k TTFT | 15.29 / 33.03 s | 17.25 / 39.77 s | **17.41 / 37.58 s** (at fp8-default family: +0.9 % / −5.5 %) |
| W_U32k_start p95 / max | 3690 / 3854 ms | 3986 / 4299 ms | **1936 / 1960 ms** (−47.5 % vs k8v4 def) |
| W_U64k_start p95 / max | 4385 / 4583 ms | 5295 / 5623 ms | **2225 / 2299 ms** (−49.3 % vs k8v4 def) |
| stalls (>2×b95) U32k / U64k | — | 36 / 60 | 60 / 90 |
| end windows p95 | — | 59–65 ms | 57.8 / 56.5 ms (clean) |

Vs the certified fp8 baseline the gaps drop **−51.4 % / −58.0 %** while TTFT
stays at the family — the same Pareto shape as B3 (4bit_nc + MNBT 4096),
transferred to the quality-safest preset (k8v4: t1.0 xhigh within 9 % of
fp8). Chunk-size note: k8v4's 3456-token chunks sit between TQ4's 3328 and
fp8's 3328-class — stalls land accordingly (1.94/2.22 s, just above the
1.5 s bar, exactly like B3's 2.00/2.05).

## Disposition vs the pre-registered rule

- Gaps clause: **FAIL (narrow)** — 1936/2225 ms vs ~1.5 s (same miss as B3).
- TTFT clause: **PASS** — +0.9 %/−5.5 % vs the fp8-default family.
- Integrity clause: PASS.

k8v4's low-stall recommendation mirrors 4bit_nc's: **MNBT 4096 is the
balanced face** (stalls ~halved, capacity +6.8 %, TTFT at family); the
sub-1.5 s regime needs page-sized chunks (B2/B5 shape, +14–40 % TTFT).

Redaction: serve_full.log +125 (8×rig +48, 11×node +77),
preexisting_arc-test.json +5 (1×host) — arithmetic-exact, raw-ref grep 0 —
`results/REDACTION-2026-11-02-p615.md`. W14 restored by trap (health 200
verified before the next leg).
