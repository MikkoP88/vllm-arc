# p615 B8 — turboquant_3bit_nc + MNBT 4096: a 4th chunk size lands on the taxonomy curve (2026-11-02)

All-presets completion set. One-variable leg off the 3bit_nc default face:
`turboquant_3bit_nc`, `ARC_MNBT=4096`, no thr. **3bit_nc's KV page is
2176** → MNBT 4096 floors to just **2176-token chunks** (1 page —
confirmed by the store log showing ONLY **2176-class × 290** lines), the
finest large-budget chunk of the whole matrix. Battery clean: flip 14.4 s
(3bit first-compile), integrity 5/5 + determinism 3/3, 0 ERROR /
0 Traceback.

## Result (`rig/p65_sum.txt`)

| Metric | 3bit_nc default (P6.14) | fp8 default (B1) | **3bit_nc + MNBT 4096 (B8)** |
|---|---|---|---|
| KV capacity | 975,329 tok (3.72×) | 405,491 (1.53×) | **1,042,793 tok (3.98×, +6.9 % — largest pool of the campaign)** |
| U32k / U64k TTFT | 16.02 / 35.74 s | 17.25 / 39.77 s | **19.41 / 42.09 s** (+21/+18 % vs 3bit def; +12.5/+5.8 % vs family) |
| W_U32k_start p95 / max | 3189 / 3288 ms | 3986 / 4299 ms | **1932 / 2013 ms** (−39.5 % vs 3bit def; p50 1241) |
| W_U64k_start p95 / max | 3729 / 3857 ms | 5295 / 5623 ms | **1637 / 2086 ms** (−56.1 % vs 3bit def; p50 1256) |
| stalls (>2×b95) U32k / U64k | — | 36 / 60 | 56 / 174 |
| end windows p95 | — | 59–65 ms | 44.5 / 60.8 ms (clean) |

**Taxonomy confirmation with a 4th chunk size:** B8's U64k p95 (1637 ms)
lands exactly between the 3328-chunk faces (~2.0–2.3 s) and the 1664-chunk
faces (1.29–1.33 s), as stall ≈ tokens-per-chunk / prefill-rate predicts
for 2176-token chunks — the curve now has four measured points
(7488/6656, 3712/3456/3328, 2176, 1664).

## Disposition vs the pre-registered rule

- Gaps clause: **FAIL (narrow)** — 1932/1637 ms vs ~1.5 s (U64k is the
  closest large-budget face to the bar in the whole matrix).
- TTFT clause: **FAIL** — +12.5/+5.8 % vs the fp8-default family.
- Integrity clause: PASS.

Reading: 3bit_nc + MNBT 4096 is the **max-capacity near-low-stall face**
(3.98× KV, stalls −40/−56 % vs its default, −52/−69 % vs fp8 default) at a
moderate TTFT cost — but its dtype carries the largest sampled-throughput
and quality face costs of the family (owner-call per P6.14).

Redaction: serve_full.log +125 (8×rig +48, 11×node +77),
preexisting_arc-test.json +5 (1×host) — arithmetic-exact, raw-ref grep 0 —
`results/REDACTION-2026-11-02-p615.md`. W14 restored by trap (health 200
verified before each subsequent leg).
