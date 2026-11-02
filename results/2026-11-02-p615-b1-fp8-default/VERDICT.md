# p615 B1 — fp8 TRUE default-posture bystander-stall baseline (2026-11-02)

**The first-ever gap measurement of the certified default posture with
NO long-prefill threshold** (every prior fp8 "gap" number — p95
1299/1569 ms — came from thr2048-profile faces). One-variable leg on
`vllm-arc:v0.30.0-b70-cert6`: `fp8_e4m3`, MNBT 8192, no thr; battery
clean (flip 8.3 s, integrity 5/5 + det 3/3, 0 ERROR / 0 Traceback);
solo TTFT 17.25 / 39.77 s (r70 family 17.36 / 40.26 — in-family ✓).

## Result (`rig/p65_sum.txt`, `rig/p65_tl.json`)

| Window | p50 | **p95** | max | stall count |
|---|---|---|---|---|
| W_U32k_start | 58.9 ms | **3985.5 ms** | 4298.5 ms | 36 >2×b95 |
| W_U64k_start | 65.0 ms | **5295.0 ms** | 5623.3 ms | 60 >2×b95 |
| end windows | 55.3 ms | 59–65 ms | — | 0 |

Store classes confirm the chunk granularity: **7488-class × 34**
(MNBT 8192 floored to the fp8 page multiple 832 = 9×832) + 2496×2
tail — the 7488-token chunk step is a ~4.0–5.3 s engine-wide decode
blackout.

## Consequence — the dtype comparison INVERTS at equal posture

| Posture | U32k p95 | U64k p95 |
|---|---|---|
| fp8 TRUE default (B1, this leg) | **3986 ms** | **5295 ms** |
| TQ4 default (cert6 gate) | 3227 ms | 3780 ms |
| fp8 thr2048 profile (historical "baseline") | ~1273–1304 ms | ~1574–1604 ms |

The TurboQuant dtype **improves** bystander stalls at equal posture
(narrower packed-KV reads make each big chunk step ~20–40 % faster);
the docs' "TQ costs ~2.5× vs fp8" was a profile-mismatch artifact
(TQ legs ran true default; the fp8 comparator numbers carried
thr2048). Both docs and the disposition math re-baseline on this leg.

Redaction: serve_full.log 8×rig + 11×node masks, +125 bytes exact
(raw-ref grep 0) — `results/REDACTION-2026-11-02-p615.md`. W14
restored by trap.
