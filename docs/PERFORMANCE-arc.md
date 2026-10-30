# Performance — vllm-arc, qwen3.8-27b-fp8 on dual Intel Arc Pro B70

Measured performance of the certified posture (standalone numbers; test
definitions and hardware in [`TESTING-arc.md`](TESTING-arc.md)). All
faces: first-traffic fresh boot, timers off, MTP k5, XPU graphs on, TP2.

## Headline faces (canonical certified posture, r43/k5)

| Face | Value |
|---|---|
| solo200 tokens/s (incl. TTFT), run ×4 | 86.39 / 86.40 / 86.50 / 87.13 |
| agg4x256 #1 tokens/s, run ×2 | 273.52 / 274.01 |
| agg4x256 #2 tokens/s, run ×2 | 303.48 / 305.42 |
| TTFT t1 8k / 16k / 32k (ms) | 3987 / 7238 / 15534 |
| TTFT t1 8k / 16k / 32k (ms, k4 leg r42) | 3445 / 7252 / 15552 |
| **TTFT t1 64k (ms)** | **40,926** (concurrent-load run — conservative upper bound) |
| prefill rate at 8k face | 2,308 tok/s (7,950 tok / 3.445 s) |
| warm t2 8k / 16k / 32k (ms) | 848 / 675 / 1178 |
| **warm-hit (prefix re-hit)** | **740 ms** (face limit 750 ms) |
| survival flip (first request after boot) | HTTP 200, 6.09 s |
| solo step time @ c1 (200 tokens / 2.30 s) | ~48.7 ms/step (k5) |
| speculative acceptance | 89.2% aggregate, 5.46 mean tokens/step, pos-1 rate 0.8919 |

Notes: the 8k t1 TTFT is first-cell-of-the-ladder (warmup-sensitive);
the k4 leg's 3445 ms shows the steady 8k face. The 64k cell was taken
while the 8 h soak battery was cycling (conservative — a quiet-window
run would be equal or faster). The 8k/16k/32k ladder and 64k face were
measured on the certified image (r42/r43 legs and r46 leg respectively).

## 12-cell load matrix (r44, certified posture)

`vllm bench serve`, random dataset, output 512, ignore-eos, 12/12 cells
zero-error. TTFT/TPOT/ITL in ms:

| Cell | TTFT | TPOT | ITL | req/s | out tok/s | Acc % | Acc len |
|---|---|---|---|---|---|---|---|
| i1024-c1 | 583.67 | 8.60 | 46.77 | 0.20 | 102.79 | 89.15 | 5.46 |
| i1024-c4 | 1985.32 | 11.07 | 64.47 | 0.52 | 264.69 | 96.92 | 5.85 |
| i1024-c16 | 6237.01 | 28.97 | 126.85 | 0.64 | 325.55 | 67.96 | 4.40 |
| i16384-c1 | 7380.80 | 9.13 | 49.11 | 0.08 | 42.50 | 88.00 | 5.40 |
| i16384-c4 | 13263.11 | 31.51 | 160.63 | 0.13 | 66.43 | 82.44 | 5.12 |
| i16384-c16 | 46421.90 | 111.19 | 523.98 | 0.14 | 72.38 | 74.70 | 4.73 |
| i32768-c1 | 16083.60 | 10.83 | 52.19 | 0.05 | 23.68 | 76.42 | 4.82 |
| i32768-c4 | 34286.64 | 44.64 | 210.74 | 0.07 | 34.47 | 74.69 | 4.73 |
| i32768-c16 | 97008.14 | 161.02 | 708.16 | 0.07 | 34.84 | 68.28 | 4.41 |
| i131072-c1 | 101094.58 | 20.71 | 66.15 | 0.01 | 4.58 | 43.73 | 3.19 |
| i131072-c4 | 162169.64 | 162.68 | 854.78 | 0.01 | 6.24 | 85.76 | 5.29 |
| i131072-c16 | 903979.67 | 184.16 | 843.54 | 0.01 | 4.78 | 71.91 | 4.60 |

Raw cells: `results/2026-10-30-p4-r44-matrix-certified/cell-*.txt`.

## Speculative decoding detail (64k cell, r46)

Acceptance 87.43%, mean length 5.37 tokens/step, 218 drafts / 1090
draft tokens / 953 accepted. Per-position acceptance:
96.79 / 96.79 / 96.79 / 96.33 / 50.46 (positions 0–4) — position 4 is
the k5 draft tail; see the 12-cell table above for per-cell positions.

## Stability

- run7n ×2 per leg agree within 1.5% (steady-state).
- Prefix battery phases byte-identical (deterministic corpus, temp 0).
- 8 h soak (r46): cycles of run7n → longctx → prefix/churn → integrity
  probe; zero-error / zero-drift result banked at close-out in
  `results/2026-10-30-p5-r46-soak/` (VERDICT + cycle log).
