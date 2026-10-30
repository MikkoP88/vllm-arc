# Performance — vllm-arc, qwen3.8-27b-fp8 on dual Intel Arc Pro B70

Measured performance of the certified posture (standalone numbers; test
definitions and hardware in [`TESTING-arc.md`](TESTING-arc.md)). All
certified faces: first-traffic fresh boot, timers off, XPU graphs on,
TP2, greedy temp 0. **Recommended MTP depth = k3** (r46t campaign,
below); the certified-greedy history at k4/k5 is retained row-labeled —
k5 is dropped from the recommendation; its evidence is
superseded, not retracted.

## Headline faces (certified greedy battery, temp 0)

| Face | k3 (recommended, r46t) | k4 (r42) | k5 (dropped, r43) |
|---|---|---|---|
| agg4x256 steady tokens/s | 299.86 | 297.65 / 297.94 | 303.48 / 305.42 |
| solo200 tokens/s (incl. TTFT) | 84.46 / 84.53 | 85.01 / 85.16 | 86.39–87.13 (×4) |
| greedy acceptance | 85.6% | ~87% | 89.2% (pos-1 0.8919) |

k3 and k4 tie within ±0.7 % on the certified face; k5 is faster there
(+1.9–2.5 % agg, +1.5–2.4 % solo, on 89.2 % greedy acceptance) — but
that greedy-era edge inverts under real-world sampling, and the faces
below decide the depth: k3.

| Face (posture-level) | Value |
|---|---|
| TTFT t1 8k / 16k / 32k (ms) | 3987 / 7238 / 15534 (k5 leg r43); 3445 / 7252 / 15552 (k4 leg r42 — 8k steady face) |
| **TTFT t1 64k (ms)** | **40,926** (concurrent-load run — conservative upper bound) |
| prefill rate at 8k face | 2,308 tok/s (7,950 tok / 3.445 s) |
| warm t2 8k / 16k / 32k (ms) | 848 / 675 / 1178 |
| **warm-hit (prefix re-hit)** | **740 ms** (face limit 750 ms) |
| survival flip (first request after boot) | HTTP 200, 6.09 s |
| solo step time @ c1 (200 tokens / 2.30 s) | ~48.7 ms/step (k5 leg) |
| agg4x256 first run after boot (k3) | warmup cost on run #1 (93.7 vs 299.9 steady) — first-traffic face, gone by run #2 |

Notes: the 8k t1 TTFT is first-cell-of-the-ladder (warmup-sensitive);
the k4 leg's 3445 ms shows the steady 8k face. The 64k cell was taken
while the 8 h soak battery was cycling (conservative — a quiet-window
run would be equal or faster). The 8k/16k/32k ladder and 64k face were
measured on the certified image (r42/r43 legs and r46 leg respectively).

## Real-world faces (r46t campaign — recommended posture k3)

Full methodology and evidence: `results/2026-10-30-p5-r46t-temps/`
(VERDICT + raw batteries + acceptance counters). Two request regimes,
both on the run7n battery (agg4x256 ×2 + solo200 ×2):

**A. xhigh thinking + real-world sampling** (temperature sweep, top_p
0.95 / top_k 20 / min_p 0 / presence_penalty 0 / repetition_penalty 1,
`enable_thinking: true, reasoning_effort: "xhigh"`). agg tok/s = 2-run
mean; steady = run-2 mean (run #1 after a fresh boot carries first-
battery warmup, heaviest on k3):

| temp | k3 agg | k4 agg | k3 solo | k4 solo | acceptance k3 / k4 |
|---|---|---|---|---|---|
| 0.2 | 174.9 * | 201.7 | 63.9 | 64.6 | 57.3 % / 50.2 % |
| 0.4 | 208.8 | 184.4 | 63.0 | 60.4 | 57.1 % / 45.9 % |
| 0.6 | 202.8 | 193.6 | 61.3 | 62.3 | 53.1 % / 46.6 % |
| 0.8 | 195.8 | 184.9 | 61.4 | 59.1 | 52.3 % / 43.5 % |
| 1.0 | 186.2 | 182.6 | 61.1 | 54.3 | 49.5 % / 43.2 % |
| mean | **193.7** (steady 205.4) | 189.4 (steady 194.2) | **62.1** | 60.2 | — |

\* run #1 = 137.2 (warmup), run #2 = 212.6. k5 measured 185.8 agg
mean before being dropped (evidence retained in the results dir).

**B. alt sampling, non-thinking** (temperature 0.7 / top_p 0.80 /
top_k 20 / min_p 0 / presence_penalty 1.5 / repetition_penalty 1.0):

| leg | agg #1 / #2 | solo #1 / #2 | acceptance |
|---|---|---|---|
| k3 | 170.6 * / 254.9 | 76.9 / 92.2 | 88.5 % |
| k4 | 236.9 / 252.6 | 85.3 / 104.3 | 82.6 % |

\* warmup-contaminated run #1. Near-deterministic continuation keeps
acceptance high — the one regime where k4's deeper draft pays (solo
104.3 vs 92.2); steady agg is a tie.

**Non-MTP baseline** (serve line minus `--speculative-config`): t1.0
xhigh thinking agg 118.8 / solo 33.5; alt face steady agg 124.2 /
solo 33.4. **MTP value on measured faces: k3 = 1.57× (xhigh t1.0) and
2.05× (alt steady) aggregate over no-spec.**

Mechanism (measured across the sweep): speculative acceptance falls as
temperature rises, and falls faster with deeper drafts (k3 49.5 % vs
k5 33.5 % at t1.0 xhigh; k3 85.6 % at greedy t0). Depth pays only at
high acceptance — real-world sampled thinking sits below the
crossover, which is why k3 is the recommendation and k5's greedy-era
win inverts under sampling.



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

Acceptance figures split by sampling condition: the headline faces
above (pos-1 0.8919, spec-gate counters) are **greedy (temperature 0)**
— k5-leg values; k3's greedy acceptance is 85.6 % (r46t). Matrix-cell
and 64k-cell acceptance below are **server-default sampling**
(`vllm bench serve` 0.30 does not force greedy — see
TESTING-arc.md suite 5). TTFT/t1 faces are sampling-insensitive.
Real-world (sampled, xhigh-thinking) acceptance by depth and
temperature: see the r46t table above.

Acceptance 87.43%, mean length 5.37 tokens/step, 218 drafts / 1090
draft tokens / 953 accepted. Per-position acceptance:
96.79 / 96.79 / 96.79 / 96.33 / 50.46 (positions 0–4) — position 4 is
the k5 draft tail; see the 12-cell table above for per-cell positions.

## Stability

- run7n ×2 per leg agree within 1.5% (steady-state).
- Prefix battery phases byte-identical (deterministic corpus, temp 0).
- **8 h soak (r46) — GREEN**: 46 cycles over a 29,156 s window (each
  cycle: run7n → longctx → prefix/churn → integrity probe), **zero
  error blocks, integrity probes 46/46 PASS, final golden 5/5
  byte-identical (zero drift), speculative pos-1 rate 0.8919 steady to
  the last cycle, health 200 end-state**. Evidence:
  `results/2026-10-30-p5-r46-soak/` (VERDICT + soak log + summary).
