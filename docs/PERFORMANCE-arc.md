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

## Prefill anatomy (p66 — analysis of record, 2026-10-31)

Solo TTFT sweep, fresh boot each, nolmc, graph, k3, greedy, unique seeds
(r66a; three faces × four boots — `--long-prefill-token-threshold`
none/8192/4096/2048):

| Face | prompt tok | thr=none | thr8192 | thr4096 | thr2048 | thr2048 Δ |
|---|---|---|---|---|---|---|
| U8k | 8158 | 3.74 s (2178 t/s) | 3.75 s | 4.04 s | 7.94 s | **+112%** |
| U32k | 32553 | 17.34 s (1878 t/s) | 17.35 s | 18.24 s | 21.49 s | +23.9% |
| U64k | 65303 | 39.20 s (1666 t/s) | 39.23 s | 42.38 s | 49.75 s | +26.9% |

thr=none and thr8192 are the same path (≤ 0.03 s apart — at MNBT 8192
the threshold IS the default chunked size). The rate falls with context
(2178 → 1666 t/s, attention KV growth) even before any chunking effect.
thr2048's cost is two mechanisms stacked: more chunk boundaries (8158 tok
= 5 × 1664-token chunks instead of 2 × 7488) **and** small-M device
inefficiency (U8k rate halves: 2178 → 1028 t/s).

**Where the time goes (r66b, per-chunked-step decomposition on the live
V2 runner)**: scheduler 0.3 ms, host prep ≤ 10 ms, TP2 allreduce ~28 ms,
sampling ≤ 107 ms — and the rest is the **device execution of the chunk
itself**, observed as a queue-drain wait inside the next step's
mamba/GDN `preprocess_state` launch window (780–1018 ms of an ~0.9 s
step at 1664-token chunks; XPU eager forward is async, so the enqueue
looks flat at ~127 ms and the device time surfaces one step later).
Prefill is **device-bound at ~34 % MFU** (1664-token chunks) to ~40 %
(7488-token chunks, 2.2–2.4 tok/ms) against the measured 294 TFLOPS
fleet fp16 GEMM ceiling (147–150 TFLOPS/card, 256 CUs, devprobe r66a).
The gap is attention over growing KV + W8A16 dequant + non-GEMM ops +
per-layer launch/dispatch — kernel territory, not host glue.

**Improvement ladder, closed out**:

- thr table (knob) — documented above; the p65 long-context profile
  (`--long-prefill-token-threshold 2048` + deferral) is the one place
  thr2048 is correct, for LMC-side reasons (see CONFIGURATION-arc).
- **MNBT 16384 (knob) — A/B'd, NOT adopted** (r66c): U32k −9.7 %
  (17.33→15.65 s), U64k −4.1 %, confirming the small-M diagnosis — but
  U8k +9.9 % (3.74→4.11 s) and the KV pool pays 18 % (405,491→332,629
  tokens; max concurrency 1.55x→1.27x) from the doubled prefill
  activation workspace. Never combine with thr2048 (the threshold pins
  chunk size — only the KV loss remains). Conditional long-prompt-only
  lever; the certified default stays 8192.
- **Host glue — closed as empty** (r66b): nothing material to cut.
- **Structural (raise device MFU)** — the only remaining large lever
  (prefill compile NONE→PIECEWISE, attention kernel work); deferred,
  A/B-gated, out of P6.6 scope. → **P6.7 ran it; see the next section.**

Evidence: `results/2026-10-31-p66-r66a-prefill-capture/` (curve + device
probe), `results/2026-10-31-p66-r66b-step-decomposition/` (step
decomposition + the engine-map correction + probe-instrumentation laws),
`results/2026-10-31-p66-r66c-mnbt16k/` (MNBT A/B).

## Prefill structural rungs (p67 — compile REVAL + op-class capture, 2026-10-31)

Zero engine-code change; cert4 remains the certified image (no cert5).
Evidence: `results/2026-10-31-p67-r67a-compile-ab/` (VERDICT.md,
r67b_capture_analysis.md, P67_LADDER_VERDICT.md, raw rank traces).

**Compile REVAL (r67a, Appendix H law #1 on 0.30) — REJECTED.**
`mode=vllm_compile` × `cudagraph_mode=FULL_AND_PIECEWISE` (the only legal
combo per the capture law), same-session A/B on the cert4 image: MTP mean
acceptance **3.40/3.57 → 2.21/2.32 (−35 %)**, agg4x256 steady 300.48 →
182.04 t/s and solo200 84.49 → 51.66 t/s (both −39 % — pure arithmetic of
fewer accepted tokens per verify step), prefill faces FLAT (U8k 3.75 vs
3.74 s, U32k 17.08 vs 17.33, U64k 39.20 vs 39.19). KV pool +5.3 %
(426,874 tokens) does not compensate. Boot cost 460 s vs 265 s stock;
3 fork-added jit_monitor WARNs (inference-time Triton JIT of spec kernels).
`mode` stays NONE. Config law banked en route: with MTP k3 and
max_num_seqs 64, every captured size must satisfy the GDN decode-only cap
`max_num_seqs×(k+1)` = 256 tokens (`gdn_attn.py:120`) — capture ladders
must top at 256 under FULL_AND_PIECEWISE (stock ladder headroom to 320
boot-dies at the first piecewise capture).

**Op-class device anatomy (r67b, torch-profiler XPU trace of the U32k
face, rank0, 15.38 s kernel time, 71.9 % busy):**

| Slice | Share | Verified launch site |
|---|---|---|
| GEMM | 39.7 % | `_xpu_C::fp8_gemm_w8a16` (the p44 W8A16 route — fastest on-image per r40) |
| TP2 allreduce (ring+ReduceCopy+barrier) | ~30.4 % | `c10d::allreduce_` |
| FA-layer attention (16 layers, DENSE) | 20.3 % | `_vllm_fa2_C::varlen_fwd` (~31 ms × 102) |
| GDN chunk-scan (48 layers) | 4.7 % | `gdn::ChunkFwdO/tiled/WU/A/InverseOpt` ×336 |
| norms + act + rope + elementwise | ~5 % | — |
| indexer/gather (sparse machinery) | 0.1 % | — |

Consequences: the radiance QSA-splitk whale does not exist on stock 0.30
(the model runs dense FA — no sparse indexer); **direct-fp8 GEMM rung
closed negative** (on-image `XPUFp8BlockScaledMMKernel`/`_xpu_C.fp8_gemm`
exists but r40 measured it 2.2–2.4× slower than the deployed w8a16 at
prefill Ms — the GEMM whale already runs the fastest on-image kernel);
**ESIMD FA-kernel rung re-scoped** to the dense varlen site (GQA 24q/4kv,
head_dim 256, fp16, paged fp8 KV PAGE 1664; est. 5–7 % compute MFU today;
ceiling −8…−12 % U32k TTFT if 1.5–2×) — open as a kernel project; and the
**TP2 allreduce family (~30 %) is the largest unattacked slice** (not in
the owner's ladder; recorded as the top recommendation beyond it).

## Stability

- run7n ×2 per leg agree within 1.5% (steady-state).
- Prefix battery phases byte-identical (deterministic corpus, temp 0).
- **8 h soak (r46) — GREEN**: 46 cycles over a 29,156 s window (each
  cycle: run7n → longctx → prefix/churn → integrity probe), **zero
  error blocks, integrity probes 46/46 PASS, final golden 5/5
  byte-identical (zero drift), speculative pos-1 rate 0.8919 steady to
  the last cycle, health 200 end-state**. Evidence:
  `results/2026-10-30-p5-r46-soak/` (VERDICT + soak log + summary).
