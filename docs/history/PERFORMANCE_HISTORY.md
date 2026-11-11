# PERFORMANCE history — superseded measurement eras

Archived from `docs/PERFORMANCE-arc.md` (and the README's provenance
narratives) in the P6.38d docs restructure (2026-11-11). Append-only
archive of superseded measurement eras; the living doc
([`../PERFORMANCE-arc.md`](../PERFORMANCE-arc.md)) carries the
current-cert record only. Authoritative narratives: the cited
`results/` VERDICTs.

## Pre-cert4 measurement history (moved from PERFORMANCE-arc.md)

**Depth-selection faces (cert-era images — the k3/k4/k5 greedy record).**
Measured on the original-cert/cert2-era engines (r42 k4 leg, r43 k5
leg, r46t k3 on the p46 image); the cert4-native reproduction
(294.7–300.5 / 83.7–84.9 across five boots, sweep acceptance within
~1 pp) superseded these as the record of the day. Depth law unchanged:
**k3**.

| Face | k3 (recommended, r46t) | k4 (r42) | k5 (dropped, r43) |
|---|---|---|---|
| agg4x256 steady tokens/s | 299.86 | 297.65 / 297.94 | 303.48 / 305.42 |
| solo200 tokens/s (incl. TTFT) | 84.46 / 84.53 | 85.01 / 85.16 | 86.39–87.13 (×4) |
| greedy acceptance | 85.6% | ~87% | 89.2% (pos-1 0.8919) |

Cert-era posture-level faces from the same legs: TTFT t1 8k/16k/32k
3987/7238/15534 ms (k5 leg) and 3445/7252/15552 ms (k4 leg); 64k t1
40,926 ms (concurrent-load run — conservative); 8k prefill rate 2,308
tok/s; warm t2 848/675/1178 ms (prewarm-v7 corpus); warm-hit 740 ms;
survival flip 6.09 s; k5-leg solo step ~48.7 ms.

**Real-world faces — the k3 decision (r46t campaign, p46 image,
2026-10-30).** Depth selection under real-world sampling picked **k3**
(full methodology: `results/2026-10-30-p5-r46t-temps/`). Regime A —
xhigh thinking + real-world sampling (top_p 0.95 / top_k 20 / min_p 0
/ presence_penalty 0 / repetition_penalty 1), agg = 2-run mean:

| temp | k3 agg | k4 agg | k3 solo | k4 solo | acceptance k3 / k4 |
|---|---|---|---|---|---|
| 0.2 | 174.9 * | 201.7 | 63.9 | 64.6 | 57.3 % / 50.2 % |
| 0.4 | 208.8 | 184.4 | 63.0 | 60.4 | 57.1 % / 45.9 % |
| 0.6 | 202.8 | 193.6 | 61.3 | 62.3 | 53.1 % / 46.6 % |
| 0.8 | 195.8 | 184.9 | 61.4 | 59.1 | 52.3 % / 43.5 % |
| 1.0 | 186.2 | 182.6 | 61.1 | 54.3 | 49.5 % / 43.2 % |
| mean | **193.7** (steady 205.4) | 189.4 (steady 194.2) | **62.1** | 60.2 | — |

\* run #1 = 137.2 (warmup), run #2 = 212.6; k5 measured 185.8 agg mean
before being dropped. Regime B — alt sampling (t0.7 / top_p 0.80 /
top_k 20 / presence_penalty 1.5), non-thinking: k3 agg 254.9 steady /
solo 92.2, acceptance 88.5 %; k4 agg 252.6 / solo 104.3, acceptance
82.6 % — the one regime where the deeper draft pays. Non-MTP baseline:
t1.0 xhigh agg 118.8 / solo 33.5; alt steady agg 124.2 / solo 33.4 —
MTP value k3 = 1.57× (xhigh t1.0) and 2.05× (alt steady). Mechanism:
acceptance falls as temperature rises, faster with deeper drafts
(k3 49.5 % vs k5 33.5 % at t1.0 xhigh) — depth pays only at high
acceptance, so k5's greedy-era win inverts under sampling.

**12-cell load matrix (r44 — cert-era image; the load-shape record).**
`vllm bench serve`, random dataset, output 512, ignore-eos, 12/12 cells
zero-error (raw cells `results/2026-10-30-p4-r44-matrix-certified/`):

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

**Speculative decoding detail (64k cell, r46 — cert-era).** Acceptance
87.43 %, mean length 5.37 tokens/step (server-default sampling);
per-position 96.79 / 96.79 / 96.79 / 96.33 / 50.46 (positions 0–4,
position 4 = the k5 draft tail).

**r46 8 h soak (cert-era — the long-soak record).** GREEN: 46 cycles
over a 29,156 s window (run7n → longctx → prefix/churn → integrity
probe), zero error blocks, integrity 46/46 PASS, final golden 5/5
byte-identical, pos-1 0.8919 steady, health 200 end-state —
`results/2026-10-30-p5-r46-soak/`. Superseded as the soak of record by
the r70 4 h cert4-native soak (56/56 cycles).

**Greedy-lane sampled sweep (r70, cert4-native — superseded by the
probabilistic-lane record).** agg steady t0.2→t1.0: 205.6 / 221.7 /
202.4 / 204.8 / 194.4; alt 264.0–280.9 / solo 72.5–73.1 — parity at
cool temperatures, the probabilistic lane pulls ahead as temperature
rises (t1.0 +19…+29 %) and on alt solo. The r46t→cert4 parity table:
`results/2026-10-31-p70-r70-cert4-comprehensive/R46T_PARITY.md`.

**TQ preset provenance (moved from README.md).** The TurboQuant
corruption era (reported under `turboquant_k8v4`, split Class A/B,
interim refusal, then the three-defect fix) is narrated in the
cert7/cert8 CHANGELOG entries; evidence
`results/2026-11-04-p619-tqverify/`. The champion-face gate chains
(P6.17 campaign on cert6, re-verified at the cert7–cert10 release
gates) are superseded by the current champion faces on cert23.

**Gemma history columns (moved from PERFORMANCE-arc.md).** The L2
mxfp8 record column (probe98 35 s / face8k 35.4 / face32k 297.4 /
face120k 1819.5 s; decode 11.9 solo / 54.6 agg8 tok/s; G1 reason-probe
miss flagged) was measured pre-FA2 on the cert17-class engine and never
re-run — the current gemma record faces live in PERFORMANCE-arc.md and
`docs/reports/`.

## Gemma prefill root-cause close — layers 1–3 (condensed)

- **Layer 1 — hd512 full attention**: the shared triton
  unified-attention 2D path ran gemma-4's 10 full-attention hd512
  layers at 10.81 s/call (0.71 TF/s) at q=8192/k=32768 fp8-KV.
  r108aa/r108ab first read a structural L2-bandwidth wall (~37 GB of
  L2-side reads, BLOCK_M SLM-capped at 64, the cert12 table measured AT
  the ceiling); r108ac's fp16-KV control refuted the bandwidth story
  (10.28 s at 2× bytes — the kernel is slow per K-tile iteration, a
  kernel defect). The engine already ships the fast kernel for the
  shape class (`vllm_xpu_kernels` ESVD FA2); the triton backend never
  dispatched to it. r108ac/r108ad measured the route at production
  geometry: FA2 0.3197 s/call = 34× (k98k ~30×, decode 1.5 ms/call).
  cert18 = the wrapper-level dispatch.
- **Layer 2 — probe-time compile**: the G2 long-prefill freeze's
  ~31 s first-probe JIT compile accepted (cert16 warm-gate v4
  cache-key split); the face120k 300 s cliff was the stock
  execute-model RPC watchdog (`ARC_RPC_TIMEOUT=1200`).
- **Layer 3 — sliding hd256**: after cert18 the dominant residual was
  the 50/60 sliding hd256 layers — triton 0.147–0.158 s/call at q8192
  fp8-KV and k-INDEPENDENT (0.1466 k8192 / 0.1580 k32k / 0.1576 k98k),
  while the same ESVD FA2 kernel windowed runs 0.006–0.009 s = 18–25×
  (decode S8×32k 4.38 → 0.3–0.4 ms = 11–15×). 50 layers × ~0.148 s ≈
  7.4 s per q8192 chunk. cert19 = the second wrapper-level dispatch
  (`patch_arc_xpu_fa2_slide_route.py`, asserts cert18 v3 bytes).
  Numeric guards: kernels agree 1.2e-4…9.8e-4 and both match
  global-softmax table-faithful window-masked GQA torch refs at
  1.3e-4…1.1e-3 prefill / 8e-5 decode; the earlier "ref misses ~4"
  were a harness bug in the TEST refs (per-chunk softmax
  normalization); the softcap theory eliminated separately.

## Prefill anatomy history — p66/p67 (condensed)

Solo TTFT sweep (r66a; `--long-prefill-token-threshold`
none/8192/4096/2048): U8k 3.74 / 3.75 / 4.04 / 7.94 s; U32k 17.34 /
17.35 / 18.24 / 21.49 s; U64k 39.20 / 39.23 / 42.38 / 49.75 s —
thr=none ≡ thr8192 at MNBT 8192; thr2048 costs more chunk boundaries
AND small-M device inefficiency (U8k rate halves 2178 → 1028 t/s).
Step decomposition (r66b): scheduler 0.3 ms, host prep ≤ 10 ms, TP2
allreduce ~28 ms, sampling ≤ 107 ms, the rest = device execution of
the chunk (queue-drain wait inside the next step's GDN preprocess
window) — prefill device-bound at ~34 % MFU (1664-token chunks) to
~40 % (7488) vs the measured 294 TFLOPS fleet fp16 GEMM ceiling.
MNBT 16384 A/B (r66c): U32k −9.7 % / U64k −4.1 % but U8k +9.9 % and
KV pool −18 % (405,491 → 332,629) — conditional long-prompt-only
lever, never adopted. Compile REVAL (r67a): `vllm_compile` collapses
acceptance 3.40/3.57 → 2.21/2.32 (−35 %) and decode −39 %, prefill
flat — `mode` stays none; capture ladders under FULL_AND_PIECEWISE
must top at `max_num_seqs × (k+1)` = 256 (the `gdn_attn.py:120` law).
Op-class anatomy (r67b, U32k rank0): GEMM 39.7 % / AR ~30.4 % / FA
20.3 % / GDN 4.7 % / norms-act ~5 % / indexer 0.1 %.

## TQ sampled-collapse mechanism (P6.15c, condensed)

The pre-fix greedy-lane t1.0 collapse: the engine step stays 41.7 ms
and wall = drafts × 41.7 ms — a collapsed request is BORN collapsed
(MTP draft acceptance dead from round 1, sticky for the request's
lifetime; pos0 rejections on 169/181 rounds in the reproduced 26-tps
request). Trigger = 4-bit-V KV noise breaking draft/target match on
low-margin sampled-trajectory regions; fp8 KV immune at the same
posture, greedy t0 immune on every dtype. Per-request lottery at
10–50 %. Probabilistic drafting (P6.16/P6.17) removed the graded
damage modes and lifts every dtype's t1.0 aggregate; the rare
total-collapse trajectory remains on TQ dtypes only (true
draft/target divergence, Σ min(p,q) ≈ 0.08 at pos0).

## Draft-sampling default-flip ledger

P6.16 shipped the posture opt-in under a pre-registered adoption rule
(default flip needs EVERY dtype ≥ +10 % t1.0 agg — k8v4 is EV-neutral,
fails). P6.17 flipped the default anyway (2026-11-03: probabilistic
ruled the superior lane on every dtype, EV-neutral on k8v4) — the
certified dtype stays fp8 (gains +29 %, collapse-immune); k8v4 users
get an EV-neutral posture change with uniform solo streams (still
strictly better than its ragged greedy face).
