# Qwen3.8-27B on 2x Intel Arc Pro B70 — FP8, MTP k3

| Field | Value |
|---|---|
| Organization | Qwen (Alibaba) |
| Parameters | 27B dense hybrid-GDN (+MTP head) |
| Precision | FP8 (fp8 checkpoint, compressed-tensors) |
| Accelerators | 2x Intel Arc Pro B70 (BMG-G31, 32 GB each, 64 GB total), TP2 |
| Host RAM | 64 GB (close-out F posture: --shm-size 16g, 16 GB LMCache CPU tier) |
| Engine | vLLM v0.30.0 — vllm-arc fork, image v0.30.0-b70-cert28, XPU graphs |
| KV cache dtype | fp8_e4m3 |
| Speculative decoding | MTP k3, probabilistic drafting |
| Max model len | 262144 |
| KV pool (this boot) | 398,363 tokens (r240b posture) |
| Date | 2026-11-24 |
| Cell outcome | FACES_OK |

## Executive summary

Infrastructure decisions require real performance data. This report measures user-facing performance — how many concurrent requests the configuration supports at a given context length before streaming speed or first-token wait degrade below the user-experience bands.

This benchmark evaluates **Qwen3.8-27B** (Qwen (Alibaba); 27B parameters, dense hybrid-GDN) running **FP8 (fp8 checkpoint, compressed-tensors)** on **2x Intel Arc Pro B70** (32 GB each, TP2) — the certified incumbent posture.

Test parameters: context lengths 1K–128K tokens; concurrency 1–8; 600 output tokens per request; unique seeded-random prompts (cold prefill, no prefix-cache hits); MTP k3 (probabilistic drafting); KV cache `fp8_e4m3`; temperature 0.0 with paired 1.0 faces.

| Metric | Result |
|---|---|
| Generation speed, single request | 93.8 tok/s (1K) → 74.4 tok/s (128K) |
| TTFT, single request | 0.84 (1K) → 105.8 (128K) |
| Peak per-user speed (solo) | **93.8 tok/s** @ 1K ctx (**very fast**) |
| Peak system throughput | **343 tok/s** @ 8 concurrent, 1K ctx — ≈ 1.23 M tokens/hour |
| Throughput scaling (1 → 8, 1K ctx) | 3.6x (45% of ideal linear) |
| Interactive chat capacity (8K) | ~3 concurrent requests (estimated between the measured 3 and 4 points) |
| Long-context assistant (RAG) capacity (32K) | thresholds exceeded at concurrency 1 - not recommended |
| Document processing (batch) capacity (128K) | 3 concurrent requests |
| Success rate | **100.0%** across 92 requests (34 faces green) |

> Every concurrent request models one user session holding a full generation (prompt prefill + 600 streamed tokens), barrier-started together — the worst-case burst. Interactive sessions idle between turns, so sustained user counts run well below request concurrency. Full definitions: [METHODOLOGY.md](METHODOLOGY.md).

## Use case guidance

The table maps this configuration to common deployment scenarios. Capacity limits are where TTFT or streaming speed fall below the accepted thresholds for a comfortable user experience (the band edges of the UX classification below).

| Use case | Context | Thresholds | Capacity |
|---|---|---|---|
| Interactive chat | 8K | TTFT < 10s · speed >= 15 tok/s | ~3 concurrent requests (estimated between the measured 3 and 4 points) |
| Long-context assistant (RAG) | 32K | TTFT < 10s · speed >= 15 tok/s | thresholds exceeded at concurrency 1 - not recommended |
| Document processing (batch) | 128K | speed >= 15 tok/s | 3 concurrent requests |

The limits are conservative: past them the system keeps serving with slower response times that may still be acceptable for your workload.

## System throughput

Aggregate token generation across all concurrent requests (output tokens only; prefill excluded). Measured as n × 600 / wall — the wall includes the barrier start and the slowest-user tail, so it is the honest saturated-burst number.

| Condition | Aggregate throughput |
|---|---|
| Peak (1K, 8 req) | 343 tok/s |
| 8K ctx, 4 req | 119 tok/s |
| 8K ctx, 8 req | 127 tok/s |
| 32K ctx, 4 req | 34 tok/s |
| 32K ctx, 8 req | 34 tok/s |

At the peak point this configuration produces approximately **1.23 M tokens/hour** — the number that matters for batch workloads (document processing, synthetic-data generation, offline analysis).

![Aggregate throughput vs context length, one line per concurrency — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_agg_ctx.svg)

<i>Aggregate throughput vs context length, one line per concurrency — FP8, MTP k3</i>

*Interpretation:* throughput peaks at 343 tok/s (8 requests, 1K); at 128K the heaviest measured load (3 requests) yields 6 tok/s (2% of peak) — long-context KV residency shrinks the resident batch. Batch workloads should batch short and parallel; long-context work is bound by KV residency, not compute.

## Per-user generation speed

The streaming rate each user experiences. As concurrency rises, per-user speed falls because the accelerators are shared; with MTP, multiple accepted tokens ride one stream chunk (spec-aware accounting).

| Context | Solo (tok/s) | class | Best @ conc (tok/s) | @ conc 8 (tok/s) | class |
|---|---|---|---|---|---|
| 1024 (1K) | 93.8 | **very fast** | 99.3 | 59.7 | **very fast** |
| 8192 (8K) | 66.8 | **very fast** | 97.2 | 36.5 | fast |
| 32768 (32K) | 91.4 | **very fast** | 92.1 | 14.7 | slow |
| 65536 (64K) | 84.9 | **very fast** | 81.1 | - | - |
| 131072 (128K) | 74.4 | **very fast** | 76.0 | - | - |

### How fast is this?

| Speed | User experience |
|---|---|
| < 15 tok/s | Slow - may be slower than reading speed |
| 15-25 tok/s | Acceptable - keeps pace with reading |
| 25-50 tok/s | Fast - exceeds reading speed |
| >= 50 tok/s | Very fast - text appears nearly instantly |

Solo speed spans **66.8–93.8 tok/s** (8K to 1K context); at concurrency 8 the floor is 14.7 tok/s (32K ctx).

![Per-user decode speed vs context length (solo) — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_decode.svg)

<i>Per-user decode speed vs context length (solo) — FP8, MTP k3</i>

![Per-user speed vs context length, one line per concurrency — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_speed_ctx.svg)

<i>Per-user speed vs context length, one line per concurrency — FP8, MTP k3</i>

![Per-user speed range: the band every deployment concurrency lands in — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_range_speed.svg)

<i>Per-user speed range: the band every deployment concurrency lands in — FP8, MTP k3</i>

*Interpretation:* at 128K a single user streams 76.0 tok/s while 3 requests share down to 25.1 tok/s (33% of solo); at 8K the spread is 36.5-97.2 tok/s — every concurrency you deploy lands between the band edges; the bands below classify the streaming feel.

## Time to first token

Time from request submission to the first streamed token — the primary responsiveness metric. Client-side on the rig loopback, it decomposes into queue wait (rising with concurrency) and prefill (rising with context); the technical analysis below separates the two.

| Context | Solo (s) | class | @ conc 8 mean (s) | @ conc 8 worst (s) | class |
|---|---|---|---|---|---|
| 1024 (1K) | 0.84 | responsive | 1.82 | 2.17 | noticeable |
| 8192 (8K) | 4.40 | noticeable | 16.7 | 25.9 | frustrating |
| 32768 (32K) | 16.3 | frustrating | 72.8 | 124.5 | frustrating |
| 65536 (64K) | 39.0 | frustrating | - | - | - |
| 131072 (128K) | 105.8 | frustrating | - | - | - |

### How responsive is this?

| TTFT | User experience |
|---|---|
| < 0.5 s | Feels instant |
| 0.5-2 s | Feels responsive |
| 2-5 s | Noticeable but still acceptable |
| 5-10 s | Feels slow; acceptable at higher context |
| > 10 s | Can be frustrating; users may retry or abandon |

> **Caching note.** Every face uses a unique seeded-random prompt — a deliberate cold prefill, the worst case. The certified posture runs prefix caching (LMCache): in production, incremental conversations pay TTFT only for new tokens, which is dramatically faster than these numbers.

![TTFT vs context length (solo) — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_ttft.svg)

<i>TTFT vs context length (solo) — FP8, MTP k3</i>

![TTFT vs context length, one line per concurrency — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_ttft_ctx.svg)

<i>TTFT vs context length, one line per concurrency — FP8, MTP k3</i>

![TTFT range: single request (bottom) to worst user at max concurrency (top) — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_range_ttft.svg)

<i>TTFT range: single request (bottom) to worst user at max concurrency (top) — FP8, MTP k3</i>

*Interpretation:* at 128K the first token takes 98.2 solo but up to 307.1 for the worst user under 3 requests — 68% of the loaded wait is queue+prefill stagger, the sharing cost the capacity tables price below.

## Capacity analysis

### Interactive chat (8K context)

Conversational assistant sessions: system prompt plus a few turns fit 8K. Both first-token wait and streaming speed matter. Thresholds: TTFT < 10s · speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 | @ 4 | @ 8 |
|---|---|---|---|---|---|
| TTFT worst user (s) | 3.06 | 6.16 | 9.56 | 12.6 *(exceeded)* | 25.9 *(exceeded)* |
| Per-user mean speed (tok/s) | 97.2 | 75.8 | 68.9 | 61.0 | 36.5 |

![Interactive chat (8K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/qwen3.8-27b-fp8-k3_uc_chat8_speed.svg)

<i>Interactive chat (8K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

![Interactive chat (8K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar](charts/qwen3.8-27b-fp8-k3_uc_chat8_ttft.svg)

<i>Interactive chat (8K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar</i>

**Capacity limit: ~3 concurrent requests (estimated between the measured 3 and 4 points)**

### Long-context assistant (RAG) (32K context)

Retrieval-augmented / large-prompt assistants at 32K context. Thresholds: TTFT < 10s · speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 | @ 4 | @ 8 |
|---|---|---|---|---|---|
| TTFT worst user (s) | 15.3 *(exceeded)* | 30.5 *(exceeded)* | 45.5 *(exceeded)* | 60.8 *(exceeded)* | 124.5 *(exceeded)* |
| Per-user mean speed (tok/s) | 92.1 | 58.2 | 41.0 | 30.7 | 14.7 *(below)* |

![Long-context assistant (RAG) (32K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/qwen3.8-27b-fp8-k3_uc_assist32_speed.svg)

<i>Long-context assistant (RAG) (32K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

![Long-context assistant (RAG) (32K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar](charts/qwen3.8-27b-fp8-k3_uc_assist32_ttft.svg)

<i>Long-context assistant (RAG) (32K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar</i>

**Capacity limit: thresholds exceeded at concurrency 1 - not recommended**

### Document processing (batch) (128K context)

Offline summarization / extraction at the largest measured context. Turnaround matters, not first-token wait: the bar is streaming speed (and total throughput). Thresholds: speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 |
|---|---|---|---|
| TTFT worst user (s) | 98.2 | 197.9 | 307.1 |
| Per-user mean speed (tok/s) | 76.0 | 25.8 | 25.1 |

![Document processing (batch) (128K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/qwen3.8-27b-fp8-k3_uc_docmax_speed.svg)

<i>Document processing (batch) (128K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

**Capacity limit: 3 concurrent requests**

### KV pool (the whole-pool bound)

This boot exposed a KV pool of **398,363 tokens** (r240b posture). A session occupies ctx + 664 (output + scheduler slack), so the pool holds:

| Context | Pool capacity (sessions) | Measured green @ conc | c8 fully resident? |
|---|---|---|---|
| 1024 (1K) | 235 | 8 (green) | yes |
| 8192 (8K) | 44 | 8 (green) | yes |
| 32768 (32K) | 11 | 8 (green) | yes |
| 65536 (64K) | 6 | 6 (green) | marginal |
| 131072 (128K) | 3 | 3 (green) | no |

*Sessions are the whole-pool bound; `--max-num-seqs 64` bounds the running batch above it. TurboQuant dtypes multiply the pool up to 3.68x: see the [KV-cache dtype report](qwen3.8-27b-fp8-kv-cache-dtype.md).*

## Technical analysis

### Queue + prefill decomposition (derived)

At concurrency 1 the queue is empty, so solo TTFT is pure prefill. Under the barrier burst the added first-token wait — `ttft_mean(n) − ttft_mean(1)` — is the queue+prefill-share stagger the Nth user pays. Derived client-side; engine-internal queueing is not exported by this engine version. The x-axis keeps the canonical rung ladder up to the highest measured context — an empty slot marks a rung the concurrency measurements skip (64K: only the solo face measured it on qwen; unmeasured entirely on the gemma boots); no stagger exists there.

| Context | added wait @ n2 | @ n4 | @ n8 |
|---|---|---|---|
| 1024 (1K) | +0.2 s | +0.5 s | +1.5 s |
| 8192 (8K) | +1.6 s | +6.1 s | +13.6 s |
| 32768 (32K) | +10.2 s | +26.0 s | +57.5 s |
| 65536 (64K) | +20.7 s | +58.5 s | - |
| 131072 (128K) | +52.2 s | - | - |

![Added first-token wait vs context length, one line per concurrency (derived queue+prefill stagger) — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_queue_ctx.svg)

<i>Added first-token wait vs context length, one line per concurrency (derived queue+prefill stagger) — FP8, MTP k3</i>

*Interpretation:* the loaded-request stagger grows from +1.5 s at 1K to +57.5 s at 32K (8 requests) — the context-carrying requests serialize behind each other's prefill.

### Per-user prefill speed

Rate at which input context is processed before the first token. Client-observed (ctx / TTFT at queue ~0); the engine-side device rate is higher — the client view includes chunk drain and the decode-transition tail. The single-request line spans **every measured context** (sweep plus solo faces — the sweep skipped 64K); loaded lines cover the contexts the concurrency sweep measured.

| Context | prefill @ n1 (tok/s) | @ n4 | @ n8 |
|---|---|---|---|
| 1024 (1K) | 3585 | 1312 | 564 |
| 8192 (8K) | 2674 | 892 | 490 |
| 32768 (32K) | 2145 | 794 | 450 |
| 65536 (64K) | 1800 | 690 | - |
| 131072 (128K) | 1334 | - | - |

![Client-observed prefill rate vs context length, one line per concurrency — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_prefill.svg)

<i>Client-observed prefill rate vs context length, one line per concurrency — FP8, MTP k3</i>

*Interpretation:* prefill peaks at 3,585 tok/s (1K) and eases as context grows — chunked prefill batches more work per pass at length, trading first-token wait for decode-window protection.

### Inter-token latency

Time between consecutive streamed tokens — the smoothness of streaming. Solo p50 spans **39.1 ms** (1K) to 50.9 ms; the worst observed p99 is **69 ms** (64K). With MTP one chunk can carry multiple accepted tokens, so the perceived stream is smoother than the raw chunk gap.

![Inter-token latency vs context length (solo) — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_itl.svg)

<i>Inter-token latency vs context length (solo) — FP8, MTP k3</i>

### Scaling efficiency

Percentage of ideal linear scaling achieved as concurrency rises (100% = n× the concurrency-1 throughput with no per-user cost). Real efficiency is always lower — the accelerators are shared; high efficiency at your target concurrency signals headroom. The x-axis keeps the canonical rung ladder up to the highest measured context — an empty slot marks a rung the concurrency measurements skip (64K: only the solo face measured it on qwen; unmeasured entirely on the gemma boots).

![Scaling efficiency vs context length, one line per concurrency — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_eff_ctx.svg)

<i>Scaling efficiency vs context length, one line per concurrency — FP8, MTP k3</i>

*Interpretation:* efficiency holds at 45% at 1K but falls to 16% at 32K under 8 requests — high efficiency signals headroom; a sharp drop signals the accelerators are already saturated.

## Temperature analysis (MTP k3)

Higher sampling temperature lowers draft acceptance (tok/step falls), so decode speed drops as sampling sharpens — the certified probabilistic drafting keeps the target distribution correct at every temperature.

| Temp | TTFT (s) | Decode (tok/s) | Tok/step | ITL p50 (ms) | ITL p90 (ms) |
|---|---|---|---|---|---|
| 0.0 | 3.51 | 96.2 | 3.92 | 40.5 | 40.6 |
| 0.2 | 3.53 | 93.5 | 3.87 | 41.1 | 41.2 |
| 0.4 | 3.52 | 92.6 | 3.85 | 41.1 | 41.2 |
| 0.6 | 3.51 | 89.5 | 3.70 | 41.1 | 41.2 |
| 0.8 | 3.53 | 92.2 | 3.82 | 41.1 | 41.2 |
| 1.0 | 3.53 | 83.7 | 3.47 | 41.1 | 41.2 |

![Decode speed by sampling temperature — MTP k3](charts/qwen3.8-27b-fp8-k3_temp_decode.svg)

<i>Decode speed by sampling temperature — MTP k3</i>

*Interpretation:* decode falls 13% from t0.0 (96.2 tok/s, 3.92 tok/step) to t1.0 (83.7 tok/s, 3.47 tok/step) — hotter sampling lowers draft acceptance, so fewer speculative tokens ride each step; the target distribution stays exact at every temperature.

### Temperature 0.0 vs 1.0 (per context)

![Solo decode speed t0.0 vs t1.0 — FP8, MTP k3](charts/qwen3.8-27b-fp8-k3_temp_cmp.svg)

<i>Solo decode speed t0.0 vs t1.0 — FP8, MTP k3</i>

## Results — detail tables

### Context sweep (solo, temperature 0.0)

| Context | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Tok/step | ITL p50 (ms) | ITL p90 (ms) | ITL p99 (ms) | Decode @ t1.0 (tok/s) |
|---|---|---|---|---|---|---|---|---|
| 1024 (1K) | 0.84 | 1,218 | 93.8 | 3.87 | 39.7 | 39.9 | 64.3 | 92.8 |
| 8192 (8K) | 4.40 | 1,864 | 66.8 | 2.71 | 40.5 | 40.6 | 41.9 | 67.3 |
| 32768 (32K) | 16.29 | 2,012 | 91.4 | 3.90 | 42.7 | 42.8 | 52.9 | 83.1 |
| 65536 (64K) | 39.03 | 1,679 | 84.9 | 3.87 | 45.7 | 45.7 | 47.9 | 55.5 |
| 131072 (128K) | 105.81 | 1,239 | 74.4 | 3.77 | 51.4 | 51.7 | 64.9 | 73.2 |

### Concurrency sweep — 1024 (1K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.29 | 0.29 | 99.3 | 99.3 | 95 | 100% | 39.1 | 39.3 |
| 2 | 0.44 | 0.56 | 76.9 | 67.0 | 129 | 68% | 41.0 | 41.3 |
| 3 | 0.77 | 0.97 | 81.3 | 77.6 | 207 | 73% | 43.0 | 43.4 |
| 4 | 0.78 | 0.89 | 73.3 | 64.8 | 237 | 62% | 45.8 | 46.3 |
| 8 | 1.82 | 2.17 | 59.7 | 50.0 | 343 | 45% | 56.0 | 61.4 |

### Concurrency sweep — 8192 (8K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.06 | 3.06 | 97.2 | 97.2 | 65 | 100% | 40.4 | 40.4 |
| 2 | 4.63 | 6.16 | 75.8 | 62.3 | 93 | 72% | 43.4 | 43.5 |
| 3 | 7.19 | 9.56 | 68.9 | 45.1 | 108 | 55% | 46.2 | 46.3 |
| 4 | 9.18 | 12.6 | 61.0 | 35.8 | 119 | 46% | 48.9 | 49.1 |
| 8 | 16.7 | 25.9 | 36.5 | 18.9 | 127 | 24% | 61.2 | 61.5 |

### Concurrency sweep — 32768 (32K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 15.3 | 15.3 | 92.1 | 92.1 | 28 | 100% | 42.5 | 42.6 |
| 2 | 25.5 | 30.5 | 58.2 | 34.2 | 32 | 57% | 47.6 | 47.7 |
| 3 | 33.5 | 45.5 | 41.0 | 17.6 | 33 | 40% | 52.5 | 53.2 |
| 4 | 41.3 | 60.8 | 30.7 | 12.2 | 34 | 31% | 56.9 | 80.6 |
| 8 | 72.8 | 124.5 | 14.7 | 5.1 | 34 | 16% | 77.1 | 3602.1 |

### Concurrency sweep — 65536 (64K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 36.4 | 36.4 | 81.1 | 81.1 | 14 | 100% | 45.4 | 45.5 |
| 2 | 57.1 | 73.3 | 42.3 | 14.7 | 15 | 53% | 53.0 | 54.6 |
| 3 | 75.5 | 110.6 | 28.4 | 7.5 | 15 | 36% | 61.1 | 83.7 |
| 4 | 94.9 | 149.4 | 17.1 | 5.1 | 15 | 27% | 67.9 | 4105.9 |
| 6 | 135.4 | 237.9 | 16.6 | 3.7 | 15 | 18% | 78.5 | 4404.5 |

### Concurrency sweep — 131072 (128K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 98.2 | 98.2 | 76.0 | 76.0 | 6 | 100% | 50.9 | 51.0 |
| 2 | 150.4 | 197.9 | 25.8 | 5.5 | 6 | 50% | 64.1 | 65.2 |
| 3 | 202.7 | 307.1 | 25.1 | 5.7 | 6 | 34% | 64.1 | 3497.3 |

*Scaling eff. = aggregate / (n x aggregate-at-1): 100% = perfect linear scaling. ITL p50 med = median of per-request p50s; ITL p90 max = worst user. Aggregate is wall-clock (n x 600 tokens / wall) and includes the slowest-user tail.*

## Configuration (copy & run on your own hardware)

### Validated invocation (what the gate ran, byte-exact)

```bash
# image vllm-arc:v0.30.0-b70-cert28 ; 2026-11-24 ; TP2 on 2x Arc Pro B70
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert28 ARC_NAME=arc-b70 \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1 ARC_SPEC=3 \
  bash serve_arc_b70.sh     # tools/serve_arc_b70.sh in this repo
```

The wrapper is the audited source of truth (banked posture echo: `qf3/posture.txt`). What it expands to:

```bash
# 1) container boot layer  [serve_arc_b70.sh:199-210]
docker run -d --name arc-b70 \
  --privileged --network host --ipc host \
  --shm-size 16g \              # close-out F: 64 GB-RAM hosts
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v ./neo_cache:/root/.neo_cache -v ./triton_cache:/root/.triton_cache \
  -v ./inductor_cache:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert28 -c 'sleep infinity'

# 2) engine layer (docker exec)  [serve_arc_b70.sh:327-375]
export PYTHONHASHSEED=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300 ZE_AFFINITY_MASK=0,1
export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0
export ARC_TRACE_FLIP=1 VLLM_XPU_ENABLE_XPU_GRAPH=1   # decode-only XPU graphs
export CCL_SYCL_ALLREDUCE_FP8=1                    # cert10 AR overlay
export LMCACHE_USE_EXPERIMENTAL_API=True LMCACHE_MAX_LOCAL_CPU_SIZE=16 LMCACHE_LOCAL_CPU=True
export LMCACHE_EVICT_POLICY=lr_u LMCACHE_CHUNK_SIZE=1024 LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_MIN_RETRIEVE_TOKENS=1024 LMCACHE_LOG_LEVEL=INFO LMCACHE_DISABLE_PINNED_MEMORY=1
export LMCACHE_TRACK_USAGE=false                   # close-out F: 16 GB CPU tier
exec vllm serve --model /models/target \
  --served-model-name qwen3.8-27b-fp8 --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 --max-model-len 262144 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \
  --dtype float16   \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 --mamba-cache-mode=align \
  --enable-prefix-caching --trust-remote-code \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"lmcache.extra_config":{"mamba_pool_ratio":2}},"engine_id":"arc-p40"}' \
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1..320]}' \   # 64-size discipline, see wrapper
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --port 8000
```

## Test parameters

- Context lengths: 1K, 8K, 32K, 64K, 128K tokens (prompt + template) — the full ladder this report covers, annex rungs included. Solo faces: concurrency 1, 600 output tokens, `ignore_eos`.
- Concurrency sweeps: 1/2/3/4/8 at contexts 1K, 8K, 32K, 64K, 128K — barrier-started simultaneous requests, 600 output tokens each, temperature 0.0, plus one temperature-1.0 solo face per context.
- Temperature sweep (MTP acceptance): 0.0, 0.2, 0.4, 0.6, 0.8, 1.0 at 8K context.
- Prompts: seeded random-token contexts, unique per face and per request slot (`p138_face.py` / `p139_conc.py`, tag-seeded) — every face is a cold prefill (no prompt-cache hits).
- Client-side timing on the rig loopback; TTFT includes the full engine prefill window (at c=1 queue wait ~0; at c=N the TTFT spread IS the queue+prefill stagger).

## Notes

- Per-user speed = output tokens / decode window (spec-decode aware; with MTP one stream chunk carries multiple accepted tokens). Aggregate = n x 600 tokens / wall clock.
- The r240b 1K t0.0 face runs first after the boot warmup at a cold spec-verify batch shape — its acceptance (2.8 tok/step) sits below the family norm (3.8 at 8K); the 8K/32K faces are the steady reference. Temperature comparisons are within-harness (r240 sweep vs r240b faces, both reported).
- Prefill rate is the client-observed ctx/TTFT; the engine-side device rate (P6.36 decomposition) is ~2.1–2.9k tok/s at fp8 — the client view includes chunk drain and the decode-transition tail.
- Metric definitions, UX bands, use-case thresholds and the honest-limits statement: [METHODOLOGY.md](METHODOLOGY.md).
- Banks: `results/2026-11-24-p647-r240-reports/` (solo + temperature faces, r240) and `results/2026-11-24-p647-r240b-conc/` (concurrency faces, r240b); tools `tools/ar/p138/p138_face.py`, `tools/ar/p139/p139_conc.py`.
