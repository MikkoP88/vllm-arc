# Qwen3.8-27B on 2x Intel Arc Pro B70 — MXFP8, no MTP

| Field | Value |
|---|---|
| Organization | Qwen (Alibaba) |
| Parameters | 27B dense hybrid-GDN (+MTP head) |
| Precision | MXFP8 (online, bf16 base checkpoint) |
| Accelerators | 2x Intel Arc Pro B70 (BMG-G31, 32 GB each, 64 GB total), TP2 |
| Host RAM | 64 GB (close-out F posture: --shm-size 16g, 16 GB LMCache CPU tier) |
| Engine | vLLM v0.30.0 — vllm-arc fork, image v0.30.0-b70-cert28, XPU graphs |
| KV cache dtype | fp8_e4m3 |
| Speculative decoding | disabled |
| Max model len | 262144 |
| KV pool (this boot) | 439,620 tokens (r240b posture) |
| Date | 2026-11-25 |
| Cell outcome | FACES_OK |

## Executive summary

Infrastructure decisions require real performance data. This report measures user-facing performance — how many concurrent requests the configuration supports at a given context length before streaming speed or first-token wait degrade below the user-experience bands.

This benchmark evaluates **Qwen3.8-27B** (Qwen (Alibaba); 27B parameters, dense hybrid-GDN) running **MXFP8 (online, bf16 base checkpoint)** on **2x Intel Arc Pro B70** (32 GB each, TP2) — the MTP-off ablation of the MXFP8 lane.

Test parameters: context lengths 1K–128K tokens; concurrency 1–8; 600 output tokens per request; unique seeded-random prompts (cold prefill, no prefix-cache hits); no speculative decoding; KV cache `fp8_e4m3`; temperature 0.0 with paired 1.0 faces.

| Metric | Result |
|---|---|
| Generation speed, single request | 28.7 tok/s (1K) → 22.1 tok/s (128K) |
| TTFT, single request | 1.42 (1K) → 106.4 (128K) |
| Peak per-user speed (solo) | **28.7 tok/s** @ 1K ctx (fast) |
| Peak system throughput | **184 tok/s** @ 8 concurrent, 1K ctx — ≈ 662.78 K tokens/hour |
| Throughput scaling (1 → 8, 1K ctx) | 6.5x (81% of ideal linear) |
| Interactive chat capacity (8K) | ~2 concurrent requests (estimated between the measured 2 and 3 points) |
| Long-context assistant (RAG) capacity (32K) | thresholds exceeded at concurrency 1 - not recommended |
| Document processing (batch) capacity (128K) | ~1 concurrent requests (estimated between the measured 1 and 2 points) |
| Success rate | **100.0%** across 86 requests (28 faces green) |

> Every concurrent request models one user session holding a full generation (prompt prefill + 600 streamed tokens), barrier-started together — the worst-case burst. Interactive sessions idle between turns, so sustained user counts run well below request concurrency. Full definitions: [METHODOLOGY.md](METHODOLOGY.md).

## Use case guidance

The table maps this configuration to common deployment scenarios. Capacity limits are where TTFT or streaming speed fall below the accepted thresholds for a comfortable user experience (the band edges of the UX classification below).

| Use case | Context | Thresholds | Capacity |
|---|---|---|---|
| Interactive chat | 8K | TTFT < 10s · speed >= 15 tok/s | ~2 concurrent requests (estimated between the measured 2 and 3 points) |
| Long-context assistant (RAG) | 32K | TTFT < 10s · speed >= 15 tok/s | thresholds exceeded at concurrency 1 - not recommended |
| Document processing (batch) | 128K | speed >= 15 tok/s | ~1 concurrent requests (estimated between the measured 1 and 2 points) |

The limits are conservative: past them the system keeps serving with slower response times that may still be acceptable for your workload.

## System throughput

Aggregate token generation across all concurrent requests (output tokens only; prefill excluded). Measured as n × 600 / wall — the wall includes the barrier start and the slowest-user tail, so it is the honest saturated-burst number.

| Condition | Aggregate throughput |
|---|---|
| Peak (1K, 8 req) | 184 tok/s |
| 8K ctx, 4 req | 68 tok/s |
| 8K ctx, 8 req | 95 tok/s |
| 32K ctx, 4 req | 27 tok/s |
| 32K ctx, 8 req | 31 tok/s |

At the peak point this configuration produces approximately **662.78 K tokens/hour** — the number that matters for batch workloads (document processing, synthetic-data generation, offline analysis).

![Aggregate throughput vs context length, one line per concurrency — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_agg_ctx.svg)

<i>Aggregate throughput vs context length, one line per concurrency — MXFP8, no MTP</i>

*Interpretation:* throughput peaks at 184 tok/s (8 requests, 1K); at 128K the heaviest measured load (3 requests) yields 6 tok/s (3% of peak) — long-context KV residency shrinks the resident batch. Batch workloads should batch short and parallel; long-context work is bound by KV residency, not compute.

## Per-user generation speed

The streaming rate each user experiences. As concurrency rises, per-user speed falls because the accelerators are shared; with MTP, multiple accepted tokens ride one stream chunk (spec-aware accounting).

| Context | Solo (tok/s) | class | Best @ conc (tok/s) | @ conc 8 (tok/s) | class |
|---|---|---|---|---|---|
| 1024 (1K) | 28.7 | fast | 29.2 | 24.5 | acceptable |
| 8192 (8K) | 28.7 | fast | 28.6 | 19.0 | acceptable |
| 32768 (32K) | 27.1 | fast | 26.9 | 9.5 | slow |
| 65536 (64K) | 25.0 | acceptable | 25.1 | - | - |
| 131072 (128K) | 22.1 | acceptable | 22.3 | - | - |

### How fast is this?

| Speed | User experience |
|---|---|
| < 15 tok/s | Slow - may be slower than reading speed |
| 15-25 tok/s | Acceptable - keeps pace with reading |
| 25-50 tok/s | Fast - exceeds reading speed |
| >= 50 tok/s | Very fast - text appears nearly instantly |

Solo speed spans **22.1–28.7 tok/s** (128K to 1K context); at concurrency 8 the floor is 9.5 tok/s (32K ctx).

![Per-user decode speed vs context length (solo) — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_decode.svg)

<i>Per-user decode speed vs context length (solo) — MXFP8, no MTP</i>

![Per-user speed vs context length, one line per concurrency — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_speed_ctx.svg)

<i>Per-user speed vs context length, one line per concurrency — MXFP8, no MTP</i>

![Per-user speed range: the band every deployment concurrency lands in — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_range_speed.svg)

<i>Per-user speed range: the band every deployment concurrency lands in — MXFP8, no MTP</i>

*Interpretation:* at 128K a single user streams 22.3 tok/s while 3 requests share down to 9.4 tok/s (42% of solo); at 8K the spread is 19.0-28.6 tok/s — every concurrency you deploy lands between the band edges; the bands below classify the streaming feel.

## Time to first token

Time from request submission to the first streamed token — the primary responsiveness metric. Client-side on the rig loopback, it decomposes into queue wait (rising with concurrency) and prefill (rising with context); the technical analysis below separates the two.

| Context | Solo (s) | class | @ conc 8 mean (s) | @ conc 8 worst (s) | class |
|---|---|---|---|---|---|
| 1024 (1K) | 1.42 | responsive | 1.61 | 1.78 | responsive |
| 8192 (8K) | 4.30 | noticeable | 17.1 | 26.1 | frustrating |
| 32768 (32K) | 17.3 | frustrating | 75.3 | 126.2 | frustrating |
| 65536 (64K) | 40.8 | frustrating | - | - | - |
| 131072 (128K) | 106.4 | frustrating | - | - | - |

### How responsive is this?

| TTFT | User experience |
|---|---|
| < 0.5 s | Feels instant |
| 0.5-2 s | Feels responsive |
| 2-5 s | Noticeable but still acceptable |
| 5-10 s | Feels slow; acceptable at higher context |
| > 10 s | Can be frustrating; users may retry or abandon |

> **Caching note.** Every face uses a unique seeded-random prompt — a deliberate cold prefill, the worst case. The certified posture runs prefix caching (LMCache): in production, incremental conversations pay TTFT only for new tokens, which is dramatically faster than these numbers.

![TTFT vs context length (solo) — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_ttft.svg)

<i>TTFT vs context length (solo) — MXFP8, no MTP</i>

![TTFT vs context length, one line per concurrency — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_ttft_ctx.svg)

<i>TTFT vs context length, one line per concurrency — MXFP8, no MTP</i>

![TTFT range: single request (bottom) to worst user at max concurrency (top) — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_range_ttft.svg)

<i>TTFT range: single request (bottom) to worst user at max concurrency (top) — MXFP8, no MTP</i>

*Interpretation:* at 128K the first token takes 98.6 solo but up to 297.0 for the worst user under 3 requests — 67% of the loaded wait is queue+prefill stagger, the sharing cost the capacity tables price below.

## Capacity analysis

### Interactive chat (8K context)

Conversational assistant sessions: system prompt plus a few turns fit 8K. Both first-token wait and streaming speed matter. Thresholds: TTFT < 10s · speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 | @ 4 | @ 8 |
|---|---|---|---|---|---|
| TTFT worst user (s) | 3.40 | 6.80 | 10.1 *(exceeded)* | 13.2 *(exceeded)* | 26.1 *(exceeded)* |
| Per-user mean speed (tok/s) | 28.6 | 26.3 | 25.4 | 24.1 | 19.0 |

![Interactive chat (8K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/qwen3.8-27b-mxfp8-k0_uc_chat8_speed.svg)

<i>Interactive chat (8K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

![Interactive chat (8K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar](charts/qwen3.8-27b-mxfp8-k0_uc_chat8_ttft.svg)

<i>Interactive chat (8K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar</i>

**Capacity limit: ~2 concurrent requests (estimated between the measured 2 and 3 points)**

### Long-context assistant (RAG) (32K context)

Retrieval-augmented / large-prompt assistants at 32K context. Thresholds: TTFT < 10s · speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 | @ 4 | @ 8 |
|---|---|---|---|---|---|
| TTFT worst user (s) | 16.1 *(exceeded)* | 31.5 *(exceeded)* | 47.2 *(exceeded)* | 62.8 *(exceeded)* | 126.2 *(exceeded)* |
| Per-user mean speed (tok/s) | 26.9 | 22.2 | 18.2 | 15.4 | 9.5 *(below)* |

![Long-context assistant (RAG) (32K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/qwen3.8-27b-mxfp8-k0_uc_assist32_speed.svg)

<i>Long-context assistant (RAG) (32K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

![Long-context assistant (RAG) (32K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar](charts/qwen3.8-27b-mxfp8-k0_uc_assist32_ttft.svg)

<i>Long-context assistant (RAG) (32K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar</i>

**Capacity limit: thresholds exceeded at concurrency 1 - not recommended**

### Document processing (batch) (128K context)

Offline summarization / extraction at the largest measured context. Turnaround matters, not first-token wait: the bar is streaming speed (and total throughput). Thresholds: speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 |
|---|---|---|---|
| TTFT worst user (s) | 98.6 | 197.9 | 297.0 |
| Per-user mean speed (tok/s) | 22.3 | 13.4 *(below)* | 9.4 *(below)* |

![Document processing (batch) (128K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/qwen3.8-27b-mxfp8-k0_uc_docmax_speed.svg)

<i>Document processing (batch) (128K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

**Capacity limit: ~1 concurrent requests (estimated between the measured 1 and 2 points)**

### KV pool (the whole-pool bound)

This boot exposed a KV pool of **439,620 tokens** (r240b posture). A session occupies ctx + 664 (output + scheduler slack), so the pool holds:

| Context | Pool capacity (sessions) | Measured green @ conc | c8 fully resident? |
|---|---|---|---|
| 1024 (1K) | 260 | 8 (green) | yes |
| 8192 (8K) | 49 | 8 (green) | yes |
| 32768 (32K) | 13 | 8 (green) | yes |
| 65536 (64K) | 6 | 6 (green) | marginal |
| 131072 (128K) | 3 | 3 (green) | no |

*Sessions are the whole-pool bound; `--max-num-seqs 64` bounds the running batch above it. TurboQuant dtypes multiply the pool up to 3.68x: see the [KV-cache dtype report](qwen3.8-27b-fp8-kv-cache-dtype.md).*

## Technical analysis

### Queue + prefill decomposition (derived)

At concurrency 1 the queue is empty, so solo TTFT is pure prefill. Under the barrier burst the added first-token wait — `ttft_mean(n) − ttft_mean(1)` — is the queue+prefill-share stagger the Nth user pays. Derived client-side; engine-internal queueing is not exported by this engine version. The x-axis keeps the canonical rung ladder up to the highest measured context — an empty slot marks a rung the concurrency measurements skip (64K: only the solo face measured it on qwen; unmeasured entirely on the gemma boots); no stagger exists there.

| Context | added wait @ n2 | @ n4 | @ n8 |
|---|---|---|---|
| 1024 (1K) | +-0.1 s | +0.1 s | +1.1 s |
| 8192 (8K) | +1.7 s | +6.4 s | +13.7 s |
| 32768 (32K) | +10.3 s | +26.9 s | +59.1 s |
| 65536 (64K) | +20.8 s | +59.8 s | - |
| 131072 (128K) | +52.2 s | - | - |

![Added first-token wait vs context length, one line per concurrency (derived queue+prefill stagger) — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_queue_ctx.svg)

<i>Added first-token wait vs context length, one line per concurrency (derived queue+prefill stagger) — MXFP8, no MTP</i>

*Interpretation:* the loaded-request stagger grows from +1.1 s at 1K to +59.1 s at 32K (8 requests) — the context-carrying requests serialize behind each other's prefill.

### Per-user prefill speed

Rate at which input context is processed before the first token. Client-observed (ctx / TTFT at queue ~0); the engine-side device rate is higher — the client view includes chunk drain and the decode-transition tail. The single-request line spans **every measured context** (sweep plus solo faces — the sweep skipped 64K); loaded lines cover the contexts the concurrency sweep measured.

| Context | prefill @ n1 (tok/s) | @ n4 | @ n8 |
|---|---|---|---|
| 1024 (1K) | 1894 | 1485 | 636 |
| 8192 (8K) | 2408 | 833 | 480 |
| 32768 (32K) | 2030 | 762 | 435 |
| 65536 (64K) | 1742 | 672 | - |
| 131072 (128K) | 1329 | - | - |

![Client-observed prefill rate vs context length, one line per concurrency — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_prefill.svg)

<i>Client-observed prefill rate vs context length, one line per concurrency — MXFP8, no MTP</i>

*Interpretation:* prefill peaks at 2,408 tok/s (8K) and eases as context grows — chunked prefill batches more work per pass at length, trading first-token wait for decode-window protection.

### Inter-token latency

Time between consecutive streamed tokens — the smoothness of streaming. Solo p50 spans **33.9 ms** (1K) to 44.5 ms; the worst observed p99 is **45 ms** (128K). With MTP one chunk can carry multiple accepted tokens, so the perceived stream is smoother than the raw chunk gap.

![Inter-token latency vs context length (solo) — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_itl.svg)

<i>Inter-token latency vs context length (solo) — MXFP8, no MTP</i>

### Scaling efficiency

Percentage of ideal linear scaling achieved as concurrency rises (100% = n× the concurrency-1 throughput with no per-user cost). Real efficiency is always lower — the accelerators are shared; high efficiency at your target concurrency signals headroom. The x-axis keeps the canonical rung ladder up to the highest measured context — an empty slot marks a rung the concurrency measurements skip (64K: only the solo face measured it on qwen; unmeasured entirely on the gemma boots).

![Scaling efficiency vs context length, one line per concurrency — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_eff_ctx.svg)

<i>Scaling efficiency vs context length, one line per concurrency — MXFP8, no MTP</i>

*Interpretation:* efficiency holds at 81% at 1K but falls to 25% at 32K under 8 requests — high efficiency signals headroom; a sharp drop signals the accelerators are already saturated.

### Temperature 0.0 vs 1.0 (per context)

![Solo decode speed t0.0 vs t1.0 — MXFP8, no MTP](charts/qwen3.8-27b-mxfp8-k0_temp_cmp.svg)

<i>Solo decode speed t0.0 vs t1.0 — MXFP8, no MTP</i>

## Results — detail tables

### Context sweep (solo, temperature 0.0)

| Context | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Tok/step | ITL p50 (ms) | ITL p90 (ms) | ITL p99 (ms) | Decode @ t1.0 (tok/s) |
|---|---|---|---|---|---|---|---|---|
| 1024 (1K) | 1.42 | 722 | 28.7 | 1.00 | 34.4 | 34.6 | 38.5 | 28.7 |
| 8192 (8K) | 4.30 | 1,905 | 28.7 | 1.00 | 34.9 | 35.0 | 35.4 | 28.2 |
| 32768 (32K) | 17.30 | 1,894 | 27.1 | 1.00 | 37.0 | 37.0 | 37.4 | 26.5 |
| 65536 (64K) | 40.77 | 1,607 | 25.0 | 1.00 | 39.7 | 39.8 | 40.5 | 24.8 |
| 131072 (128K) | 106.43 | 1,232 | 22.1 | 1.00 | 45.0 | 45.2 | 46.0 | 22.1 |

### Concurrency sweep — 1024 (1K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.54 | 0.54 | 29.2 | 29.2 | 28 | 100% | 33.9 | 34.1 |
| 2 | 0.40 | 0.51 | 28.3 | 28.3 | 56 | 98% | 34.3 | 34.5 |
| 3 | 0.59 | 0.69 | 27.6 | 27.3 | 80 | 94% | 34.9 | 35.1 |
| 4 | 0.69 | 0.81 | 27.0 | 26.6 | 105 | 92% | 35.2 | 35.4 |
| 8 | 1.61 | 1.78 | 24.5 | 23.5 | 184 | 81% | 37.0 | 37.3 |

### Concurrency sweep — 8192 (8K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.40 | 3.40 | 28.6 | 28.6 | 25 | 100% | 34.9 | 34.9 |
| 2 | 5.12 | 6.80 | 26.3 | 24.4 | 43 | 87% | 35.4 | 35.5 |
| 3 | 7.75 | 10.1 | 25.4 | 21.3 | 57 | 77% | 36.1 | 36.2 |
| 4 | 9.83 | 13.2 | 24.1 | 19.0 | 68 | 69% | 36.7 | 36.8 |
| 8 | 17.1 | 26.1 | 19.0 | 12.9 | 95 | 48% | 40.2 | 40.3 |

### Concurrency sweep — 32768 (32K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 16.1 | 16.1 | 26.9 | 26.9 | 16 | 100% | 36.8 | 36.8 |
| 2 | 26.5 | 31.5 | 22.2 | 18.2 | 22 | 71% | 37.4 | 37.5 |
| 3 | 35.0 | 47.2 | 18.2 | 12.2 | 25 | 54% | 38.4 | 38.5 |
| 4 | 43.0 | 62.8 | 15.4 | 9.1 | 27 | 44% | 39.7 | 39.8 |
| 8 | 75.3 | 126.2 | 9.5 | 4.5 | 31 | 25% | 46.2 | 46.7 |

### Concurrency sweep — 65536 (64K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 37.6 | 37.6 | 25.1 | 25.1 | 10 | 100% | 39.4 | 39.4 |
| 2 | 58.4 | 75.2 | 17.5 | 10.4 | 12 | 62% | 40.0 | 40.1 |
| 3 | 78.5 | 112.6 | 13.4 | 6.3 | 13 | 44% | 41.3 | 41.4 |
| 4 | 97.5 | 150.3 | 10.7 | 4.5 | 14 | 35% | 43.4 | 43.6 |
| 6 | 135.4 | 225.4 | 7.6 | 2.8 | 14 | 24% | 49.3 | 52.0 |

### Concurrency sweep — 131072 (128K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 98.6 | 98.6 | 22.3 | 22.3 | 5 | 100% | 44.5 | 44.6 |
| 2 | 150.8 | 197.9 | 13.4 | 5.0 | 5 | 56% | 45.1 | 45.2 |
| 3 | 200.8 | 297.0 | 9.4 | 2.7 | 6 | 39% | 47.0 | 47.3 |

*Scaling eff. = aggregate / (n x aggregate-at-1): 100% = perfect linear scaling. ITL p50 med = median of per-request p50s; ITL p90 max = worst user. Aggregate is wall-clock (n x 600 tokens / wall) and includes the slowest-user tail.*

## Configuration (copy & run on your own hardware)

### Validated invocation (what the gate ran, byte-exact)

```bash
# image vllm-arc:v0.30.0-b70-cert28 ; 2026-11-25 ; TP2 on 2x Arc Pro B70
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert28 ARC_NAME=arc-b70 \
  ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b ARC_QUANTCFG={"ignore":["*visual*","*linear_attn*","mtp.fc"]} ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0 ARC_QUANT=mxfp8 ARC_SPEC=0 \
  bash serve_arc_b70.sh     # tools/serve_arc_b70.sh in this repo
```

The wrapper is the audited source of truth (banked posture echo: `q80/posture.txt`). What it expands to:

```bash
# 1) container boot layer  [serve_arc_b70.sh:199-210]
docker run -d --name arc-b70 \
  --privileged --network host --ipc host \
  --shm-size 16g \              # close-out F: 64 GB-RAM hosts
  -v /models/qwen3.8-27b:/models/target:ro \
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
export LMCACHE_USE_EXPERIMENTAL_API=True LMCACHE_MAX_LOCAL_CPU_SIZE=16 LMCACHE_LOCAL_CPU=True
export LMCACHE_EVICT_POLICY=lr_u LMCACHE_CHUNK_SIZE=1024 LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_MIN_RETRIEVE_TOKENS=1024 LMCACHE_LOG_LEVEL=INFO LMCACHE_DISABLE_PINNED_MEMORY=1
export LMCACHE_TRACK_USAGE=false                   # close-out F: 16 GB CPU tier
exec vllm serve --model /models/target \
  --served-model-name qwen3.8-27b --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 --max-model-len 262144 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \
  --dtype float16 --quantization mxfp8 --quantization-config '{"ignore":["*visual*","*linear_attn*","mtp.fc"]}' \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 --mamba-cache-mode=align \
  --enable-prefix-caching --trust-remote-code \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"lmcache.extra_config":{"mamba_pool_ratio":2}},"engine_id":"arc-p40"}' \
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1..320]}' \   # 64-size discipline, see wrapper
  --port 8000
```

## Test parameters

- Context lengths: 1K, 8K, 32K, 64K, 128K tokens (prompt + template) — the full ladder this report covers, annex rungs included. Solo faces: concurrency 1, 600 output tokens, `ignore_eos`.
- Concurrency sweeps: 1/2/3/4/8 at contexts 1K, 8K, 32K, 64K, 128K — barrier-started simultaneous requests, 600 output tokens each, temperature 0.0, plus one temperature-1.0 solo face per context.
- Prompts: seeded random-token contexts, unique per face and per request slot (`p138_face.py` / `p139_conc.py`, tag-seeded) — every face is a cold prefill (no prompt-cache hits).
- Client-side timing on the rig loopback; TTFT includes the full engine prefill window (at c=1 queue wait ~0; at c=N the TTFT spread IS the queue+prefill stagger).

## Notes

- Per-user speed = output tokens / decode window (spec-decode aware; with MTP one stream chunk carries multiple accepted tokens). Aggregate = n x 600 tokens / wall clock.
- The r240b 1K t0.0 face runs first after the boot warmup at a cold spec-verify batch shape — its acceptance (2.8 tok/step) sits below the family norm (3.8 at 8K); the 8K/32K faces are the steady reference. Temperature comparisons are within-harness (r240 sweep vs r240b faces, both reported).
- Prefill rate is the client-observed ctx/TTFT; the engine-side device rate (P6.36 decomposition) is ~2.1–2.9k tok/s at fp8 — the client view includes chunk drain and the decode-transition tail.
- Metric definitions, UX bands, use-case thresholds and the honest-limits statement: [METHODOLOGY.md](METHODOLOGY.md).
- Banks: `results/2026-11-24-p647-r240-reports/` (solo + temperature faces, r240) and `results/2026-11-24-p647-r240b-conc/` (concurrency faces, r240b); tools `tools/ar/p138/p138_face.py`, `tools/ar/p139/p139_conc.py`.
