# Gemma-4-31B on 2x Intel Arc Pro B70 — FP8, no MTP

| Field | Value |
|---|---|
| Organization | Google |
| Parameters | 31B dense hybrid (two-head-size GDN, no MTP head) |
| Precision | FP8 (online, bf16 base checkpoint) |
| Accelerators | 2x Intel Arc Pro B70 (BMG-G31, 32 GB each, 64 GB total), TP2 |
| Host RAM | 64 GB (close-out F posture: --shm-size 16g, 16 GB LMCache CPU tier) |
| Engine | vLLM v0.30.0 — vllm-arc fork, image v0.30.0-b70-cert28, XPU graphs |
| KV cache dtype | fp8_e4m3 |
| Speculative decoding | disabled |
| Max model len | 40960 |
| KV pool (this boot) | 56,128 tokens (r240b posture) |
| Date | 2026-11-25 |
| Cell outcome | FACES_OK |

## Executive summary

The 64K points in the tables and charts below come from the **64K annex boot** — same image and quantization posture; the annex section quantifies both boots' agreement at the shared contexts.

Infrastructure decisions require real performance data. This report measures user-facing performance — how many concurrent requests the configuration supports at a given context length before streaming speed or first-token wait degrade below the user-experience bands.

This benchmark evaluates **Gemma-4-31B** (Google; 31B parameters, dense hybrid) running **FP8 (online, bf16 base checkpoint)** on **2x Intel Arc Pro B70** (32 GB each, TP2) — the gemma online-FP8 capability lane.

Test parameters: context lengths 1K–64K tokens; concurrency 1–8; 600 output tokens per request; unique seeded-random prompts (cold prefill, no prefix-cache hits); no speculative decoding; KV cache `fp8_e4m3`; temperature 0.0 with paired 1.0 faces.

| Metric | Result |
|---|---|
| Generation speed, single request | 27.0 tok/s (1K) → 24.1 tok/s (64K) |
| TTFT, single request | 1.01 (1K) → 56.0 (64K) |
| Peak per-user speed (solo) | **27.0 tok/s** @ 1K ctx (fast) |
| Peak system throughput | **178 tok/s** @ 8 concurrent, 1K ctx — ≈ 641.92 K tokens/hour |
| Throughput scaling (1 → 8, 1K ctx) | 6.6x (82% of ideal linear) |
| Interactive chat capacity (8K) | ~2 concurrent requests (estimated between the measured 2 and 3 points) |
| Long-context assistant (RAG) capacity (32K) | thresholds exceeded at concurrency 1 - not recommended |
| Document processing (batch) capacity (64K) | 1 concurrent requests |
| Success rate | **100.0%** across 44 requests (16 faces green) |

> Every concurrent request models one user session holding a full generation (prompt prefill + 600 streamed tokens), barrier-started together — the worst-case burst. Interactive sessions idle between turns, so sustained user counts run well below request concurrency. Full definitions: [METHODOLOGY.md](METHODOLOGY.md).

## Use case guidance

The table maps this configuration to common deployment scenarios. Capacity limits are where TTFT or streaming speed fall below the accepted thresholds for a comfortable user experience (the band edges of the UX classification below).

| Use case | Context | Thresholds | Capacity |
|---|---|---|---|
| Interactive chat | 8K | TTFT < 10s · speed >= 15 tok/s | ~2 concurrent requests (estimated between the measured 2 and 3 points) |
| Long-context assistant (RAG) | 32K | TTFT < 10s · speed >= 15 tok/s | thresholds exceeded at concurrency 1 - not recommended |
| Document processing (batch) | 64K | speed >= 15 tok/s | 1 concurrent requests |

The limits are conservative: past them the system keeps serving with slower response times that may still be acceptable for your workload.

## System throughput

Aggregate token generation across all concurrent requests (output tokens only; prefill excluded). Measured as n × 600 / wall — the wall includes the barrier start and the slowest-user tail, so it is the honest saturated-burst number.

| Condition | Aggregate throughput |
|---|---|
| Peak (1K, 8 req) | 178 tok/s |
| 8K ctx, 4 req | 63 tok/s |

At the peak point this configuration produces approximately **641.92 K tokens/hour** — the number that matters for batch workloads (document processing, synthetic-data generation, offline analysis).

![Aggregate throughput vs context length, one line per concurrency — FP8, no MTP](charts/gemma-4-31b-fp8_agg_ctx.svg)

<i>Aggregate throughput vs context length, one line per concurrency — FP8, no MTP</i>

*Interpretation:* throughput peaks at 178 tok/s (8 requests, 1K); at 64K the heaviest measured load (1 request) yields 7 tok/s (4% of peak) — long-context KV residency shrinks the resident batch. Batch workloads should batch short and parallel; long-context work is bound by KV residency, not compute.

## Per-user generation speed

The streaming rate each user experiences. As concurrency rises, per-user speed falls because the accelerators are shared; with MTP, multiple accepted tokens ride one stream chunk (spec-aware accounting).

| Context | Solo (tok/s) | class | Best @ conc (tok/s) | @ conc 8 (tok/s) | class |
|---|---|---|---|---|---|
| 1024 (1K) | 27.0 | fast | 28.1 | 23.6 | acceptable |
| 8192 (8K) | 26.6 | fast | 26.7 | - | - |
| 32768 (32K) | 25.4 | fast | 25.4 | - | - |
| 65536 (64K) | 24.1 | acceptable | 24.1 | - | - |

### How fast is this?

| Speed | User experience |
|---|---|
| < 15 tok/s | Slow - may be slower than reading speed |
| 15-25 tok/s | Acceptable - keeps pace with reading |
| 25-50 tok/s | Fast - exceeds reading speed |
| >= 50 tok/s | Very fast - text appears nearly instantly |

Solo speed spans **24.1–27.0 tok/s** (64K to 1K context); at concurrency 8 the floor is 23.6 tok/s (1K ctx).

![Per-user decode speed vs context length (solo) — FP8, no MTP](charts/gemma-4-31b-fp8_decode.svg)

<i>Per-user decode speed vs context length (solo) — FP8, no MTP</i>

![Per-user speed vs context length, one line per concurrency — FP8, no MTP](charts/gemma-4-31b-fp8_speed_ctx.svg)

<i>Per-user speed vs context length, one line per concurrency — FP8, no MTP</i>

![Per-user speed range: the band every deployment concurrency lands in — FP8, no MTP](charts/gemma-4-31b-fp8_range_speed.svg)

<i>Per-user speed range: the band every deployment concurrency lands in — FP8, no MTP</i>

*Interpretation:* at 64K a single user streams 24.1 tok/s while 1 request share down to 24.1 tok/s (100% of solo); at 8K the spread is 18.7-26.7 tok/s — every concurrency you deploy lands between the band edges; the bands below classify the streaming feel.

## Time to first token

Time from request submission to the first streamed token — the primary responsiveness metric. Client-side on the rig loopback, it decomposes into queue wait (rising with concurrency) and prefill (rising with context); the technical analysis below separates the two.

| Context | Solo (s) | class | @ conc 8 mean (s) | @ conc 8 worst (s) | class |
|---|---|---|---|---|---|
| 1024 (1K) | 1.01 | responsive | 1.45 | 1.60 | responsive |
| 8192 (8K) | 4.03 | noticeable | - | - | - |
| 32768 (32K) | 21.8 | frustrating | - | - | - |
| 65536 (64K) | 56.0 | frustrating | - | - | - |

### How responsive is this?

| TTFT | User experience |
|---|---|
| < 0.5 s | Feels instant |
| 0.5-2 s | Feels responsive |
| 2-5 s | Noticeable but still acceptable |
| 5-10 s | Feels slow; acceptable at higher context |
| > 10 s | Can be frustrating; users may retry or abandon |

> **Caching note.** Every face uses a unique seeded-random prompt — a deliberate cold prefill, the worst case. The certified posture runs prefix caching (LMCache): in production, incremental conversations pay TTFT only for new tokens, which is dramatically faster than these numbers.

![TTFT vs context length (solo) — FP8, no MTP](charts/gemma-4-31b-fp8_ttft.svg)

<i>TTFT vs context length (solo) — FP8, no MTP</i>

![TTFT vs context length, one line per concurrency — FP8, no MTP](charts/gemma-4-31b-fp8_ttft_ctx.svg)

<i>TTFT vs context length, one line per concurrency — FP8, no MTP</i>

![TTFT range: single request (bottom) to worst user at max concurrency (top) — FP8, no MTP](charts/gemma-4-31b-fp8_range_ttft.svg)

<i>TTFT range: single request (bottom) to worst user at max concurrency (top) — FP8, no MTP</i>

*Interpretation:* at 64K the first token takes 56.0 solo but up to 56.0 for the worst user under 1 request — 0% of the loaded wait is queue+prefill stagger, the sharing cost the capacity tables price below.

## Capacity analysis

### Interactive chat (8K context)

Conversational assistant sessions: system prompt plus a few turns fit 8K. Both first-token wait and streaming speed matter. Thresholds: TTFT < 10s · speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 | @ 4 | @ 6 |
|---|---|---|---|---|---|
| TTFT worst user (s) | 3.75 | 7.19 | 10.7 *(exceeded)* | 14.3 *(exceeded)* | 21.2 *(exceeded)* |
| Per-user mean speed (tok/s) | 26.7 | 24.5 | 22.3 | 21.2 | 18.7 |

![Interactive chat (8K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/gemma-4-31b-fp8_uc_chat8_speed.svg)

<i>Interactive chat (8K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

![Interactive chat (8K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar](charts/gemma-4-31b-fp8_uc_chat8_ttft.svg)

<i>Interactive chat (8K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar</i>

**Capacity limit: ~2 concurrent requests (estimated between the measured 2 and 3 points)**

### Long-context assistant (RAG) (32K context)

Retrieval-augmented / large-prompt assistants at 32K context. Thresholds: TTFT < 10s · speed >= 15 tok/s.

| Metric | @ 1 |
|---|---|
| TTFT worst user (s) | 20.6 *(exceeded)* |
| Per-user mean speed (tok/s) | 25.4 |

![Long-context assistant (RAG) (32K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/gemma-4-31b-fp8_uc_assist32_speed.svg)

<i>Long-context assistant (RAG) (32K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

![Long-context assistant (RAG) (32K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar](charts/gemma-4-31b-fp8_uc_assist32_ttft.svg)

<i>Long-context assistant (RAG) (32K): worst-user TTFT by concurrency; the dashed line marks the 10.0 s use-case bar</i>

**Capacity limit: thresholds exceeded at concurrency 1 - not recommended**

### Document processing (batch) (64K context)

Offline summarization / extraction at the largest measured context. Turnaround matters, not first-token wait: the bar is streaming speed (and total throughput). Thresholds: speed >= 15 tok/s.

| Metric | @ 1 |
|---|---|
| TTFT worst user (s) | 56.0 |
| Per-user mean speed (tok/s) | 24.1 |

![Document processing (batch) (64K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor](charts/gemma-4-31b-fp8_uc_docmax_speed.svg)

<i>Document processing (batch) (64K): per-user speed by concurrency; the dashed line marks the 15.0 tok/s use-case floor</i>

**Capacity limit: 1 concurrent requests**

### KV pool (the whole-pool bound)

This boot exposed a KV pool of **56,128 tokens** (r240b posture). A session occupies ctx + 664 (output + scheduler slack), so the pool holds:

| Context | Pool capacity (sessions) | Measured green @ conc | c8 fully resident? |
|---|---|---|---|
| 1024 (1K) | 33 | 8 (green) | yes |
| 8192 (8K) | 6 | 6 (green) | marginal |
| 32768 (32K) | 1 | 1 (green) | no |
| 65536 (64K) | 1 | 1 (green) | no |

*The 64K row's pool is the annex boot's (80,619 tokens) — the main boot's pool cannot hold that context.*

*Sessions are the whole-pool bound; `--max-num-seqs 64` bounds the running batch above it. TurboQuant dtypes multiply the pool up to 3.68x: see the [KV-cache dtype report](qwen3.8-27b-fp8-kv-cache-dtype.md).*

## Technical analysis

### Queue + prefill decomposition (derived)

At concurrency 1 the queue is empty, so solo TTFT is pure prefill. Under the barrier burst the added first-token wait — `ttft_mean(n) − ttft_mean(1)` — is the queue+prefill-share stagger the Nth user pays. Derived client-side; engine-internal queueing is not exported by this engine version. The x-axis keeps the canonical rung ladder up to the highest measured context — an empty slot marks a rung the concurrency measurements skip (64K: only the solo face measured it on qwen; unmeasured entirely on the gemma boots); no stagger exists there.

| Context | added wait @ n2 | @ n4 | @ n8 |
|---|---|---|---|
| 1024 (1K) | +-0.3 s | +0.1 s | +0.7 s |
| 8192 (8K) | +1.7 s | +5.5 s | - |

![Added first-token wait vs context length, one line per concurrency (derived queue+prefill stagger) — FP8, no MTP](charts/gemma-4-31b-fp8_queue_ctx.svg)

<i>Added first-token wait vs context length, one line per concurrency (derived queue+prefill stagger) — FP8, no MTP</i>

### Per-user prefill speed

Rate at which input context is processed before the first token. Client-observed (ctx / TTFT at queue ~0); the engine-side device rate is higher — the client view includes chunk drain and the decode-transition tail. The single-request line spans **every measured context** (sweep plus solo faces — the sweep skipped 64K); loaded lines cover the contexts the concurrency sweep measured.

| Context | prefill @ n1 (tok/s) | @ n4 | @ n8 |
|---|---|---|---|
| 1024 (1K) | 1410 | 1278 | 708 |
| 8192 (8K) | 2183 | 885 | - |
| 32768 (32K) | 1594 | - | - |
| 65536 (64K) | 1170 | - | - |

![Client-observed prefill rate vs context length, one line per concurrency — FP8, no MTP](charts/gemma-4-31b-fp8_prefill.svg)

<i>Client-observed prefill rate vs context length, one line per concurrency — FP8, no MTP</i>

*Interpretation:* prefill peaks at 2,183 tok/s (8K) and eases as context grows — chunked prefill batches more work per pass at length, trading first-token wait for decode-window protection.

### Inter-token latency

Time between consecutive streamed tokens — the smoothness of streaming. Solo p50 spans **35.7 ms** (1K) to 41.7 ms; the worst observed p99 is **42 ms** (64K). With MTP one chunk can carry multiple accepted tokens, so the perceived stream is smoother than the raw chunk gap.

![Inter-token latency vs context length (solo) — FP8, no MTP](charts/gemma-4-31b-fp8_itl.svg)

<i>Inter-token latency vs context length (solo) — FP8, no MTP</i>

### Scaling efficiency

Percentage of ideal linear scaling achieved as concurrency rises (100% = n× the concurrency-1 throughput with no per-user cost). Real efficiency is always lower — the accelerators are shared; high efficiency at your target concurrency signals headroom. The x-axis keeps the canonical rung ladder up to the highest measured context — an empty slot marks a rung the concurrency measurements skip (64K: only the solo face measured it on qwen; unmeasured entirely on the gemma boots).

![Scaling efficiency vs context length, one line per concurrency — FP8, no MTP](charts/gemma-4-31b-fp8_eff_ctx.svg)

<i>Scaling efficiency vs context length, one line per concurrency — FP8, no MTP</i>

### Temperature 0.0 vs 1.0 (per context)

![Solo decode speed t0.0 vs t1.0 — FP8, no MTP](charts/gemma-4-31b-fp8_temp_cmp.svg)

<i>Solo decode speed t0.0 vs t1.0 — FP8, no MTP</i>

## Results — detail tables

### Context sweep (solo, temperature 0.0)

| Context | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Tok/step | ITL p50 (ms) | ITL p90 (ms) | ITL p99 (ms) | Decode @ t1.0 (tok/s) |
|---|---|---|---|---|---|---|---|---|
| 1024 (1K) | 1.01 | 1,013 | 27.0 | 1.00 | 37.1 | 37.3 | 37.5 | 27.6 |
| 8192 (8K) | 4.03 | 2,034 | 26.6 | 1.00 | 37.7 | 37.7 | 38.0 | 26.1 |
| 32768 (32K) | 21.76 | 1,506 | 25.4 | 1.00 | 39.5 | 39.6 | 40.0 | 25.1 |
| 65536 (64K) | 55.99 | 1,170 | 24.1 | - | 41.7 | 41.8 | 42.3 | 23.8 |

*Rows sourced from the r240b/r240c concurrency-1 faces where no r240 solo face exists (same cold-prefill measurement).*

### Concurrency sweep — 1024 (1K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.73 | 0.73 | 28.1 | 28.1 | 27 | 100% | 35.7 | 36.5 |
| 2 | 0.38 | 0.49 | 27.7 | 27.6 | 54 | 100% | 36.1 | 36.8 |
| 3 | 0.65 | 0.81 | 27.0 | 26.7 | 79 | 97% | 36.9 | 37.7 |
| 4 | 0.80 | 0.92 | 27.3 | 26.9 | 105 | 97% | 36.5 | 37.3 |
| 8 | 1.45 | 1.60 | 23.6 | 22.7 | 178 | 82% | 42.4 | 44.5 |

### Concurrency sweep — 8192 (8K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.75 | 3.75 | 26.7 | 26.7 | 23 | 100% | 37.6 | 37.7 |
| 2 | 5.43 | 7.19 | 24.5 | 22.7 | 40 | 87% | 38.3 | 38.4 |
| 3 | 7.29 | 10.7 | 22.3 | 19.5 | 52 | 76% | 39.9 | 40.0 |
| 4 | 9.25 | 14.3 | 21.2 | 17.6 | 63 | 69% | 39.9 | 40.0 |
| 6 | 12.8 | 21.2 | 18.7 | 14.3 | 78 | 57% | 41.3 | 41.5 |

### Concurrency sweep — 32768 (32K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 20.6 | 20.6 | 25.4 | 25.4 | 14 | 100% | 39.5 | 39.5 |

### Concurrency sweep — 65536 (64K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 56.0 | 56.0 | 24.1 | 24.1 | 7 | 100% | 41.7 | 41.8 |

*Scaling eff. = aggregate / (n x aggregate-at-1): 100% = perfect linear scaling. ITL p50 med = median of per-request p50s; ITL p90 max = worst user. Aggregate is wall-clock (n x 600 tokens / wall) and includes the slowest-user tail.*

## 64K annex — max_model_len 65536

Raising the certified gemma maxlen (40960) to 65536 is a KV-headroom question. The 64k closure boot — one 64k session fits the fp8-class pool, so the ladder's 64k rung is a measurement; 128k is the 8-bit refusal class only 4-bit weight budgets clear — the boot below is its own cell (`gf8x64`, outcome `CONC_OK`). Its measurements are merged into the tables and charts above (the executive summary notes the provenance); the same-contexts table below quantifies both boots' agreement, and the pools differ — this section holds the boot record.

| Field | Value |
|---|---|
| Posture | this lane + `ARC_MAXLEN=131072` |
| KV pool (annex boot) | 80,619 tokens (r240c posture) |
| Contexts measured | 64K |
| Cell outcome | CONC_OK |

![Annex per-user speed vs context length, one line per concurrency — FP8](charts/gemma-4-31b-fp8_annex_speed.svg)

<i>Annex per-user speed vs context length, one line per concurrency — FP8</i>

![Annex TTFT vs context length, one line per concurrency — FP8](charts/gemma-4-31b-fp8_annex_ttft.svg)

<i>Annex TTFT vs context length, one line per concurrency — FP8</i>

![Annex per-user speed range — the band every deployment concurrency lands in (FP8)](charts/gemma-4-31b-fp8_annex_range_speed.svg)

<i>Annex per-user speed range — the band every deployment concurrency lands in (FP8)</i>

![Annex TTFT range: single request (bottom) to worst user at max concurrency (top) — FP8](charts/gemma-4-31b-fp8_annex_range_ttft.svg)

<i>Annex TTFT range: single request (bottom) to worst user at max concurrency (top) — FP8</i>

### Same contexts, both boots

| Context | solo speed, maxlen 40960 | solo speed, 131072 | TTFT, 40960 | TTFT, 131072 | @8 speed, 40960 | @8 speed, 131072 |
|---|---|---|---|---|---|---|
| 65536 (64K) | 24.1 | 24.1 | 56.0 | 56.0 | - | - |

Two boots, one lane: the shared contexts agree within 0% on solo speed (worst relative delta) — the maxlen change costs nothing at short context; what it buys is the 128K point above.

> **Prompt-budget rule (banked).** The API server enforces prompt + max_tokens <= max_model_len — the completion budget counts — and gemma's tokenizer round-trip inflates decoded text ~1.026x. The annex faces size the server-side prompt budget as ctx - max_tokens - 64 (`p139_conc.py` v2, verify-and-trim); the boundary probes are banked in the r240c bank (`p140_probe_gm4x_probe_b131000.json` = the server stating the rule verbatim).

### Annex invocation (byte-exact)

### Validated invocation (what the gate ran, byte-exact)

```bash
# image vllm-arc:v0.30.0-b70-cert28 ; 2026-11-25 ; TP2 on 2x Arc Pro B70
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert28 ARC_NAME=arc-b70 \
  ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=65536 ARC_FP8=0 ARC_QUANT=fp8 \
  bash serve_arc_b70.sh     # tools/serve_arc_b70.sh in this repo
```

The wrapper is the audited source of truth (banked posture echo: `gf8x64/posture.txt`). What it expands to:

```bash
# 1) container boot layer  [serve_arc_b70.sh:199-210]
docker run -d --name arc-b70 \
  --privileged --network host --ipc host \
  --shm-size 16g \              # close-out F: 64 GB-RAM hosts
  -v /models/gemma4-31b:/models/target:ro \
  -v <chat-template.json>:/root/chat_template.json:ro \
  -v ./neo_cache:/root/.neo_cache -v ./triton_cache:/root/.triton_cache \
  -v ./inductor_cache:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert28 -c 'sleep infinity'

# 2) engine layer (docker exec)  [serve_arc_b70.sh:327-375]
export PYTHONHASHSEED=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 ZE_AFFINITY_MASK=0,1
export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0
export ARC_TRACE_FLIP=1 VLLM_XPU_ENABLE_XPU_GRAPH=1   # decode-only XPU graphs
exec vllm serve --model /models/target \
  --served-model-name gemma4-31b --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 --max-model-len 65536 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \
  --dtype float16 --quantization fp8  \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 --mamba-cache-mode=align \
  --enable-prefix-caching --trust-remote-code \
  --chat-template /root/chat_template.json \   # gemma: no qwen parsers
  --no-disable-hybrid-kv-cache-manager \
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1..320]}' \   # 64-size discipline, see wrapper
  --port 8000
```

## Configuration (copy & run on your own hardware)

### Validated invocation (what the gate ran, byte-exact)

```bash
# image vllm-arc:v0.30.0-b70-cert28 ; 2026-11-25 ; TP2 on 2x Arc Pro B70
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert28 ARC_NAME=arc-b70 \
  ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 ARC_QUANT=fp8 \
  bash serve_arc_b70.sh     # tools/serve_arc_b70.sh in this repo
```

The wrapper is the audited source of truth (banked posture echo: `gf8/posture.txt`). What it expands to:

```bash
# 1) container boot layer  [serve_arc_b70.sh:199-210]
docker run -d --name arc-b70 \
  --privileged --network host --ipc host \
  --shm-size 16g \              # close-out F: 64 GB-RAM hosts
  -v /models/gemma4-31b:/models/target:ro \
  -v <chat-template.json>:/root/chat_template.json:ro \
  -v ./neo_cache:/root/.neo_cache -v ./triton_cache:/root/.triton_cache \
  -v ./inductor_cache:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert28 -c 'sleep infinity'

# 2) engine layer (docker exec)  [serve_arc_b70.sh:327-375]
export PYTHONHASHSEED=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 ZE_AFFINITY_MASK=0,1
export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0
export ARC_TRACE_FLIP=1 VLLM_XPU_ENABLE_XPU_GRAPH=1   # decode-only XPU graphs
exec vllm serve --model /models/target \
  --served-model-name gemma4-31b --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 --max-model-len 40960 \
  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \
  --dtype float16 --quantization fp8  \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 --mamba-cache-mode=align \
  --enable-prefix-caching --trust-remote-code \
  --chat-template /root/chat_template.json \   # gemma: no qwen parsers
  --no-disable-hybrid-kv-cache-manager \
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1..320]}' \   # 64-size discipline, see wrapper
  --port 8000
```

## Test parameters

- Context lengths: 1K, 8K, 32K, 64K tokens (prompt + template) — the full ladder this report covers, annex rungs included. Solo faces: concurrency 1, 600 output tokens, `ignore_eos`.
- Concurrency sweeps: 1/2/3/4/8 at contexts 1K, 8K, 32K, 64K — barrier-started simultaneous requests, 600 output tokens each, temperature 0.0, plus one temperature-1.0 solo face per context.
- Prompts: seeded random-token contexts, unique per face and per request slot (`p138_face.py` / `p139_conc.py`, tag-seeded) — every face is a cold prefill (no prompt-cache hits).
- Client-side timing on the rig loopback; TTFT includes the full engine prefill window (at c=1 queue wait ~0; at c=N the TTFT spread IS the queue+prefill stagger).

## Notes

- Per-user speed = output tokens / decode window (spec-decode aware; with MTP one stream chunk carries multiple accepted tokens). Aggregate = n x 600 tokens / wall clock.
- The r240b 1K t0.0 face runs first after the boot warmup at a cold spec-verify batch shape — its acceptance (2.8 tok/step) sits below the family norm (3.8 at 8K); the 8K/32K faces are the steady reference. Temperature comparisons are within-harness (r240 sweep vs r240b faces, both reported).
- Prefill rate is the client-observed ctx/TTFT; the engine-side device rate (P6.36 decomposition) is ~2.1–2.9k tok/s at fp8 — the client view includes chunk drain and the decode-transition tail.
- Metric definitions, UX bands, use-case thresholds and the honest-limits statement: [METHODOLOGY.md](METHODOLOGY.md).
- Banks: `results/2026-11-24-p647-r240-reports/` (solo + temperature faces, r240) and `results/2026-11-24-p647-r240b-conc/` (concurrency faces, r240b) and `results/2026-11-24-p647-r240c-gemma128k/` (64K annex); tools `tools/ar/p138/p138_face.py`, `tools/ar/p139/p139_conc.py`.
