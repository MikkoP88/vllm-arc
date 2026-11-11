# Qwen3.8-27B on 2x Intel Arc Pro B70 — FP8, no MTP

| Field | Value |
|---|---|
| Organization | Qwen (Alibaba) |
| Parameters | 27B dense hybrid-GDN (+MTP head) |
| Precision | FP8 (fp8 checkpoint, compressed-tensors) |
| Accelerators | 2x Intel Arc Pro B70 (BMG-G31, 32 GB each, 64 GB total), TP2 |
| Host RAM | 64 GB (close-out F posture: --shm-size 16g, 16 GB LMCache CPU tier) |
| Engine | vLLM v0.30.0 — vllm-arc fork, image v0.30.0-b70-cert23, XPU graphs |
| KV cache dtype | fp8_e4m3 |
| Speculative decoding | disabled |
| Max model len | 262144 |
| KV pool (this boot) | 494,979 tokens (r139 posture) |
| Date | 2026-11-11 |
| Cell outcome | FACES_OK |

## Executive summary

Infrastructure decisions require real performance data. This report measures user-facing performance — how many concurrent requests the configuration supports at a given context length before streaming speed or first-token wait degrade below the user-experience bands.

This benchmark evaluates **Qwen3.8-27B** (Qwen (Alibaba); 27B parameters, dense hybrid-GDN) running **FP8 (fp8 checkpoint, compressed-tensors)** on **2x Intel Arc Pro B70** (32 GB each, TP2) — the MTP-off ablation of the incumbent.

Test parameters: context lengths 1K–128K tokens; concurrency 1–8; 600 output tokens per request; unique seeded-random prompts (cold prefill, no prefix-cache hits); no speculative decoding; KV cache `fp8_e4m3`; temperature 0.0 with paired 1.0 faces.

| Metric | Result |
|---|---|
| Generation speed, single request | 34.0 tok/s (1K) → 24.8 tok/s (128K) |
| TTFT, single request | 1.79 (1K) → 101.7 (128K) |
| Peak per-user speed (solo) | **34.0 tok/s** @ 1K ctx (fast) |
| Peak system throughput | **187 tok/s** @ 8 concurrent, 1K ctx — ≈ 672.84 K tokens/hour |
| Throughput scaling (1 → 8, 1K ctx) | 6.0x (75% of ideal linear) |
| Interactive chat capacity (8K) | ~2 concurrent requests (estimated between the measured 2 and 3 points) |
| Long-context assistant (RAG) capacity (32K) | thresholds exceeded at concurrency 1 - not recommended |
| Document processing (batch) capacity (128K) | ~1 concurrent requests (estimated between the measured 1 and 2 points) |
| Success rate | **100.0%** across 69 requests (23 faces green) |

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
| Peak (1K, 8 req) | 187 tok/s |
| 8K ctx, 4 req | 71 tok/s |
| 8K ctx, 8 req | 97 tok/s |
| 32K ctx, 4 req | 28 tok/s |
| 32K ctx, 8 req | 32 tok/s |

At the peak point this configuration produces approximately **672.84 K tokens/hour** — the number that matters for batch workloads (document processing, synthetic-data generation, offline analysis).


## Per-user generation speed

The streaming rate each user experiences. As concurrency rises, per-user speed falls because the accelerators are shared; with MTP, multiple accepted tokens ride one stream chunk (spec-aware accounting).

| Context | Solo (tok/s) | class | Best @ conc (tok/s) | @ conc 8 (tok/s) | class |
|---|---|---|---|---|---|
| 1024 (1K) | 34.0 | fast | 34.0 | 28.8 | fast |
| 8192 (8K) | 33.4 | fast | 33.4 | 21.2 | acceptable |
| 32768 (32K) | 31.2 | fast | 31.3 | 10.5 | slow |
| 65536 (64K) | 28.9 | fast | - | - | - |
| 131072 (128K) | 24.8 | acceptable | 24.8 | - | - |

### How fast is this?

| Speed | User experience |
|---|---|
| < 15 tok/s | Slow - may be slower than reading speed |
| 15-25 tok/s | Acceptable - keeps pace with reading |
| 25-50 tok/s | Fast - exceeds reading speed |
| >= 50 tok/s | Very fast - text appears nearly instantly |

Solo speed spans **24.8–34.0 tok/s** (128K to 1K context); at concurrency 8 the floor is 10.5 tok/s (32K ctx).

![Per-user decode speed vs context length (solo) — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_decode.svg)

<i>Per-user decode speed vs context length (solo) — FP8, no MTP</i>

![Per-user decode speed vs concurrency — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_conc_speed.svg)

<i>Per-user decode speed vs concurrency — FP8, no MTP</i>

## Time to first token

Time from request submission to the first streamed token — the primary responsiveness metric. Client-side on the rig loopback, it decomposes into queue wait (rising with concurrency) and prefill (rising with context); the technical analysis below separates the two.

| Context | Solo (s) | class | @ conc 8 mean (s) | @ conc 8 worst (s) | class |
|---|---|---|---|---|---|
| 1024 (1K) | 1.79 | responsive | 4.74 | 5.26 | slow |
| 8192 (8K) | 4.18 | noticeable | 19.7 | 28.3 | frustrating |
| 32768 (32K) | 15.9 | frustrating | 74.9 | 126.8 | frustrating |
| 65536 (64K) | 38.3 | frustrating | - | - | - |
| 131072 (128K) | 101.7 | frustrating | - | - | - |

### How responsive is this?

| TTFT | User experience |
|---|---|
| < 0.5 s | Feels instant |
| 0.5-2 s | Feels responsive |
| 2-5 s | Noticeable but still acceptable |
| 5-10 s | Feels slow; acceptable at higher context |
| > 10 s | Can be frustrating; users may retry or abandon |

> **Caching note.** Every face uses a unique seeded-random prompt — a deliberate cold prefill, the worst case. The certified posture runs prefix caching (LMCache): in production, incremental conversations pay TTFT only for new tokens, which is dramatically faster than these numbers.

![TTFT vs context length (solo) — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_ttft.svg)

<i>TTFT vs context length (solo) — FP8, no MTP</i>

![TTFT vs concurrency (mean and worst user) — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_conc_ttft.svg)

<i>TTFT vs concurrency (mean and worst user) — FP8, no MTP</i>

## Capacity analysis

### Interactive chat (8K context)

Conversational assistant sessions: system prompt plus a few turns fit 8K. Both first-token wait and streaming speed matter. Thresholds: TTFT < 10s · speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 | @ 4 | @ 8 |
|---|---|---|---|---|---|
| TTFT worst user (s) | 3.78 | 7.32 | 10.8 *(exceeded)* | 14.4 *(exceeded)* | 28.3 *(exceeded)* |
| Per-user mean speed (tok/s) | 33.4 | 31.5 | 29.3 | 27.8 | 21.2 |

**Capacity limit: ~2 concurrent requests (estimated between the measured 2 and 3 points)**

### Long-context assistant (RAG) (32K context)

Retrieval-augmented / large-prompt assistants at 32K context. Thresholds: TTFT < 10s · speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 | @ 4 | @ 8 |
|---|---|---|---|---|---|
| TTFT worst user (s) | 16.0 *(exceeded)* | 31.8 *(exceeded)* | 47.5 *(exceeded)* | 63.4 *(exceeded)* | 126.8 *(exceeded)* |
| Per-user mean speed (tok/s) | 31.3 | 24.9 | 20.6 | 17.2 | 10.5 *(below)* |

**Capacity limit: thresholds exceeded at concurrency 1 - not recommended**

### Document processing (batch) (128K context)

Offline summarization / extraction at the largest measured context. Turnaround matters, not first-token wait: the bar is streaming speed (and total throughput). Thresholds: speed >= 15 tok/s.

| Metric | @ 1 | @ 2 | @ 3 |
|---|---|---|---|
| TTFT worst user (s) | 101.7 | 203.6 | 305.9 |
| Per-user mean speed (tok/s) | 24.8 | 14.7 *(below)* | 10.2 *(below)* |

**Capacity limit: ~1 concurrent requests (estimated between the measured 1 and 2 points)**

### KV pool (the whole-pool bound)

This boot exposed a KV pool of **494,979 tokens** (r139 posture). A session occupies ctx + 664 (output + scheduler slack), so the pool holds:

| Context | Pool capacity (sessions) | Measured green @ conc | c8 fully resident? |
|---|---|---|---|
| 1024 (1K) | 293 | 8 (green) | yes |
| 8192 (8K) | 55 | 8 (green) | yes |
| 32768 (32K) | 14 | 8 (green) | yes |
| 65536 (64K) | 7 | - | marginal |
| 131072 (128K) | 3 | 3 (green) | no |

*Sessions are the whole-pool bound; `--max-num-seqs 64` bounds the running batch above it. TurboQuant dtypes multiply the pool up to 3.68x: see the [KV-cache dtype report](qwen3.8-27b-fp8-kv-cache-dtype.md).*

## Technical analysis

### Queue + prefill decomposition (derived)

At concurrency 1 the queue is empty, so solo TTFT is pure prefill. Under the barrier burst the added first-token wait — `ttft_mean(n) − ttft_mean(1)` — is the queue+prefill-share stagger the Nth user pays. Derived client-side; engine-internal queueing is not exported by this engine version.

| Context | added wait @ n2 | @ n4 | @ n8 |
|---|---|---|---|
| 1024 (1K) | +-0.2 s | +1.0 s | +3.1 s |
| 8192 (8K) | +3.0 s | +8.1 s | +15.9 s |
| 32768 (32K) | +9.8 s | +26.9 s | +58.9 s |
| 131072 (128K) | +53.3 s | - | - |

![Added first-token wait vs concurrency (derived queue+prefill stagger) — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_queue.svg)

<i>Added first-token wait vs concurrency (derived queue+prefill stagger) — FP8, no MTP</i>

### Per-user prefill speed

Rate at which input context is processed before the first token. Client-observed (ctx / TTFT at queue ~0); the engine-side device rate is higher — the client view includes chunk drain and the decode-transition tail.

| Context | prefill @ n1 (tok/s) | @ n4 | @ n8 |
|---|---|---|---|
| 1024 (1K) | 605 | 378 | 216 |
| 8192 (8K) | 2164 | 692 | 416 |
| 32768 (32K) | 2047 | 763 | 437 |
| 131072 (128K) | 1289 | - | - |

![Client-observed prefill rate vs context length (solo) — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_prefill.svg)

<i>Client-observed prefill rate vs context length (solo) — FP8, no MTP</i>

### Inter-token latency

Time between consecutive streamed tokens — the smoothness of streaming. Solo p50 spans **29.5 ms** (1K) to 40.0 ms; the worst observed p99 is **41 ms** (128K). With MTP one chunk can carry multiple accepted tokens, so the perceived stream is smoother than the raw chunk gap.

![Inter-token latency vs context length (solo) — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_itl.svg)

<i>Inter-token latency vs context length (solo) — FP8, no MTP</i>

### Scaling efficiency

Percentage of ideal linear scaling achieved as concurrency rises (100% = n× the concurrency-1 throughput with no per-user cost). Real efficiency is always lower — the accelerators are shared; high efficiency at your target concurrency signals headroom.

![Scaling efficiency vs concurrency — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_conc_eff.svg)

<i>Scaling efficiency vs concurrency — FP8, no MTP</i>

### Temperature 0.0 vs 1.0 (per context)

![Solo decode speed t0.0 vs t1.0 — FP8, no MTP](charts/qwen3.8-27b-fp8-k0_temp_cmp.svg)

<i>Solo decode speed t0.0 vs t1.0 — FP8, no MTP</i>

## Results — detail tables

### Context sweep (solo, temperature 0.0)

| Context | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Tok/step | ITL p50 (ms) | ITL p90 (ms) | ITL p99 (ms) | Decode @ t1.0 (tok/s) |
|---|---|---|---|---|---|---|---|---|
| 1024 (1K) | 1.79 | 573 | 34.0 | 1.00 | 29.5 | 29.7 | 30.1 | 33.3 |
| 8192 (8K) | 4.18 | 1,958 | 33.4 | 1.00 | 30.0 | 30.0 | 30.4 | 32.8 |
| 32768 (32K) | 15.85 | 2,067 | 31.2 | 1.00 | 32.0 | 32.1 | 32.5 | 30.7 |
| 65536 (64K) | 38.29 | 1,712 | 28.9 | 1.00 | 34.7 | 34.8 | 35.0 | - |
| 131072 (128K) | 101.67 | 1,289 | 24.8 | 1.00 | 40.1 | 40.2 | 41.1 | 24.4 |

### Concurrency sweep — 1024 (1K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.69 | 1.69 | 34.0 | 34.0 | 31 | 100% | 29.5 | 29.6 |
| 2 | 1.45 | 2.07 | 32.5 | 31.4 | 60 | 97% | 29.8 | 30.0 |
| 3 | 1.71 | 2.14 | 31.6 | 30.2 | 87 | 93% | 30.6 | 30.8 |
| 4 | 2.71 | 3.30 | 31.0 | 28.4 | 109 | 88% | 30.9 | 31.1 |
| 8 | 4.74 | 5.26 | 28.8 | 24.5 | 187 | 75% | 32.6 | 32.8 |

### Concurrency sweep — 8192 (8K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.78 | 3.78 | 33.4 | 33.4 | 28 | 100% | 30.0 | 30.0 |
| 2 | 6.79 | 7.32 | 31.5 | 30.7 | 46 | 84% | 30.6 | 30.7 |
| 3 | 9.20 | 10.8 | 29.3 | 25.6 | 60 | 73% | 31.4 | 31.5 |
| 4 | 11.8 | 14.4 | 27.8 | 22.0 | 71 | 64% | 31.9 | 32.0 |
| 8 | 19.7 | 28.3 | 21.2 | 14.0 | 97 | 44% | 35.2 | 35.5 |

### Concurrency sweep — 32768 (32K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 16.0 | 16.0 | 31.3 | 31.3 | 17 | 100% | 32.0 | 32.0 |
| 2 | 25.8 | 31.8 | 24.9 | 19.1 | 23 | 69% | 32.6 | 32.7 |
| 3 | 34.8 | 47.5 | 20.6 | 12.6 | 27 | 52% | 33.8 | 33.9 |
| 4 | 42.9 | 63.4 | 17.2 | 9.4 | 28 | 42% | 35.1 | 35.2 |
| 8 | 74.9 | 126.8 | 10.5 | 4.6 | 32 | 23% | 41.7 | 41.8 |

### Concurrency sweep — 131072 (128K) context (temperature 0.0)

| Concurrency | TTFT mean (s) | TTFT max (s) | Per-user mean (tok/s) | Per-user worst (tok/s) | Aggregate (tok/s) | Scaling eff. | ITL p50 med (ms) | ITL p90 max (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 | 101.7 | 101.7 | 24.8 | 24.8 | 5 | 100% | 40.0 | 40.2 |
| 2 | 155.0 | 203.6 | 14.7 | 5.0 | 5 | 55% | 40.7 | 40.8 |
| 3 | 206.7 | 305.9 | 10.2 | 2.7 | 5 | 38% | 42.7 | 43.1 |

*Scaling eff. = aggregate / (n x aggregate-at-1): 100% = perfect linear scaling. ITL p50 med = median of per-request p50s; ITL p90 max = worst user. Aggregate is wall-clock (n x 600 tokens / wall) and includes the slowest-user tail.*

## Configuration (copy & run on your own hardware)

### Validated invocation (what the gate ran, byte-exact)

```bash
# image vllm-arc:v0.30.0-b70-cert23 ; 2026-11-11 ; TP2 on 2x Arc Pro B70
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert23 ARC_NAME=arc-b70 \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1 ARC_SPEC=0 \
  bash serve_arc_b70.sh     # tools/serve_arc_b70.sh in this repo
```

The wrapper is the audited source of truth (banked posture echo: `qf0/posture.txt`). What it expands to:

```bash
# 1) container boot layer  [serve_arc_b70.sh:199-210]
docker run -d --name arc-b70 \
  --privileged --network host --ipc host \
  --shm-size 16g \              # close-out F: 64 GB-RAM hosts
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v ./neo_cache:/root/.neo_cache -v ./triton_cache:/root/.triton_cache \
  -v ./inductor_cache:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert23 -c 'sleep infinity'

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
  --port 8000
```

## Test parameters

- Context lengths: 1K, 8K, 32K, 64K, 128K tokens (prompt + template). Solo faces: concurrency 1, 600 output tokens, `ignore_eos`.
- Concurrency sweeps: 1/2/3/4/8 at contexts 1K, 8K, 32K, 128K — barrier-started simultaneous requests, 600 output tokens each, temperature 0.0, plus one temperature-1.0 solo face per context.
- Prompts: seeded random-token contexts, unique per face and per request slot (`p138_face.py` / `p139_conc.py`, tag-seeded) — every face is a cold prefill (no prompt-cache hits).
- Client-side timing on the rig loopback; TTFT includes the full engine prefill window (at c=1 queue wait ~0; at c=N the TTFT spread IS the queue+prefill stagger).

## Notes

- Per-user speed = output tokens / decode window (spec-decode aware; with MTP one stream chunk carries multiple accepted tokens). Aggregate = n x 600 tokens / wall clock.
- The r139 1K t0.0 face runs first after the boot warmup at a cold spec-verify batch shape — its acceptance (2.8 tok/step) sits below the family norm (3.8 at 8K); the 8K/32K faces are the steady reference. Temperature comparisons are within-harness (r138 sweep vs r139 faces, both reported).
- Prefill rate is the client-observed ctx/TTFT; the engine-side device rate (P6.36 decomposition) is ~2.1–2.9k tok/s at fp8 — the client view includes chunk drain and the decode-transition tail.
- Metric definitions, UX bands, use-case thresholds and the honest-limits statement: [METHODOLOGY.md](METHODOLOGY.md).
- Banks: `results/2026-11-10-p637-reports/` (solo + temperature faces, r138) and `results/2026-11-11-p637b-conc/` (concurrency faces, r139); tools `tools/ar/p138/p138_face.py`, `tools/ar/p139/p139_conc.py`.
