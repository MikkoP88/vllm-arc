# Gemma-4-31B on 2x Intel Arc Pro B70 — FP8, no MTP

| Field | Value |
|---|---|
| Organization | Google |
| Parameters | 31B dense hybrid (two-head-size GDN, no MTP head) |
| Precision | FP8 (online, bf16 base checkpoint) |
| Accelerators | 2x Intel Arc Pro B70 (BMG-G31, 32 GB each, 64 GB total), TP2 |
| Host RAM | 64 GB (close-out F posture: --shm-size 16g, 16 GB LMCache CPU tier) |
| Engine | vLLM v0.30.0 — vllm-arc fork, image v0.30.0-b70-cert23, XPU graphs |
| KV cache dtype | fp8_e4m3 |
| Speculative decoding | disabled |
| Date | 2026-11-10 |
| Cell outcome | FACES_OK |

## Executive summary

Gemma-4-31B — the gemma online-FP8 capability lane — measured on the shipping cert23 image, solo streaming, seeded-random unique prompts (no cache hits), `ignore_eos`.

| Metric | Result |
|---|---|
| TTFT @ 1K ctx | 1.02 s |
| TTFT @ 32K ctx | 21.8 s |
| Peak decode speed | 27.0 tok/s @ 1K ctx |
| Decode @ largest ctx | 25.4 tok/s |
| Prefill rate (client, ctx/TTFT) | 1,006 (min) … 2,036 tok/s (max) |
| Success rate | 3/3 faces (ctx sweep + temp sweep) |

## Test parameters

- Context lengths: 1K, 8K, 32K tokens (prompt + template). Concurrency 1, 600 output tokens per request, `ignore_eos`.
- Prompts: seeded random-token contexts, unique per face (`p138_face.py`, tag-seeded) — every face is a cold prefill (no prompt-cache hits).
- Client-side timing on the rig loopback; TTFT includes the full engine prefill window (queue ~0 at concurrency 1).

## Configuration (copy & run on your own hardware)

#### Validated invocation (what the r138 gate ran, byte-exact)

```bash
# image vllm-arc:v0.30.0-b70-cert23 ; 2026-11-10 ; TP2 on 2x Arc Pro B70
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert23 ARC_NAME=arc-b70 \
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
  -v ./neo_cache:/root/.neo_cache -v ./triton_cache:/root/.triton_cache \
  -v ./inductor_cache:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert23 -c 'sleep infinity'

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
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --no-disable-hybrid-kv-cache-manager \
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1..320]}' \   # 64-size discipline, see wrapper
  --port 8000
```

## Results — context sweep (temperature 0.0)

| Context | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Tok/step | ITL p50 (ms) | ITL p90 (ms) | ITL p99 (ms) |
|---|---|---|---|---|---|---|---|
| 1024 (1K) | 1.02 | 1,006 | 27.0 | 1.00 | 37.1 | 37.3 | 37.4 |
| 8192 (8K) | 4.02 | 2,036 | 26.6 | 1.00 | 37.7 | 37.7 | 38.1 |
| 32768 (32K) | 21.81 | 1,502 | 25.4 | 1.00 | 39.6 | 39.6 | 39.9 |

## Charts

![TTFT vs context length — FP8, no MTP](charts/gemma-4-31b-fp8_ttft.svg)

<i>TTFT vs context length — FP8, no MTP</i>

![Decode speed vs context length — FP8, no MTP](charts/gemma-4-31b-fp8_decode.svg)

<i>Decode speed vs context length — FP8, no MTP</i>

![Prefill rate vs context length — FP8, no MTP](charts/gemma-4-31b-fp8_prefill.svg)

<i>Prefill rate vs context length — FP8, no MTP</i>

![Inter-token latency vs context length — FP8, no MTP](charts/gemma-4-31b-fp8_itl.svg)

<i>Inter-token latency vs context length — FP8, no MTP</i>

## Notes

- Decode speed = output tokens / decode window (spec-decode aware; with MTP one stream chunk carries multiple accepted tokens — Tok/step is the acceptance view).
- Prefill rate is the client-observed ctx/TTFT; the engine-side device rate (P6.36 decomposition) is ~2.1–2.9k tok/s at fp8 — the client view includes chunk drain and the decode-transition tail.
- Bank: `results/2026-11-10-p637-reports/` (faces, postures, serve logs); face tool: `tools/ar/p138/p138_face.py`.
