# Performance reports — 2x Intel Arc Pro B70 (vllm-arc cert28)

World-class close-out reports per model x quantization x MTP posture: executive summary and key findings, use-case guidance with capacity verdicts, system throughput, UX-banded speed and TTFT, capacity analysis, technical analysis (queue decomposition, prefill, ITL, scaling efficiency), concurrency sweeps {1,2,3,4,8}, temperature analysis, the full copyable configuration, and charts. Every line chart measures context length on the x-axis (concurrency is the series), range bands span the measured envelope, and each section carries its table, its chart, and an interpretation together. Generated 2026-11-24 from the r240 bank (`results/2026-11-24-p647-r240-reports/`) + r240b bank (`results/2026-11-24-p647-r240b-conc/`) + gemma 128K annex / 64K closure (`results/2026-11-24-p647-r240c-gemma128k/`) — the full battery re-run on the certified image by the 2026-11-24 currency campaign.

Metric definitions, UX bands and thresholds: [METHODOLOGY.md](METHODOLOGY.md).

## Qwen3.8-27B (dense hybrid-GDN, MTP k3 and no-MTP)

| Report | Precision | MTP | Peak decode (solo) | Per-user @ c8, 8K | Agg @ c8, 8K | Tokens/hr @ c8 | TTFT max @ c8 (s) | Chat capacity (8K) | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| [qwen3.8-27b-fp8-k3](qwen3.8-27b-fp8-k3.md) | FP8 (fp8 checkpoint, compressed-tensors) | MTP k3 | 93.8 tok/s @ 1K | 36.5 | 127 | 458K | 25.9 | ~3 concurrent requests | FACES_OK |
| [qwen3.8-27b-fp8-k0](qwen3.8-27b-fp8-k0.md) | FP8 (fp8 checkpoint, compressed-tensors) | no MTP | 34.0 tok/s @ 1K | 21.5 | 107 | 385K | 23.7 | ~3 concurrent requests | FACES_OK |
| [qwen3.8-27b-mxfp8-k3](qwen3.8-27b-mxfp8-k3.md) | MXFP8 (online, bf16 base checkpoint) | MTP k3 | 81.4 tok/s @ 8K | 32.7 | 116 | 416K | 28.3 | ~2 concurrent requests | FACES_OK |
| [qwen3.8-27b-mxfp8-k0](qwen3.8-27b-mxfp8-k0.md) | MXFP8 (online, bf16 base checkpoint) | no MTP | 28.7 tok/s @ 1K | 19.0 | 95 | 343K | 26.1 | ~2 concurrent requests | FACES_OK |
| [qwen3.8-27b-mxfp4-k3](qwen3.8-27b-mxfp4-k3.md) | MXFP4 W4A16 (online, bf16 base checkpoint) | MTP k3 | 103.2 tok/s @ 8K | 37.8 | 127 | 457K | 28.6 | ~2 concurrent requests | FACES_OK |
| [qwen3.8-27b-mxfp4-k0](qwen3.8-27b-mxfp4-k0.md) | MXFP4 W4A16 (online, bf16 base checkpoint) | no MTP | 36.9 tok/s @ 1K | 22.4 | 105 | 376K | 26.3 | ~2 concurrent requests | FACES_OK |

## Gemma-4-31B (dense hybrid, no MTP head; 128K annex on MXFP4)

| Report | Precision | MTP | Peak decode (solo) | Per-user @ c8, 8K | Agg @ c8, 8K | Tokens/hr @ c8 | TTFT max @ c8 (s) | Chat capacity (8K) | Outcome | 128K annex |
|---|---|---|---|---|---|---|---|---|---|---|
| [gemma-4-31b-fp8](gemma-4-31b-fp8.md) | FP8 (online, bf16 base checkpoint) | no MTP | 27.0 tok/s @ 1K | - | - | - | - | ~2 concurrent requests | FACES_OK | CONC_OK |
| [gemma-4-31b-mxfp8](gemma-4-31b-mxfp8.md) | MXFP8 (online, bf16 base checkpoint) | no MTP | 26.5 tok/s @ 1K | - | - | - | - | ~2 concurrent requests | FACES_OK | CONC_OK |
| [gemma-4-31b-mxfp4](gemma-4-31b-mxfp4.md) | MXFP4 W4A16 (online, bf16 base checkpoint) | no MTP | 38.0 tok/s @ 1K | 18.2 | 92 | 329K | 29.0 | ~2 concurrent requests | FACES_OK | 28.3 tok/s solo · cap 2 |

## Cross-cutting

- [KV-cache dtype comparison — both models + TurboQuant + capacity cliffs](qwen3.8-27b-fp8-kv-cache-dtype.md)
- [KV-cache dtype comparison — gemma-4-31B: the attempted MX KV lanes (refused at decode: hd512) + the auto/e5m2 refusals](gemma-4-31b-kv-cache-dtype.md)
- [Benchmark methodology](METHODOLOGY.md)

![Qwen: solo decode speed vs context (all configs)](charts/qwen3.8-27b_all_decode.svg)

<i>Qwen: solo decode speed vs context (all configs)</i>

![Qwen: TTFT vs context (all configs)](charts/qwen3.8-27b_all_ttft.svg)

<i>Qwen: TTFT vs context (all configs)</i>

![Gemma: solo decode speed vs context (all configs)](charts/gemma-4-31b_all_decode.svg)

<i>Gemma: solo decode speed vs context (all configs)</i>

![Gemma: TTFT vs context (all configs)](charts/gemma-4-31b_all_ttft.svg)

<i>Gemma: TTFT vs context (all configs)</i>

![Qwen: per-user speed @ c8 (8K ctx)](charts/qwen3.8-27b_all_c8user.svg)

<i>Qwen: per-user speed @ c8 (8K ctx)</i>

![Qwen: aggregate throughput @ c8 (8K ctx)](charts/qwen3.8-27b_all_c8agg.svg)

<i>Qwen: aggregate throughput @ c8 (8K ctx)</i>

![Gemma: per-user speed @ c8 (8K ctx)](charts/gemma-4-31b_all_c8user.svg)

<i>Gemma: per-user speed @ c8 (8K ctx)</i>

![Gemma: aggregate throughput @ c8 (8K ctx)](charts/gemma-4-31b_all_c8agg.svg)

<i>Gemma: aggregate throughput @ c8 (8K ctx)</i>

## Reading the numbers

- **Per-user speed (streaming UX)**: < 15 tok/s slow · 15–25 acceptable · 25–50 fast · ≥ 50 very fast.
- **TTFT**: < 0.5 s instant · 0.5–2 s responsive · 2–5 s noticeable · 5–10 s slow · > 10 s frustrating.
- **Concurrency**: r240b barrier-starts N simultaneous requests, each a full cold prefill + 600 streamed tokens — the worst-case burst; sustained interactive user counts run well below request concurrency.
- Solo faces (r240): concurrency 1, 600 output tokens, temperature 0.0 unless stated; unique per-face prompts (cold prefill). MTP lanes add the 6-point temperature sweep; r240b adds a t1.0 solo face per context on every config.
- Hardware: 2x Arc Pro B70 (32 GB each, one card per socket, TP2), host 64 GB RAM; close-out F defaults (--shm-size 16g, 16 GB LMCache CPU tier).
