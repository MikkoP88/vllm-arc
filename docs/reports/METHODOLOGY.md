# Benchmark methodology — the performance reports

| Field | Value |
|---|---|
| Harnesses | `tools/ar/p138/p138_face.py` (solo), `tools/ar/p139/p139_conc.py` (concurrency) |
| Banks | `results/2026-11-10-p637-reports/` (r138), `results/2026-11-11-p637b-conc/` (r139), `results/2026-11-11-r140-gemma128k/` (gemma 128K annex) |
| Engine | vLLM v0.30.0 — vllm-arc fork, image `vllm-arc:v0.30.0-b70-cert23`, XPU graphs, TP2 |
| Hardware | 2x Intel Arc Pro B70 (BMG-G31, 32 GB each), host 64 GB RAM, close-out F defaults |
| Date | 2026-11-11 |

## What each report measures

User-facing performance: how many concurrent requests a configuration supports at a given context length before streaming speed or first-token wait degrade below the user-experience bands. Every number comes from a banked face JSON; nothing is estimated except where explicitly marked (`~` capacity estimates, derived queue decomposition).

## Metrics

| Metric | Definition |
|---|---|
| TTFT | Request submission (common barrier) to the first streamed token, client-side on the rig loopback. At concurrency 1 this is pure prefill (queue ~0). |
| Per-user speed | 600 output tokens / decode window (`total - ttft`) per request — spec-aware: with MTP one stream chunk carries multiple accepted tokens, and the numerator is the true token count, not the chunk count. |
| Aggregate throughput | n x 600 tokens / wall clock, where wall is the slowest request's total. Includes the staggered-start and slowest-user tail: the honest saturated-burst number. |
| Prefill rate | Context length / TTFT at concurrency 1 (client-observed; includes chunk drain and the decode-transition tail — the engine-side device rate is higher). |
| ITL p50/p90/p99 | Gaps between consecutive streamed chunks, per request; aggregated as median-of-p50s and max-of-p90/p99 (worst user). |
| Added wait (n) | ttft_mean(n) - ttft_mean(1): derived queue+prefill-share stagger under the barrier burst. |
| Scaling efficiency | aggregate(n) / (n x aggregate(1)); 100% = ideal linear scaling. |
| KV pool | Parsed from the booted engine log (`KV cache size: N tokens`) and banked per cell in `posture.txt`. |

## Presentation conventions

Every line chart in these reports measures **context length on the x-axis** — the deployment variable you choose; concurrency is the series (one line per request count, 1-8). The x-axis is **linear**: every measured context length gets one equally spaced slot (1K, 8K, 32K, 64K, 128K … in even steps, no logarithmic compression, no skipped middle labels), and the axis stops at the highest context actually measured — an unmeasured trailing context never occupies a slot. Where the concurrency sweep skipped a context (64K on the qwen sweep) the single-request line and its band edge are completed from that configuration's solo faces — the same single-request measurement on the same certified image; loaded lines keep their honest gaps, and band shading never bridges an unmeasured context. Derived-metric charts (queue stagger, scaling efficiency) keep the full ladder on the axis with an empty slot where the metric was not measured, and a break inside any line is always a real gap — no line chords across an unmeasured context. The gemma MXFP4 tables and charts carry the 128K annex-boot measurements, same image and quantization posture (the annex section quantifies both boots' agreement at the shared contexts). Range-band charts show the measured envelope across concurrency (top edge = single request, bottom edge = maximum tested concurrency; for TTFT the envelope inverts — the loaded worst user is the top edge). Every chart point carries its value label, and dashed horizontal lines mark the user-experience thresholds from the bands below. Categorical comparisons (KV dtype, sampling temperature, per-config leaderboards) use bar charts. Each section places its table and its chart under the same heading, with an *Interpretation:* paragraph that reads the banked numbers — the analysis ships with the data. Presentation format follows the public millstoneai.com inference-benchmark pages, extended with copyable byte-exact configurations, banked refusal records, and spec-aware MTP accounting.

## User-experience bands

### Streaming speed

| Speed | Experience |
|---|---|
| < 15 tok/s | Slow - may be slower than reading speed |
| 15-25 tok/s | Acceptable - keeps pace with reading |
| 25-50 tok/s | Fast - exceeds reading speed |
| >= 50 tok/s | Very fast - text appears nearly instantly |

### Time to first token

| TTFT | Experience |
|---|---|
| < 0.5 s | Feels instant |
| 0.5-2 s | Feels responsive |
| 2-5 s | Noticeable but still acceptable |
| 5-10 s | Feels slow; acceptable at higher context |
| > 10 s | Can be frustrating; users may retry or abandon |

## Use-case thresholds

The use-case guidance in each report applies the band edges: interactive work holds TTFT < 10 s (top of the *slow* band) and speed >= 15 tok/s (the *acceptable* floor); batch document processing holds the speed bar only. Capacity is the largest measured concurrency where both bars hold; `~` marks a linear estimate between two measured points.

## Prompt law (cold prefill, always)

Every face and every concurrency slot uses a unique seeded-random prompt (seed derives from the face tag and slot). Two reasons: shared prefixes would retrieve from the prefix cache and stop measuring prefill; and identical prompts across slots would corrupt the concurrency measurement. The flip side: all TTFT numbers here are worst-case cold prefill — production traffic with prefix caching (LMCache, enabled in the certified posture) pays TTFT only for new tokens.

## Honest limits

- Client-side timing only: queue wait is derived (ttft_mean(n) - ttft_mean(1)), not exported by the engine.
- No GPU telemetry (utilization, power, temperature) is captured in these banks.
- A concurrency face requires all n requests to succeed (`n_fail = 0`); failed faces are banked and shown, never averaged away.
- By-design capacity skips (pool < n x (ctx + 664)) are recorded as `KV_CAP` rows in the gate log — the measured capacity wall, not a failure.
- A dtype or posture that refuses to boot is a valid, banked outcome with its serve log.

## Reproducing

Each report's **Configuration** block is the byte-exact validated invocation (`tools/serve_arc_b70.sh` + the banked posture echo). The gate scripts are `tools/ar/r139_conc.sh` (concurrency matrix) and `tools/ar/r140_gemma128k.sh` (gemma 128K annex); reports regenerate with `tools/ar/p139/p139_report.py`.
