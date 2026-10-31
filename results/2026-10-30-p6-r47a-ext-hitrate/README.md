# r47a — P6.1 external-hit-rate root cause (E5) + P6.3 log-audit capture

**Date**: 2026-10-30 (evening leg)
**Serve**: fresh `arc-test` from `vllm-arc:v0.30.0-b70-cert2`, certified posture
(graph=1, spec k=3, LMCache on), `LMCACHE_LOG_LEVEL=DEBUG` via
`/root/build/p61_serve.sh`; health 200; log `/root/serve_full.log`.
**Question (owner directive)**: LMCache enabled and working, but the engine log
reads `External prefix cache hit rate: 0.0%` forever — is the cache broken?

## VERDICT: EXTERNAL_RETRIEVAL_WORKS — the 0.0% is upstream-designed behavior

Eviction probe (`lmc_evict_probe.py`, salt `evictprobe-fixed-20261030`,
8021-token prompt, 64×7.4k-token churn):

| Phase | queries_total delta | hits_total delta | Wall |
|---|---|---|---|
| baseline | 0 | 0 | — |
| 1-STORE (P, max_tokens 16) | +16315 | +0 | 11.15 s |
| 2-EVICT (64 churn prompts) | +474654 | +0 | 227 s |
| 3-RETRIEVE (byte-identical P) | **+8021** | **+6656** | 1.68 s |

`3-RETRIEVE` is the discriminator: the churn evicted every local copy of P, so
the local APC could not absorb the resend; the scheduler asked the connector
and LMCache returned **6656 of 8021 tokens (13 of 15 full 512-token chunks)**
in 1.68 s. Evidence:

- `serve_retrieved_keylines.txt` —
  `LMC joint lookup: req=… attn_hit=7488 mamba_hit=6656 -> 6656` (the v136
  joint clamp bounding the resume depth to the deepest BANKED mamba boundary),
  then both workers: `Retrieved 6656 out of 6656 required tokens … size:
  0.1079 gb, cost 178.5 ms, throughput: 0.6044 GB/s` (+ the mamba-state slice
  `Retrieved 832 out of 832`).
- `metrics_external.txt` — final counters:
  `external_prefix_cache_queries_total 498990.0`, `external_prefix_cache_hits_total 6656.0`.
- `hitrate_log_lines.txt` — the engine log line itself moving off zero for the
  first time once real external hits existed:
  `External prefix cache hit rate: 1.3%` (21:09:24 onward). 6656/498990 = 1.33%
  — the line is a **cumulative lifetime average** (loggers.py:219), which is
  why even real hits read ~0.0% on a high-query serve.

## E5 — 100% decomposition of every zero (upstream-aligned)

1. **Local-first absorption (the dominant term).** Upstream vLLM 0.30 consults
   the connector at admission with the block-aligned LOCAL hit
   (`scheduler.py:936`, `partial_tail`/`block_aligned_local`) and books
   `connector_prefix_cache_hits = num_external_computed_tokens` — the *used*
   external portion — only at `scheduler.py:1196` when `queries != 0`. The
   external tier is designed as a **supplement** (KVConnectorBase_V1: "beyond
   what is already computed"): when the local APC already covers a repeated
   prefix (owner's steady serve: 48% internal hit rate), the external
   contribution is 0 by construction. Not a bug — the upstream contract.
2. **Fresh prefixes.** No repeat anywhere → connector answers 0 → true zero.
3. **Cumulative-average display.** Even when hits occur, the logged rate is
   `hits_total/queries_total` over engine lifetime; low external-hit volume
   reads 0.0% indefinitely (our +6656 shows as 1.3% against 499k queries).
4. **Probe-forced recovery works.** With local evicted, the chain
   store→evict→retrieve returns 83% of full chunks (6656/7680; the shortfall is
   the v136 joint clamp to the deepest banked mamba boundary — by design, only
   materializable scheduler-chunk states can be resumed).
5. **Arithmetic note (benign).** STORE phase booked +16315 queries for an
   8021-token prompt ≈ 2×+273: the scheduler queried the connector more than
   once for the same request across admission attempts (queries denominator
   only; hits unaffected). RETRIEVE booked exactly +8021.

**Consequences (P6.1 fix scope):** no engine-code metric change is warranted —
the metric is truthful. The gaps are observability/docs: the operator-facing
retrieval signal is the `Retrieved N out of M … GB/s` line (already INFO from
`cache_engine.py:962`) plus the joint-lookup DEBUG line, not the 0.0% metric;
CONFIGURATION/envs.md get a "what this metric means" note; the probe ships in
this directory as the verification tool; 0.5.6's L1/L2 hit-count Prometheus
metrics (P6.2) add per-tier counters on top.

## P6.3 capture — certified-posture WARN/ERROR inventory (63 lines)

Full inventory: `serve_warn_error_full.txt`. Triage:

**Fork-added → fix (p63 / P6.2 kit):**
- `ArcXpuPlatform: graph-safe output install failed at register() (expected
  mid-resolution; platform hooks retry)` ×12 — ours, `vllm_gaudi/xpu/__init__.py:74`;
  self-described expected → demote to DEBUG.
- `ArcXpuPlatform: ARC_GRAPH_TRACE state->False at execute_model-entry …` ×2 —
  ours, `graph_safe_output.py:594` diagnostic trace at WARNING → demote to DEBUG.
- LMCache `ContinuousUsageContext instance already created withdifferent
  metadata` **ERROR** ×2 (both workers) — upstream code, fork-triggered: our
  dual engines (attn + mamba) carry different kv metadata and
  `observability.py:1965` calls `GetOrCreate` unconditionally. Fix: kit patch
  gating that call on `is_usage_tracking_enabled()` + serve-chain env
  `LMCACHE_TRACK_USAGE=false` (see law note below).
- Telemetry law note: LMCache usage tracking is **enabled by default**
  (`usage_telemetry/identity.py:25` — opt-out `LMCACHE_TRACK_USAGE=false` or
  `DO_NOT_TRACK=1`); the champion chain sets neither, so the serve has been
  attempting anonymous usage sends (silently failing on the air-gapped rig).
  P6.2/P6.4 add `LMCACHE_TRACK_USAGE=false` to the certified chain —
  no-internet-publishing law compliance, upstream-documented knob.
- `Unknown vLLM environment variable detected: VLLM_QUANTIZE_Q40_LIB` ×1 —
  ours: `Dockerfile:37-40` sets the var to EMPTY to "not carry" it, but the
  bare name still trips vllm 0.30 env checking. Fix: delete the ENV stanza.

**Upstream — keep (documented, no action):** XPU Graph experimental ×2;
`vllm.utils.jit_monitor` Triton-JIT-during-inference ×3 (MTP kernels' first
use; advisory warmup-extension hint); LMCache no-config-file ×3 (env-var
config is our validated mode); builtin-hash fallback ×4 (vllm 0.30 lacks the
'builtin' hash export; per-worker local CPU backends don't cross-match);
controller-sender not initialized ×4 (controller-less single-node posture);
`NL_X_NB_BS_NH_TWO_HS` deprecation ×2 (GQA 2LTD layout; still shipped in
0.5.6 `gpu_connector/kv_format/specs/`); KVConnectorBase_V1 experimental ×3;
MTP num_speculative_tokens>1; mamba_ssm_dtype override; padding layers 6.25%;
fp8 k_scale 1.0 ×2; bf16→fp16 casting ×2; DeepSelect missing ×3; argparse
`--model` deprecation; Inductor disabled ×2; generation_config override;
raw-prompts deprecation (probe client style).

## Artifacts
- `lmc_evict_probe.py` / `evict_probe.log` — the probe and its full run
- `serve_retrieved_keylines.txt` — joint-lookup + Retrieved INFO lines
- `retrieved_line_count.txt` (0) — the adapter-layer layerwise "Retrieved"
  line never fires on our non-layerwise path (engine INFO line is the signal)
- `hitrate_log_lines.txt`, `metrics_external.txt` — metric wiring proof
- `serve_warn_error_full.txt` — P6.3 inventory (63 lines)
