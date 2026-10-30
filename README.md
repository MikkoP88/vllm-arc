# vllm-arc

**Certified vLLM stack for `qwen3.8-27b-fp8` (dense hybrid-GDN + MTP) on dual Intel® Arc™ Pro B70 — TP2.**

A fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) at **v0.30.0** (package name `vllm_gaudi` preserved; the Arc/XPU work lives in `vllm_gaudi/xpu/`). Certified release: **`rel/v0.30.0-b70-cert`** — every gate GREEN with banked evidence (see [Repository model](#repository-model)).

## What this fork adds

| Area | What | Where |
|---|---|---|
| Platform | Arc XPU platform: device detection, mem-info patch, GMU conditional relax | `vllm_gaudi/xpu/platform.py` |
| Graphs | `FULL_DECODE_ONLY` XPU graphs (capture sizes 1–320), stream hygiene, first-request warmup interlock | `patches/`, `vllm_gaudi/xpu/patches/` |
| Speculative decoding | MTP (multi-token prediction) — **k3 recommended**, k4 documented alternative; acceptance gate instrument | `tools/arc_spec_gate.py` |
| KV cache | LMCache hybrid-KV connector: mamba/GDN boundary snap, TWO_HS transfer, lookup memo, pinned-memory safety law | `vllm_gaudi/xpu/patches/lmcache_hma_connector.py` |
| Quantization | FP8 weights with a block-quant **W8A16** route (vs stock per-token W8A8) | `vllm_gaudi/xpu/patches/fp8_block_w8a16.py` |
| Scheduler | Tail-split v4 for hybrid prefix-cache loads | `vllm_gaudi/xpu/patches/sched_tail_split.py` |
| Structured output | XGrammar backend, compact-JSON posture | `tools/serve_arc_b70.sh` |
| Distribution | Certified Docker image with `IMAGE_MANIFEST` (fork SHA, pinned wheels, vendored lmcache closure) | `Dockerfile`, `tools/build_image.sh` |

## Certified performance (dual Arc Pro B70, TP2, qwen3.8-27b-fp8)

| Face | Value |
|---|---|
| agg4x256 steady throughput (greedy, MTP k3) | **299.9 tok/s** |
| solo200 throughput (incl. TTFT, k3) | **84.5 tok/s** |
| TTFT t1 8k / 16k / 32k | 3.4 s / 7.3 s / 15.6 s |
| TTFT t1 64k | 40.9 s (concurrent-load conservative) |
| Warm prefix re-hit (8k) | **740 ms** |
| MTP value over no-spec (real-world sampled faces) | **1.6–2.1×** |
| 8 h soak | 46 cycles, zero errors, integrity 46/46, zero golden drift |

Full methodology and numbers: [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md).

## Quick start

```bash
# 1. Build the certified image from the tag (wheel pins enforced by md5;
#    the lmcache wheel is rebuildable from external_patches/lmcache)
git checkout rel/v0.30.0-b70-cert
tools/build_image.sh          # -> vllm-arc:v0.30.0-b70-cert

# 2. Boot (privileged is required — pidfd IPC law)
docker run -d --name arc-test --privileged --network host --ipc host \
  --shm-size 64g \
  -v /path/to/qwen3.8-27b-fp8:/models/target \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint bash vllm-arc:v0.30.0-b70-cert -c "sleep infinity"

# 3. Serve (recommended MTP depth k3)
docker exec arc-test /opt/venv/bin/vllm serve --model /models/target \
  --served-model-name qwen3.8-27b-fp8 --tensor-parallel-size 2 \
  --quantization fp8 --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 --mamba-cache-mode=align \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1..320]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8000
```

The complete validated line (all flags + the 24-var env chain) is in
[`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md); the uniform boot
interface with posture knobs is `tools/serve_arc_b70.sh`.

## Documentation

- [`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md) — how to run the certified posture
- [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md) — measured numbers (real-world temperature × MTP-depth campaign included)
- [`docs/TESTING-arc.md`](docs/TESTING-arc.md) — 13 test suites, gates, validation record
- [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) — swap an incumbent stack for this one (boot, verify, rollback)
- [`docs/envs.md`](docs/envs.md) — complete env/knob control surface + naming map

## Repository model

- **`main` is linear and protected**: changes land as squash commits from per-item lanes (`feat/arc-pNN-*`); direct pushes, force pushes, and deletions are blocked; PRs are required.
- **Certified releases are annotated tags** (`rel/v0.30.0-b70-cert`) pointing at the certified `main` commit; the Docker image embeds the fork SHA in `/etc/vllm-arc/IMAGE_MANIFEST` for A/B verification.
- **Evidence is in-tree**: every validation leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT + raw outputs) — 60+ legs, P0 through P5.
- The `feat/arc-*` branches are kept deliberately: squash-merging leaves the per-item development history nowhere else.

## Upstream & license

Fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) @ tag `v0.30.0` (the vLLM hardware plugin for Intel® Gaudi®). Upstream's README and docs for the base plugin: [vllm-gaudi docs](https://vllm-gaudi.readthedocs.io/en/latest/index.html). License: Apache-2.0 (see [LICENSE](LICENSE)).
