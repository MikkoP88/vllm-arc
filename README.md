# vllm-arc

**Certified vLLM stack for `qwen3.8-27b-fp8` (dense hybrid-GDN + MTP) on dual Intel® Arc™ Pro B70 — TP2.**

A fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) at **v0.30.0** (package name `vllm_gaudi` preserved; the Arc/XPU work lives in `vllm_gaudi/xpu/`). Certified release: **`rel/v0.30.0-b70-cert4`** — every gate GREEN with banked evidence (cert4 = cert3 engine + P6.5 length-gated chunk-store deferral [opt-in long-context profile, default posture identical to cert3]; see [Repository model](#repository-model)). P6.6 (prefill performance) closed **analysis-only**: the anatomy is documented in PERFORMANCE-arc ("Prefill anatomy" — device-bound ~34–40 % MFU, MNBT/thr knobs A/B'd and documented, host glue empty), no engine change — cert4 remains current.

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
# 1. Build the certified image from the certified tag — two phases,
#    validated end-to-end (r46v/r46x evidence: the r46u glue repro passes
#    image-natively on a tag-built image).
#    The build runs ON THE RIG (or an equally prepared Linux host): it
#    needs the pinned-wheel stashes (/root/build/radiance_wheels,
#    wheels-arc-vxk, wheels-arc-lmc — md5s enforced against
#    external_patches/wheels.pins.md5 by tools/wheel_guard.sh), the
#    rig-local base image (vllm-iexperimental:base-v0.26.0), and internet
#    on the FIRST build only (torch+xpu from download.pytorch.org/whl/xpu,
#    the rest from PyPI — the stack layers cache in docker after).
#    Every patch ships in the archive itself (COPY src/ + pip install .);
#    the lmcache/vxk wheels are rebuildable from external_patches/ if a
#    stash is ever lost.
#
#    repo side (your checkout of this fork):
RIG=10.20.3.65                                  # <- your build host
SHA=$(git rev-parse rel/v0.30.0-b70-cert4)      # the certified commit
git archive --format=tar -o /tmp/fork.tar rel/v0.30.0-b70-cert4
scp /tmp/fork.tar root@$RIG:/root/build/
#    rig side (via ssh — same effect as running them on the host):
ssh root@$RIG 'tar -xOf /root/build/fork.tar tools/build_image.sh > /root/build/build_image.sh'
ssh root@$RIG bash /root/build/build_image.sh /root/build/fork.tar $SHA vllm-arc:v0.30.0-b70-cert4
#    -> wheel_guard PASSes (4/4 md5s), in-build gates run, IMAGE_MANIFEST
#       records fork_sha == $SHA (verify: docker run --rm --entrypoint cat
#       vllm-arc:v0.30.0-b70-cert4 /etc/vllm-arc/IMAGE_MANIFEST).
#       ~15 min cold; ~1 min when only the source changed (stack layers
#       cached — layer-order law in the Dockerfile).
#
#    Tag lineage: cert4 = the certified release (cert3's engine + the
#    P6.5 chunk-store deferral in the lmcint adapter — opt-in via
#    LMC_DEFER_CHUNK_STORE=1 + LMC_DEFER_MIN_TOKENS=40960 under
#    --long-prefill-token-threshold 2048; default posture identical to
#    cert3; gates results/2026-10-31-p65-cert4/). Superseded:
#    rel/v0.30.0-b70-cert3 (identical default behavior, lacks the
#    deferral hunk), rel/v0.30.0-b70-cert2 (0.5.5-era wheel — still
#    valid, older lmcache line), rel/v0.30.0-b70-cert (predates r46u —
#    never build from it).

# 2. Boot — the validated posture, annotated:
#    --privileged   REQUIRED: with CCL_ZE_IPC_EXCHANGE=pidfd it keeps
#                   oneCCL's ze_fd_manager off the opendir device walk
#                   that kills non-privileged TP2 init
#    --ipc host     share host /dev/shm
#    --shm-size 64g tmpfs ceiling, not a reservation (moot under
#                   --ipc host; guards the private-IPC fallback)
#    model :ro      the checkpoint is an input, never an output
#    -e pair        the 2 boot-layer vars of the 25-var chain (23 in step 3)
docker run -d --name arc-test --privileged --network host --ipc host \
  --shm-size 64g \
  -v /path/to/qwen3.8-27b-fp8:/models/target:ro \
  -v ~/.cache/arc/neo:/root/.neo_cache \
  -v ~/.cache/arc/triton:/root/.triton_cache \
  -v ~/.cache/arc/inductor:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint bash vllm-arc:v0.30.0-b70-cert4 -c "sleep infinity"

# 3. Serve — the certified line as an annotated script (the rig pattern:
#    script + docker cp + exec; logs land in /root/serve_full.log)
cat > serve_k3.sh <<'EOF'
#!/bin/bash
# --- serve environment: 23 of the certified 25-var chain (2 set at boot) ---
export PYTHONHASHSEED=0                 # init determinism
export ZE_AFFINITY_MASK=0,1             # one Arc Pro B70 per socket
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ZE_IPC_EXCHANGE=pidfd        # --privileged pair (TP2 init law)
export CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576
export CCL_SYCL_ALLGATHERV_TMP_BUF=0
export CCL_SYCL_ALLREDUCE_TMP_BUF=0
export VLLM_XPU_ENABLE_XPU_GRAPH=1      # graph posture law
export ARC_TRACE_FLIP=1                 # flip-payload interlock
# --- LMCache certified set (10 — keep as a set) ---
export LMCACHE_USE_EXPERIMENTAL_API=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=100
export LMCACHE_LOCAL_CPU=True
export LMCACHE_EVICT_POLICY=lr_u
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_MIN_RETRIEVE_TOKENS=1024
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_DISABLE_PINNED_MEMORY=1  # law: pinned USM >16-32 GiB/proc segfaults
export LMCACHE_TRACK_USAGE=false        # P6.3: LMCache telemetry is default-ON; rig serves offline
# --- the certified CLI: MTP k3 (recommended), LMCache ON, mode:none
#     (no torch.compile) + decode-only graphs, explicit capture list ---
exec /opt/venv/bin/vllm serve --model /models/target \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --block-size 64 \
  --dtype float16 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --mamba-cache-mode=align \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both","kv_connector_extra_config":{"use_native":true,"lmcache.extra_config":{"mamba_pool_ratio":2}},"engine_id":"arc-p40"}' \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --no-disable-hybrid-kv-cache-manager \
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8000 \
  > /root/serve_full.log 2>&1
EOF
docker cp serve_k3.sh arc-test:/root/
docker exec -d arc-test bash /root/serve_k3.sh

# 4. Health — first boot takes ~10 min to reach 200 (model load + graph capture)
curl -s -o /dev/null -w "%{http_code}\n" -m 5 http://127.0.0.1:8000/health

# Optional — OpenAI reasoning_effort compatibility (high/max→xhigh aliasing,
# none→thinking-off; default path identical either way; ships in-repo):
#   boot:  add -v $(pwd)/assets/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro
#   serve: add --chat-template /root/chat_template_qwen38_high.jinja
```

This quick start is the certified line (verified against rig
`/root/build/r42/r43_serve_k5.sh`) minus the optional chat template —
semantics per flag/env, knobs, and posture laws in
[`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md); the uniform
boot interface with posture knobs is `tools/serve_arc_b70.sh`
(rig paths).

## Documentation

- [`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md) — how to run the certified posture
- [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md) — measured numbers (real-world temperature × MTP-depth campaign included)
- [`docs/TESTING-arc.md`](docs/TESTING-arc.md) — 13 test suites, gates, validation record
- [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) — swap an incumbent stack for this one (boot, verify, rollback)
- [`docs/envs.md`](docs/envs.md) — complete env/knob control surface + naming map

## Repository model

- **`main` is linear and protected**: changes land as squash commits from per-item lanes (`feat/arc-pNN-*`); direct pushes, force pushes, and deletions are blocked; PRs are required.
- **Certified releases are annotated tags** pointing at the certified `main` commit — current: `rel/v0.30.0-b70-cert4`; superseded: `rel/v0.30.0-b70-cert3` (identical default behavior, lacks the p65 deferral hunk), `rel/v0.30.0-b70-cert2` (0.5.5-era lmcache line — valid, older), `rel/v0.30.0-b70-cert` (predates the r46u glue-alias fix — never build from it). The Docker image embeds the fork SHA in `/etc/vllm-arc/IMAGE_MANIFEST` for A/B verification.
- **Evidence is in-tree**: every validation leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT + raw outputs) — 60+ legs, P0 through P6.
- The `feat/arc-*` branches are kept deliberately: squash-merging leaves the per-item development history nowhere else.

## Upstream & license

Fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) @ tag `v0.30.0` (the vLLM hardware plugin for Intel® Gaudi®). Upstream's README and docs for the base plugin: [vllm-gaudi docs](https://vllm-gaudi.readthedocs.io/en/latest/index.html). License: Apache-2.0 (see [LICENSE](LICENSE)).
