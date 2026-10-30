# Configuration — vllm-arc certified posture (dual Intel Arc Pro B70, TP2)

How to run the certified configuration, and the recommended knob/param
settings. Semantics of every env/knob are in [`envs.md`](envs.md) (the
complete control surface + naming map); test definitions in
[`TESTING-arc.md`](TESTING-arc.md). The uniform boot interface is
`tools/serve_arc_b70.sh` — this document is the readable form of it.

## Image

| Item | Value |
|---|---|
| Image | `vllm-arc:v0.30.0-b70-p46` (validation) / `vllm-arc:v0.30.0-b70-cert` (certified release) |
| Manifest | `/etc/vllm-arc/IMAGE_MANIFEST` inside the image — fork SHA, wheel pins, dependency closure (`lmcache_closure=`), build host |
| Base | upstream vllm-gaudi v0.30.0 + pinned wheels (vllm 0.30.0+xpu, triton 3.7.2+xpu, vllm_xpu_kernels 0.1.14.1+arc3, lmcache 0.5.5+arc1) |

Verify before any leg: `docker run --rm --entrypoint bash <img> -c "cat
/etc/vllm-arc/IMAGE_MANIFEST"` and check `fork_sha` matches the tree you
built from (container-freshness law: a stale container silently runs old
code).

## Docker run (validated posture)

```bash
docker run -d --name arc-test \
  --privileged \
  --device /dev/dri --ipc host --network host \
  --shm-size 64m \
  -v /root/build/neo_cache:/root/.neo_cache \
  -v /root/build/triton_cache:/root/.triton_cache \
  -v /root/build/inductor_cache:/root/.inductor_cache \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-p46 -c 'sleep infinity'
```

`--privileged` is required with `CCL_ZE_IPC_EXCHANGE=pidfd` (keeps
oneCCL's ze_fd_manager off the opendir device walk that kills
non-privileged TP2 init — validated posture, env-inspected 2026-10-27).

## vllm serve CLI (certified line)

```bash
/opt/venv/bin/vllm serve --model /models/target \
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
  --chat-template <chat_template_qwen38_high.jinja> \
  --no-disable-hybrid-kv-cache-manager \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1..320]}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8000
```

The full capture-size list is in `tools/serve_arc_b70.sh:105`.

### MTP depth (num_speculative_tokens) — recommended 3

Data-driven recommendation from the r46t campaign
(`results/2026-10-30-p5-r46t-temps/`, measured on the p46 image,
real-world sampling sets + xhigh thinking):

| Depth | Status | Where it wins |
|---|---|---|
| **3** | **recommended (canonical)** | real-world sampled serving: +5.8 % steady agg / +3.8 % solo over k4 across temps 0.2–1.0 xhigh thinking; ties k4 on the certified greedy face (±0.7 %) |
| 4 | documented alternative | near-deterministic non-thinking workloads (acceptance ≥ ~82 %): steady solo 104.3 vs k3's 92.2 at t0.7/tp0.80/pp1.5 |
| 5 | **dropped** | was certified r43 (greedy acceptance 89.2 %); under real-world sampling acceptance collapses to 33–42 % and its depth wastes — superseded evidence retained |
| none (omit `--speculative-config`) | non-MTP baseline | 1.5–2.1× slower than MTP on every measured face — never recommended for this model |

Mechanism (measured): speculative acceptance falls as temperature
rises and falls faster with deeper drafts — at t1.0 xhigh thinking:
k3 49.5 % / k4 43.2 % / k5 33.5 %; at greedy t0: k3 85.6 %. Depth only
pays when acceptance is high; real-world sampled thinking sits well
below the crossover. Spec depth is serve-time CLI only — changing it
rebuilds no image.

## Runtime environment (24 vars — export exactly this set)

Runtime/system (15):

```bash
export PYTHONHASHSEED=0                # init determinism
export ZE_AFFINITY_MASK=0,1            # one Arc Pro B70 per socket
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576
export CCL_SYCL_ALLGATHERV_TMP_BUF=0
export CCL_SYCL_ALLREDUCE_TMP_BUF=0
export VLLM_XPU_ENABLE_XPU_GRAPH=1     # graph posture law
export ARC_TRACE_FLIP=1                # flip-payload interlock
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 # 262144 > derived max
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

LMCache certified set (9 — keep as a set):

```bash
export LMCACHE_USE_EXPERIMENTAL_API=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=100
export LMCACHE_LOCAL_CPU=True
export LMCACHE_EVICT_POLICY=lr_u
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_MIN_RETRIEVE_TOKENS=1024
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_DISABLE_PINNED_MEMORY=1 # law: pinned USM >16-32 GiB/proc segfaults
```

## vllm-arc knobs — recommended settings

**Recommended = the certified posture: export NONE of the engine knobs.**
All patches are active at their code defaults; the certified serve line
sets zero of them (rig-verified). The complete truth table with every
default and consumer site is [`envs.md`](envs.md). Summary of what you
may change and when:

| Knob | Recommended | Change only if |
|---|---|---|
| `VLLM_ARC_WARMUP_DISABLE` | unset (interlock ACTIVE) | `=1` for A/B first-traffic reproduction |
| `VLLM_ARC_GDN_FASTPATH_DISABLE` | unset (fastpath ACTIVE, saves ~6.5 ms/req) | any non-`0` value isolates the stock metadata path |
| `VLLM_ARC_LOOKUP_MEMO` | unset (=1, memo ON) | `=0` to debug lookup replay |
| `VLLM_ARC_LMC_HMA` | unset (=1, connector ON) | `=0` A/B only (boot then dies at the HMA gate) |
| `VLLM_ARC_LMC_TORCH_KV` | unset (=1) | **never `=0` in production** — native pinned ops segfault >16–32 GiB/process |
| `VLLM_ARC_LMC_MAMBA_SNAP` | unset (=1) | **never `=0`** — breaks output integrity |
| `VLLM_MAMBA_FP16_SEED_REPAIR` | unset (=1) | `=0` only to reproduce the fp16 seed defect |
| `VLLM_ARC_R35_TAIL_SPLIT` | unset (=1, v4 tail-split) | `=0` restores upstream backoff exactly (A/B control) |
| `VLLM_ARC_FP8_W8A16_DISABLE` | unset (=0, W8A16 reroute ACTIVE) | `=1` falls back to stock per-token W8A8 quant |
| `VLLM_ARC_LOOKUP_ELIDE` | unset (**0 = OFF — int tokens, opt-in**) | diagnostic-only; forfeits real joint loads |
| `VLLM_ARC_XG_COMPACT` | unset (`false`) | `=1` when you want compact-JSON xgrammar structured output (appends `--structured-outputs-config`) |
| `VLLM_ARC_GMU_RELAX` | unset (`0`) | set `0.8` if boot aborts on headroom at higher utilization (relaxes only the engine default 0.92) |
| `LMC_BOUNDARY_KEEP` / `LMC_BOUNDARY_CAP` | unset (8 / 48) | raise only with boundary-registry pressure evidence |
| `LMC_BOUND_STATS` / `LMC_BOUND_MD5` | unset (logging ON) | `=0` to quiet boundary telemetry |
| `ARC_SNAP_PROBE` | unset (silent) | diagnostics only (leg telemetry) |
| `VLLM_COMPACT_GDN` | unset | auto-detected (runner → platform); not operator-set |

Host-script layer (`tools/serve_arc_b70.sh`): `ARC_IMAGE`, `ARC_NAME`,
`ARC_GRAPH` (0 eager / 1 graph — certified = 1), `ARC_SPEC` (MTP depth
= `num_speculative_tokens`: 0 = off, default 3 = the recommendation,
4 = the D-champion alternative; only 0/3/4/5 accepted — the legged
depths), `ARC_LMC` (1 = the certified LMCache connector + the 9-var
`LMCACHE_*` chain — default; 0 = no LMCache), `ARC_GMU` (0.8),
`ARC_MNBT` (8192), `ARC_MAXSEQS` (64), `ARC_PORT` (8000).

## Posture laws (never change in production)

1. **align law** — never remove `--mamba-cache-mode=align`; never ship
   `--no-enable-prefix-caching` (flips cache mode to "none", kills the
   boundary registry and long-context cells).
2. **graph law** — always `VLLM_XPU_ENABLE_XPU_GRAPH=1` on the certified
   posture.
3. **pinned-KV law** — `VLLM_ARC_LMC_TORCH_KV` default 1 is a segfault
   guard, not a tuning knob.
4. **container freshness** — recreate the serve container whenever the
   image changes (a stale container silently runs old code).
5. **model mounts read-only** (`:ro`) — the checkpoint and chat template
   are inputs, never outputs.
