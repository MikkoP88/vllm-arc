# vllm-arc

**Certified vLLM stack for `qwen3.8-27b-fp8` (dense hybrid-GDN + MTP) on dual Intel® Arc™ Pro B70 — TP2.**

[![Release](https://img.shields.io/badge/release-rel%2Fv0.30.0--b70--cert28-blue)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue)](#dependencies)
[![vLLM core](https://img.shields.io/badge/vLLM-0.30.0%2Bxpu-blue)](#dependencies)

A fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) at **v0.30.0** (package name `vllm_gaudi` preserved; the Arc/XPU work lives in `vllm_gaudi/xpu/`).

- Certified release: **`rel/v0.30.0-b70-cert28`** — every gate GREEN with banked evidence: the full release battery passed on all three MX-native KV lanes (mxfp8 / mxfp4 / mxk8v4) plus the fp8 control, gemma comparative PASS, and every record face re-discriminated on the shipping image itself. Release content, tag lineage, and validation legs: [`CHANGELOG.md`](CHANGELOG.md); release summaries: [`docs/release_notes.md`](docs/release_notes.md); current state: [`docs/STATUS-arc.md`](docs/STATUS-arc.md).
- **Project status: P6.47 certified — the MX KV-cache decode line is closed.** All three MX-native KV lanes serve at their record from shallow through long context, with 1.28–1.88× the fp8 KV-token capacity at 0.81–0.84× the fp8 decode speed. The measured close-out and the open-channel disposition live in [`docs/STATUS-arc.md`](docs/STATUS-arc.md).
- **Per-config performance reports** — [`docs/reports/`](docs/reports/README.md): every model × quantization × MTP posture (qwen3.8-27b fp8/MXFP8/MXFP4-W4A16 with k3 + no-MTP, gemma-4-31b fp8/MXFP8/MXFP4), decode speed / TTFT / prefill rate / ITL at 1K–128K context, the MTP-k3 sampling-temperature sweep, the `--kv-cache-dtype` comparison (stock set + TurboQuant presets), line charts, and the complete copyable configuration for each cell. Every patch module documented with provenance + evidence: [`docs/PATCHES-arc.md`](docs/PATCHES-arc.md).
- **Improvement-lane status: closed** — every prefill device slice is dispositioned by measurement and the stack sits at the practical in-scope floor. Per-slice table + closed-route disposition: [`docs/STATUS-arc.md`](docs/STATUS-arc.md).

## Contents

- [What this fork adds](#what-this-fork-adds)
- [Dependencies](#dependencies)
- [Certified performance](#certified-performance-dual-arc-pro-b70-tp2)
- [TurboQuant KV-cache dtypes](#turboquant-kv-cache-dtypes)
- [Quick start](#quick-start)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Repository model](#repository-model)
- [Upstream & license](#upstream--license)

## What this fork adds

Everything below ships in the certified image and is load-bearing on at
least one certified lane (full per-patch provenance, evidence and
load-layer mechanics: [`docs/PATCHES-arc.md`](docs/PATCHES-arc.md)):

| Area | What | Where |
|---|---|---|
| Platform | Arc XPU platform: B70 device detection, XPU mem-info patch, GMU conditional relax; admission + platform routing for the MX KV lanes | `vllm_gaudi/xpu/platform.py` |
| XPU graphs | `FULL_DECODE_ONLY` graphs (capture sizes 1–320) with stream hygiene, **first-request warmup interlock**, **graph-safe output copies**, **speculator full-capture coercion** (PIECEWISE→FULL), and **graph-exec double-buffer** (every graph captured ×2, alternate-exec replays — same-exec replay blocking eliminated; E2E-neutral hygiene fix) | `patches/`, `vllm_gaudi/xpu/patches/{first_request_warmup,graph_safe_output,speculator_full_capture,graph_exec_doublebuffer}.py` |
| Speculative decoding | MTP (multi-token prediction) — **k3: the certified, overall-best depth** (acceptance 83.7 % strict-gate at t0; deeper drafts measured and dropped — decision record in the CHANGELOG); probabilistic drafting default; acceptance-gate instrument | `tools/arc_spec_gate.py`, `tools/serve_arc_b70.sh` |
| **MX-native KV cache** | The certified `ARC_KVDTYPE=mxfp8 / mxfp4 / mxk8v4` + `ARC_MXKV_NATIVE=1` lane: `MxAttentionBackend` engine integration, slot layout + pool arithmetic, bit-exact quant reference, fused Triton quantize+pack+scatter store, paged flash-decode split-KV v2, whole-cache dequant for continuation prefill, and the arc7mx native-decode bridge | `vllm_gaudi/xpu/patches/mx_kv/`, `vllm_gaudi/xpu/patches/mx_dtype_map.py`, `vllm_gaudi/xpu/patches/mx_kv_mode.py` |
| Hybrid-KV cache | LMCache hybrid-KV connector (HMA + 0.5.5 port): mamba/GDN boundary snap, TWO_HS transfer, KV-format detector, lookup memo, pinned-memory safety law; **off-thread store** (default ON — the engine thread never waits on the store body's D2H gathers); Phase-6 auto-off on the MX lanes | `vllm_gaudi/xpu/patches/lmcache_hma_connector.py`, `vllm_gaudi/xpu/patches/lmcint/`, `vllm_gaudi/xpu/patches/mamba_*_glue.py` |
| TurboQuant KV dtypes | All four `turboquant_*` presets **serve clean** (three stacked spec-verify defects fixed wheel-side, fix active by default); opt-in MQ verify kernel (+21 % agg8, experimental) | `vllm_gaudi/xpu/patches/tq_spec_fix.py`, `vllm_gaudi/xpu/patches/tq_mq_aos.py` |
| Quantization | FP8 checkpoint lane with a block-quant **W8A16** route (vs stock per-token W8A8); **online MXFP4-W4A16** (`ARC_MXFP4_MODE=w4a16`, the best online-quant decode lane); **online MXFP8-W8A16**; skinny-M decode reroute (banked OFF) | `vllm_gaudi/xpu/patches/{fp8_block_w8a16,mxfp4_w4a16,mxfp8_w8a16,skinny_m}.py` |
| Attention routing | Image-overlay patches on the stock attention stack: B70 prefill launch-config table, engine warmup gate, full-attention FA2 dispatch, sliding-window FA2 dispatch (gemma hd256 → ESVD); GDN metadata fast path | `tools/patch_arc_xpu_*.py`, `vllm_gaudi/xpu/patches/gdn_metadata_fastpath.py` |
| Scheduler | Tail-split v4 for hybrid prefix-cache loads | `vllm_gaudi/xpu/patches/sched_tail_split.py` |
| Telemetry | Silent-off diagnostics instruments: per-step host timeline (`ARC_STEP_TRACE`) + per-call distributed-collective census (`ARC_AR_TRACE`) — registry-exempt, zero cost when off | `vllm_gaudi/xpu/patches/{step_trace,ar_trace}.py` |
| Comm (image overlay) | **e4m3-wire allreduce** for the TP2 fp16 large ring (oneCCL 2022.0.0 same-version rebuild + patch, default-OFF env `CCL_SYCL_ALLREDUCE_FP8`; ships as a two-file lib overlay — opt-in 1.65× at the dominant AR) | `external_patches/oneccl/`, `tools/build_oneccl_arc.sh` |
| Structured output | XGrammar backend, compact-JSON posture; qwen3 reasoning + tool-call parsers | `tools/serve_arc_b70.sh` |
| Wheels | **arc7mx kernel wheel** (`vllm_xpu_kernels 0.1.14.1+arc7mx` build28: MX decode kernels + the r231d seqused-tightened split clamp) pinned by md5 manifest and enforced at image-build time, with in-image pin assertions on the installed `.so`s | `tools/pin_kit.sh`, `tools/wheel_guard.sh`, `external_patches/wheels.pins.md5` |
| Distribution | Certified Docker image with `IMAGE_MANIFEST` (fork SHA, pinned wheels, vendored lmcache closure); the uniform boot interface `serve_arc_b70.sh` (25-variable chain); push gates: leak sweep, currency check, docs validation | `Dockerfile`, `tools/build_image.sh`, `tools/serve_arc_b70.sh` |

- **Change log** — every change to this fork is recorded in [`CHANGELOG.md`](CHANGELOG.md): per-cert and per-phase entries with what shipped, the release tag, fork SHA, key numbers and the evidence path. Release summaries: [`docs/release_notes.md`](docs/release_notes.md).

## Dependencies

The certified image pins its full stack — nothing floats. Wheels are
md5-pinned at build time (`tools/wheel_guard.sh`) and the installed
kernel `.so`s are asserted in-image (`IMAGE_MANIFEST`).

| Component | Certified version | Notes |
|---|---|---|
| Hardware | 2 × Intel Arc Pro B70 (BMG-G31, Xe2), TP2 | 32 GB each; both cards on one CPU socket |
| Host runtime | Intel XPU stack (compute runtime + Level-Zero), Docker | validated base lineage; `--privileged` + `CCL_ZE_IPC_EXCHANGE=pidfd` for TP2 init |
| Python | 3.12 (`/opt/venv`) | base-image venv |
| vLLM core | `0.30.0+xpu` (upstream vllm-gaudi v0.30.0) | pinned wheel, md5-guarded |
| PyTorch | `2.13.0+xpu` | the validated stack freeze |
| Triton | `3.7.2+xpu` | pinned wheel |
| `vllm_xpu_kernels` | `0.1.14.1+arc7mx` (build28) | our custom wheel — MX-native decode kernels + the seqused-tightened split clamp; installed `.so` md5s asserted in-image |
| LMCache | `0.5.6+arc1` | internal wheel: upstream 0.5.6 sdist + the 4-patch arc kit, SYCL XPU build |
| oneCCL | `2022.0.0` | torch-pinned exact; the e4m3-wire AR overlay is a same-version rebuild (opt-in) |

Full pin inventory — every wheel md5, the in-image asserts, and the
rebuild recipe: [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md),
[`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md), and the
CHANGELOG cert28 entry.

## Certified performance (dual Arc Pro B70, TP2)

**Current record — measured on the shipping cert28 image** (the full
validation matrix + the per-config reports, concurrency 1 unless noted;
full reports with charts and copyable configs:
[`docs/reports/`](docs/reports/README.md)):

| Cell | Peak decode | Notes |
|---|---|---|
| qwen fp8 · MTP k3 | 93.8 tok/s @ 1K | the certified default lane |
| qwen mxfp4-W4A16 · MTP k3 | 103.3 tok/s @ 8K | best online-quant lane |
| qwen `--kv-cache-dtype mxfp4` · native · k3 | 65.2 tok/s @ 32K (n1) · 258.4 agg8 (r233) | cert28 native MX-decode record, full five-face ITL 43.4→58.1 ms (2K→32K): the r231d clamp −7.6% at 2K with long-ctx parity exact; same-ctx gap +8.2% vs fp8 @2K (ctrl rows 40.1/40.0/40.4, r235); agg8 0.84× ctrl, clamp-neutral (r236) |
| gemma mxfp4-W4A16 | 38.0 tok/s @ 1K | +43 % vs gemma mxfp8 |
| TTFT (qwen fp8 · MTP k3) | 3.76 s @ 8K | fresh-context, no caching; best-TTFT opt-in `CCL_SYCL_ALLREDUCE_FP8=1`: 3.27 s |

**Champion record faces** — the probabilistic-drafting campaign of
record (provenance and per-gate re-verification history:
[`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md); re-discriminated
on the cert28 engine — E2E-neutral vs the measurement eras). Certified
posture: probabilistic
drafting — `ARC_SPEC_METHOD=probabilistic`, the serve script's default.
qwen3.8-27b-fp8, dtype `fp8_e4m3`:

| Face | Value |
|---|---|
| t0 agg4x256 / solo200 | **291.4–303.1 / 83.7–84.9 tok/s** |
| t1.0-xhigh agg4x256 | **218–241 tok/s steady** cross-boot (highest-variance face) |
| t1.0-xhigh solo | **64–71 tok/s, uniform (no collapse face in 20 measured pairs)** |
| Temperature sweep agg (t0.2/0.4/0.6/0.8 steady) | **214.9 / 215.0 / 222.3 / 224.7 tok/s** |
| Alt sampling face (t0.7/tp0.80/pp1.5) agg / solo | **271.5–281.3 / 82.7–92.1 tok/s** |
| TTFT solo 8k / 16k / 32k / 64k | **3.76 / 7.90 / 16.93 / 38.93 s** default posture; **opt-in `CCL_SYCL_ALLREDUCE_FP8=1`: 3.27 / — / 16.29 / 35.73 s** (every face −3.8…−13 %) |
| Warm prefix re-hit 8k / 16k / 32k (t2) | **0.99 / 1.08 / 1.09 s** (warm-hit face 0.80–0.85 s) |
| MTP k3 acceptance, t0 | **83.7 %** strict-gate / **85.9 %** battery window (pos1 0.9258; byte-stable across release gates) |
| MTP over no-spec (same-boot A/B) | **t0 2.35–2.47× · t1.0 1.84–2.09× · alt 2.42–2.53×** |

Per-face provenance and methodology: [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md); gate-by-gate history: [`CHANGELOG.md`](CHANGELOG.md).

### TurboQuant KV-cache dtypes

Every `turboquant_*` kv-cache dtype serves clean (zero corruption on
all four presets, all-dtype long-return gate with the fp8 control;
the fix is proven carried across release gates). The historical
corruption was three stacked spec-verify defects, all fixed wheel-side
by `vllm_gaudi/xpu/patches/tq_spec_fix.py` (active by default on TQ
dtypes; fp8 never imports the hooked module) — root-cause narrative:
[`CHANGELOG.md`](CHANGELOG.md) cert7/cert8 entries; evidence
`results/2026-11-04-p619-tqverify/`. Knobs: `VLLM_ARC_TQ_SPEC_FIX=1`
(default), `VLLM_ARC_TQ_VERIFY_MODE=split|mq|synth` (split ships; mq =
single-pass kernel, +21 % agg8, experimental), `VLLM_ARC_TQ_MQ_SPLITS=32`
— `docs/envs.md`. Known character: identity faces (t0 variant flips /
t1-seed sporadic divergence) remain on some classes — the engine's
probabilistic-drafting class, variant hashes cross-boot-stable; the
coarsest preset (3bit) shows the widest t0 variant drift.

Preset table (capacity = the certified same-day gate values — capacity
tracks boot-day free memory; TTFT/t1.0-agg columns are the dtype
campaign faces of record. Full per-dtype guide:
`docs/CONFIGURATION-arc.md` "KV-cache dtype" and
[`docs/reports/qwen3.8-27b-fp8-kv-cache-dtype.md`](docs/reports/qwen3.8-27b-fp8-kv-cache-dtype.md)):

| Preset | KV capacity | U32k / U64k TTFT | t1.0 agg | Status |
|---|---|---|---|---|
| `fp8_e4m3` (default) | 400,739 tok | 23.16 / 50.77 s¹ | **241.10** | **the certified default — fastest aggregate; the TQ fix never engages (control-proven)** |
| `turboquant_k8v4` | 519,972 tok (1.98×) | 15.29 / 33.03 s | 172.95 | servable — spec tps 54.4 solo / 236.0 agg8 (split); **285.4 MQ mode** (experimental) |
| `turboquant_4bit_nc` | 748,549 tok (2.86×) | 15.91 / 34.02 s | 134.42 | servable — identity face varies per boot, dominant hash stable |
| `turboquant_k3v4_nc` | 843,202 tok (3.22×) | 15.48 / 33.78 s | 137.11 | servable — sporadic t1-seed pairs, t0 boot-stable |
| `turboquant_3bit_nc` | **963,764 tok (3.68×)** | 16.02 / 35.74 s | 171.32 | servable — widest t0 variant drift (coherent output, not corruption) |

¹ fp8 comparator as run in the dtype A/B (thr2048-profile face);
true-default fp8 TTFT is 17.25/39.77 s — the honest TQ TTFT edge is
−10…−16 %.

For stall relief on the certified fp8 line the measured levers remain first
choice: `ARC_MNBT=4096` or the p65 long-context profile — see "Optional
postures" below.

## Quick start

```bash
# 1. Build the certified image from the certified tag.
#    Runs ON THE RIG (or an equally prepared Linux host): needs the pinned-wheel
#    stashes, the rig-local base image, and internet on the FIRST build only
#    (stack layers cache in docker after). Every patch ships in the archive
#    (COPY src/ + pip install .); wheel pins enforced by tools/wheel_guard.sh.
#    Full provenance + the wheel-stash inventory: docs/CONFIGURATION-arc.md;
#    rebuildability audit (every image input mapped to a tracked artifact
#    or a pinned external prerequisite): results/2026-11-03-p618-cert7/
#    REPRODUCIBILITY-AUDIT.md.
RIG=<build-host>                                # <- set: your rig/build host
                                                 #    (never commit the real
                                                 #    address — publishing-
                                                 #    hygiene law, and run
                                                 #    tools/leak_sweep.sh
                                                 #    before pushing)
SHA=$(git rev-parse rel/v0.30.0-b70-cert28)      # the certified commit
git archive --format=tar -o /tmp/fork.tar rel/v0.30.0-b70-cert28
scp /tmp/fork.tar root@$RIG:/root/build/
ssh root@$RIG 'tar -xOf /root/build/fork.tar tools/build_image.sh > /root/build/build_image.sh'
ssh root@$RIG bash /root/build/build_image.sh /root/build/fork.tar $SHA vllm-arc:v0.30.0-b70-cert28
#    -> wheel_guard PASSes, IMAGE_MANIFEST records fork_sha == $SHA
#       (verify: docker run --rm --entrypoint cat
#        vllm-arc:v0.30.0-b70-cert28 /etc/vllm-arc/IMAGE_MANIFEST)
#       ~15 min cold; ~1 min when only the source changed (stack layers
#       cached — layer-order law in the Dockerfile).
#
#    cert28 = cert27 content + the r231d seqused-tightened split clamp
#    wheel (arc7mx build28: deep splits at short contexts no longer
#    serialize tiles the attention window cannot fill; the cert27
#    numerics fixes carry through). The certified bytes were cut by the
#    staged bake tools/ar/r232_build_cert28.sh (byte-pinned arc7mx
#    build28 wheel on the cert27 base); the from-source path above
#    reproduces the same engine from the tagged tree. The full tag
#    lineage — every cert's content, validation legs, and the oneCCL
#    overlay reproduction recipe — lives in CHANGELOG.md.

# 2. Boot — the validated posture (annotated semantics in
#    docs/CONFIGURATION-arc.md Step 1; key points: --privileged REQUIRED
#    with CCL_ZE_IPC_EXCHANGE=pidfd [TP2 init law]; --ipc host shares
#    host /dev/shm; --shm-size 16g is a ceiling, not a reservation;
#    model :ro — the checkpoint is an input; the -e pair are the 2
#    boot-layer vars of the 25-var chain).
docker run -d --name arc-test --privileged --network host --ipc host \
  --shm-size 16g \
  -v /path/to/qwen3.8-27b-fp8:/models/target:ro \
  -v ~/.cache/arc/neo:/root/.neo_cache \
  -v ~/.cache/arc/triton:/root/.triton_cache \
  -v ~/.cache/arc/inductor:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint bash vllm-arc:v0.30.0-b70-cert28 -c "sleep infinity"

# 3. Serve — the certified line (probabilistic drafting = the serve
#    script's default posture), byte-identical to
#    docs/CONFIGURATION-arc.md Step 2 (that file is canonical; per-var
#    semantics there + docs/envs.md). The rig pattern: script + docker cp
#    + exec; logs land in /root/serve_full.log.
#    (Best-TTFT opt-in: add export CCL_SYCL_ALLREDUCE_FP8=1 to
#    the env block — first bullet in Optional postures below.)
cat > serve_k3.sh <<'EOF'
#!/bin/bash
# --- serve environment: the certified chain as the serve script resolves
#     it for the default lane (13 runtime/system + the graph-posture var +
#     the 10-var LMCACHE_* set; 2 more vars are set at boot in Step 1) ---
export PYTHONHASHSEED=0                 # init determinism
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300  # RPC watchdog, = vLLM stock default; the ARC_RPC_TIMEOUT passthrough (only the gemma-4 lanes override: 1200 s for long-prefill faces, r108y)
export ZE_AFFINITY_MASK=0,1             # expose both B70s (device indices 0,1)
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ZE_IPC_EXCHANGE=pidfd        # pairs with --privileged (Step 1)
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
export LMCACHE_MAX_LOCAL_CPU_SIZE=16
export LMCACHE_LOCAL_CPU=True
export LMCACHE_EVICT_POLICY=lr_u
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_MIN_RETRIEVE_TOKENS=1024
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_DISABLE_PINNED_MEMORY=1  # law: pinned USM >16-32 GiB/proc segfaults
export LMCACHE_TRACK_USAGE=false        # usage telemetry is default-ON;
                                        # this rig serves offline (pairs with
                                        # the e04 wheel patch gating the
                                        # per-request usage-context creation)
# --- the certified CLI: MTP k3 + probabilistic drafting (the CLI
#     form of the serve-script default posture; ARC_SPEC_METHOD=probabilistic
#     on the boot interface produces identical runtime bytes), LMCache ON,
#     mode:none (no torch.compile) + decode-only graphs, explicit capture
#     list. ---
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
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --port 8000 \
  > /root/serve_full.log 2>&1
EOF
docker cp serve_k3.sh arc-test:/root/
docker exec -d arc-test bash /root/serve_k3.sh

# 4. Health — first boot takes ~5-6 min to reach 200 (model load + tune
#    warmup + the ×2 graph captures of the graph-exec double-buffer)
curl -s -o /dev/null -w "%{http_code}\n" -m 5 http://127.0.0.1:8000/health
```

Optional postures (all documented in full in
[`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md)):

- **Best TTFT / prefill — the recommended perf opt-in** — one env line in the serve block: `export CCL_SYCL_ALLREDUCE_FP8=1` (e4m3-wire allreduce on the dominant prefill ARs — every long-context TTFT face improves: U8k −13 %, U64k −8.2 %, U32k −3.8 %, C32k −4.6 % vs the no-overlay baseline; decode/spec neutral, task-level quality parity; silent no-op on images without the overlay; CONFIGURATION "Best-performance postures")
- **OpenAI `reasoning_effort` compatibility** — mount + pass
  `assets/chat_template_qwen38_high.jinja` (high/max→xhigh aliasing,
  none→thinking-off; default path identical either way).
- **Bystander-stall tuning — first choice** — `ARC_MNBT=4096`
  on fp8: bystander p95 2030/2587 ms (vs 3986/5295 default) at TTFT
  +8–13 %, KV +6.8 %, decode-neutral;
  menu in `docs/CONFIGURATION-arc.md`.
- **Long-context serve profile** — the deeper-stall niche for
  ≥48k prompts into concurrent decoders:
  `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` +
  `LMC_DEFER_MIN_TOKENS=40960` (U64k TTFT 51.16 s, bystander p95/max
  1580/1599 ms). All three together; never lower the min-tokens gate
  below 40960 (32k-class prompts regress +4.2 s). Prefer
  `ARC_MNBT=4096` unless the absolute stall floor is the binding
  constraint.

The uniform boot interface with posture knobs (rig paths) is
`tools/serve_arc_b70.sh`.

## Documentation

- [`CHANGELOG.md`](CHANGELOG.md) — **the changelog**: per-cert and per-phase entries (what shipped, tag, fork SHA, key numbers, evidence path)
- [`docs/release_notes.md`](docs/release_notes.md) — **release notes**: one summary per certified release
- [`docs/history/`](docs/history/) — **archived doc history** (prior certs, phase ledgers, validation record, superseded measurement eras); the living docs carry current-cert state only
- [`docs/STATUS-arc.md`](docs/STATUS-arc.md) — current release detail + improvement-lane status
- [`docs/CONFIGURATION-arc.md`](docs/CONFIGURATION-arc.md) — how to run the certified posture (the canonical serve block, knob table, posture laws)
- [`docs/PERFORMANCE-arc.md`](docs/PERFORMANCE-arc.md) — measured numbers: **the current certified record, the comprehensive dtype faces (fp8 + all four TurboQuant presets + MQ experimental mode), the posture matrices**
- [`docs/reports/`](docs/reports/README.md) — **the per-config performance reports**: model × quantization × MTP, decode/TTFT/prefill/ITL at 1K–128K context, the MTP temperature sweep, the kv-cache-dtype comparison, line charts, copyable configs
- [`docs/PATCHES-arc.md`](docs/PATCHES-arc.md) — the complete patch inventory: every module's purpose, mechanism, provenance, knobs, evidence, and status across the four loading layers
- [`docs/TESTING-arc.md`](docs/TESTING-arc.md) — 14 test suites, gates, laws
- [`docs/MIGRATION-arc.md`](docs/MIGRATION-arc.md) — swap an incumbent stack for this one (boot, verify, rollback)
- [`docs/envs.md`](docs/envs.md) — complete env/knob control surface + naming map

## Contributing

- One branch per change (`feat/arc-pNN-<slug>` or `docs/<slug>`), Conventional Commit messages, squash-merge to `main` via PR — the full flow and the branch-protection rules are in the [Repository model](#repository-model) below.
- Engineering conventions — architecture, code style, testing, and PR review rules — live in [`AGENTS.md`](AGENTS.md).
- Every change ships with its evidence: validation legs bank under `results/`, and the push gates (`tools/leak_sweep.sh`, `tools/check_currency.sh`, `tools/validate_docs.py`) must pass.

## Repository model

- **`main` is linear and protected**: changes land as squash commits from per-item lanes (`feat/arc-pNN-*`); direct pushes, force pushes, and deletions are blocked; PRs are required.
- **Certified releases are annotated tags** pointing at the certified `main` commit — current: `rel/v0.30.0-b70-cert28` (cert27 + the r231d seqused-tightened split clamp wheel — arc7mx build28 BAKED with in-image md5 asserts; native posture `ARC_KVDTYPE=mxfp8|mxfp4|mxk8v4` + `ARC_MXKV_NATIVE=1`, all three MX lanes native at their record — mxfp4 five-face ITL 43.4→58.1 ms; evidence `results/2026-11-24-p647-cert28-battery/` + `results/2026-11-24-p647-r234-cert28-itl/`). The full tag lineage — every cert's content, wheel deltas, validation legs, and supersession notes — lives in [`CHANGELOG.md`](CHANGELOG.md); release summaries: [`docs/release_notes.md`](docs/release_notes.md). The Docker image embeds the fork SHA in `/etc/vllm-arc/IMAGE_MANIFEST` for A/B verification.
- **Evidence is in-tree**: every validation leg banks to `results/YYYY-MM-DD-<phase>-r<leg>-<slug>/` (VERDICT + raw outputs) — **233 evidence dirs, 203 VERDICTs of record**, P0 through P6.47.
- **No secrets or real host addresses in-tree (publishing-hygiene law)**: tracked == pushable, so credentials, rig/host addresses, and private identities never enter tracked files; banked logs carry redacted host addresses (each banking-time scrub documented in its `results/REDACTION-*.md` note). `tools/leak_sweep.sh` must exit 0 before every push.
- The `feat/arc-*` branches are kept deliberately: squash-merging leaves the per-item development history nowhere else.

## Upstream & license

Fork of [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) @ tag `v0.30.0` (the vLLM hardware plugin for Intel® Gaudi®). Upstream's README and docs for the base plugin: [vllm-gaudi docs](https://vllm-gaudi.readthedocs.io/en/latest/index.html). License: Apache-2.0 (see [LICENSE](LICENSE)).
