# Configuration — vllm-arc certified posture (dual Intel Arc Pro B70, TP2)

How to run the certified configuration, and the recommended knob/param
settings. Semantics of every env/knob are in [`envs.md`](envs.md) (the
complete control surface + naming map); test definitions in
[`TESTING-arc.md`](TESTING-arc.md). The uniform boot interface is
`tools/serve_arc_b70.sh` — this document is the readable form of it.

## Image

| Item | Value |
|---|---|
| Image | `vllm-arc:v0.30.0-b70-cert22` — **the certified release**, tag `rel/v0.30.0-b70-cert22` (cert21 + the P6.33 skinny-M kernel family **banked OFF**: arc6 `vllm_xpu_kernels` wheel `0.1.14.1+arc6` (adds the three GEMV-regime M≤8 weight-only ops `_xpu_C.skinny_fp8_gemm_w8a16`/`skinny_fp8_gemm_w8a16_mxfp8`/`skinny_fp4_gemm_w4a16` — register dequant, deterministic split-K; plus W4A8 dispatch case arms + the v0.1.17 Option-B backports #611/#645/#650/#600; carries every arc5/arc4 op; sha256 `484e06c1…b7156`, ~352 MB rig-only per the 100 MB artifact law) + `vllm_gaudi/xpu/patches/skinny_m.py` + the platform `_XPU_PATCH_MODULES` entry + the envs `ARC_SKINNY_M` registry entry, via `tools/cert22.Dockerfile`; knob `ARC_SKINNY_M` (unset/`off` = stock routes — **the certified default** after the r127 e2e falsification: under XPU-graph replay stock oneDNN runs 420–680 GB/s device-side and the recorded skinny kernels regressed gemma solo decode −38/−42%; `on` = opt-in with per-lane M caps w16 M=1 / m816,g16 M≤2); **serve behavior byte-identical to cert21** — r127g2 cert leg: gemma default posture 27.7 solo / 192.9 agg8 = the exact r124b reference values, zero route markers; evidence `results/2026-11-08-p633-skinny-m/`). Prior certified image: `vllm-arc:v0.30.0-b70-cert21` — tag `rel/v0.30.0-b70-cert21` (cert20 + the P6.31 online-MXFP8 W8A16 route: arc5 `vllm_xpu_kernels` wheel `0.1.14.1+arc5` (adds `_xpu_C.fp8_gemm_w8a16_mxfp8` — bf16 acts × f8 weights, group-32 e8m0 weight scales, zero per-call act quant; also carries `_xpu_C.fp4_gemm_w4a8`, the banked oneDNN-refused W4A8 closed-negative) + `vllm_gaudi/xpu/patches/mxfp8_w8a16.py` + the `mxfp4_w4a16.py` w4a8 probe branch + the platform `_XPU_PATCH_MODULES` entry, via `tools/cert21.Dockerfile`; knob `ARC_MXFP8_MODE` (unset/`w8a16` = route — default; `w8a8` = stock control); validated r124/r124b same-image + r125 three-lane gate: qwen online mxfp8 U8k 3.61 / U32k 17.07 s, decode 53.8–55.6 / 265.9 agg8 tok/s = 1.37–1.55× over stock W8A8 (spec 0.9804), gemma 1.57–1.85× over stock at `ARC_MAXLEN=40960`; adjudication: checkpoint-fp8 stays the perf default, mxfp4-w4a16 the superior online lane; evidence `results/2026-11-08-p631-mxfp8-w8a16/`). Prior certified image: `vllm-arc:v0.30.0-b70-cert20` — tag `rel/v0.30.0-b70-cert20` (cert19 + the P6.30 online-MXFP4 W4A16 route: arc4 `vllm_xpu_kernels` wheel `0.1.14.1+arc4` (adds `_xpu_C.fp4_gemm_w4a16` — bf16 acts × f4 weights, group-32 e8m0 weight scales → oneDNN `jit:gemm` DPAS via weight decomposition) + `vllm_gaudi/xpu/patches/mxfp4_w4a16.py` + the platform `_XPU_PATCH_MODULES` entry, via `tools/cert20.Dockerfile`; knob `ARC_MXFP4_MODE` (unset/`w4a16` = route — default; `w4a4` = stock control); validated r120c/r120g same-image + r121 gate: qwen online mxfp4 U8k 3.66 / U32k 17.12 s, decode 62.5–65.2 / ~305 agg8 tok/s — superior to online mxfp8 on every axis 1.47–1.75×, qwen fp8 discriminator in-band = incumbent lanes untouched; evidence `results/2026-11-08-p630-mxfp4-perf/`). Prior certified image: `vllm-arc:v0.30.0-b70-cert19` — tag `rel/v0.30.0-b70-cert19` (cert18 + the P6.29 sliding-window FA2 route: sliding-window head-256 calls — gemma-4's 50/60 local layers, window (1023,0) — dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel windowed in the same triton unified-attention wrapper; `external_patches/vllm/patch_arc_xpu_fa2_slide_route.py` via `tools/cert19.Dockerfile`; 18–25x prefill / 11–15x decode in microbench, k-independent (r109a/r109a3), guards green vs global-softmax GQA torch refs (r109a6); validated on the image (r111): face8k 4.9 / face32k 17.2 (qwen-parity, 1.016x of the 16.93 record) / face120k 110.9 / face32k_2 0.9 s, decode 28.4 solo / 195 agg8 tok/s (+45 % / 2.64x over cert18), warm 32/32, G1 3/3, erraudit 0/0, BOTH route markers live per rank; qwen discriminator U8k 3.83 / U32k 17.58 s in-band vs 3.76/16.93; ngram spec for gemma-4 closed negative (upstream v1 spec path asserts "no mamba layers" on pure-attention hybrids — banked r111 traceback); engine env `ARC_ATTN_FA2` (default 1; `=0` = both routes off); evidence `results/2026-11-07-p629/`). Prior certified image: `vllm-arc:v0.30.0-b70-cert18` — tag `rel/v0.30.0-b70-cert18` (cert17 + the P6.28 FA2 route: full-attention head-512 calls — gemma-4's 10 global layers — dispatch to the `vllm_xpu_kernels` ESVD FA2 kernel in the triton unified-attention wrapper; `external_patches/vllm/patch_arc_xpu_fa2_route.py` via `tools/cert18.Dockerfile`; 34× at production geometry in microbench (r108ac/r108ad), validated on the image (r108ae): gemma faces 2.2–16.3× faster, decode +6 %, G1 3/3, erraudit 0/0; qwen3.8 discriminator U8k/U32k green with route marker 0 = the incumbent lane untouched; engine env `ARC_ATTN_FA2` (default 1; `=0` = the pure-triton escape hatch); evidence `results/2026-11-06-p628-gemma4/phaseC/cert18_l1/`). Prior certified image: `vllm-arc:v0.30.0-b70-cert17` — tag `rel/v0.30.0-b70-cert17` (the cert11 engine + the P6.28 gemma-4 chain: cert12 B70 prefill launch-config table for the triton unified-attention launcher — head-256/512 keyed points, upstream PR#60418 pattern, `external_patches/vllm/patch_arc_xpu_attn_launcher.py`; cert16 warm-gate v4 — the engine-exact int-softcap triton cache-key fix, G2 freeze layer 1; cert17 lmcint kit hardening — detector BS floor 16 + the fail-loud two-head-size `[ARC-P628-LMC-KVGATE]` boot gate, r108z gate-fire proven; **certified gemma posture: `ARC_LMC=0` + `ARC_RPC_TIMEOUT=1200`**; the qwen3.8 fp8 engine bytes carry from cert11; evidence `results/2026-11-06-p628-gemma4/`). Prior certified image: `vllm-arc:v0.30.0-b70-cert11` (P6.26 platform patch; **overlay on cert10** — every byte carries from cert10 (engine, oneCCL fp8-wire overlay rows, venv, freeze) except `vllm_gaudi/xpu/platform.py` + its regenerated pyc, r105b walk-proven 294→294 files, pip freeze diff 0) **+ the P6.26 platform patch**: `ArcXpuPlatform.supported_quantization` gains `quark` + `supports_mx() → True` — opens the **MXFP4 checkpoint lane** (`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` and `RedHatAI/Qwen3.8-27B-MXFP4xFP8`; both validated end-to-end, spec gates above the fp8 incumbent, decode penalty 1.21×/1.29× — weight-constrained lane, **fp8 stays the performance default**; `results/2026-11-06-p626-mxfp4/`). Existing lanes: zero dispatch change (kernel selection is table-keyed; the patch is config-validation-surface only — runtime-inertness grep-proven + same-day face-for-face discriminator proof r105t/t2) |
| Manifest | `/etc/vllm-arc/IMAGE_MANIFEST` inside the image — fork SHA (== cert9's `aa2ee17d…`, the vllm tree carries from cert9 through cert10), wheel pins, dependency closure (`lmcache_closure=`), build host, the cert10 overlay rows (`oneccl=2022.0.0+arc1` + lib/patch shas — **carried into cert11**), plus the cert11 rows (`cert11=P6.26 quark quant + supports_mx…`, `cert11_fork_sha=c987c2cd…`, `cert11_built=<utc>`), the P6.28 rows (`cert12=…launch-config table`, `cert12_fork_sha=a964e047`, the cert16/cert17 description rows — their fork_sha/built stamps ship EMPTY, a cosmetic build-script wart; content proven by the banked build logs + the r108z gate-fire boot), and the cert18 rows (`cert18=P6.28 FA2 route: …`, `cert18_fork_sha=3136adfe…`, `cert18_built=<utc>`), and the cert19 rows (`cert19=P6.29 sliding-window FA2 route: …`, `cert19_fork_sha=904a3de8…`, `cert19_built=<utc>`), the cert20 rows (`cert20=P6.30 MXFP4 W4A16 route: …`, `cert20_fork_sha=a918bee9…`, `cert20_vxk_wheel_sha256=c675f1a3…`), and the cert21 rows (`cert21=P6.31 MXFP8 W8A16 route: …`, `cert21_fork_sha=c8fdf32a…`, `cert21_vxk_wheel_sha256=219268cf…`), and the cert22 rows (`cert22=P6.33 skinny-M kernel family banked OFF (ARC_SKINNY_M gate: unset/off = stock routes, the certified default; on = opt-in)…`, `cert22_fork_sha=e2457389…`, `cert22_vxk_wheel_sha256=484e06c1…`) |
| Base | upstream vllm-gaudi v0.30.0 + pinned wheels (vllm 0.30.0+xpu, triton 3.7.2+xpu, lmcache 0.5.6+arc1; `vllm_xpu_kernels` 0.1.14.1 base, overlaid by the certified arc wheels — current `+arc5`, lineage + md5s in `external_patches/wheels.pins.md5`) |

Superseded images (never presented as current — the full lineage table
with per-image evidence lives in PERFORMANCE-arc "Image lineage"):

| Image | Status |
|---|---|
| `v0.30.0-b70-cert20` | superseded by cert21 — fully valid; every byte carries into cert21 plus the arc5 wheel swap + the mxfp8_w8a16 patch + the mxfp4 w4a8 probe branch + the platform entry (fp8/checkpoint/mxfp4 lanes byte-identical, r125 discriminator + w4a16 carry leg). Its online-mxfp4 records stand |
| `v0.30.0-b70-cert19` | superseded by cert20 — fully valid; every byte carries into cert20 plus the arc4 wheel swap + the mxfp4_w4a16 patch + the platform entry (fp8/checkpoint lanes byte-identical, r121 discriminator). Its gemma-4 records stand |
| `v0.30.0-b70-cert18` | superseded by cert19 — fully valid; every byte carries into cert19 plus the slide-route patch (sliding hd256 windowed calls only — every other call path identical). Its gemma-4 records are the pre-slide baseline (faces 14.2/41.5/181.5 s, decode 19.6/73.8) |
| `v0.30.0-b70-cert17` | superseded by cert18 — fully valid; every byte carries into cert18 plus the FA2-route patch on the triton unified-attention wrapper (hd512 full-attention calls only — every other call path identical; proven by the marker-0 qwen discriminator). Its gemma-4 prefill/decode records are the pre-fix baseline |
| `v0.30.0-b70-cert10` | superseded by cert11 — fully valid; every byte carries into cert11 except `vllm_gaudi/xpu/platform.py` (+pyc). Its oneCCL overlay rows ride cert11. The MXFP4/quark lane does NOT boot on it (the P6.26 rejection is the pre-fix behavior) |
| `v0.30.0-b70-cert9` | superseded by cert10 — fully valid; its engine bytes carry unchanged into cert10 (cert10 = this engine + the default-OFF oneCCL overlay; default-posture behavior identical) |
| `v0.30.0-b70-cert8` | superseded by cert9 — fully valid; all four TQ presets serve clean (P6.19 fix); its prefill TTFT faces carry the pre-P6.20 inline store path (cert9's off-thread store is strictly faster) |
| `v0.30.0-b70-cert7` | superseded by cert8 — fp8 behavior byte-identical; its `turboquant_*` boot refusal superseded by the P6.19 fix (TQ dtypes serve clean on cert8) — do not serve TQ dtypes on cert7 (refused) |
| `v0.30.0-b70-cert6` | superseded — fp8 behavior byte-identical; its servable `turboquant_*` dtypes carry the pre-P6.19 corrupted spec-verify path — **do not serve TQ dtypes on cert6** (gates `results/2026-11-02-p614-cert6/`) |
| `v0.30.0-b70-cert5` | superseded by cert6 — identical default behavior, lacks the P6.14 TQ-aware store hunk (gates `results/2026-11-01-p65-cert5/`) |
| `v0.30.0-b70-cert4` | superseded by cert5 — identical default behavior, lacks the default-OFF slice hunk (gates `results/2026-10-31-p65-cert4/`) |
| `v0.30.0-b70-cert3` | superseded by cert4 — identical default behavior, lacks the deferral hunk (gates `results/2026-10-31-p6-r47c-p48-056gates/`) |
| `v0.30.0-b70-cert2` | superseded — 0.5.5-era lmcache wheel line, still valid |
| `v0.30.0-b70-p48` | r47c gate image (same engine tree as cert3) |
| `v0.30.0-b70-p47` | r46v build-validation image |
| `v0.30.0-b70-p46` | r46t depth-campaign image |
| `v0.30.0-b70-cert` | **never build from it** — predates the r46u glue fix |

Verify before any leg: `docker run --rm --entrypoint bash <img> -c "cat
/etc/vllm-arc/IMAGE_MANIFEST"` and check `fork_sha` matches the tree you
built from (container-freshness law: a stale container silently runs old
code).

## Deploy — in execution order

The certified deployment is **three steps in this order**: boot the
container (Step 1), then serve with the environment chain and the CLI as
**one unit** (Step 2), then health-check (in Step 2). The serve CLI is
only certified with its 25-var environment exported first — Step 2
therefore ships env + CLI in a single script block, so the two can never
be separated by following the guide top to bottom.

### Step 1 — Boot the container (validated posture)

```bash
docker run -d --name arc-test \
  --privileged --network host --ipc host \
  --shm-size 64g \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v /root/build/neo_cache:/root/.neo_cache \
  -v /root/build/triton_cache:/root/.triton_cache \
  -v /root/build/inductor_cache:/root/.inductor_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert19 -c 'sleep infinity'
```

These are the validated boot bytes (rig `r46t_campaign.sh:54-63`,
`r42/r46_soak.sh:33-42`): `--shm-size 64g`, four mounts, and the two
boot-layer vars of the 25-var chain below (`:ro` on the model is the
read-only posture law — the leg scripts omit it, the law is stricter).
The optional chat-template mount (below) is added to this block only if
you use the template.

`--shm-size` is a tmpfs *ceiling*, not a reservation, and is
moot under `--ipc host` (host `/dev/shm` is used); 64g guards the
private-IPC fallback, where small values ENOSPC-crash TP2 IPC buffers.

**Hardware floor**: the hard floor is the **GPUs** — dual B70
(**31.9 GiB each, measured** — r66a devprobe: Intel [0xe223], 256 CUs
per card; the older "16 G each" in this doc was wrong): ~27 G of fp8
weights split TP2 (≈ 13.5 GiB/card) plus KV/mamba/graph pools inside the
GMU-0.8 budget (0.8 × 31.9 ≈ 25.5 GiB/card) — the measured XPU KV pool
at MNBT 8192 is **400,739 tokens** (the value reproduced byte-exact on
three separate days: p618b, the cert7 gate, the cert8 gate; the
405,491 figure in older tables was the P6.14/cert6-gate day's
free-memory state — KV capacity tracks boot-day free memory). At
MNBT 16384: 332,629 tokens (r66c). Host RAM is not
gated by the cache:
`LMCACHE_MAX_LOCAL_CPU_SIZE=100` is a single *virtual* reservation —
one malloc wrapped by `torch.frombuffer`, first-touch physical pages,
evicted (lr_u) at the cap (wheel-verified:
`lmcache/v1/memory_management.py:593`; eager allocation is the pinned
path, disabled by the p40d law). Smaller hosts fit by scaling that one
env down — a ~100 G single mmap will not survive a small-RAM kernel's
overcommit heuristic at boot. Such postures are mechanically sound but
unvalidated; the certified legs ran on the rig class (dual B70, 377 G
RAM).

`--privileged` is required with `CCL_ZE_IPC_EXCHANGE=pidfd` (keeps
oneCCL's ze_fd_manager off the opendir device walk that kills
non-privileged TP2 init — validated posture, env-inspected 2026-10-27).

### Step 2 — Serve (environment chain + CLI as ONE unit)

The CLI below is the **certified line** — the rig-certified line
(`/root/build/r42/r43_serve_k5.sh`, depth 3 per r46t) plus the
`draft_sample_method` key (probabilistic drafting — the P6.17 certified
default posture and the serve script's default since P6.17); drop that
one key (or set
`ARC_SPEC_METHOD=greedy` on the script interface) for the historical
greedy lane — byte-identical at t0, lower on every sampled face —
every flag byte-verified. It is certified **only**
together with these 23 exported vars (13 runtime/system + 10 LMCache —
the other 2 of the 25-var chain are the `-e` pair in Step 1). Ship it as
one script inside the container — never a bare interactive
`vllm serve`, which silently drops the graph law, the oneCCL block, and
the whole LMCache set:

```bash
# serve_k3.sh — run INSIDE the container (docker cp + exec below)
#!/bin/bash
# --- serve environment: 23 of the certified 25-var chain (2 set at boot) ---
export PYTHONHASHSEED=0                 # init determinism
export ZE_AFFINITY_MASK=0,1             # one Arc Pro B70 per socket
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
export LMCACHE_MAX_LOCAL_CPU_SIZE=100
export LMCACHE_LOCAL_CPU=True
export LMCACHE_EVICT_POLICY=lr_u
export LMCACHE_CHUNK_SIZE=1024
export LMCACHE_SAVE_DECODE_CACHE=True
export LMCACHE_MIN_RETRIEVE_TOKENS=1024
export LMCACHE_LOG_LEVEL=INFO
export LMCACHE_DISABLE_PINNED_MEMORY=1  # law: pinned USM >16-32 GiB/proc segfaults
export LMCACHE_TRACK_USAGE=false        # P6.3: usage telemetry is default-ON;
                                        # this rig serves offline (pairs with
                                        # the e04 wheel patch gating the
                                        # per-request usage-context creation)
# --- the certified CLI: MTP k3 + probabilistic drafting (P6.17 — the CLI
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
```

Two compile-posture laws (p67, do not re-try blind — evidence
`results/2026-10-31-p67-r67a-compile-ab/`):

1. **`mode` stays `none`** — the 0.30 REVAL of the compile-off acceptance
   law (Appendix H #1) FIRED: `mode=vllm_compile` ×
   `cudagraph_mode=FULL_AND_PIECEWISE` collapses MTP mean acceptance
   3.40/3.57 → 2.21/2.32 (−35 %) and decode t/s −39 %, with
   prefill flat — nothing compensates (KV pool +5.3 % aside).
2. **If a FULL_AND_PIECEWISE experiment is ever repeated**, the capture
   ladder must top at `max_num_seqs × (num_spec+1)` = 64 × 4 = **256**
   tokens: `gdn_attn.py:120` hard-caps captured tokens there for GDN
   models, and the stock ladder's 320 headroom (kept above for k4) makes
   the first piecewise capture boot-die with "GDN only supports
   decode-only full CUDAGraph capture". The ladder above is correct for
   the certified `mode:none` posture (FULL_DECODE_ONLY never builds the
   offending piecewise captures; the real k3 verify shape is 256).

Launch and health-check (the rig pattern; first boot ~5 min to 200 —
285–295 s at the cert7–cert10 gates: model load + graph capture):

```bash
docker cp serve_k3.sh arc-test:/root/
docker exec -d arc-test bash /root/serve_k3.sh
curl -s -o /dev/null -w "%{http_code}\n" -m 5 http://127.0.0.1:8000/health
```

`[1..320]` shorthand is **invalid JSON** and is never shipped anywhere;
copy-paste the block as-is. `--kv-transfer-config` loads the LMCache
connector — the certified posture runs LMCache **ON** (semantics of
every var: [`envs.md`](envs.md)).

**Reading the `External prefix cache hit rate: 0.0%` log line (P6.1
metric truth):** a steady 0.0 % does **not** mean the external cache is
dead. Root cause (r47a, `results/2026-10-30-p6-r47a-ext-hitrate/`):
the local prefix cache absorbs all repeats in steady state, so the
scheduler books every hit locally and never consults the external
store (`scheduler.py:936`), and the logged rate is a cumulative
average over requests that mostly had nothing external to find
(`loggers.py:219`). External retrieval itself is proven end-to-end —
the eviction probe (local cache forced small, real evictions driven)
retrieved 13/15 chunks from the external store (+6656/8021 tokens).
Verification tool for any future doubt: the r47a evict-probe leg.

**Chat template — optional, ships in-repo**
(`assets/chat_template_qwen38_high.jinja`, md5 `eb69f1c4…` =
checkpoint's own template + two edits: `reasoning_effort`
`"high"`/`"max"` → `xhigh`, `"none"` → `enable_thinking=false`).
The engine serves fine without it — it falls back to the checkpoint's
own template (identical in `tokenizer_config.json` and
`chat_template.jinja`, md5 `519239a4…`), and the default path (no
`reasoning_effort` sent) is byte-identical: xhigh thinking either way.
Mount it only if you want OpenAI-standard `reasoning_effort`
compatibility — without it, `"high"` and `"none"` requests fail with
400 (`raise_exception` in the checkpoint template). The certified line
includes it (that is the validated posture). To use it, add to the boot
block:
`-v /path/to/repo/assets/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro`
and to the serve line:
`--chat-template /root/chat_template_qwen38_high.jinja`.

### Best-performance postures (P6.24: opt-in — what to enable when)

The Step 2 block above is the certified default posture — best steady
sampled throughput, zero opt-ins. One measured lever improves a specific
objective on top of it; each recipe is a single delta, each links to its
full section, and **no two opt-ins were measured combined** (the
mechanisms are orthogonal — wire format / chunk size — but un-measured
combinations are unvalidated; one lever at a time, the matrix law):

| Objective | Recipe (delta on Step 2) | Measured effect |
|---|---|---|
| **Best TTFT / prefill — the recommended perf opt-in** | add `export CCL_SYCL_ALLREDUCE_FP8=1` to the serve env block (**cert10+ images only** — silent no-op on ≤ cert9, which lack the overlay; the overlay rows carry into cert11) | e4m3-wire allreduce on the dominant prefill ARs: **U8k 3.76→3.27 s (−13 %), U64k 38.93→35.73 (−8.2 %), U32k 16.93→16.29 (−3.8 %), C32k 1.09→1.04 (−4.6 %)** — every p65 face beats the cert9 baseline (r99 A/B/A/B interleave); 1.65× at the dominant AR; decode/spec neutral (r98: tps 0.998–1.005, spec +0.008 ≪ 0.03 bar); task-level quality parity |
| Best steady sampled throughput | **none — Step 2 as-is** (MTP k3 + probabilistic drafting is the measured optimum; the FP8 line changes nothing decode-side) | t0 agg 291.4–303.1 / solo 83.7–84.9 tok/s family |
| Lowest bystander stalls (balanced) | `ARC_MNBT=4096` | stalls −49/−51 % at TTFT +8–13 %, KV +6.8 %, decode-neutral — "fp8 stall levers" below |
| Deeper stall floor (≥48k prompts) | p65 profile: `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` | U64k bystander p95/max 1580/1599 ms — below |
| Max KV capacity (trades speed) | `--kv-cache-dtype turboquant_*` (all four presets servable since cert8) | up to 3.68× KV capacity, slower aggregate — "KV-cache dtype" below |

FP8 line placement — anywhere in the serve env block (it is read by the
**overlaid oneCCL** at the first large allreduce, not by engine code —
so it also works as a `docker run -e`, but the env block next to the
other `CCL_*` exports is the canonical spot):

```bash
# in serve_k3.sh, alongside the other CCL_* exports:
export CCL_SYCL_ALLREDUCE_FP8=1
```

On the uniform boot interface: `ARC_FP8=1` (cert10+ images; default
unset = the certified posture, exec-block bytes unchanged). The
companion knobs (`_MIN_COUNT` crossover, `_STATS` diagnostics) are
diagnostics-class — leave them unset; rows in the knob table below.
Evidence: `results/2026-11-05-p624-fp8-ar/` (r96–r100).

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

### fp8 stall levers — `ARC_MNBT=4096` (preferred) and the p65 long-context profile

A bystander stall is the inter-token gap a decoding stream suffers
while another stream's long prefill runs. Two opt-in levers cut it —
both shrink the prefill chunk and trade stall depth against TTFT;
neither changes the certified default posture, and both are
decode-neutral. **Preferred: `ARC_MNBT=4096`** — one knob, the milder
TTFT cost, a larger KV pool. The p65 profile buys the deeper stall
floor for fleets serving ≥48k prompts into concurrent decoders, at a
higher TTFT cost.

| fp8 dtype | certified default | `ARC_MNBT=4096` (preferred) | p65 profile |
|---|---|---|---|
| Configure | — (MNBT 8192) | `ARC_MNBT=4096` | thr2048 + defer pair (below) |
| Prefill chunk | 7488 tok | 3328 tok | 1664 tok |
| TTFT U32k / U64k | 17.25 / 39.77 s | 19.49 / 42.83 s | 22.88–23.15 / 51.16 s |
| Bystander gap p95 U32k / U64k | 3986 / 5295 ms | 2030 / 2587 ms | — / 1580 ms |
| Bystander gap max U64k | — | — | 1599 ms |
| KV capacity | 400,739 tok (cert7-native) | 427,666 tok (cert7-native) | 405,491 tok (P6.15 value) |
| Decode (t0) | certified family | unchanged (knob decode-neutral) | zero cost (run7n) |

Era note: the default and `MNBT 4096` columns are the cert6 P6.15
matrix (`results/2026-11-02-p615-stall-analysis/`; KV row above
re-verified natively on cert7, p618e — 400,739/427,666); the p65 column
is the cert4/cert5-era record face (`results/2026-10-31-p65-cert4/`,
`results/2026-11-01-p65-cert5/`). Both live levers were re-verified
natively on cert7 (p618e, same harness): default TTFT 18.27/39.91 s +
gaps p95 4001/5273 ms; MNBT 4096 TTFT 19.51/42.34 s + gaps p95
2052/2523 ms — all in-family
(`results/2026-11-03-p618-cert7/PERF.md`).

**Enable MNBT 4096** — one knob; the serve script maps the host var to
the CLI flag:

```bash
ARC_MNBT=4096 bash tools/serve_arc_b70.sh    # -> --max-num-batched-tokens 4096
```

Raw CLI form: add `--max-num-batched-tokens 4096` to the `vllm serve`
line. Verify: the launch echo prints `mnbt=4096`, and prefill steps
run 3328-token chunks (fp8 KV page 832 × 4).

**Enable the p65 profile** — three settings, always together (the rig
copy: `/root/build/serve_arc_b70_defer2.sh`):

```bash
# 1) engine CLI (with the serve line's other flags):
--long-prefill-token-threshold 2048
# 2) env, inside the LMC_ENV set (keeps the 25-var chain + 2):
export LMC_DEFER_CHUNK_STORE=1
export LMC_DEFER_MIN_TOKENS=40960
```

**Default posture is unchanged** — the certified serve block above sets
none of this; the deferral hunk is inert without the envs (cert4 gate:
default-posture boot + run7n + WARN/ERROR audit byte-class-identical to
cert3). Enable the p65 profile only when long prompts (≥ ~48k tokens)
arrive into a fleet with concurrent decoders.

**The problem it fixes** (P6.5, `results/2026-10-31-p65-r48a-stall-capture/`):
when a ~64k-token prompt is prefilled alongside ~12 decoding bystanders
under `--long-prefill-token-threshold 2048`, every prefill chunk emits
its own LMCache store event; the ~38 events × ~79 ms fixed overhead
amplify into **+24.6 s** TTFT for the long stream (76.89 s vs 52.27 s
with no connector) and bystander inter-token gap p95 **4829 ms** vs
~2000 ms baseline. Attribution: ~70 % connector store path ([vii-a]),
~25 % engine-side chunk cost ([viii], the P6.6 lane). At 32k the
per-chunk events are ≈ free (LMC ≡ no-LMC, 22.78 vs 22.39 s) — the fix
is a 64k-class phenomenon only.

Semantics: intermediate prefill-chunk rows keep their row (loads and
mamba addressing still ride it) but carry `can_save=False` and leave
`num_saved_tokens` un-advanced, so the final chunk stores the whole
span in **one** event. `LMC_DEFER_MIN_TOKENS` gates by prompt length —
below it, per-chunk stores are kept (they are free at 32k, and the
one-shot store would *regress* that face by +4.2 s: the ~0.5 GB
contended store runs at ~0.24 GB/s). 40960 splits the two regimes.

**Validated faces** (r48l + cert4, W2MT 4000, fresh boot):

| Face | no-thr baseline | thr2048 no-defer | thr2048 + defer (this profile) |
|---|---|---|---|
| U64k TTFT | 41.61 s (12 bys disrupted, p95 4829 ms) | 56.10 s | **51.90–51.95 s** |
| U64k bystander gap p95 / max | 2027 / 2347 ms (no-LMC) | 1969 / 2242 ms | **1585 / 1612–1997 ms** |
| U32k TTFT | — | 22.88–23.15 s | **22.88–23.15 s** (gate: unchanged) |
| run7n t0 agg / solo | 294.7–300.5 / 83.7–84.9 (cert4-native family, 5 boots) | same family under thr2048 | **zero cost** |

Store signature of the fix working: `1664`-class chunk stores = 0 for a
64k prompt, replaced by one `63232` final; at 32k the ~20 `1664`-class
stores are present as normal (the gate held). Do **not** lower
`LMC_DEFER_MIN_TOKENS` below 40960 for mixed fleets — r48k measured the
32k regression; raising it (e.g. 49152) only narrows coverage.

**Experimental, NOT certified (P6.13, gated 2026-11-01):
`LMC_STORE_SLICE_TOKENS=4096`** — spreads the deferred one-shot final-span
store across ~19 bounded windows (aligned 3328 tok each at runtime chunk
1664), one per step, re-using the exact per-chunk store call shape; default
unset = byte-identical certified behavior. Gate
(`results/2026-11-01-p65-rung3-slice/`): all safety faces green (integrity
5/5 + determinism 3/3, run7n 300.48 / 84.4, U32k family unchanged, zero
fork-added WARN/ERROR), mechanism landed exactly (`63232`-class = 0 all
session; 38 = 2×19 windows; U64k TTFT 50.72 s / 48.79 s-cold vs the 52.25 s
bar; full-hit + mid-drain replay integrity clean at 1.74/1.82 s). NOT
adopted into the certified profile: the pre-registered ≤600 ms bystander
max-gap bar was mechanically missed (1598 ms) and is recorded as mis-set
(the r48e nolmc floor is 2347 ms with zero stores — chunk compute [viii]
is the floor; the corrected claim, store-side contribution removed, is
confirmed). Enabling it is a caller decision on the corrected bar.
**Reproducible on the certified image since cert5** (the hunk ships in
`rel/v0.30.0-b70-cert5`, default-OFF): the cert5 release gate re-ran the
slice face on the tag-built image — U64k TTFT **50.77 s** (cert4 same-face
51.95), W_U64k_start max **1602 ms** vs cert4's 1997 (the one-shot tail
gone), `63232`-class **0** / `3328`×38 exact, U32k unchanged
(`results/2026-11-01-p65-cert5/`). The cert6 release gate re-ran the same
face in-family (U64k TTFT 51.66 s, `63232`-class 0, `3328` signature
exact — `results/2026-11-02-p614-cert6/`).

## MXFP4 / MXFP8 weight lanes — pre-quantized checkpoints (P6.26/P6.27; certified on cert11)

cert11 opens the non-default weight-format lanes: OCP-MX FP4 (P6.26)
and MXFP8 W8A8 (P6.27) checkpoints boot and serve under the full
certified posture (XPU graph FULL_DECODE_ONLY,
MTP k3 probabilistic, LMCache on) on the native kernels
(`XPUMxFp4LinearKernel` / `XPUMxFp8LinearKernel`). mxfp4 is the
**certified weight-constrained lane**; fp8 stays the certified
performance default — no posture change, and the incumbent fp8
regression battery (r105) passed at cert10 refs on the same image.

Three pre-quantized checkpoints validated (phase-A boot + phase-B full
posture; evidence `results/2026-11-06-p626-mxfp4/`,
`results/2026-11-06-p627-mxfp8/`):

| checkpoint | quant surface | weights/rank | KV pool | U32k TTFT | bystander p50 |
|---|---|---|---|---|---|
| `qwen3.8-27b-mxfp4xfp8` (RedHatAI) | CT W8A8-fp8 attn + W4A4-mxfp4 MLP | — | 10.38 GiB | 29.01 s | 64.7 ms |
| `qwen3.8-27b-quark-awq-mxfp4` (amd) | quark OCP-MX W4A4, all quantized linears | **9.03 GiB** | 12.16 GiB | 32.28 s | 68.8 ms |
| `qwen3.8-27b-mxfp8` (intel-ai AutoRound, P6.27) | CT W8A8 MXFP8-dynamic group-32 E8M0 + quantized MTP drafter | 14.74 GiB | 6.25 GiB | **26.67 s** | 81.8 ms |
| (fp8 incumbent, ref) | W8A8-fp8 block-scaled | ~13.6 GiB | — | 16.13–16.45 s | 53.5 ms |

Prefill/TTFT pays 1.6–2.4× (per-call dynamic activation quant; qurk's
all-layer W4A4 pays more than the CT-mixed split); decode-side penalty
is 1.21–1.29× — under the pre-registered 2× decode-collapse trigger.
In exchange, weights drop to 9.03 GiB/rank and KV headroom rises ~2
GiB/rank. Spec gate PASS on both (pos1 0.9853 / 0.9951 — above the fp8
incumbent's 0.9258).

The **mxfp8** lane (P6.27, same knobs) inverts that split: every TTFT
face sits BELOW both mxfp4 lanes (26.67 s U32k — prefill amortizes the
per-call quant pass), while decode p50 81.8 ms (1.53×) sits above both
(W8 weights read 2× the W4 bytes per decode step + group-32 E8M0 scale
handling vs the incumbent's fused block-scaled lane). Weights 14.74
GiB/rank ≈ fp8 — no capacity win; the lane's value is running the
AutoRound import as-is: spec pos1 0.9626 / acceptance 0.9174 on the
QUANTIZED MTP drafter, health 285 s incumbent-identical, 0/0 erraudit,
zero engine change (rides cert11 unmodified).

### Online MXFP4/MXFP8 quantization of base checkpoints (P6.30/P6.31; native since cert20/cert21, carried on cert22)

The lanes above quantize checkpoints that ship pre-quantized
weights. P6.30 opens **online** quantization of the BASE bf16 models
(the same shape as the online-fp8 lane): `ARC_QUANT=mxfp4` or `mxfp8`
on `/models/qwen3.8-27b` or `/models/gemma4-31b` quantizes every
standard linear at load (240 layers on gemma — the hybrid GDN
linear-attn/mamba layers are auto-skipped by the wheel's own dispatch;
no ignore list needed; qwen runs with
`ARC_QUANTCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'`).

**mxfp4 runs the W4A16 route** (the P6.30 root-cause fix, landed
cert20; op carried unchanged on the cert21/arc5 and cert22/arc6
wheels):
stock W4A4 quantized activations per call (≈ the GEMM cost at decode)
AND its (f4, f4) joint dtype can never pass oneDNN gemmstone's JIT
gate (the counter-operand list has no f4) — `matmul:ref`. The arc4
wheel's `_xpu_C.fp4_gemm_w4a16` keeps activations bf16: `jit:gemm`
DPAS via weight decomposition, group-32 e8m0 scales on the weights.
Gate knob `ARC_MXFP4_MODE`: unset/`w4a16` = the route (default),
`w4a4` = stock control (A/B), anything else WARNs and keeps stock.
Same-image scorecard (r120c, qwen3.8-27b, GMU 0.85, MTP k3):
**U8k 3.66 s / U32k 17.12 s / solo 62.5–65.2 / agg8 ~305 tps** vs
stock online-mxfp8 (W8A8) 5.59 / 25.20 / 36 / ~197 — **superior on
every axis (1.47–1.75×)**; vs stock W4A4 control 6.84 / 29.74 / 43.4 / ~210 —
1.74–1.87× TTFT, 1.43–1.45× decode. Gemma (r120g, same image, gemma
posture): **U8k 3.94 s (2,070 t/s) / U32k 21.16 s / 40.5 solo /
262 agg8** — 1.90–2.21× faster than the W4A4 control and faster than
gemma's plain-fp8 online record on 3 of 4 axes (U32k the honest
exception — the bf16-DPAS-par physics ceiling). The qwen plain-fp8
online lane is an upstream interface dead-end (`fp8` is not an online
shorthand on this vLLM; `--quantization-config` is consumed by the
online framework and requires linear/moe/targets — no ignore channel
for qwen's hybrid layers; r120h traceback + r120h2 `fp8_per_channel`
probe banked). Full evidence:
`results/2026-11-08-p630-mxfp4-perf/`.

**mxfp8 runs the W8A16 route on cert21, carried unchanged on
cert22** (the P6.31 fix — the same
act-quant-elimination as the mxfp4 lane): stock online mxfp8 is W8A8 —
per-call activation quant ≈ the GEMM at decode Ms AND the (f8,f8)
joint rate measured 2.05–2.64× slower than bf16 at prefill on every
scale flavor (r122). The arc5 wheel's `_xpu_C.fp8_gemm_w8a16_mxfp8`
keeps activations bf16 against f8 weights with group-32 e8m0 weight
scales. Gate knob `ARC_MXFP8_MODE`: unset/`w8a16` = the route
(default), `w8a8` = stock control (A/B), anything else WARNs and keeps
stock. Same-image scorecard: **qwen (r124) U8k 3.61 s / U32k 17.17→
17.07 s / solo 53.8–55.6 / agg8 265.9 tps vs stock W8A8 5.59 / 25.05 /
35.6–36.3 / 194.2 — 1.37–1.55× on every face**, spec pos1 0.9804
(control 0.9758); **gemma (r124b) 3.87 / 20.94 s / 27.7 / 192.4 vs
6.88 / 32.86 / 15.0 / 111.0 — 1.57–1.85×**. Adjudication
(pre-registered from the r123 kernel matrix): the op sits between
stock and the checkpoint class (m816 GEMM 1.74×/2.68× faster than
stock at decode/prefill medians, 1.50×/1.06× behind w16pc) — e2e
**checkpoint-fp8 stays the perf default, `ARC_MXFP8_MODE=w8a16` is
the online-mxfp8 lane route, and mxfp4-w4a16 stays the superior
online lane** (qwen solo 62.5–65.2 vs 55.6). The **W4A8** (e4m3×f4)
probe — `ARC_MXFP4_MODE=w4a8`, the arc5 op `_xpu_C.fp4_gemm_w4a8` —
is a banked closed-negative: oneDNN primitive-create refuses the
joint dtype ("Only support int4 and fp8 gemm", r123). Full evidence:
`results/2026-11-08-p631-mxfp8-w8a16/`.

**Gemma online-mxfp8 capacity law**: 8-bit weights double mxfp4's
footprint on the 31B model (~29 GiB under TP2) — at gmu 0.8 only
5.31 GiB KV remains, short of one 262144-token seq (8.33 GiB needed;
estimated max len 103296) → boot refusal. NOT a lane defect: set
`ARC_MAXLEN=40960` (the serve-script `--max-model-len` knob) — the
battery's longest probe is the U32k face at 32 800 tokens, and the
cap gates admission only, so faces/decode stay comparable with the
banked 262144 refs.

### Skinny-M decode kernels (P6.33; cert22) — opt-in, default OFF

The cert22 arc6 wheel ships three GEMV-regime M≤8 weight-only GEMM
kernels (`skinny_fp8_gemm_w8a16` / `skinny_fp8_gemm_w8a16_mxfp8` /
`skinny_fp4_gemm_w4a16`) — correct (r126 gates) and faster than stock
**in the eager/host-bound regime**, but e2e-falsified for the serve
path: the certified decode replays XPU graphs, under which stock
oneDNN runs its tensor-core kernels at 420–680 GB/s device-side — the
skinny scalar-FMA streaming design loses 1.6–2.6× recorded into the
replayed graph (gemma solo −38/−42% in the r127 battery), and the
size-1 decode graph pads the token dim to 2, so even M=1-capped
routing executes as M=2 in-graph.

Gate knob `ARC_SKINNY_M` (a serve-env pass-through in the rig
`serve_arc_b70.sh`, mirroring `ARC_MXFP8_MODE`): **unset/`off` =
stock routes — the certified default**; `on` = opt-in (per-lane M
caps: w16 M=1, m816/g16 M≤2; anything else WARNs and keeps stock).
Leave unset for all certified lanes. Full evidence:
`results/2026-11-08-p633-skinny-m/`.

Boot = the same uniform interface; three knobs route a non-default
model (host dir, served name, quant flag):

```bash
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert22 \
  ARC_MODEL=/models/qwen3.8-27b-quark-awq-mxfp4 \
  ARC_SERVED=qwen3.8-27b-quark-awq-mxfp4 \
  ARC_QUANT= \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 bash /root/build/serve_arc_b70.sh
```

- `ARC_MODEL` — host model dir mounted at `/models/target`
  (default `/models/qwen3.8-27b-fp8`).
- `ARC_SERVED` — `--served-model-name` (default `qwen3.8-27b-fp8`).
- `ARC_QUANT` — `--quantization` value (unset-default `fp8`); an
  **explicitly empty** `ARC_QUANT=` omits the flag so the checkpoint's
  own `quant_config` drives scheme selection — required for quark /
  CT-MX checkpoints (passing `--quantization fp8` would fight the
  config).

**Image floor: quark checkpoints need cert11+** — cert10 rejects them
at config validation (`quark quantization is currently not supported
in xpu`). cert11's `ArcXpuPlatform` override (commit 7ea0f27b) admits
the format and states truthful Xe2 MX capability; it is runtime-inert
for the fp8 lane (kernel selection is table-keyed and never consults
the changed paths — r105 battery PASS at cert10 refs proves zero
dispatch change).

## KV-cache dtype — `fp8_e4m3` default + the four TurboQuant presets (comprehensive; all servable on cert8, re-gated clean on cert9, k8v4 carry proven on cert10)

> **FIXED (2026-11-04, P6.19 / cert8): every `turboquant_*` preset serves
> clean.** The P6.18 boot refusal is lifted; the corruption it guarded
> against was root-caused to **three stacked spec-verify defects** (none
> the quantizer's precision) and fixed wheel-side by the
> `tq_spec_fix` hook, active by default on TQ dtypes (fp8 never imports
> the hooked module — posture unchanged, control-proven in the gate):
> captured verify graphs baking the raw-KV flash fast-path
> (**#TQ-blind-verify**), a replay-broken multi-row synthetic-decode
> dispatch in the continuation branch, and — for the first single-pass
> MQ draft — the wheel's two TQ ops trees (AoS = the engine's XPU
> default vs a FlyDSL-gated SoA tree; the draft targeted SoA and read
> the live cache as zeros). Zero REP-LOOP/JUNK/BLEED/EMPTY on all four
> presets, all-dtype gate with the fp8 certified control
> (`results/2026-11-04-p619-tqverify/`).

The certified default remains `--kv-cache-dtype fp8_e4m3` — do not set
the flag at all unless you want a TQ preset. **All five dtypes are
servable on cert8**; three knobs govern the TQ verify path (section
below).

### The dtype menu (all five; capacity = cert8-gate same-day values)

| dtype (`--kv-cache-dtype`) | KV pool (cert8) | pool × | t0 solo ×2 (P6.15) | t1.0 agg (P6.14, prob lane) | U32k/U64k TTFT (P6.14)¹ | identity record (cert8 gate) |
|---|---|---|---|---|---|---|
| `fp8_e4m3` — **certified default** | 400,739 | — (baseline dtype) | 84.5 / 84.4 | **241.10** | 23.16 / 50.77 s | t0 lanes 100 % clean, code/simple byte == cert7 |
| `turboquant_k8v4` | 519,972 (== t6p) | 1.98× | 88.5 / 88.6 | 172.95 (−28 %) | 15.29 / 33.03 s | long-return bar PASS full + 3 churn cycles (multi = the Class-A variant trio) |
| `turboquant_4bit_nc` | 748,549 | 2.86× | 86.2 / 86.0 | 134.42 (−44 %) | 15.91 / 34.02 s | **cleanest record — zero flags of any kind, incl. the t1 lane** |
| `turboquant_k3v4_nc` | 843,202 | 3.22× | 80.8 / 80.5 | 137.11 (−43 %) | 15.48 / 33.78 s | rehit1 takes the Class-A variant; cold + rehit2 byte-match the t9 primaries on all 4 classes |
| `turboquant_3bit_nc` | **963,764** | **3.68×** | 85.8 / 85.6 | 171.32 (−29 %) | 16.02 / 35.74 s | 3-bit precision floor: essay/code coherent adjacent variants per round; multi/simple stable |

¹ The TTFT column is the P6.14 same-harness A/B (fp8 comparator as run
there = the thr2048-profile face, hence its 23.16/50.77); the
true-default fp8 TTFT is 17.25/39.77 s (P6.15) — the honest TQ
TTFT edge vs true-default fp8 is **−10…−16 %**. The t1.0/t0 columns
are the P6.14/P6.15 campaign faces (prob lane); full matrices with
per-face provenance in PERFORMANCE-arc ("TurboQuant KV-cache dtype
faces" + "Master matrix"). KV pool tracks boot-day free memory — the
P6.14-era table there (526k/757k/853k/975k) was that campaign's
memory state.

### fp8_e4m3 — the certified default

The quality/speed reference: fastest aggregate on every measured face,
the tightest identity record, and the P6.19 fix never engages (the
hooked module is not imported — control-proven t7/t9 + cert8 gate,
`fix-engaged-on-fp8=0`). Choose it unless KV capacity is the binding
constraint. Records: PERFORMANCE-arc "Current certified image".

### What each TurboQuant preset buys and costs

- **`turboquant_k8v4`** — the balanced TQ choice: 4-bit K/V with 8-level
  scales, the decode+prefill sweet spot of the family (fastest t0
  decode 88.5+ AND fastest default prefill 2097/1945 t/s, P6.15),
  mildest sampled-aggregate cost (−28 %), P6.14's quality-safest
  preset, spec-verify tps solo 54.42/54.59 / agg8 235.97 (cert8 gate,
  split dispatch; t6p dev-leg 54.5/240.9 in-family).
- **`turboquant_4bit_nc`** — the capacity lane with the cleanest
  identity record (2.86× pool; zero flags anywhere in the cert8 gate);
  pays the largest sampled-aggregate cost (−44 %). Cert9-gate note: that
  boot sampled the essay variant set (t0 A-B-A + one t1-seed pair —
  r84b re-char 7/8 essay pairs identical, code 4/4, all coherent, zero
  garble, dominant hashes byte-identical to the cert8 gate boot): the
  per-boot draft lottery chooses which identity face shows; the variant
  families are cross-cert stable.
- **`turboquant_k3v4_nc`** — 3-level K scales: 3.22× pool at −43 %
  sampled aggregate, the slowest t0 decode of the family (80.5–83.3).
- **`turboquant_3bit_nc`** — the max-capacity lane: 3.68× pool and only
  −29 % sampled aggregate (its t1.0 agg beats the sub-4-bit presets'),
  but the coarsest precision — 8 affine V levels, upstream's own table
  +20.59 % PPL on gpt-oss — and the widest t0 variant drift (each round
  takes a different coherent adjacent variant on essay/code; outputs
  stay coherent English/code, garble-clean — the P6.18c-documented
  value-precision floor, NOT corruption).

Identity faces on TQ dtypes are the same engine-class
probabilistic-drafting faces the certified fp8 posture fires (fp8
control in the P6.19 gate): warm-resume variant flips (p618b Class-A)
and, on `3bit_nc`, the precision-floor drift. The deterministic serving
guarantee — t0 within-path byte-identity — holds wherever t0 is
identical; the corruption bar (REP-LOOP/JUNK/BLEED/EMPTY,
garble) is **zero across all five dtypes** on cert8 AND cert9, cold and
warm, churn included (cert9 evidence: the r84 gate + r84b re-char —
dominant identity hashes byte-identical across the two cert gates, so
the P6.20 off-thread store changed no output distribution; cert10
carried the k8v4 leg — essay/code/simple hashes byte-identical to the
stock cert9 k8v4 leg at the r99 gate, so the oneCCL overlay is absent
from the TQ decision path).

### Serving a TurboQuant preset

Exactly one flag changes on the certified line — everything else (the
25-var chain, graphs, MTP k3 + probabilistic drafting, LMCache) is
identical:

```bash
# in the serve script, replace the fp8 default:
#   --kv-cache-dtype fp8_e4m3
# with the preset, e.g.:
    --kv-cache-dtype turboquant_k8v4
# (the uniform boot interface has no dtype knob — edit the flag in the
#  script copy, exactly how the cert8 gate boots each preset)
```

Verify the fix is engaged in `/root/serve_full.log` — the posture line
`v6 installed (active=True mode=split mq_kernel=True)` must appear on
every TQ boot (cert8 gate posture check), and
`tq_spec_fix` in the hook-install table. On fp8 neither appears
(the control proof).

### Verify dispatch — split (certified) / MQ (experimental) / synth (diagnosis-only)

TurboQuant spec-decode **verify** (scoring the q_len+1 draft rows
against the quantized cache) has three dispatch modes on cert8:

- **`split` — the certified default** (`VLLM_ARC_TQ_VERIFY_MODE=split`
  or unset): q_len separate single-row stock decode calls with per-row
  causal limits as recorded device ops on the live `seq_lens` buffer —
  graph-replay-safe, all-dtype gate-proven. This is the posture behind
  every cert8 TQ number above.
- **`mq` — experimental, opt-in** (`VLLM_ARC_TQ_VERIFY_MODE=mq`): a
  single-pass multi-query kernel (`patches/tq_mq_aos.py`) re-ported
  onto the engine's slot-interleaved **AoS** layout (the wheel's own MQ
  kernel targets the SoA tree, which is never used on XPU — that
  mismatch was root cause RC3). Offline-exact vs the engine's own
  store/decode pair on all four presets (MQVAL5). Measured on the
  cert8 gate annex: **agg8 284.81 tok/s = 1.21× split** (235.97),
  solo unchanged (53.83/54.31) — the win is concurrency, not latency;
  in-family with the t8g dev leg (289.23). **Experimental because its
  identity faces are wider than split's** (multi/simple variant drift
  across rounds) — serve it when +20 % aggregate matters more than
  variant stability. Knob `VLLM_ARC_TQ_MQ_SPLITS=32` (KV split count).
- **`synth` — never serve** (`VLLM_ARC_TQ_VERIFY_MODE=synth`): the
  pre-fix multi-row synthetic-decode dispatch, kept for differential
  diagnosis only — it is replay-broken under XPU graphs by construction
  (root cause RC2).

Kill-switch: `VLLM_ARC_TQ_SPEC_FIX=0` disables the whole fix (returns
the pre-fix corrupt spec-verify path — diagnosis only). Full knob
semantics: [`envs.md`](envs.md).

### Stall levers per dtype

The P6.15 generalization holds on every preset: **`ARC_MNBT=4096`
halves bystander stalls** at TTFT ≤ the fp8-default family, with the
chunk ladder quantized to the dtype's KV page (fp8 page 832 →
1664/3328/7488-token chunks; TQ page 1664 → e.g. 4bit 1664/3328/6656,
k8v4 3456@4096 / 8064 default, k3v4 3712/7424, 3bit 2176/6528). The
full per-face Pareto matrix (TTFT / gaps / capacity for every
dtype × knob cell): PERFORMANCE-arc "Bystander-stall tuning" +
"Master matrix" — the fp8 rows are the live first-choice menu (the TQ
rows are the banked P6.15 record, servable again since cert8).

The P6.18 record (reproduction battery, roundtest, guard gate) stays
banked at `results/2026-11-03-p618{b-repro,c-roundtest,-cert7}/` — its
Class-B/Class-A split stands; its precision-ceiling ROOT-CAUSE
conclusion was corrected by P6.19 (the roundtest exonerated the kernels,
and the true spec-verify defects were found above them).

## Draft-sampling posture (P6.17: CERTIFIED DEFAULT — `ARC_SPEC_METHOD`)

The certified posture is **probabilistic MTP drafting** — MTP drafts
sampled from the draft distribution + the rejection-ratio verify (the
mathematically-correct target sampler; greedy drafts + sampled verify
is biased at temperature), and the superior lane on every measured
sampled face (P6.16 matrix; P6.17 comprehensive re-certification,
2026-11-03). Since P6.17 this is the **serve script's
default posture** (unset = probabilistic); at t0 the two postures are
provably identical (plain-argmax drafts + accept-iff-argmax verify) —
measured in-band on every dtype and every p617 gate window.

The certified line on the uniform boot interface (no image change, no
CLI edit — the env states the default explicitly):

```bash
ARC_SPEC_METHOD=probabilistic ARC_IMAGE=vllm-arc:v0.30.0-b70-cert19 \
  ARC_GRAPH=1 ARC_SPEC=3 bash /root/build/serve_arc_b70.sh
# appends "draft_sample_method":"probabilistic" to --speculative-config
# (P6.17: this is also the DEFAULT when ARC_SPEC_METHOD is unset;
#  ARC_SPEC_METHOD=greedy opts out to the historical lane bytes)
```

Measured effect (t1.0-xhigh, vs the greedy lane; full matrix in
PERFORMANCE-arc "Draft-sampling posture"):

| dtype | agg | solo pair |
|---|---|---|
| fp8 (default dtype) | **+29 %** (186.48 → 241.10) | 61.73/61.79 → **67.46/64.10** (beats its greedy bank) |
| k8v4 | EV-neutral | ragged → uniform |
| tq4 / k3v4 / 3bit | **+44 / +54 / +91 %** | ~77+~26 ragged → uniform mid-50s–60s |

Menu: **probabilistic is the default — nothing to enable**. Opt out
with `ARC_SPEC_METHOD=greedy` only for the historical lane (byte-
identical at t0; −15…−23 % t1.0 agg on fp8). The historical TQ-dtype
lane notes (ragged greedy solo, the ~26 tok/s divergent-trajectory
collapse residual) died with the dtype — that collapse face was
TQ-only and fp8 was always immune (P6.18 closed the dtype itself).
Flip validation: flag probe 5/5 (unset/probabilistic/
junk → prob bytes; `greedy` → the exact pre-P6.17 bytes; `ARC_SPEC=0`
→ no spec flag) + live-boot audits — default-path engine bytes ==
env-form record-leg bytes; greedy opt-out boot carries the old
certified bytes (`results/2026-11-03-p617a-probcert/`; P6.16 lineage
`results/2026-11-03-p616{a,b,c}-*/`).

## Bystander-stall tuning (P6.15, opt-in — certified defaults unchanged)

A bystander inter-token gap during another stream's long prefill =
one chunked-prefill engine step (no decode piggyback on this stack).
Stall duration ≈ tokens-per-chunk / prefill-rate, so the lever is
chunk granularity, tuned by `ARC_MNBT` (`--max-num-batched-tokens`)
or `--long-prefill-token-threshold`, floored to the dtype's KV-page
multiple (fp8 page 832 → 1664/3328/7488-token chunks). Smaller
chunks = shorter stalls but more prefix re-reads = longer TTFT — a
measured Pareto frontier, no free knob (all legs on cert6, integrity
5/5 + det 3/3, 0 errors; `results/2026-11-02-p615-stall-analysis/`).
The former TQ-preset rows of this menu are servable again since cert8
(P6.19 — serve a `turboquant_*` dtype to re-enter that lane; the p615
legs remain the banked record of those faces). **On the certified
fp8 default the levers below are the whole menu** (P6.17l guide):

| Fleet priority | Recommended face | Cost |
|---|---|---|
| **Balanced — first choice (P6.15/P6.17l)** | `fp8_e4m3` + `ARC_MNBT=4096` | stalls −49/−51 % (bystander p95 2030/2587 ms vs 3986/5295 default), KV +6.8 %, TTFT +8–13 %, decode-neutral |
| Long-context stall floor (p65 profile, P6.17l) | fp8 + `--long-prefill-token-threshold 2048` + `LMC_DEFER_CHUNK_STORE=1` + `LMC_DEFER_MIN_TOKENS=40960` | U64k TTFT 51.16 s, bystander p95/max 1580/1599 ms; all three together — never lower the min-tokens gate below 40960 (32k-class prompts regress +4.2 s) |
| Max sampled throughput | certified defaults as-is | fp8 true-default bystander p95 3.99/5.30 s (worst stall face, best sampled throughput); **best TTFT overall on cert10 = the P6.24 FP8 opt-in** (one env line, prefill-only, decode-neutral — "Best-performance postures" above) |

The 2026-11-03 completion set (B6–B8, `results/2026-11-02-p615-b{6,7,8}-*/`)
had generalized the balanced row across the then-servable presets
(**MNBT 4096 halves stalls at TTFT ≤ fp8-default family**; chunk =
KV page × n) — the TQ instances of that generalization were suspended
during P6.18–P6.19 and are **servable again since cert8** (the p615
legs remain the banked record of those faces); the fp8 instance is the
live first-choice row above. Solo decode/prefill/TTFT for every face:
`results/2026-11-02-p615-probes/VERDICT.md`.

Laws: never combine with the experimental slice profile unless
re-validated; one knob at a time (the matrix is one-variable);
`ARC_MNBT=2048` under fp8 floors to 1664-token chunks — the fp8
thr2048 profile is the historical equivalent (r66a).

**First choice for stall-sensitive fp8 fleets: `ARC_MNBT=4096`** —
one knob, stalls −49/−51 % (p95 2030/2587 ms), TTFT +8–13 % (U64k
42.83 s), KV +6.8 %, decode-neutral. The head-to-head with the p65
profile — usage blocks and the performance table — is the
"fp8 stall levers" section above.

## The 25-var chain (reference)

The step blocks above already carry all 25. This section is the
reference view: what each layer is and where it is delivered. Delivery
split (validated bytes): **23 serve-time** (13 runtime/system + the
LMCache 10 — the export block inside Step 2) and **2 boot-time**
(`VLLM_ALLOW_LONG_MAX_MODEL_LEN`, `VLLM_WORKER_MULTIPROC_METHOD` — the
`-e` pair in Step 1's `docker run`).

Serve-time runtime/system (13 — in Step 2's block):

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
```

Serve-time LMCache certified set (10 — keep as a set; in Step 2's block):

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
export LMCACHE_TRACK_USAGE=false       # P6.3: telemetry default-ON; offline rig
```

Boot-time (2 — in Step 1's `docker run`):

```bash
-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1   # 262144 > derived max
-e VLLM_WORKER_MULTIPROC_METHOD=spawn
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
| `VLLM_ARC_LMC_MAMBA_SNAP` | unset (=1) | **never `=0`** — breaks output integrity |
| `VLLM_MAMBA_FP16_SEED_REPAIR` | unset (=1) | `=0` only to reproduce the fp16 seed defect |
| `VLLM_ARC_R35_TAIL_SPLIT` | unset (=1, v4 tail-split) | `=0` restores upstream backoff exactly (A/B control) |
| `VLLM_ARC_FP8_W8A16_DISABLE` | unset (=0, W8A16 reroute ACTIVE) | `=1` falls back to stock per-token W8A8 quant |
| `VLLM_ARC_TQ_SPEC_FIX` | unset (=1, spec-verify fix ACTIVE on every `turboquant_*` dtype; fp8 never imports the hooked module) | `=0` only to reproduce the pre-fix corrupt TQ spec-verify (diagnosis) |
| `VLLM_ARC_LMC_ASYNC_STORE` | unset (=1, off-thread store ON — the cert9 default) | `=0` restores the cert≤8 inline store posture (A/B control) |
| `CCL_SYCL_ALLREDUCE_FP8` | unset (`0` = **OFF — the certified default**; cert10+; oneCCL overlay env, not a vllm var) | `=1` opts into the **P6.24 e4m3-wire allreduce** for the TP2 fp16 large ring: dominant prefill ARs move 1 B/elem on the wire (1.65× at 76,677,120 B; every p65 TTFT face beats the cert9 baseline — U8k −13 %/U64k −8.2 %/U32k −3.8 %). Numerics: one e4m3 round trip per output element = rel-rms ~0.023 on the bench metric (the measured converter floor; r98 task-level parity green). Sub-`_MIN_COUNT` allreduces take the stock path (the `simple`-class byte-identity proof) |
| `CCL_SYCL_ALLREDUCE_FP8_MIN_COUNT` | unset (`2097152` elements) | only messages ≥ this count use the fp8 wire (measured crossover: fp8 is slower below ~2M elements — 0.353 vs 0.327 ms at 2.6 MB); leave at default |
| `CCL_SYCL_ALLREDUCE_FP8_STATS` | unset (`0` = silent-off) | `=1` per-stage telemetry to stderr (k1 amax / k2 quant / barriers / k3 res-scale / k4 reduce / k5 dequant) — diagnostic; declared perturbation (drains the pipeline between stages), numbers attribute cost, they do not race |
| `VLLM_ARC_TQ_VERIFY_MODE` / `VLLM_ARC_TQ_MQ_SPLITS` | unset (`split` = the certified dispatch) | `=mq` opt-in experimental (+21 % agg8, wider identity faces — TQ section above); `=synth` never (replay-broken); `VLLM_ARC_TQ_MQ_SPLITS` = KV split count (`32`) |
| `VLLM_ARC_LMC_EVSYNC` | unset (=1, flush-scoped event sync in the store body) | `=0` reverts to full `torch.xpu.synchronize()` (r80: neutral alone — the in-order stream law) |
| `VLLM_ARC_LOOKUP_ELIDE` | unset (**0 = OFF — int tokens, opt-in**) | diagnostic-only; forfeits real joint loads |
| `VLLM_ARC_XG_COMPACT` | unset (`false`) | `=1` when you want compact-JSON xgrammar structured output (appends `--structured-outputs-config`) |
| `VLLM_ARC_GMU_RELAX` | unset (`0`) | set `0.8` if boot aborts on headroom at higher utilization (relaxes only the engine default 0.92) |
| `LMC_DEFER_CHUNK_STORE` | unset (`0` = OFF — certified default) | `=1` only in the long-context profile (above), together with `--long-prefill-token-threshold 2048` and the min-tokens gate |
| `LMC_DEFER_MIN_TOKENS` | unset (`0` = defer every chunked prefill) | certified profile value **40960**; lowering it below ~40960 regresses 32k-class prompts (+4.2 s, r48k) |
| `LMC_STORE_SLICE_TOKENS` | unset (`0` = OFF — certified default, ships since cert5) | **experimental** `=4096` — slices the deferred one-shot final-span store into ~19 bounded windows; long-context profile section above |
| `LMC_BOUNDARY_KEEP` / `LMC_BOUNDARY_CAP` | unset (8 / 48) | raise only with boundary-registry pressure evidence |
| `LMC_BOUND_STATS` / `LMC_BOUND_MD5` | unset (logging ON) | `=0` to quiet boundary telemetry |
| `ARC_SNAP_PROBE` | unset (silent) | diagnostics only (leg telemetry) |
| `VLLM_COMPACT_GDN` | unset | auto-detected (runner → platform); not operator-set |

Host-script layer (`tools/serve_arc_b70.sh`): `ARC_IMAGE`, `ARC_NAME`,
`ARC_GRAPH` (0 eager / 1 graph — certified = 1), `ARC_SPEC` (MTP depth
= `num_speculative_tokens`: 0 = off, default 3 = the recommendation,
4 = the D-champion alternative; only 0/3/4/5 accepted — the legged
depths), `ARC_SPEC_METHOD` (draft-sampling posture, P6.17: unset =
`probabilistic` = the CERTIFIED default —
section above; `greedy` = the opt-out historical lane), `ARC_LMC` (1 = the certified
LMCache connector + the 10-var `LMCACHE_*` chain — default; 0 = no
LMCache), `ARC_GMU` (0.8), `ARC_MNBT` (8192 — the certified default;
r66c A/B'd 16384: long-face
solo TTFT −4…−10 % but −18 % KV pool (405,491→332,629 tokens,
concurrency 1.55x→1.27x) and U8k +10 % — a conditional long-prompt-only
lever, **never** combined with `--long-prefill-token-threshold` (the
threshold pins chunk size; only the KV loss would remain); see
PERFORMANCE-arc "Prefill anatomy"), `ARC_MAXSEQS` (64), `ARC_PORT`
(8000), `ARC_FP8` (0 = the certified stock allreduce; `1` = the P6.24
cert10 overlay opt-in — appends `export CCL_SYCL_ALLREDUCE_FP8=1` to
the exec env, "Best-performance postures" above; no-op on images
without the overlay; default emits no env line).

## Posture laws (never change in production)

1. **align law** — never remove `--mamba-cache-mode=align`; never ship
   `--no-enable-prefix-caching` (flips cache mode to "none", kills the
   boundary registry and long-context cells).
2. **graph law** — always `VLLM_XPU_ENABLE_XPU_GRAPH=1` on the certified
   posture.
3. **pinned-KV law** — `LMCACHE_DISABLE_PINNED_MEMORY=1` (in the
   certified 25-var chain) is a segfault guard, not a tuning knob:
   pinned USM >16–32 GiB/process segfaults, and the 5-op
   torch-baseline transfer re-pin is unconditional wheel-side
   (`e01-xpu-platform-ops.patch`). The old `VLLM_ARC_LMC_TORCH_KV`
   knob is **dead since r47b** — setting it is a no-op on every
   certified image (see [`envs.md`](envs.md)).
4. **container freshness** — recreate the serve container whenever the
   image changes (a stale container silently runs old code).
5. **model mounts read-only** (`:ro`) — the checkpoint and chat template
   are inputs, never outputs.
