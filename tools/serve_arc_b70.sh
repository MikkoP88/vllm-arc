#!/bin/bash
# serve_arc_b70.sh — uniform boot interface for vllm-arc on the dual-B70 rig
# (plan §1 harness law: ONE boot script per image; arc_matrix.sh drives legs
# through this interface).
#
# Posture knobs (environment; defaults = P1 gate posture, plan §10):
#   ARC_IMAGE   image tag            (default vllm-arc:v0.30.0-b70)
#   ARC_NAME    container name       (default arc-test)
#   ARC_GRAPH   0 eager | 1 VLLM_XPU_ENABLE_XPU_GRAPH=1 graph posture
#               (standing rig law: every graph leg runs with it set)
#   ARC_SPEC    MTP depth = num_speculative_tokens (0 = MTP off, default 3
#               = the r46t recommendation; 4 = the D-champion alternative)
#               — appends --speculative-config (method mtp; rig
#               /root/serve_lmc_v133.sh:53 parity). Spec-config law:
#               CLI flag, not env (plan p30). P2-candidate posture — the
#               §10 P2 gate ("FULL graph + MTP boots and survives first
#               request") is proven by the P2 spec legs, not assumed.
#   ARC_SPEC_METHOD  draft-sampling posture (P6.17; default unset =
#               probabilistic = the CERTIFIED posture). Appends
#               "draft_sample_method":"probabilistic" to the spec config —
#               sampled drafts + rejection-ratio verify: the superior lane
#               on every measured face (p617 certification: t1.0 agg +29%
#               fp8 over greedy, uniform solo streams, mathematically
#               correct target distribution — the P6.17 certified default).
#               =greedy is the OPT-OUT historical posture (pre-P6.17
#               certified bytes). Any other value = default posture.
#               Evidence: results/2026-11-03-p617a-probcert/ (+p616 lineage).
#   ARC_LMC     0 no LMCache | 1 D-champion kv-transfer config (P3; default —
#               the certified posture: LMCacheConnectorV1 + the 10-var
#               LMCACHE_* chain, keep as a set). P6.47 exception: on
#               ARC_KVDTYPE mxfp4/mxk8v4 the DEFAULT is 0 (auto-off) —
#               LMCache prefix-reload double-quantizes V and e2m1-V does
#               not survive the second quantization (r168 discriminator:
#               same m4q posture, ARC_LMC=1 essay-rehit REP-LOOP garble
#               vs ARC_LMC=0 all classes CLEAN; explicit ARC_LMC=1 still
#               forces it, with a WARN)
#   ARC_GMU     --gpu-memory-utilization (default 0.8, D-champion certified
#               posture — V1227_GMU lineage, plan Appendix F)
#   ARC_MAXLEN  --max-model-len (default 262144 = the certified incumbent
#               context, exact fit for qwen3.8's KV/token). P6.28: gemma-4's
#               heterogeneous head dims (full_attention 512 vs sliding 256)
#               double the per-token KV on full layers — at GMU 0.8 the pool
#               supports only ~120k. Legs needing a lower cap set this knob;
#               all G2 faces top out at U64k so a 120k cap costs nothing.
#   ARC_MNBT    --max-num-batched-tokens (default 8192, D champion)
#   ARC_KVDTYPE --kv-cache-dtype (default fp8_e4m3 = the certified
#               incumbent; close-out G 2026-11-10 adds the knob so the
#               kv-dtype report sweep can drive auto/fp8/fp8_e5m2 without
#               editing the wrapper)
#   ARC_MAXSEQS --max-num-seqs (default 64, D champion)
#   ARC_PORT    serve port (default 8000)
#   ARC_MODEL   host model dir mounted at /models/target (default
#               /models/qwen3.8-27b-fp8 — the certified incumbent). P6.26:
#               mxfp4 legs point this at their staged checkpoints. Changing
#               it force-recreates the container (mount is part of the
#               freshness law below — a stale /models/target silently serves
#               the wrong weights).
#   ARC_SERVED  --served-model-name (default qwen3.8-27b-fp8)
#   ARC_QUANT   --quantization value (default fp8 = the incumbent's native
#               method). EMPTY STRING = omit the flag entirely so the
#               checkpoint's own quantization_config drives resolution
#               (P6.26 mxfp4 legs: compressed-tensors / quark must come
#               from the checkpoint, not a CLI override).
#   ARC_FP8     0 (default) = stock allreduce — the certified posture | 1 =
#               export CCL_SYCL_ALLREDUCE_FP8=1: the P6.24 cert10 oneCCL
#               overlay opt-in (e4m3-wire large-ring AR — 1.65x at the
#               dominant AR, every p65 TTFT face beats cert9; decode/spec
#               neutral, task-level quality parity). No-op on images
#               without the overlay (<= cert9). Default adds NO env line
#               (the expansion slot expands blank; the served process env
#               is byte-identical to the pre-knob script).
#               Evidence: results/2026-11-05-p624-fp8-ar/ (r96-r100).
#   VLLM_ARC_XG_COMPACT=1  appends --structured-outputs-config
#               {"backend":"xgrammar","disable_any_whitespace":true}
#               (port of D's V1227_XGCOMPACT; core-0.30 native surface —
#               patch p06)
#
# Model flags are the D-champion surface for qwen3.8-27b-fp8 (rig
# /root/serve_lmc_v133.sh, plan Appendix F): TP2, block 64, fp8 quant +
# fp8_e4m3 KV, float16 mamba SSM cache, prefix caching, hybrid KV manager on.
# Certified env = the 25-var chain: 23 exported in the exec block below
# (13 runtime/system incl. ARC_TRACE_FLIP + 10 LMCACHE_*) + 2 at boot
# (-e VLLM_ALLOW_LONG_MAX_MODEL_LEN / VLLM_WORKER_MULTIPROC_METHOD).
# P6.3: LMCACHE_TRACK_USAGE=false joined the chain — LMCache usage
# telemetry is DEFAULT-ON (usage_telemetry/identity.py:25), and the
# ungated per-request usage-context creation also raised the dual-engine
# "different metadata" ERROR ×2/boot (observability.py:1965; gated
# wheel-side by e04-usage-tlm-gate.patch — env + patch together).
# --compilation-config carries mode:none (no torch.compile) + the explicit
# 85-entry capture list — [1..320] shorthand is INVALID JSON, never ship it.
set -euo pipefail

#   ARC_MXFP4_MODE  passed into the serve env when set (p630 FIX-P A/B:
#                  w4a4 = stock W4A4 control; unset = patch default w4a16)
#   ARC_SKINNY_M   passed into the serve env when set (p633: on = the
#                  M<=8 skinny kernels opt-in; unset/off = stock routes,
#                  the certified default after the r127 e2e falsification)
#   VLLM_ARC_GRAPH_EXEC_DB passed into the serve env when set (p635 D1:
#                  unset = patch default ON (graphs captured x2 into the
#                  shared pool, replays alternate — torch 2.13 same-exec
#                  replay blocks on the prior launch); 0 = stock
#                  single-buffer replay, the A/B control)
ARC_IMAGE=${ARC_IMAGE:-vllm-arc:v0.30.0-b70}
ARC_NAME=${ARC_NAME:-arc-test}
ARC_GRAPH=${ARC_GRAPH:-0}
ARC_SPEC=${ARC_SPEC:-3}
# P6.47 posture (r168 discriminator, 2026-11-13): LMCache's prefix-reload
# path re-quantizes V on write-back; e2m1-V (mxfp4/mxk8v4) does not survive
# the second quantization — r161 m4q ARC_LMC=1 banked essay-rehit REP-LOOP
# garble on full+rehit faces while r168's byte-identical posture with
# ARC_LMC=0 came back CLEAN on all 4 classes with deterministic hashes.
# Auto-off LMC on the e2m1-V lanes unless the operator forces ARC_LMC=1.
KVD_EFF=${ARC_KVDTYPE:-fp8_e4m3}
if [ -z "${ARC_LMC:-}" ] && { [ "$KVD_EFF" = mxfp4 ] || [ "$KVD_EFF" = mxk8v4 ]; }; then
  ARC_LMC=0
  echo "NOTE: ARC_LMC auto-off for ARC_KVDTYPE=$KVD_EFF (LMCache double-quant of e2m1 V; P6.47 r168 evidence)"
elif [ "${ARC_LMC:-1}" = 1 ] && { [ "$KVD_EFF" = mxfp4 ] || [ "$KVD_EFF" = mxk8v4 ]; }; then
  echo "WARN: ARC_LMC=1 forced on ARC_KVDTYPE=$KVD_EFF — r168 showed essay-rehit REP-LOOP garble on this combo"
fi
ARC_LMC=${ARC_LMC:-1}
ARC_GMU=${ARC_GMU:-0.8}
ARC_MAXLEN=${ARC_MAXLEN:-262144}
ARC_MNBT=${ARC_MNBT:-8192}
ARC_MAXSEQS=${ARC_MAXSEQS:-64}
ARC_PORT=${ARC_PORT:-8000}
ARC_MODEL=${ARC_MODEL:-/models/qwen3.8-27b-fp8}
ARC_SERVED=${ARC_SERVED:-qwen3.8-27b-fp8}
# NOTE: unset-only default (`-`, not `:-`) — an explicitly EMPTY ARC_QUANT is
# the config-driven posture (P6.26) and must not collapse to fp8.
ARC_QUANT=${ARC_QUANT-fp8}
ARC_FP8=${ARC_FP8:-0}
ARC_SPEC_METHOD=${ARC_SPEC_METHOD:-}
RIG=/root/build

# Refusal posture (p30 merge): MTP-on (ARC_SPEC>0) is a P2-candidate
# posture — allowed by default; its validation IS the §10 P2 spec-leg row.
# (The pre-P3 ARC_LMC refusal gate is retired: p40..p43 landed and the
# cache posture is certified — ARC_LMC=1 is now the default.)
if [ "$ARC_SPEC" != 0 ] && [ "$ARC_GRAPH" = 0 ]; then
  echo "NOTE: ARC_SPEC=$ARC_SPEC with ARC_GRAPH=0 — non-champion combo (D v133 runs MTP"
  echo "with FULL_DECODE_ONLY graphs); allowed for spec-eager A/B isolation only."
fi
# Depth guard (no-unevidenced law): only depths with certification legs
# (r42 k4 / r43 k5 / r46t k3) are bootable; 0 = MTP off.
case "$ARC_SPEC" in
  0|3|4|5) ;;
  *) echo "REFUSE: ARC_SPEC='$ARC_SPEC' — expected 0 (off) or MTP depth 3/4/5"; exit 3;;
esac

# --- container: create if absent (W14 pattern: persistent container, serve
# via exec — fast re-boots between legs; same device/ipc/net posture as the
# validated radiance-test container). privileged + oneCCL env = the universal
# validated TP2 posture on this rig: both donor lines boot it identically
# (radiance-test W14 env == lsv-test iexperimental v1.0.0 env, byte-compared
# 2026-10-27; CCL_ZE_IPC_EXCHANGE=pidfd is what keeps oneCCL's ze_fd_manager
# off the opendir device walk that kills non-privileged TP2 init). ---
# Container freshness law: the serve container must come from $ARC_IMAGE. A
# stale container from a previous leg's image silently runs old code — the
# p2r3 leg (2026-10-27) reproduced a fixed defect verbatim because the new
# image was built but the old container was reused (zero new-code log lines
# was the tell). Recreate whenever the container is missing OR its image
# differs from the requested one.
WANT_IMG_ID=$(docker image inspect "$ARC_IMAGE" --format '{{.Id}}') || {
  echo "REFUSE: image $ARC_IMAGE not present" >&2; exit 2; }
# Model-mount freshness (P6.26): the /models/target bind source is part of
# the container identity — a container built for another ARC_MODEL serves
# the wrong weights just as silently as a stale image does.
HAVE_IMG_ID=$(docker inspect "$ARC_NAME" --format '{{.Image}}' 2>/dev/null || true)
HAVE_MODEL_SRC=$(docker inspect "$ARC_NAME" --format '{{range .Mounts}}{{if eq .Destination "/models/target"}}{{.Source}}{{end}}{{end}}' 2>/dev/null || true)
# skin label: template changes with same image+model must also recreate
# (P6.28 — label-based, mirrors the model-mount freshness law)
HAVE_SKIN=$(docker inspect "$ARC_NAME" --format '{{index .Config.Labels "arc_skin"}}' 2>/dev/null || true)
# Chat template (OPTIONAL, in-repo copy at assets/chat_template_qwen38_high.jinja):
# without it the engine uses the checkpoint's own template — default xhigh
# thinking is identical; only OpenAI-style reasoning_effort high/max/none
# requests differ (they 400 without the shim). On the rig the template is
# present and the certified line mounts it (r43_serve_k5.sh parity).
#
# P6.28 serving-skin knobs (ARC_TEMPLATE / ARC_RPARSER / ARC_TPARSER):
# UNSET = the certified qwen skin, byte-identical (defaults below). EMPTY
# (e.g. `ARC_RPARSER= `) omits the flag so the model-native skin applies —
# gemma-4 has no reasoning channel and ships no chat template. A PATH in
# ARC_TEMPLATE mounts that file for --chat-template (gemma-4 route:
# assets/chat_template_gemma4.jinja, canonical start_of_turn format).
ARC_TEMPLATE=${ARC_TEMPLATE-__qwen_default__}
CT_SRC="$RIG/v127_stage/chat_template_qwen38_high.jinja"
CT_MOUNT=""
CT_FLAG=""
case "$ARC_TEMPLATE" in
  __qwen_default__)
    if [ -f "$CT_SRC" ]; then
      CT_MOUNT="-v $CT_SRC:/root/chat_template_qwen38_high.jinja:ro"
      CT_FLAG="--chat-template /root/chat_template_qwen38_high.jinja"
    else
      echo "NOTE: chat template not found at $CT_SRC — serving WITHOUT it (optional"
      echo "OpenAI reasoning_effort compat; in-repo copy: assets/chat_template_qwen38_high.jinja)"
    fi ;;
  "") : ;;  # explicitly no template — checkpoint-native skin
  *)  if [ -f "$ARC_TEMPLATE" ]; then
        CT_BASE=$(basename "$ARC_TEMPLATE")
        CT_MOUNT="-v $ARC_TEMPLATE:/root/$CT_BASE:ro"
        CT_FLAG="--chat-template /root/$CT_BASE"
      else
        echo "FATAL: ARC_TEMPLATE path not found: $ARC_TEMPLATE"; exit 1
      fi ;;
esac
if [ "$HAVE_IMG_ID" != "$WANT_IMG_ID" ] || [ "$HAVE_MODEL_SRC" != "$ARC_MODEL" ] || [ "$HAVE_SKIN" != "$ARC_TEMPLATE" ]; then
  HAVE_SRC=${HAVE_IMG_ID:7:12}
  [ -n "$HAVE_SRC" ] || HAVE_SRC=none
  docker rm -f "$ARC_NAME" >/dev/null 2>&1 || true
  echo "ARC_CONTAINER_FRESH image=$ARC_IMAGE id=${WANT_IMG_ID:7:12} model=$ARC_MODEL (recreated from $HAVE_SRC)"
  # Boot block = the validated bytes (rig r46t_campaign.sh:54-63 /
  # r42/r46_soak.sh:33-42): --privileged --network host --ipc host, 5
  # mounts, and the 2 boot-layer -e vars of the 25-var chain (the other
  # 23 are exported in the exec block below). --shm-size 16g: the 64g of
  # the r42/r46 rigs is far beyond TP2 IPC need and breaks 64 GB-RAM
  # hosts (close-out F, 2026-11-10; validated by the r138 report run).
  docker run -d --name "$ARC_NAME" \
    --label arc_skin="$ARC_TEMPLATE" \
    --privileged --network host --ipc host \
    --shm-size 16g \
    -v "$ARC_MODEL":/models/target:ro \
    $CT_MOUNT \
    -v /root/build/neo_cache:/root/.neo_cache \
    -v /root/build/triton_cache:/root/.triton_cache \
    -v /root/build/inductor_cache:/root/.inductor_cache \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --entrypoint /bin/bash "$ARC_IMAGE" -c 'sleep infinity'
fi
docker exec "$ARC_NAME" pkill -f "vllm serve" 2>/dev/null || true
# Wait out the old engine: /health can answer for seconds after pkill, and a
# fresh leg's health probe would race it and "pass" against a corpse.
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$ARC_PORT/health" >/dev/null 2>&1 || break
  sleep 1
done
sleep 2

GRAPH_ENV=""
COMPILATION_FLAG=""
if [ "$ARC_GRAPH" = 1 ]; then
  # Standing rig law (plan Appendix F): VLLM_XPU_ENABLE_XPU_GRAPH=1 on all
  # graph legs. P1 posture = FULL_DECODE_ONLY pool identical to the
  # D-champion certified set (54/64-size discipline lands with p12/p32).
  GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1;"
  # "mode":"none" is part of the certified line (rig r43_serve_k5.sh /
  # r46t active_serve3.sh): NO torch.compile — decode-only cudagraphs only.
  COMPILATION_FLAG="--compilation-config '{\"mode\":\"none\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}'"
fi

XG_FLAG=""
if [ "${VLLM_ARC_XG_COMPACT:-0}" = 1 ]; then
  XG_FLAG="--structured-outputs-config '{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}'"
fi

# Quantization flag (P6.26): default emits the incumbent's certified
# --quantization fp8. ARC_QUANT="" (empty) omits the flag so the
# checkpoint's quantization_config resolves the method itself — required
# for compressed-tensors/quark legs where a CLI override would either
# misresolve or bypass the checkpoint's own scheme map.
QUANT_FLAG="--quantization $ARC_QUANT"
if [ -z "$ARC_QUANT" ]; then
  QUANT_FLAG=""
fi

# Online-quant ignore list (P6.30): upstream-native --quantization-config
# JSON passed through to the engine (resolve_quantization_config merges
# quantization_config.ignore over the online shorthand base,
# config/quantization.py:276). Used by online-MX legs on the qwen VL
# checkpoint: the vision tower MLP (linear_fc2 input 4304 -> 2152 @TP2)
# violates the MX block-32 rule (online/mxfp4.py:102); the validated
# offline mxfp8 checkpoint additionally keeps all GDN linear_attn
# projections and mtp.fc in bf16 — this knob mirrors that exclusion set.
# JSON quoting mirrors SPEC_FLAG (single quotes survive the bash -c layer).
QUANTCFG_FLAG=""
if [ -n "${ARC_QUANTCFG:-}" ]; then
  QUANTCFG_FLAG="--quantization-config '${ARC_QUANTCFG}'"
fi

# Parser skin (P6.28): unset = the certified qwen3 parsers; EMPTY
# (`ARC_RPARSER= ` / `ARC_TPARSER= `) omits the flag(s) for models with
# no reasoning channel / tool format (gemma-4). Tool-choice rides the
# tool parser (enable-auto-tool-choice requires one).
RP_FLAG="--reasoning-parser ${ARC_RPARSER-qwen3}"
if [ -z "${ARC_RPARSER-unset}" ]; then RP_FLAG=""; fi
TP_FLAG="--tool-call-parser ${ARC_TPARSER-qwen3_coder} --enable-auto-tool-choice"
if [ -z "${ARC_TPARSER-unset}" ]; then TP_FLAG=""; fi

# P6.24 opt-in pass-through (cert10 overlay): default 0 emits NO env line —
# the certified default posture serves the stock allreduce (proven
# FP8_ENV_SET=0 at the cert10 gate). =1 appends the single overlay env to
# the exec block (read by the patched libccl at the first large AR).
FP8_ENV=""
if [ "$ARC_FP8" = 1 ]; then
  FP8_ENV="export CCL_SYCL_ALLREDUCE_FP8=1;"
fi

# MTP (p30): spec config, CLI flag not env — rig /root/serve_lmc_v133.sh:53
# parity. Depth = ARC_SPEC — default 3 (r46t recommendation), 4 = the certified
# D value (plan Appendix F champion chain). The draft side resolves entirely inside
# upstream 0.30: qwen3_5 model_type -> Qwen3_5MTP arch, n_predict from
# text_config.mtp_num_hidden_layers (p30 README assessment) — no draft-model
# path, the head ships inside the target checkpoint (mtp.* weights).
#
# P6.17 CERTIFIED draft-sampling posture (upstream-config-native on the
# pinned 0.30.0 wheel; spec config key draft_sample_method,
# config/speculative.py:588): DEFAULT (unset) = probabilistic drafting =
# sampled drafts + rejection-ratio verify — the certified posture on every
# measured face (p617 campaign: t1.0-xhigh 4-way agg +29% over greedy at
# fp8, uniform solo streams — no 26-tok/s collapse legs — and the
# mathematically-correct target distribution; greedy drafts + sampled
# verify is biased at temperature). ARC_SPEC_METHOD=greedy is the OPT-OUT
# historical posture (the pre-P6.17 certified bytes, behavior
# byte-identical at t0: drafts are plain argmax at temp 0 (gumbel.py:102,117)
# and verify accepts iff draft==target argmax
# (rejection_sampler_utils.py:571-585) — measured in-band on every dtype).
# Residual on TQ dtypes: rare divergent-trajectory requests can still
# collapse (~26 tok/s) — worst-case-sensitive fleets use fp8 (whose sampled
# face improves under this posture) or cooler sampling. Any value other
# than "greedy" = default posture (no unevidenced config surface).
SPEC_METHOD_FLAG=",\"draft_sample_method\":\"probabilistic\""
if [ "$ARC_SPEC_METHOD" = "greedy" ]; then
  SPEC_METHOD_FLAG=""
fi
SPEC_FLAG=""
if [ "$ARC_SPEC" != 0 ]; then
  SPEC_FLAG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$ARC_SPEC$SPEC_METHOD_FLAG}'"
fi

# LMCache (P3, certified posture): the D-champion kv-transfer config — exact
# bytes of the certified line (rig /root/build/r42/r43_serve_k5.sh:53) — plus
# the 10-var LMCACHE_* chain as one set (plan Appendix E; keep as a set).
# P6.3 adds the 10th: LMCACHE_TRACK_USAGE=false (telemetry default-ON law,
# header comment above).
LMC_ENV=""
LMC_FLAG=""
if [ "$ARC_LMC" = 1 ]; then
  # LMCACHE_MAX_LOCAL_CPU_SIZE=16 (GB): close-out F 2026-11-10 — the 100 GB
  # virtual reservation of the r42 rig breaks 64 GB-RAM hosts; 16 GB keeps
  # a meaningful CPU tier with headroom for weights load + page cache.
  LMC_ENV="export LMCACHE_USE_EXPERIMENTAL_API=True; export LMCACHE_MAX_LOCAL_CPU_SIZE=16; export LMCACHE_LOCAL_CPU=True; export LMCACHE_EVICT_POLICY=lr_u; export LMCACHE_CHUNK_SIZE=1024; export LMCACHE_SAVE_DECODE_CACHE=True; export LMCACHE_MIN_RETRIEVE_TOKENS=1024; export LMCACHE_LOG_LEVEL=INFO; export LMCACHE_DISABLE_PINNED_MEMORY=1; export LMCACHE_TRACK_USAGE=false;"
  LMC_FLAG="--kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"use_native\":true,\"lmcache.extra_config\":{\"mamba_pool_ratio\":2}},\"engine_id\":\"arc-p40\"}'"
fi

docker exec -d "$ARC_NAME" bash -c "
export PYTHONHASHSEED=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${ARC_RPC_TIMEOUT:-300}  # p628: ARC_RPC_TIMEOUT knob (300 stock; envs.py:242 execute-model RPC watchdog; r108y face120k died at 300.5s)
export ZE_AFFINITY_MASK=0,1
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
# oneCCL TP2 block — identical in W14 incumbent and D lsv-test (validated
# posture, cited above); provenance: rig container env inspect 2026-10-27.
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576
export CCL_SYCL_ALLGATHERV_TMP_BUF=0
export CCL_SYCL_ALLREDUCE_TMP_BUF=0
$FP8_ENV
export ARC_TRACE_FLIP=1
${ARC_MXFP4_MODE:+export ARC_MXFP4_MODE=${ARC_MXFP4_MODE};}
${ARC_MXFP8_MODE:+export ARC_MXFP8_MODE=${ARC_MXFP8_MODE};}
${ARC_SKINNY_M:+export ARC_SKINNY_M=${ARC_SKINNY_M};}
${VLLM_ARC_GRAPH_EXEC_DB:+export VLLM_ARC_GRAPH_EXEC_DB=${VLLM_ARC_GRAPH_EXEC_DB};}
$LMC_ENV
$GRAPH_ENV
exec /opt/venv/bin/vllm serve --model /models/target \
  --served-model-name $ARC_SERVED \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization $ARC_GMU \
  --max-model-len $ARC_MAXLEN \
  --max-num-batched-tokens $ARC_MNBT \
  --max-num-seqs $ARC_MAXSEQS \
  --block-size 64 \
  --dtype float16 \
  $QUANT_FLAG \
  $QUANTCFG_FLAG \
  --kv-cache-dtype ${ARC_KVDTYPE:-fp8_e4m3} \
  --mamba-ssm-cache-dtype float16 \
  --mamba-cache-mode=align \
  --enable-prefix-caching \
  --trust-remote-code \
  $RP_FLAG \
  $TP_FLAG \
  $CT_FLAG \
  --no-disable-hybrid-kv-cache-manager \
  $LMC_FLAG \
  $COMPILATION_FLAG \
  $SPEC_FLAG \
  $XG_FLAG \
  --port $ARC_PORT \
  > /root/serve_full.log 2>&1"

echo "ARC_SERVE_LAUNCHED image=$ARC_IMAGE name=$ARC_NAME model=$ARC_MODEL served=$ARC_SERVED quant=${ARC_QUANT:-<config-driven>} qcfg=${ARC_QUANTCFG:-} graph=$ARC_GRAPH spec=$ARC_SPEC gmu=$ARC_GMU maxlen=$ARC_MAXLEN mnbt=$ARC_MNBT seqs=$ARC_MAXSEQS fp8=$ARC_FP8 xgc=${VLLM_ARC_XG_COMPACT:-0} tmpl=${ARC_TEMPLATE-__qwen_default__} rp=${ARC_RPARSER-<qwen3>} tp=${ARC_TPARSER-<qwen3_coder>} port=$ARC_PORT $(date +%H:%M:%S)"
