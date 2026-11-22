#!/bin/bash
# r222_build_cert26.sh -- P6.47 Phase-8 convergence: build
# vllm-arc:v0.30.0-b70-cert26 (cert25zc + the arc7 MX-native decode
# line: the arc7mx wheel 62f234c8 BAKED + the fork overlay with the
# native_verify bridge). The ctx tree (the 8 mx_kv overlay .py files
# from fork HEAD) is staged by scp before this runs; the wheel is the
# rig's /root/build/wheels-arc-vxk arc7mx build (md5-pinned). Refuses
# unless the staged tree carries the content markers AND the evidence
# banks stand: r219b auto-at-128 A/A (ARC_MXKV_NSPLIT retired; C++ auto
# chain 341 bw -> 130 reduce -> 128 policy lands the validated optimum
# by itself), r220b ctl e2e (posture behavior-identical -0.64 pct vs
# the banked ns128 posture), r220c value gate v2 (d_16_64 <= 0.000977,
# det 0, nan 0 -- P64NATIVE value-correct as an env-gated instrument).
# Posture shipped: _16 sub-tile frame + C++ auto splits (kv_tile 16->64
# closed e2e-negative by the r220 arbiter; P64NATIVE stays default-off).
# FORK_SHA must be passed explicitly (cert-release-sequence law:
# manifest fork_sha = the content commit).
set -euo pipefail
R=/root/build
CTX=$R/cert26ctx
FORK_SHA=${1:?usage: r222_build_cert26.sh <FORK_SHA>}
WHEELMD5=62f234c8d0aa89c4aec3e8d2c6fa22ad

SP=$CTX
# --- staged-tree content refuses (fork HEAD mx_kv overlay) ---
for f in engine.py native_decode.py triton_mx_decode.py __init__.py \
         mx_dequant.py mx_layout.py mx_quant.py triton_mx_store.py; do
  [ -f "$SP/$f" ] || { echo "R222_MISSING staged file $f"; exit 1; }
done
grep -q 'from .native_decode import mx_decode_native' "$SP/engine.py" \
  || { echo "R222_STALE_TREE (no native bridge import)"; exit 1; }
grep -q 'os.environ.get("ARC_MXKV_NATIVE", "") == "1"' "$SP/engine.py" \
  || { echo "R222_STALE_TREE (native route not env-gated default-off)"; exit 1; }
grep -q '"mxfp8": (16384, 16384)' "$SP/engine.py" \
  || { echo "R222_STALE_TREE (_NATIVE_MINLEN defaults drifted)"; exit 1; }
grep -q 'ARC_MXKV_NSPLIT' "$SP/native_decode.py" \
  || { echo "R222_STALE_TREE (no NSPLIT pin)"; exit 1; }
grep -q '_max_sk = int(bt_rows.shape\[1\]) \* int(kv_cache.shape\[1\])' "$SP/native_decode.py" \
  || { echo "R222_STALE_TREE (no capacity-priced walk -- r212f)"; exit 1; }
grep -q 'MXKV native call fmt=' "$SP/native_decode.py" \
  || { echo "R222_STALE_TREE (no capture telemetry)"; exit 1; }
# cert25zc inheritance must be intact in the staged tree
grep -q 'os.environ.get("ARC_MXKV_V5_TFLOOR", "2") or 2' "$SP/triton_mx_decode.py" \
  || { echo "R222_STALE_TREE (TILE_FLOOR launcher default not 2)"; exit 1; }
grep -q "_V5_G_MIN" "$SP/engine.py" \
  || { echo "R222_STALE_TREE (no G-adaptive knee)"; exit 1; }
grep -q "mx_decode_rows_dot" "$SP/__init__.py" \
  || { echo "R222_STALE_TREE (no package export)"; exit 1; }
# --- wheel refuse: the certified arc7mx build, byte-pinned ---
WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R222_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R222_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"; exit 1; }
echo "wheel md5 $GOTMD5 (pinned)"
# --- evidence refuses (banked on the rig) ---
grep -q "MXKV native call fmt=mxfp4.*splits=auto" "$R/p647/r219b/auto/serve.log" 2>/dev/null \
  || { echo "R222_NO_R219B_EVIDENCE (auto posture A/A)"; exit 1; }
grep -q "ERROR=0" "$R/p647/r219b/auto/erraudit.txt" 2>/dev/null \
  || { echo "R222_NO_R219B_ERRAUDIT"; exit 1; }
ls "$R"/p199/out/p220a_*32768*.json >/dev/null 2>&1 \
  || { echo "R222_NO_R220A_ARTIFACTS (offline tile ladder)"; exit 1; }
grep -q "ERROR=0" "$R/p647/r220b/ctl/erraudit.txt" 2>/dev/null \
  || { echo "R222_NO_R220B_EVIDENCE (e2e A/A ctl)"; exit 1; }
docker image inspect vllm-arc:v0.30.0-b70-cert25zc >/dev/null 2>&1 \
  || { echo "R222_NO_BASE (cert25zc)"; exit 1; }

mkdir -p "$CTX"
cp "$WHEEL" "$CTX/$(basename "$WHEEL")"
cat > "$CTX/Dockerfile" <<'DOCKERFILE_EOF'
# SPDX-License-Identifier: Apache-2.0
# Dockerfile.cert26 -- P6.47 Phase-8 convergence image: cert25zc + the
# arc7 MX-native decode line. Content: (1) the arc7mx wheel
# vllm_xpu_kernels-0.1.14.1+arc7mx (md5 62f234c8..., byte-pinned by the
# builder) BAKED into the image -- it carries the MX-native
# paged-decode CUTLASS/Xe2 kernel (mx_attn_interface.mx_paged_decode),
# the r219 C++ auto-split heuristic at the validated optimum (chain 341
# bw -> 130 reduce -> 128 policy; ARC_MXKV_NSPLIT retired as a posture
# requirement, kept as probe instrument), the r219b ELSE-branch policy
# cap 64 -> 128 + block16 alignment, the corrected WG-canonical
# coordinate frames (r220a3: bwidth TiledCopy builders compose
# sg_to_vnk into svB; copies slice WG-linear over unsliced tiles; wi16
# only at fragment-tv decodes; sg_row_off only at the SLM V image WG
# frame), and the value-correct env-gated P64NATIVE instrument
# (default OFF -- the r220 e2e arbiter closed the 16->64 tile lever:
# offline -46..-50 pct @1K but e2e +0.86/+5.63/+13.40 pct itl monotone
# in ctx, NOT_RESCUED at any split depth). (2) The fork mx_kv overlay
# with the native_verify bridge (engine.py: ARC_MXKV_NATIVE env gate,
# default-off; _NATIVE_MINLEN per-lane thresholds mxfp8 16384,
# mxfp4 12288:8192, mxk8v4 8192; r212b capture telemetry; native
# __pycache__ purged). Certified MX posture on this image: the serve
# chain passes ARC_MXKV_NATIVE=1 with ARC_KVDTYPE (NSPLIT and
# P64NATIVE unset = C++ auto + _16 frame); the Triton v2/v5 lanes stay
# the below-threshold fallback and the certified-signature freeze
# (kernel constexpr TILE_FLOOR stays 4) is inherited from cert25zc
# unchanged. Built by r222_build_cert26.sh with FORK_SHA=<content
# commit>.
FROM vllm-arc:v0.30.0-b70-cert25zc

ARG FORK_SHA=unknown
ARG BUILT=unknown

COPY engine.py /tmp/mx_engine.py
COPY native_decode.py /tmp/mx_native_decode.py
COPY triton_mx_decode.py /tmp/mx_triton_mx_decode.py
COPY __init__.py /tmp/mx_init.py
COPY mx_dequant.py /tmp/mx_dequant.py
COPY mx_layout.py /tmp/mx_layout.py
COPY mx_quant.py /tmp/mx_quant.py
COPY triton_mx_store.py /tmp/mx_triton_mx_store.py
COPY vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl /tmp/mx_wheel.whl

RUN set -euxo pipefail; \
    SP=/opt/venv/lib/python3.12/site-packages/vllm_gaudi; \
    cp /tmp/mx_engine.py            $SP/xpu/patches/mx_kv/engine.py; \
    cp /tmp/mx_native_decode.py     $SP/xpu/patches/mx_kv/native_decode.py; \
    cp /tmp/mx_triton_mx_decode.py  $SP/xpu/patches/mx_kv/triton_mx_decode.py; \
    cp /tmp/mx_init.py              $SP/xpu/patches/mx_kv/__init__.py; \
    cp /tmp/mx_dequant.py           $SP/xpu/patches/mx_kv/mx_dequant.py; \
    cp /tmp/mx_layout.py            $SP/xpu/patches/mx_kv/mx_layout.py; \
    cp /tmp/mx_quant.py             $SP/xpu/patches/mx_kv/mx_quant.py; \
    cp /tmp/mx_triton_mx_store.py   $SP/xpu/patches/mx_kv/triton_mx_store.py; \
    rm -rf $SP/xpu/patches/__pycache__/mx_kv; \
    /opt/venv/bin/pip install --quiet --force-reinstall --no-deps /tmp/mx_wheel.whl; \
    rm -f /tmp/mx_*.py /tmp/mx_wheel.whl; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import inspect, os;\
       os.environ.pop('ARC_MXKV_NATIVE', None);\
       import vllm_gaudi.xpu.patches.mx_kv.engine as e;\
       import vllm_gaudi.xpu.patches.mx_kv.native_decode as n;\
       import vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode as t;\
       import vllm_gaudi.xpu.patches.mx_kv as pkg;\
       from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode;\
       esrc = inspect.getsource(e); nsrc = inspect.getsource(n);\
       tsrc = inspect.getsource(t); isrc = inspect.getsource(pkg);\
       assert 'from .native_decode import mx_decode_native' in esrc, 'native bridge import';\
       assert 'os.environ.get(\"ARC_MXKV_NATIVE\", \"\") == \"1\"' in esrc, 'native route default-off';\
       assert e._NATIVE_MINLEN == {'mxfp8': (16384, 16384), 'mxfp4': (12288, 8192), 'mxk8v4': (8192, 8192)}, 'per-lane native thresholds';\
       assert 'ARC_MXKV_NSPLIT' in nsrc, 'NSPLIT pin retained as instrument';\
       assert '_max_sk = int(bt_rows.shape[1]) * int(kv_cache.shape[1])' in nsrc, 'capacity-priced walk (r212f)';\
       assert 'MXKV native call fmt=' in nsrc, 'capture telemetry';\
       assert callable(n.mx_decode_native), 'bridge callable';\
       assert callable(mx_paged_decode), 'wheel native kernel live';\
       assert '_split_verify_decode' in esrc, 'batched MTP-verify decode';\
       assert '_V5_DEFAULT' in esrc and e._V5_DEFAULT == ('mxfp8', 'mxk8v4'), 'v5 default flip unchanged';\
       assert '_V5_G_MIN' in esrc and e._V5_G_MIN == 8, 'G-adaptive knee unchanged';\
       assert 'os.environ.get(\"ARC_MXKV_V5_TFLOOR\", \"2\") or 2' in tsrc, 'tf2 launcher default unchanged';\
       assert 'TILE_FLOOR: tl.constexpr = 4' in tsrc, 'kernel constexpr stays 4 (signature freeze)';\
       assert '_mx_decode_split_dot2' in tsrc, 'v7 research kernel present';\
       assert 'os.environ.get(\"ARC_MXKV_V7\", \"0\") == \"1\"' in tsrc, 'v7 dormant default-off';\
       assert 'mx_decode_rows_dot' in isrc, 'package export unchanged';\
       print('CERT26_NATIVE_LINE_LIVE wheel+overlay OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert26=P6.47 Phase-8 convergence -- arc7 MX-native decode line: the arc7mx wheel (vllm_xpu_kernels 0.1.14.1+arc7mx, md5 62f234c8d0aa89c4aec3e8d2c6fa22ad, byte-pinned) BAKED + the fork mx_kv overlay native_verify bridge. Wheel carries the MX-native paged-decode Xe2 kernel (mx_attn_interface.mx_paged_decode), the r219 auto-split heuristic at the validated optimum (chain 341 bw -> 130 reduce -> 128 policy; ARC_MXKV_NSPLIT retired as posture, kept as instrument), ELSE-branch policy cap 64 -> 128 + block16 alignment, the r220a3 corrected WG-canonical coordinate frames, and the value-correct env-gated P64NATIVE instrument (default OFF; the r220 e2e arbiter closed the 16->64 tile lever: offline wins inverted e2e +13.4 pct @32K, not rescued at any split depth; r221a/r221b eliminated shape/row-count/cache-state as offline-transferable mechanisms). Overlay carries the ARC_MXKV_NATIVE gate (default-off; certified MX posture passes it with ARC_KVDTYPE), _NATIVE_MINLEN per-lane thresholds (mxfp8 16384, mxfp4 12288:8192, mxk8v4 8192), the r212f capacity-priced walk (max_seqlen_k = bt_width x block count, per-row device limits bound the true walk), and capture-time bucket telemetry. Evidence: r219b auto-at-128 A/A (auto 48.5/51.2/56.8 within 0.08 pct of pinned; tps 65.4 @32K; chunks 53-54; 0 errors), r220b ctl e2e (posture -0.64 pct vs banked), r220c value gate v2 (d_16_64 <= 0.000977, det 0, nan 0). Triton v2/v5 lanes + certified-signature freeze inherited from cert25zc unchanged."; \
      echo "cert26_fork_sha=${FORK_SHA}"; \
      echo "cert26_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p647; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p647 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
DOCKERFILE_EOF

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r222] building vllm-arc:v0.30.0-b70-cert26 FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert26 \
  "$CTX"
echo "R222_CERT26_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert26 -c \
  'tail -3 /etc/vllm-arc/IMAGE_MANIFEST'
