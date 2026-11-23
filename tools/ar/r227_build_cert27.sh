#!/bin/bash
# r227_build_cert27.sh -- P6.47 cert27: build
# vllm-arc:v0.30.0-b70-cert27 (cert26 + the r225-fixed wheel: the
# arc7mx build25 wheel aeca45da BAKED over the cert26 wheel + the fork
# HEAD mx_kv overlay, which is cert26's overlay plus the r224a
# ARC_MXKV_AB diagnostic instrument in engine.py, default-off and
# self-guarding under capture). The ctx tree (the 8 mx_kv overlay .py
# files from fork HEAD) is staged by scp before this runs; the wheel
# is the rig's /root/build/wheels-arc-vxk arc7mx build25 (md5-pinned,
# SO md5-pinned). Refuses unless the staged tree carries the content
# markers AND the evidence banks stand: r225t gate 12/12 (the
# q-cast-matched offline gate on the r224a payloads), r226a e2e
# re-entry (engine A/B collapsed + 8/8 fresh COVERED payloads at the
# same criterion), build25 log rc=0.
# What build25 carries over the cert26 wheel (r225 fixes, all in the
# fork-owned kernel tree, provenance tools/ar/r225_fix*_apply.py):
# fix1 e2m1 subnormal mantissa (implicit bit + OCP decode), fix2
# sycl::exp2 in the split-KV combine rescale, fix3 fp32 split partials
# (ElementOAccum float; the single bf16 quantization point is the final
# combine write), fix5 the S==1 sentinel-LSE predicate aligned to the
# O-normalize arm. Certified posture change vs cert26: mkq (mxk8v4)
# native is UN-REFUSED — the r223 mkq amendment flips back at the r228
# gate (ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1).
# FORK_SHA must be passed explicitly (cert-release-sequence law:
# manifest fork_sha = the content commit).
set -euo pipefail
R=/root/build
CTX=$R/cert27ctx
FORK_SHA=${1:?usage: r227_build_cert27.sh <FORK_SHA>}
WHEELMD5=aeca45daa740e9973a1c51bfc5ea2bc4
SOMD5=190e261edc79b3b497d8d8e801e65757

SP=$CTX
# --- staged-tree content refuses (fork HEAD mx_kv overlay) ---
for f in engine.py native_decode.py triton_mx_decode.py __init__.py \
         mx_dequant.py mx_layout.py mx_quant.py triton_mx_store.py; do
  [ -f "$SP/$f" ] || { echo "R227_MISSING staged file $f"; exit 1; }
done
# cert26 native line intact
grep -q 'from .native_decode import mx_decode_native' "$SP/engine.py" \
  || { echo "R227_STALE_TREE (no native bridge import)"; exit 1; }
grep -q 'os.environ.get("ARC_MXKV_NATIVE", "") == "1"' "$SP/engine.py" \
  || { echo "R227_STALE_TREE (native route not env-gated default-off)"; exit 1; }
grep -q '"mxfp8": (16384, 16384)' "$SP/engine.py" \
  || { echo "R227_STALE_TREE (_NATIVE_MINLEN defaults drifted)"; exit 1; }
grep -q 'ARC_MXKV_NSPLIT' "$SP/native_decode.py" \
  || { echo "R227_STALE_TREE (no NSPLIT pin)"; exit 1; }
grep -q '_max_sk = int(bt_rows.shape\[1\]) \* int(kv_cache.shape\[1\])' "$SP/native_decode.py" \
  || { echo "R227_STALE_TREE (no capacity-priced walk -- r212f)"; exit 1; }
grep -q 'MXKV native call fmt=' "$SP/native_decode.py" \
  || { echo "R227_STALE_TREE (no capture telemetry)"; exit 1; }
# cert25zc inheritance intact
grep -q 'os.environ.get("ARC_MXKV_V5_TFLOOR", "2") or 2' "$SP/triton_mx_decode.py" \
  || { echo "R227_STALE_TREE (TILE_FLOOR launcher default not 2)"; exit 1; }
grep -q "_V5_G_MIN" "$SP/engine.py" \
  || { echo "R227_STALE_TREE (no G-adaptive knee)"; exit 1; }
grep -q "mx_decode_rows_dot" "$SP/__init__.py" \
  || { echo "R227_STALE_TREE (no package export)"; exit 1; }
# cert27 delta: the r224a AB diagnostic instrument, default-off
grep -q 'os.environ.get("ARC_MXKV_AB", "") == "1"' "$SP/engine.py" \
  || { echo "R227_STALE_TREE (no ARC_MXKV_AB default-off gate)"; exit 1; }
grep -q 'ARC_MXKV_AB_MINSL' "$SP/engine.py" \
  || { echo "R227_STALE_TREE (no AB MINSL budget bound)"; exit 1; }
# --- wheel refuse: the r225-gated build25, byte-pinned ---
WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R227_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R227_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"; exit 1; }
echo "wheel md5 $GOTMD5 (pinned)"
# --- evidence refuses (banked on the rig) ---
# launch2.log = the final r225t run's stdout (gate.log is the cell log;
# launch.log is the band4 first run that FAILed 10/12 -- history)
grep -q "R225T_GATE: PASS (12/12 cells, wheel_md5=$WHEELMD5)" \
  "$R/p647/r225t/launch2.log" 2>/dev/null \
  || { echo "R227_NO_R225T_EVIDENCE (offline q-cast-matched gate)"; exit 1; }
grep -q "R226A_LEG_DONE" "$R/p647/r226a/launch.log" 2>/dev/null \
  || { echo "R227_NO_R226A_EVIDENCE (e2e re-entry leg)"; exit 1; }
grep -q "score=8/8" "$R/p647/r226a/launch.log" 2>/dev/null \
  || { echo "R227_NO_R226A_SCORE (8/8 fresh payloads)"; exit 1; }
grep -q "VXK_ARC7MX_BUILD_EXIT rc=0" "$R/arc7mx_build25_r225.log" 2>/dev/null \
  || { echo "R227_NO_BUILD25_LOG (wheel build rc)"; exit 1; }
docker image inspect vllm-arc:v0.30.0-b70-cert26 >/dev/null 2>&1 \
  || { echo "R227_NO_BASE (cert26)"; exit 1; }

mkdir -p "$CTX"
cp "$WHEEL" "$CTX/$(basename "$WHEEL")"
cat > "$CTX/Dockerfile" <<'DOCKERFILE_EOF'
# SPDX-License-Identifier: Apache-2.0
# Dockerfile.cert27 -- P6.47 cert27 image: cert26 + the r225-fixed
# arc7mx wheel. Content: (1) the arc7mx build25 wheel
# vllm_xpu_kernels-0.1.14.1+arc7mx (md5 aeca45da..., byte-pinned by the
# builder; installed .so md5 190e261e..., asserted in-image) BAKED over
# the cert26 wheel -- it carries the r225 kernel fixes: fix1 e2m1
# subnormal mantissa (implicit bit for the normal band + OCP m*0.5
# subnormal decode in e2m1_scaled_bf16), fix2 sycl::exp2 in the
# split-KV combine rescale (evicting native::exp2's ~1e-3 relative
# error), fix3 fp32 split partials (temp_out always a dedicated fp32
# buffer; ElementOAccum float; the reduce combine is the ONE bf16
# quantization point), fix5 the S==1 sentinel-LSE arm aligned to the
# O-normalize predicate (chunk_prefill_epilogue.hpp:570). Validated:
# r225t offline gate 12/12 at the q-cast-matched criterion; r226a e2e
# re-entry 8/8 fresh COVERED payloads + engine A/B nan=0, lane gap
# collapsed an order of magnitude. (2) The fork HEAD mx_kv overlay =
# cert26's overlay plus the r224a ARC_MXKV_AB diagnostic instrument
# (default-off, self-guarding under capture, never kills serving).
# Certified MX posture on this image: qwen legs ARC_MXKV_NATIVE=1 with
# ARC_KVDTYPE (m8q/m4q/mkq alike -- mkq native UN-REFUSED by
# r225/r226; NSPLIT and P64NATIVE unset); the Triton v2/v5 lanes stay
# the below-threshold fallback. Built by r227_build_cert27.sh with
# FORK_SHA=<content commit>.
FROM vllm-arc:v0.30.0-b70-cert26

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
COPY vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl /tmp/

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
    /opt/venv/bin/pip install --quiet --force-reinstall --no-deps /tmp/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl; \
    rm -f /tmp/mx_*.py /tmp/vllm_xpu_kernels-*.whl; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import hashlib, inspect, os, pathlib;\
       os.environ.pop('ARC_MXKV_NATIVE', None);\
       import vllm_gaudi.xpu.patches.mx_kv.engine as e;\
       import vllm_gaudi.xpu.patches.mx_kv.native_decode as n;\
       import vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode as t;\
       import vllm_gaudi.xpu.patches.mx_kv as pkg;\
       import vllm_xpu_kernels;\
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
       assert 'ARC_MXKV_AB' in esrc, 'r224a AB instrument present';\
       assert 'os.environ.get(\"ARC_MXKV_AB\", \"\") == \"1\"' in esrc, 'AB instrument default-off';\
       assert 'ARC_MXKV_AB_MINSL' in esrc, 'AB budget bound';\
       assert '_split_verify_decode' in esrc, 'batched MTP-verify decode';\
       assert '_V5_DEFAULT' in esrc and e._V5_DEFAULT == ('mxfp8', 'mxk8v4'), 'v5 default flip unchanged';\
       assert '_V5_G_MIN' in esrc and e._V5_G_MIN == 8, 'G-adaptive knee unchanged';\
       assert 'os.environ.get(\"ARC_MXKV_V5_TFLOOR\", \"2\") or 2' in tsrc, 'tf2 launcher default unchanged';\
       assert 'TILE_FLOOR: tl.constexpr = 4' in tsrc, 'kernel constexpr stays 4 (signature freeze)';\
       assert '_mx_decode_split_dot2' in tsrc, 'v7 research kernel present';\
       assert 'os.environ.get(\"ARC_MXKV_V7\", \"0\") == \"1\"' in tsrc, 'v7 dormant default-off';\
       assert 'mx_decode_rows_dot' in isrc, 'package export unchanged';\
       sos = sorted(pathlib.Path(vllm_xpu_kernels.__file__).parent.rglob('libmqa_logits*.so'));\
       assert sos, 'no libmqa_logits .so in wheel install';\
       somd5 = hashlib.md5(sos[0].read_bytes()).hexdigest();\
       assert somd5 == '190e261edc79b3b497d8d8e801e65757', 'build25 libmqa_logits .so md5 pin: ' + somd5 + ' ' + sos[0].name;\
       print('CERT27_FIXED_WHEEL_LIVE overlay+so OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert27=P6.47 -- r225-fixed arc7mx wheel baked over cert26: build25 vllm_xpu_kernels 0.1.14.1+arc7mx (md5 aeca45daa740e9973a1c51bfc5ea2bc4, byte-pinned; .so md5 190e261edc79b3b497d8d8e801e65757, asserted). Kernel fixes over the cert26 wheel (fork-owned tree, provenance tools/ar/r225_fix*_apply.py): fix1 e2m1 subnormal mantissa (implicit bit + OCP subnormal decode), fix2 sycl::exp2 in the split-KV combine rescale, fix3 fp32 split partials (ElementOAccum float; the reduce combine is the single bf16 quantization point), fix5 S==1 sentinel-LSE predicate aligned to the O-normalize arm. Evidence: r225t offline q-cast-matched gate 12/12 ({base,nsplit1,nsplit2,nsplit8} x 3 r224a payloads), r226a e2e re-entry (engine A/B nan=0/120, raw lane gap p50 3.05e-1 -> 5.5e-2; 8/8 fresh COVERED payloads at the same criterion; tie-outlier allowance respected). Overlay = cert26 overlay + the r224a ARC_MXKV_AB diagnostic instrument (default-off, self-guarding under capture). Posture change: mkq (mxk8v4) native UN-REFUSED -- qwen MX legs certify ARC_MXKV_NATIVE=1 with ARC_KVDTYPE (m8q/m4q/mkq); Triton v2/v5 lanes stay the below-threshold fallback."; \
      echo "cert27_fork_sha=${FORK_SHA}"; \
      echo "cert27_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p647; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p647 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
DOCKERFILE_EOF

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r227] building vllm-arc:v0.30.0-b70-cert27 FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert27 \
  "$CTX"
echo "R227_CERT27_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert27 -c \
  'tail -3 /etc/vllm-arc/IMAGE_MANIFEST'
