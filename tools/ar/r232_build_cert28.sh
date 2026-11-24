#!/bin/bash
# r232_build_cert28.sh -- P6.47 cert28: build
# vllm-arc:v0.30.0-b70-cert28 (cert27 + the r231d arc7mx build28
# wheel: the seqused-tightened split clamp BAKED over the build25
# wheel). Kernel-only delta -- the mx_kv overlay is byte-identical
# to cert27's (inherited via FROM; no overlay staging needed):
#   mx_paged_decode_kernel.hpp legacy path + mx_reduce_split.hpp
#   (mirrored): seq_num_kv_splits = min(num_kv_splits,
#   max(1, windowed_k_blocks/4)) -- the host count prices from
#   max_seqlen_k capacity (r212f no-sync law) and over-provisioned
#   shallow-ctx decode; the clamp derives the env-pin semantics
#   per-seq ON DEVICE. Buffers keep the capacity layout; the reduce
#   reads only i < effective_splits.
# Evidence (refused below): r231d offline gates (G1/G2/G3 bitwise
# + 12 r229 band cells + ladder 2K -43/-39/-51 pct, 8K+ neutral);
# r231de e2e (3 boots, A/A 0.0-0.1 pct: c2048 -7.6 pct, c4096
# -5.6 pct, c8192 0.984x, c32768 1.000x -- the banked r230 record
# reproduced exactly by the baseline); build28 rc=0.
# FORK_SHA must be passed explicitly (cert-release-sequence law:
# manifest fork_sha = the content commit = 43f84cf9... the r231d
# offline+e2e evidence merge).
set -euo pipefail
R=/root/build
CTX=$R/cert28ctx
FORK_SHA=${1:?usage: r232_build_cert28.sh <FORK_SHA>}
WHEELMD5=e6d8d337e43b907504537061414128f8
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_FA2=6e6c48508d40cb4c5007f27f79df97cf
SO_MQA=190e261edc79b3b497d8d8e801e65757

# --- base image refuses ---
docker image inspect vllm-arc:v0.30.0-b70-cert27 >/dev/null 2>&1 \
  || { echo "R232_NO_BASE (cert27)"; exit 1; }
MF27=$(docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert27 -c \
  'grep -o "cert27_fork_sha=[0-9a-f]*" /etc/vllm-arc/IMAGE_MANIFEST | head -1')
[ "$MF27" = "cert27_fork_sha=d39ab5a26bf737d006d47a3f3125a404dca6e928" ] \
  || { echo "R232_BASE_STALE ($MF27)"; exit 1; }

# --- wheel refuse: the r231d-gated build28, byte-pinned ---
WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R232_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R232_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"; exit 1; }
echo "wheel md5 $GOTMD5 (pinned build28)"

# --- evidence refuses (banked on the rig) ---
grep -q "R231D_ORACLE_GATES PASS" "$R/p647/r231d/offline_launch2.log" 2>/dev/null \
  || { echo "R232_NO_OFFLINE_EVIDENCE (r231d oracle gates)"; exit 1; }
grep -q "R231D_R229CELLS pass=12 fail=0" "$R/p647/r231d/offline_launch2.log" 2>/dev/null \
  || { echo "R232_NO_OFFLINE_BANDS (r231d band cells)"; exit 1; }
grep -q "R231DE_GATE: PASS" "$R/p647/r231d_e2e/launch.log" 2>/dev/null \
  || { echo "R232_NO_E2E_EVIDENCE (r231de gate)"; exit 1; }
grep -q "BUILD_EXIT rc=0" "$R/arc7mx_build28_r231d.log" 2>/dev/null \
  || { echo "R232_NO_BUILD28_LOG (wheel build rc)"; exit 1; }

mkdir -p "$CTX"
cp "$WHEEL" "$CTX/$(basename "$WHEEL")"
cat > "$CTX/Dockerfile" <<'DOCKERFILE_EOF'
# SPDX-License-Identifier: Apache-2.0
# Dockerfile.cert28 -- P6.47 cert28 image: cert27 + the r231d
# arc7mx build28 wheel (seqused-tightened splits). Kernel-only
# delta over cert27, provenance tools/ar/r231d_apply.py (mirrored
# edit in mx_paged_decode_kernel.hpp + mx_reduce_split.hpp):
# device-side split clamp min(num_kv_splits, max(1,
# windowed_k_blocks/4)) on the legacy scalar path -- the host count
# prices from max_seqlen_k capacity (r212f no-sync law) and
# over-provisioned shallow-ctx decode (+92 pct at 2K walked ctx);
# the clamp derives the env-pin semantics per-seq on device and is
# a no-op at >=8K walked ctx (capacity). The mx_kv overlay is
# byte-identical to cert27's (inherited). Wheel
# vllm_xpu_kernels-0.1.14.1+arc7mx md5 e6d8d337..., byte-pinned;
# installed .so md5s asserted in-image: libattn_kernels_xe_2
# 61d2f5a3... (the MX paged-decode carrier), _vllm_fa2_C
# 6e6c4850..., libmqa_logits 190e261e... (not relinked by the
# header-only patch). Evidence: r231d offline gates (G1 clamp==pin
# / G2 pin-no-op / G3 capacity, 12+12+3 bitwise; 12 r229 band
# cells; ladder 2K -43/-39/-51 pct with 8K+ neutral) and r231de
# e2e (3 boots, A/A 0.0-0.1 pct: c2048 -7.6 pct ITL, c4096 -5.6,
# c8192 0.984x, c32768 1.000x -- record parity exact). Certified
# posture unchanged from cert27: qwen MX legs ARC_MXKV_NATIVE=1
# with ARC_KVDTYPE (m8q/m4q/mkq), NSPLIT unset (C++ auto; the
# clamp now guards the shallow-ctx tail of that auto). Built by
# r232_build_cert28.sh with FORK_SHA=<content commit>.
FROM vllm-arc:v0.30.0-b70-cert27

ARG FORK_SHA=unknown
ARG BUILT=unknown

COPY vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl /tmp/

RUN set -euxo pipefail; \
    /opt/venv/bin/pip install --quiet --force-reinstall --no-deps /tmp/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl; \
    rm -f /tmp/vllm_xpu_kernels-*.whl; \
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
       assert 'from .native_decode import mx_decode_native' in esrc, 'native bridge import (inherited)';\
       assert 'os.environ.get(\"ARC_MXKV_NATIVE\", \"\") == \"1\"' in esrc, 'native route default-off (inherited)';\
       assert e._NATIVE_MINLEN == {'mxfp8': (16384, 16384), 'mxfp4': (12288, 8192), 'mxk8v4': (8192, 8192)}, 'per-lane native thresholds (inherited)';\
       assert 'ARC_MXKV_NSPLIT' in nsrc, 'NSPLIT pin retained as instrument (inherited)';\
       assert '_max_sk = int(bt_rows.shape[1]) * int(kv_cache.shape[1])' in nsrc, 'capacity-priced walk (r212f, inherited)';\
       assert 'MXKV native call fmt=' in nsrc, 'capture telemetry (inherited)';\
       assert 'ARC_MXKV_AB' in esrc, 'r224a AB instrument (inherited)';\
       assert callable(n.mx_decode_native) and callable(mx_paged_decode), 'bridge + wheel kernel live';\
       vp = pathlib.Path(vllm_xpu_kernels.__file__).parent;\
       md5 = lambda nm: hashlib.md5((vp / nm).read_bytes()).hexdigest();\
       assert md5('libattn_kernels_xe_2.so') == '61d2f5a365f05084b3dd72c4ae1da3e0', 'build28 libattn pin';\
       assert md5('_vllm_fa2_C.abi3.so') == '6e6c48508d40cb4c5007f27f79df97cf', 'build28 fa2 stub pin';\
       mqa = sorted(vp.rglob('libmqa_logits*.so'));\
       assert mqa and hashlib.md5(mqa[0].read_bytes()).hexdigest() == '190e261edc79b3b497d8d8e801e65757', 'libmqa pin (unrelinked)';\
       print('CERT28_CLAMPED_WHEEL_LIVE overlay+so OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert28=P6.47 -- r231d seqused-tightened split clamp wheel baked over cert27: arc7mx build28 vllm_xpu_kernels 0.1.14.1+arc7mx (md5 e6d8d337e43b907504537061414128f8, byte-pinned; libattn_kernels_xe_2 .so 61d2f5a365f05084b3dd72c4ae1da3e0, _vllm_fa2_C 6e6c48508d40cb4c5007f27f79df97cf, libmqa_logits 190e261edc79b3b497d8d8e801e65757, all asserted). Kernel delta (fork-owned tree, provenance tools/ar/r231d_apply.py, mirrored in mx_paged_decode_kernel.hpp + mx_reduce_split.hpp): device-side split clamp min(num_kv_splits, max(1, windowed_k_blocks/4)) on the legacy scalar path -- the host count prices from max_seqlen_k capacity (r212f no-sync law); the clamp derives the env-pin semantics per-seq on device, no-op at >=8K walked ctx. Overlay byte-identical to cert27 (inherited). Evidence: r231d offline gates G1/G2/G3 bitwise 12/12/3 + 12 r229 band cells + ladder (2K auto -43/-39/-51 pct, 8K+ within dice); r231de e2e 3 boots A/A 0.0-0.1 pct (c2048 47.0->43.4 ms -7.6 pct, c4096 -5.6 pct, c8192 0.984x, c32768 58.1->58.0/58.1 1.000x -- the r230 cert27 record reproduced exactly by the same-day baseline). Posture unchanged: qwen MX legs ARC_MXKV_NATIVE=1 + ARC_KVDTYPE, NSPLIT unset."; \
      echo "cert28_fork_sha=${FORK_SHA}"; \
      echo "cert28_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p647; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p647 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
DOCKERFILE_EOF

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r232] building vllm-arc:v0.30.0-b70-cert28 FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert28 \
  "$CTX"
echo "R232_CERT28_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert28 -c \
  'tail -3 /etc/vllm-arc/IMAGE_MANIFEST'
