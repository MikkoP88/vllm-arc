#!/bin/bash
# r270_build_cert30.sh -- P6.47 cert30 CANDIDATE (run on the rig):
# vllm-arc:v0.30.0-b70-cert30 = the r263-rebuilt cert29 + the ONE-file
# vllm_gaudi delta of the 906e8326 content commit (PR #188): the r269
# MX-KV decode page fix in mx_kv/triton_mx_decode.py. Root cause (r268c,
# banked results/2026-11-28-p647-r268c-trace): gemma's two KV pools page
# differently -- slide (2240,64,8,528) page 64, glob (2240,128,2,1056)
# page 128 (bt strides 640 vs 320) -- while the engine's five mx_decode
# call sites omit block_size, so the Triton decode walked the 128-paged
# glob table at BLOCK_SIZE=64 pitch: tokens 64+ read past the valid row
# prefix (never-written -> 400-nan, stale -> 'our'). The fix: all three
# decode entry points (mx_decode / mx_decode_rows / mx_decode_rows_dot)
# take block_size None and resolve the cache's OWN page cache.shape[1]
# at body start; explicit-page callers unchanged. Validated by the r269
# fix-gate A/B on the same posture under ARC_GRAPH=1: matrix RED
# 323/324 (r268b baseline) -> 0/329 (chat 108/108, raw 216/216, gate
# 5/5; http_err=0 in every cell; the 'Paris our our' gate battery 5/5
# coherent) -- engine bytes md5-identical to the image's own, so cert30
# = exactly the validated runtime (engine d1fa3c17 + triton e33aa7b7).
# Wheel half unchanged (inherited from cert29: build34 arc7mx, page-128
# admission; wheel md5 95451e404b463d9d6efa4bd4a4939b26 asserted at
# cert29 build; installed .so md5s re-asserted in-image here).
# FORK_SHA must be passed explicitly (cert-release-sequence law:
# manifest fork_sha = the content commit = 906e8326661a447d1c0cec49575
# aad3249912f49 -- the PR #188 squash merge). 4dd3a1b9 (cert29) and
# 16effb49 (the rc1 crash image) are refused.
# Ship step (separate ssh call, md5-verified both sides) stages the one
# vllm_gaudi file at /root/build/cert30stage_r270 BEFORE this runner
# (tools/ar/r270_ship.sh):
#   triton_mx_decode.py  e33aa7b749e34ea350441da2e339ff30  (the r269 fix)
set -euo pipefail
R=/root/build
CTX=$R/cert30ctx_r270
STAGE=$R/cert30stage_r270
IMG=vllm-arc:v0.30.0-b70-cert30
BASE=vllm-arc:v0.30.0-b70-cert29
FORK_SHA=${1:?usage: r270_build_cert30.sh <FORK_SHA>}
TMD5=e33aa7b749e34ea350441da2e339ff30
ENGINE_MD5=d1fa3c17cde044211cbf62e28ddcc1b1
CERT29_SHA=4dd3a1b9710f1c638f7f0b5ac1885525cf57c3bc
REFUSED1=$CERT29_SHA
REFUSED2=16effb497cd36cb25df7c19cb6f1938d2d45cdc5

[ "$FORK_SHA" != "$REFUSED1" ] \
  || { echo "R270_REFUSED_CERT29_SHA (4dd3a1b9 is the base image, not this delta)"; exit 1; }
[ "$FORK_SHA" != "$REFUSED2" ] \
  || { echo "R270_REFUSED_CRASH_SHA (16effb49 is the rc1 crash image)"; exit 1; }
[ "$FORK_SHA" = "906e8326661a447d1c0cec49575aad3249912f49" ] \
  || { echo "R270_REFUSED_UNEXPECTED_SHA (expected the PR #188 content commit)"; exit 1; }

# --- base image refuses: the r263 rebuild of cert29, never rc1 ---
docker image inspect "$BASE" >/dev/null 2>&1 \
  || { echo "R270_NO_BASE (the r263-rebuilt cert29)"; exit 1; }
MFB=$(docker run --rm --entrypoint /bin/bash "$BASE" -c \
  'grep -o "cert29_fork_sha=[0-9a-f]*" /etc/vllm-arc/IMAGE_MANIFEST | head -1')
[ "$MFB" = "cert29_fork_sha=$CERT29_SHA" ] \
  || { echo "R270_BASE_STALE ($MFB -- need the r263 rebuild 4dd3a1b9, not rc1)"; exit 1; }
if docker image inspect "$IMG" >/dev/null 2>&1; then
  MFOLD=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
    'grep -o "cert30_fork_sha=[0-9a-f]*" /etc/vllm-arc/IMAGE_MANIFEST | head -1' 2>/dev/null) || MFOLD=""
  case "$MFOLD" in
    "cert30_fork_sha=$FORK_SHA")
      echo "R270_ALREADY_BUILT ($MFOLD) -- rebuilding the same content commit" ;;
    "")
      echo "R270_PRIOR_TAG_UNMARKED -- refusing to move an unknown image"; exit 1 ;;
    *)
      echo "R270_UNEXPECTED_CERT30_TAG ($MFOLD) -- refusing"; exit 1 ;;
  esac
fi

# --- evidence refuses (banked on the rig): root cause + fix A/B ---
grep -q "^R268C_DONE rc=0" "$R/p647/r268c_trace/logs/launch.log" 2>/dev/null \
  || { echo "R270_NO_ROOT_CAUSE_EVIDENCE (r268c trace leg)"; exit 1; }
grep -q "^R269_DONE rc=0" "$R/p647/r269_fixgate/logs/launch.log" 2>/dev/null \
  || { echo "R270_NO_FIX_EVIDENCE (r269 fix-gate leg)"; exit 1; }
grep -q "P266_MATRIX_DONE .*oracle=True" "$R/p647/r269_fixgate/logs/launch.log" 2>/dev/null \
  || { echo "R270_NO_MATRIX_EVIDENCE (the fix-gate matrix oracle)"; exit 1; }
grep -q "KERNEL_HEALTH\[f3\]: OK fault_lines=0" "$R/p647/r269_fixgate/logs/launch.log" 2>/dev/null \
  || { echo "R270_NO_KERNEL_HEALTH_EVIDENCE"; exit 1; }
echo "R270_EVIDENCE_OK (r268c root cause + r269 fix-gate A/B 323/324 -> 0/329)"

# --- staged delta refuses ---
[ -f "$STAGE/triton_mx_decode.py" ] \
  || { echo "R270_NO_STAGED_FILE triton_mx_decode.py (run the ship step)"; exit 1; }
GOT=$(md5sum "$STAGE/triton_mx_decode.py" | cut -d" " -f1)
[ "$GOT" = "$TMD5" ] \
  || { echo "R270_MD5_MISMATCH triton_mx_decode.py $GOT != $TMD5"; exit 1; }
echo "R270_STAGE_OK triton_mx_decode.py md5 verified (the r269 fix bytes)"

rm -rf "$CTX"
mkdir -p "$CTX"
cp "$STAGE/triton_mx_decode.py" "$CTX/"
cat > "$CTX/Dockerfile" <<DOCKERFILE_EOF
# SPDX-License-Identifier: Apache-2.0
# Dockerfile.cert30 (r270) -- P6.47 cert30 candidate: the r263-rebuilt
# cert29 + the ONE-file vllm_gaudi delta of the ${FORK_SHA} content
# commit (PR #188) = the r269 MX-KV decode page fix in mx_kv/
# triton_mx_decode.py: all three decode entry points resolve
# block_size None -> the cache's OWN page (cache.shape[1]) at body
# start. Root cause r268c: gemma's glob pool pages at 128 while the
# engine call sites omitted block_size -> the default-64 walk read
# past the valid table prefix at tokens 64+ (never-written -> 400-nan,
# stale -> 'our'; qwen's uniform p64 pools immune). Validated r269:
# matrix 323/324 RED -> 0/329 GREEN under ARC_GRAPH=1 on the same
# posture; engine bytes md5-identical to the image's own, so this
# image = exactly the validated runtime. triton_mx_decode.py md5
# ${TMD5} asserted in-image; engine.py stays ${ENGINE_MD5}; wheel + .so
# pins inherited from the r263 cert29 build and re-asserted here.
FROM vllm-arc:v0.30.0-b70-cert29

ARG FORK_SHA=unknown
ARG BUILT=unknown
ARG TMD5=unknown
ARG ENGINE_MD5=unknown

COPY triton_mx_decode.py /tmp/vg/

RUN set -euxo pipefail; \\
    SP=/opt/venv/lib/python3.12/site-packages; \\
    cp /tmp/vg/triton_mx_decode.py \$SP/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py; \\
    rm -rf /tmp/vg; \\
    find \$SP/vllm_gaudi -name __pycache__ -type d -exec rm -rf {} + || true; \\
    export LD_LIBRARY_PATH=/opt/venv/lib:\$SP/torch/lib; \\
    /opt/venv/bin/python -c \\
      "import hashlib, inspect, os, pathlib;\\
       os.environ.pop('ARC_MXKV_NATIVE', None);\\
       import vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode as t;\\
       import vllm_gaudi.xpu.patches.mx_kv.engine as e;\\
       import vllm_xpu_kernels;\\
       tsrc = inspect.getsource(t);\\
       assert tsrc.count('block_size: int | None = None') == 3, 'r269 fix: three None-default entry points';\\
       assert tsrc.count('block_size = cache.shape[1]') == 3, 'r269 fix: three own-page resolutions';\\
       assert callable(t.mx_decode) and callable(t.mx_decode_rows) and callable(t.mx_decode_rows_dot), 'three entry points live';\\
       tmd5 = hashlib.md5(pathlib.Path(t.__file__).read_bytes()).hexdigest();\\
       assert tmd5 == '\${TMD5}', 'triton_mx_decode.py bytes ' + tmd5;\\
       emd5 = hashlib.md5(pathlib.Path(e.__file__).read_bytes()).hexdigest();\\
       assert emd5 == '\${ENGINE_MD5}', 'engine.py bytes ' + emd5;\\
       vp = pathlib.Path(vllm_xpu_kernels.__file__).parent;\\
       md5 = lambda nm: hashlib.md5((vp / nm).read_bytes()).hexdigest();\\
       assert md5('libattn_kernels_xe_2.so') == 'c20ca75550f29513a0f73b70ebb1243a', 'build34 libattn pin (inherited)';\\
       assert md5('_vllm_fa2_C.abi3.so') == '046e915ebce0eb33a37832377f8c2aed', 'build34 fa2 stub pin (inherited)';\\
       mqa = sorted(vp.rglob('libmqa_logits*.so'));\\
       assert mqa and hashlib.md5(mqa[0].read_bytes()).hexdigest() == '190e261edc79b3b497d8d8e801e65757', 'libmqa pin (inherited)';\\
       print('CERT30_DELTA_LIVE asserts OK triton_mx_decode.py', tmd5, 'engine.py', emd5)"; \\
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \\
      echo "cert30=P6.47 -- the r263 cert29 + the ONE-file vllm_gaudi delta of the \${FORK_SHA} content commit (PR #188): the r269 MX-KV decode page fix (mx_kv/triton_mx_decode.py -- all three decode entry points mx_decode/mx_decode_rows/mx_decode_rows_dot take block_size None and resolve the cache's OWN page cache.shape[1] at body start; explicit-page callers unchanged). Root cause r268c (banked results/2026-11-28-p647-r268c-trace): gemma's two KV pools page differently -- slide (2240,64,8,528) page 64, glob (2240,128,2,1056) page 128, bt strides 640 vs 320 -- while the engine's five mx_decode call sites omit block_size, so the default-64 walk read the 128-paged glob table past its valid row prefix at tokens 64+ (never-written rows -> 400-nan logprobs, stale rows -> the 'our' attractor; slot alignment in the 128-space = the runtime-state dependence; qwen uniform-p64 immune). Fix validated r269 (banked results/2026-11-28-p647-r269-fixgate): same posture A/B, ARC_GRAPH=1 production graphs, matrix RED 323/324 (r268b baseline) -> 0/329 (chat 108/108, raw 216/216, gate 5/5; http_err=0 every cell; finite logprobs everywhere; the 'Paris our our' gate battery 5/5 coherent; KERNEL_HEALTH OK) -- engine bytes md5-identical to the image's own, so this image = exactly the validated runtime. triton_mx_decode.py \${TMD5}, engine.py \${ENGINE_MD5}, wheel + .so pins inherited from the r263 cert29 build."; \\
      echo "cert30_fork_sha=\${FORK_SHA}"; \\
      echo "cert30_built=\${BUILT}"; \\
      echo "cert30_built_by=r270"; \\
    } > /etc/vllm-arc/IMAGE_MANIFEST.p647; \\
    mv /etc/vllm-arc/IMAGE_MANIFEST.p647 /etc/vllm-arc/IMAGE_MANIFEST; \\
    cat /etc/vllm-arc/IMAGE_MANIFEST
DOCKERFILE_EOF

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r270] building $IMG FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  --build-arg TMD5="$TMD5" \
  --build-arg ENGINE_MD5="$ENGINE_MD5" \
  -t "$IMG" \
  "$CTX"
echo "R270_CERT30_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'tail -4 /etc/vllm-arc/IMAGE_MANIFEST'
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'cat /etc/vllm-arc/IMAGE_MANIFEST' | grep -q "^cert30_fork_sha=$FORK_SHA\$" \
  || { echo "R270_MANIFEST_FORK_SHA_MISMATCH"; exit 1; }
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'md5sum /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py' \
  | grep -q "$TMD5" \
  || { echo "R270_INSTALLED_BYTES_MISMATCH"; exit 1; }
echo "R270_IMAGE_OK $IMG cert30_fork_sha=$FORK_SHA"
