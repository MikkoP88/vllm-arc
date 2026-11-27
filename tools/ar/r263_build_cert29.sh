#!/bin/bash
# r263_build_cert29.sh -- P6.47 cert29 REBUILD (run on the rig):
# vllm-arc:v0.30.0-b70-cert29 = cert28 + the build34 arc7mx wheel + the
# six-file vllm_gaudi delta of the 4dd3a1b9 content commit (PR #167).
# Why a rebuild: the first cert29 image (r254, content commit 16effb49)
# carries the expand-view block-table stride defect -- the windowed
# continuation-prefill synth path hands the native wheel a stride-0
# expand VIEW while the wheel addresses rows at a contiguous pitch
# (rows 1..63 read the null block, rows >= 64 read past the table); it
# crashed the gemma MX e2e at every multi rehit (DEVICE_LOST, r255-r260)
# and is e2e-fixed by the r262 bridge guard (R262_SUMMARY a_probe=0
# c_suite=0 b_ctrl=2). The delta vs the r254 image is exactly two
# python files (git diff --stat 16effb49..4dd3a1b9 -- vllm_gaudi/):
#   mx_kv/native_decode.py  the bridge materializes a non-contiguous
#                           bt_rows (metadata check + device copy, one
#                           log line per (row pitch, pool) class)
#   mx_kv/engine.py         the contract comment at the synth call site
# The four other delta files are byte-identical to the r254 pins.
# Wheel half unchanged from r254 (build34, page-128 admission; wheel
# md5 95451e404b463d9d6efa4bd4a4939b26 + the .r245a.md5 sidecar;
# installed .so md5s asserted in-image). Installed engine.py /
# native_decode.py md5s are asserted in-image against the content
# commit's blobs (ENGINE_MD5 / NATIVE_MD5 build args).
# The crash image keeps an addressable alias BEFORE the tag moves:
#   vllm-arc:v0.30.0-b70-cert29-rc1  (cert29_fork_sha=16effb49; the
#   image the r255-r262 banks ran on).
# FORK_SHA must be passed explicitly (cert-release-sequence law:
# manifest fork_sha = the content commit = 4dd3a1b9710f1c638f7f0b5ac18
# 5525cf57c3bc -- the PR #167 squash merge). 16effb49 is refused.
# Ship step (separate ssh call, md5-verified both sides) stages the six
# vllm_gaudi files at /root/build/cert29stage_r263 BEFORE this runner
# (tools/ar/r263_ship.sh):
#   envs.py              31ca4ac7cfcd996aed6446d4b24c2bcd  (unchanged)
#   platform.py          0733f0b8c7dd2fb89422bafa56ec35f5  (unchanged)
#   lmc_mx_boot_gate.py  04408d2e12cbdd8587cac4633624b7bc  (unchanged)
#   vllm_v1_adapter.py   7ca7cff4e1e04d4e0ea73573b1234664  (unchanged)
#   engine.py            d1fa3c17cde044211cbf62e28ddcc1b1  (r262 comment)
#   native_decode.py     81e427590e7ad3482e46bdc25f4a62b2  (r262 guard)
set -euo pipefail
R=/root/build
CTX=$R/cert29ctx_r263
STAGE=$R/cert29stage_r263
IMG=vllm-arc:v0.30.0-b70-cert29
ALIAS=vllm-arc:v0.30.0-b70-cert29-rc1
FORK_SHA=${1:?usage: r263_build_cert29.sh <FORK_SHA>}
WHEELMD5=95451e404b463d9d6efa4bd4a4939b26
SO_ATTN=c20ca75550f29513a0f73b70ebb1243a
SO_FA2=046e915ebce0eb33a37832377f8c2aed
SO_MQA=190e261edc79b3b497d8d8e801e65757
CRASH_SHA=16effb497cd36cb25df7c19cb6f1938d2d45cdc5
ENGINE_MD5=d1fa3c17cde044211cbf62e28ddcc1b1
NATIVE_MD5=81e427590e7ad3482e46bdc25f4a62b2

[ "$FORK_SHA" != "$CRASH_SHA" ] \
  || { echo "R263_REFUSED_CRASH_SHA (16effb49 is the image being replaced)"; exit 1; }

# --- base image refuses ---
docker image inspect vllm-arc:v0.30.0-b70-cert28 >/dev/null 2>&1 \
  || { echo "R263_NO_BASE (cert28)"; exit 1; }
MF28=$(docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert28 -c \
  'grep -o "cert28_fork_sha=[0-9a-f]*" /etc/vllm-arc/IMAGE_MANIFEST | head -1')
[ "$MF28" = "cert28_fork_sha=43f84cf977f9444900ff21387715ca5865db4c1a" ] \
  || { echo "R263_BASE_STALE ($MF28)"; exit 1; }

# --- wheel refuse: the r245a-gated build34, byte-pinned ---
WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R263_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R263_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"; exit 1; }
[ -f "$WHEEL.r245a.md5" ] || { echo "R263_NO_WHEEL_SIDECAR"; exit 1; }
grep -q "$WHEELMD5" "$WHEEL.r245a.md5" || { echo "R263_SIDECAR_MISMATCH"; exit 1; }
echo "wheel md5 $GOTMD5 (pinned build34, page-128 admission)"

# --- evidence refuses (banked on the rig) ---
grep -q "VXK_ARC7MX_BUILD_EXIT rc=0" "$R/arc7mx_build34_r245a.log" 2>/dev/null \
  || { echo "R263_NO_BUILD34_LOG (wheel build rc)"; exit 1; }
grep -q "VERDICT: PASS" "$R/p245a/out/r245a_summary.txt" 2>/dev/null \
  || { echo "R263_NO_OFFLINE_EVIDENCE (r245a page-128 oracle)"; exit 1; }
grep -q "R245_DONE failed=\[\] 2026-11-26T13:04:09Z" "$R/logs/r245_e2e_launch.log" 2>/dev/null \
  || { echo "R263_NO_E2E_EVIDENCE (r245 grid, post-wheel)"; exit 1; }
grep -q "== p246 faces ok=11/11" "$R/p246/out/r246_summary.txt" 2>/dev/null \
  || { echo "R263_NO_PD_EVIDENCE (r246 probe)"; exit 1; }
grep -q "INTEROP_PASS" "$R/p249/outcome.txt" 2>/dev/null \
  || { echo "R263_NO_PH_EVIDENCE (r249 interop e2e)"; exit 1; }
grep -q "== p253 faces ok=15/15" "$R/p253/out/r253_summary.txt" 2>/dev/null \
  || { echo "R263_NO_PH4_EVIDENCE (r253 posture-aware flip probe)"; exit 1; }
grep -q "R262_SUMMARY a_probe=0 c_suite=0 b_ctrl=2" "$R/logs/r262_launch.log" 2>/dev/null \
  || { echo "R263_NO_STRIDE_FIX_EVIDENCE (r262 e2e A/B)"; exit 1; }
echo "R263_EVIDENCE_OK (wheel oracle/grid + P-D + P-H + P-H4 + the r262 stride A/B)"

# --- the crash image keeps its alias before the tag moves ---
if docker image inspect "$IMG" >/dev/null 2>&1; then
  MFOLD=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
    'grep -o "cert29_fork_sha=[0-9a-f]*" /etc/vllm-arc/IMAGE_MANIFEST | head -1' 2>/dev/null) || MFOLD=""
  case "$MFOLD" in
    "cert29_fork_sha=$CRASH_SHA")
      docker tag "$IMG" "$ALIAS"
      echo "R263_CRASH_IMAGE_ALIASED $ALIAS ($MFOLD id=$(docker image inspect -f '{{.Id}}' "$IMG" | cut -c8-19))"
      ;;
    "cert29_fork_sha=$FORK_SHA")
      echo "R263_ALREADY_BUILT ($MFOLD) -- rebuilding the same content commit"
      ;;
    *)
      echo "R263_UNEXPECTED_CERT29_TAG ($MFOLD) -- refusing to move an unknown image"
      exit 1
      ;;
  esac
else
  echo "R263_NO_PRIOR_CERT29 (fresh tag)"
fi

# --- staged vllm_gaudi delta refuses ---
declare -A VG=( \
  [envs.py]=31ca4ac7cfcd996aed6446d4b24c2bcd \
  [platform.py]=0733f0b8c7dd2fb89422bafa56ec35f5 \
  [lmc_mx_boot_gate.py]=04408d2e12cbdd8587cac4633624b7bc \
  [vllm_v1_adapter.py]=7ca7cff4e1e04d4e0ea73573b1234664 \
  [engine.py]=$ENGINE_MD5 \
  [native_decode.py]=$NATIVE_MD5 )
for f in envs.py platform.py lmc_mx_boot_gate.py vllm_v1_adapter.py \
         engine.py native_decode.py; do
  [ -f "$STAGE/$f" ] || { echo "R263_NO_STAGED_FILE $f (run the ship step)"; exit 1; }
  got=$(md5sum "$STAGE/$f" | cut -d" " -f1)
  [ "$got" = "${VG[$f]}" ] || { echo "R263_MD5_MISMATCH $f $got != ${VG[$f]}"; exit 1; }
done
echo "R263_STAGE_OK md5s verified (the 4dd3a1b9 content-commit files)"

rm -rf "$CTX"
mkdir -p "$CTX"
cp "$WHEEL" "$CTX/$(basename "$WHEEL")"
cp "$STAGE"/envs.py "$STAGE"/platform.py "$STAGE"/lmc_mx_boot_gate.py \
   "$STAGE"/vllm_v1_adapter.py "$STAGE"/engine.py "$STAGE"/native_decode.py "$CTX/"
cat > "$CTX/Dockerfile" <<'DOCKERFILE_EOF'
# SPDX-License-Identifier: Apache-2.0
# Dockerfile.cert29 (r263 rebuild) -- P6.47 cert29 image: cert28 + the
# r245a build34 wheel (page-128 admission) + the six-file vllm_gaudi
# delta of the 4dd3a1b9 content commit = the P-C2 window plumbing, the
# P-D boot gate + knob, the P-H2 packed-row interop fix + marker, the
# P-H4 posture-aware gate flip (all as the r254 image) + the r262
# stride fix (the native bridge materializes a non-contiguous block
# table; the windowed synth path's stride-0 expand VIEW reached the
# wheel unmaterialized and read past the table = the gemma DEVICE_LOST).
# Wheel vllm_xpu_kernels-0.1.14.1+arc7mx md5 95451e404b463d9d6efa4bd4a4939b26
# byte-pinned; installed .so md5s asserted in-image: libattn_kernels_xe_2
# c20ca755... (the MX paged-decode carrier incl. the page-128
# admission), _vllm_fa2_C 046e915e..., libmqa_logits 190e261e... (not
# relinked); installed engine.py / native_decode.py md5s asserted
# against the content commit's blobs. Engine posture: sliding-window
# impls boot only on the native decode lane (P-C2 init refusal,
# boot-loud); LMCache x MX is POSTURE-AWARE at connector construction
# (P-H4). Evidence: r245a offline oracle, r245 e2e grid R245_DONE
# failed=[] post-wheel, r246 probe 11/11 (P-D), r249 INTEROP_PASS
# (P-H interop), r253 probe 15/15 (P-H4 both postures), r262 e2e A/B
# R262_SUMMARY a_probe=0 c_suite=0 b_ctrl=2 (the stride fix). Built by
# r263_build_cert29.sh with FORK_SHA=<content commit>.
FROM vllm-arc:v0.30.0-b70-cert28

ARG FORK_SHA=unknown
ARG BUILT=unknown
ARG ENGINE_MD5=unknown
ARG NATIVE_MD5=unknown

COPY vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl /tmp/
COPY envs.py platform.py lmc_mx_boot_gate.py vllm_v1_adapter.py \
     engine.py native_decode.py /tmp/vg/

RUN set -euxo pipefail; \
    /opt/venv/bin/pip install --quiet --force-reinstall --no-deps /tmp/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl; \
    rm -f /tmp/vllm_xpu_kernels-*.whl; \
    SP=/opt/venv/lib/python3.12/site-packages; \
    cp /tmp/vg/envs.py              $SP/vllm_gaudi/envs.py; \
    cp /tmp/vg/platform.py          $SP/vllm_gaudi/xpu/platform.py; \
    cp /tmp/vg/lmc_mx_boot_gate.py  $SP/vllm_gaudi/xpu/patches/lmc_mx_boot_gate.py; \
    cp /tmp/vg/vllm_v1_adapter.py   $SP/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py; \
    cp /tmp/vg/engine.py            $SP/vllm_gaudi/xpu/patches/mx_kv/engine.py; \
    cp /tmp/vg/native_decode.py     $SP/vllm_gaudi/xpu/patches/mx_kv/native_decode.py; \
    find $SP/vllm_gaudi -name __pycache__ -type d -exec rm -rf {} + || true; \
    export LD_LIBRARY_PATH=/opt/venv/lib:$SP/torch/lib; \
    /opt/venv/bin/python -c \
      "import hashlib, inspect, os, pathlib;\
       os.environ.pop('ARC_MXKV_NATIVE', None);\
       import vllm_gaudi.xpu.patches.mx_kv.engine as e;\
       import vllm_gaudi.xpu.patches.mx_kv.native_decode as n;\
       import vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode as t;\
       import vllm_gaudi.xpu.patches.lmc_mx_boot_gate as g;\
       import vllm_xpu_kernels;\
       from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode;\
       esrc = inspect.getsource(e); nsrc = inspect.getsource(n);\
       gsrc = inspect.getsource(g);\
       psrc = pathlib.Path(e.__file__).parent.parent.parent.joinpath('platform.py').read_text();\
       vsrc = pathlib.Path(g.__file__).parent.parent.parent.joinpath('envs.py').read_text();\
       atxt = pathlib.Path(n.__file__).parent.parent.joinpath('lmcint','vllm_v1_adapter.py').read_text();\
       assert 'from .native_decode import mx_decode_native' in esrc, 'native bridge import (inherited)';\
       assert 'os.environ.get(\"ARC_MXKV_NATIVE\", \"\") == \"1\"' in esrc, 'native route default-off (inherited)';\
       assert e._NATIVE_MINLEN == {'mxfp8': (16384, 16384), 'mxfp4': (12288, 8192), 'mxk8v4': (8192, 8192)}, 'per-lane native thresholds (inherited)';\
       assert 'ARC_MXKV_AB' in esrc, 'r224a AB instrument (inherited)';\
       assert 'ARC_MXKV_NSPLIT' in nsrc, 'NSPLIT pin retained as instrument (inherited)';\
       assert '_max_sk = int(bt_rows.shape[1]) * int(kv_cache.shape[1])' in nsrc, 'capacity-priced walk (r212f, inherited)';\
       assert 'MXKV native call fmt=' in nsrc, 'capture telemetry (inherited)';\
       assert 'flash_attn_varlen_func as _fa_varlen' in esrc, 'P-C2 fa_utils window shim';\
       assert '_HAS_FA_VARLEN' in esrc, 'P-C2 shim posture flag';\
       assert 'requires the native decode lane' in esrc, 'P-C2 sliding-window init refusal';\
       assert 'sliding_window: int | None = None' in nsrc, 'P-C2 window param';\
       assert '_tkey = (fmt, n_rows, _max_sk, int(bt_rows.shape[1]), _w)' in nsrc, 'P-C2 window in the kernel tkey';\
       assert '(\"vllm.distributed.kv_transfer.kv_connector.factory\", \"lmc_mx_boot_gate\")' in psrc, 'P-D factory hook entry';\
       assert 'VLLM_ARC_LMC_MX_GATE: bool = True' in vsrc, 'P-D knob registration';\
       assert 'def _interop_live' in gsrc, 'P-H4 posture probe';\
       assert 'gate=interop-live' in gsrc, 'P-H4 interop-live posture line';\
       assert 'gate=bypassed' in gsrc, 'P-H4 knob bypass line';\
       assert 'ARC_LMC_MX_REFUSED' in gsrc, 'P-D/P-H4 refusal retained';\
       assert 'ARC_LMC_MX_INTEROP = \"packed-row/r249\"' in atxt, 'P-H4 capability marker on the adapter';\
       assert 'hidden_dim_size = _tq_hidden if _packed else num_kv_head * head_size' in atxt, 'P-H2 packed-row hidden-dim branch';\
       assert 'if not bt_rows.is_contiguous():' in nsrc, 'r262 stride fix: the bridge guard';\
       assert 'bt_rows = bt_rows.contiguous()' in nsrc, 'r262 stride fix: the materialization';\
       assert 'MXKV native bt materialized: rows=%d bt_width=%d stride0=%d ' in nsrc, 'r262 stride fix: one log line per (pitch, pool) class';\
       assert 'r262 stride law (2026-11-27): this expand VIEW has stride' in esrc, 'r262 stride fix: the synth call-site contract comment';\
       assert callable(g._install) and callable(n.mx_decode_native) and callable(mx_paged_decode), 'gate + bridge + wheel kernel live';\
       emd5 = hashlib.md5(pathlib.Path(e.__file__).read_bytes()).hexdigest();\
       nmd5 = hashlib.md5(pathlib.Path(n.__file__).read_bytes()).hexdigest();\
       assert emd5 == '$ENGINE_MD5', 'engine.py bytes ' + emd5;\
       assert nmd5 == '$NATIVE_MD5', 'native_decode.py bytes ' + nmd5;\
       vp = pathlib.Path(vllm_xpu_kernels.__file__).parent;\
       md5 = lambda nm: hashlib.md5((vp / nm).read_bytes()).hexdigest();\
       assert md5('libattn_kernels_xe_2.so') == 'c20ca75550f29513a0f73b70ebb1243a', 'build34 libattn pin (page-128 admission)';\
       assert md5('_vllm_fa2_C.abi3.so') == '046e915ebce0eb33a37832377f8c2aed', 'build34 fa2 stub pin';\
       mqa = sorted(vp.rglob('libmqa_logits*.so'));\
       assert mqa and hashlib.md5(mqa[0].read_bytes()).hexdigest() == '190e261edc79b3b497d8d8e801e65757', 'libmqa pin (unrelinked)';\
       print('CERT29_WHEEL_PLUS_DELTA_LIVE asserts OK engine.py', emd5, 'native_decode.py', nmd5)"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert29=P6.47 -- cert28 + the build34 arc7mx wheel (page-128 admission at the gemma full-attention geometry: TORCH_CHECK admission + the head_size==512/block_size==128 dispatch branch riding the _512/_16 sub-tile frame; carries the r231d split-clamp lineage; wheel md5 95451e404b463d9d6efa4bd4a4939b26 byte-pinned, libattn_kernels_xe_2 c20ca755..., _vllm_fa2_C 046e915e..., libmqa_logits 190e261e... all asserted) + the six-file vllm_gaudi delta of the ${FORK_SHA} content commit: P-C2 window plumbing (mx_kv/engine.py + native_decode.py: the fa_utils window shim, the sliding-window init refusal -- windowed impls boot only on the native decode lane -- and the window riding the masked kernel twin as (W-1, 0) per row), P-D boot gate (platform.py factory exec-wrap hook + envs.py VLLM_ARC_LMC_MX_GATE knob), P-H2 packed-row MX interop fix (lmcint/vllm_v1_adapter.py: _mx_fmt classes MX packed-row pools, kv_shape=(L,2,chunk,1,H_kv*slot/2), hidden_dim packed -- the 32 B/token cumulative misaddress behind the r168 garble and the r240b wedge) + the P-H4 posture-aware flip (ARC_LMC_MX_INTEROP marker; gate: interop live -> gate=interop-live allow with telemetry, absent -> ARC_LMC_MX_REFUSED, knob bypass A/B only) + the r262 stride fix (mx_kv/native_decode.py: the bridge materializes a non-contiguous block table before the native call -- the windowed continuation-prefill synth path hands the wheel a stride-0 expand VIEW while the wheel addresses rows at a contiguous pitch, so rows 1..63 read the null block and rows >= 64 read past the table: the gemma DEVICE_LOST at every multi rehit, r255-r260; engine.py carries the call-site contract comment). Evidence: r245a offline oracle VERDICT PASS (page-128 exact/bitwise/boundary/splits + same-wheel regression guards), r245 e2e grid R245_DONE failed=[] (post-wheel gemma annex), r246 probe 11/11 (P-D), r249 INTEROP_PASS (P-H e2e: re-hit A/A byte-equal 6.18->1.89 s wall, r240b kill sequence clean, zero transfer errors), r253 probe 15/15 (P-H4 both postures), r262 e2e A/B R262_SUMMARY a_probe=0 c_suite=0 b_ctrl=2 (fix on: 30/30 generations clean incl. all 13 attempts of the 118-row multi rehit; fix off: the crash recurs). Posture: qwen MX legs ARC_MXKV_NATIVE=1 + ARC_KVDTYPE as certified; gemma sliding layers ride the native lane with the window."; \
      echo "cert29_fork_sha=${FORK_SHA}"; \
      echo "cert29_built=${BUILT}"; \
      echo "cert29_rebuilt=r263 (supersedes the 16effb49 crash image, kept as vllm-arc:v0.30.0-b70-cert29-rc1)"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p647; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p647 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
DOCKERFILE_EOF

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r263] building $IMG FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  --build-arg ENGINE_MD5="$ENGINE_MD5" \
  --build-arg NATIVE_MD5="$NATIVE_MD5" \
  -t "$IMG" \
  "$CTX"
echo "R263_CERT29_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'tail -4 /etc/vllm-arc/IMAGE_MANIFEST'
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'cat /etc/vllm-arc/IMAGE_MANIFEST' | grep -q "^cert29_fork_sha=$FORK_SHA\$" \
  || { echo "R263_MANIFEST_FORK_SHA_MISMATCH"; exit 1; }
echo "R263_IMAGE_OK $IMG cert29_fork_sha=$FORK_SHA"
