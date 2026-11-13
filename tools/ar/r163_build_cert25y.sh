#!/bin/bash
# r163_build_cert25y.sh -- P6.47: build vllm-arc:v0.30.0-b70-cert25y
# (cert25x + the A9/A10 decode line: s_eff floor, native-q, batched
# MTP-verify decode) from the p147 work tree. Refuses unless the work
# tree carries the markers (s_eff floor, _split_verify_decode, cast-free
# decode path) and A9 PASSed (the offline evidence leg for this content).
# FORK_SHA must be passed explicitly (the content commit the tree was
# shipped from).
set -euo pipefail
R=/root/build
W=$R/p147/work
CTX=$R/cert25yctx
FORK_SHA=${1:?usage: r163_build_cert25y.sh <FORK_SHA>}

SP=$W/vllm_gaudi/xpu/patches
grep -q "s_eff = tl.minimum(NUM_SPLITS, tl.maximum(1, n_tiles))" \
  "$SP/mx_kv/triton_mx_decode.py" \
  || { echo "R163_STALE_TREE (no v3 s_eff floor)"; exit 1; }
grep -q "_split_verify_decode" "$SP/mx_kv/engine.py" \
  || { echo "R163_STALE_TREE (no batched verify decode)"; exit 1; }
grep -q "verify-batch" "$W/tools/ar/p147_mxkv/p157_v3_shortctx.py" \
  || { echo "R163_STALE_TREE (probe lacks the verify-batch face)"; exit 1; }
grep -q "A9_PROBE_PASS" "$R/p147/outA9/legA9_summary.txt" \
  || { echo "R163_NO_A9_EVIDENCE (run r160 first)"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/Dockerfile.cert25y" ] \
  || { echo "R163_MISSING Dockerfile.cert25y"; exit 1; }

# the decode path must be cast-free (native q)
docker run --rm --entrypoint /bin/bash -v "$W":/work:ro \
  vllm-arc:v0.30.0-b70-cert25x -c \
  '! grep -q "query.float()" /work/vllm_gaudi/xpu/patches/mx_kv/engine.py' \
  || { echo "R163_STALE_TREE (engine still upcasts query)"; exit 1; }

rm -rf "$CTX"; mkdir -p "$CTX"
cp "$SP/mx_kv/engine.py" "$SP/mx_kv/triton_mx_decode.py" "$CTX/"
cp "$W/tools/ar/p147_mxkv/Dockerfile.cert25y" "$CTX/Dockerfile"

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r163] building vllm-arc:v0.30.0-b70-cert25y FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert25y \
  "$CTX"
echo "R163_CERT25Y_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert25y -c \
  'tail -5 /etc/vllm-arc/IMAGE_MANIFEST'
