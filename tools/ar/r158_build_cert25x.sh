#!/bin/bash
# r158_build_cert25x.sh -- P6.47: build vllm-arc:v0.30.0-b70-cert25x
# (cert24 + the v3 GQA-batched decode line) from the p147 work tree.
# Refuses unless the work tree carries the bake markers (_mx_decode_split_gqa,
# the 4-tuple _MX_TUNE with the mxk8v4 shift flavor, decoder default v3)
# — a stale tree would build yesterday's kernel. FORK_SHA must be passed
# explicitly (the content commit the tree was shipped from).
# A8 (offline battery on the same tree) must have PASSed before this runs.
set -euo pipefail
R=/root/build
W=$R/p147/work
CTX=$R/cert25ctx
FORK_SHA=${1:?usage: r158_build_cert25x.sh <FORK_SHA>}

SP=$W/vllm_gaudi/xpu/patches
grep -q "_mx_decode_split_gqa" "$SP/mx_kv/triton_mx_decode.py" \
  || { echo "R158_STALE_TREE (no v3 kernel)"; exit 1; }
grep -q '"mxk8v4": (64, 8, 2, True)' "$SP/mx_kv/triton_mx_decode.py" \
  || { echo "R158_STALE_TREE (no baked flavor table)"; exit 1; }
grep -q 'ARC_MXKV_DECODER", "v3"' "$SP/mx_kv/triton_mx_decode.py" \
  || { echo "R158_STALE_TREE (decoder default not v3)"; exit 1; }
for f in __init__ engine mx_dequant mx_layout mx_quant triton_mx_decode \
         triton_mx_store mx_dtype_map mx_kv_mode; do
  [ -f "$SP/mx_kv/$f.py" ] || [ -f "$SP/$f.py" ] \
    || { echo "R158_MISSING $f.py"; exit 1; }
done
[ -f "$W/vllm_gaudi/xpu/platform.py" ] || { echo "R158_MISSING platform.py"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/Dockerfile.cert25x" ] \
  || { echo "R158_MISSING Dockerfile.cert25x"; exit 1; }

rm -rf "$CTX"; mkdir -p "$CTX"
cp "$SP/mx_kv/"{__init__,engine,mx_dequant,mx_layout,mx_quant,triton_mx_decode,triton_mx_store}.py "$CTX/"
cp "$SP/mx_dtype_map.py" "$SP/mx_kv_mode.py" "$CTX/"
cp "$W/vllm_gaudi/xpu/platform.py" "$CTX/"
cp "$W/tools/ar/p147_mxkv/Dockerfile.cert25x" "$CTX/Dockerfile"

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r158] building vllm-arc:v0.30.0-b70-cert25x FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert25x \
  "$CTX"
echo "R158_CERT25X_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert25x -c \
  'tail -6 /etc/vllm-arc/IMAGE_MANIFEST'
