#!/bin/bash
# r169_build_cert25z.sh -- P6.47: build vllm-arc:v0.30.0-b70-cert25z
# (cert25y + the A11 v4/v5 verify-kernel line + the v5 default flip)
# from the p147 work tree. Refuses unless the work tree carries the
# content markers AND the A12/A13 probe evidence PASSed (r166/r167).
# FORK_SHA must be passed explicitly (the content commit the tree was
# shipped from; cert-release-sequence law: manifest fork_sha = content
# commit).
set -euo pipefail
R=/root/build
W=$R/p147/work
CTX=$R/cert25zctx
FORK_SHA=${1:?usage: r169_build_cert25z.sh <FORK_SHA>}

SP=$W/vllm_gaudi/xpu/patches
grep -q "_mx_decode_split_rows" "$SP/mx_kv/triton_mx_decode.py" \
  || { echo "R169_STALE_TREE (no v4 kernel)"; exit 1; }
grep -q "_mx_decode_split_dot" "$SP/mx_kv/triton_mx_decode.py" \
  || { echo "R169_STALE_TREE (no v5 kernel)"; exit 1; }
grep -q "_V5_DEFAULT" "$SP/mx_kv/engine.py" \
  || { echo "R169_STALE_TREE (no v5 default flip)"; exit 1; }
grep -q "ARC_MXKV_VERIFY_DECODER" "$SP/mx_kv/engine.py" \
  || { echo "R169_STALE_TREE (no verify-decoder override)"; exit 1; }
grep -q "mx_decode_rows_dot" "$SP/mx_kv/__init__.py" \
  || { echo "R169_STALE_TREE (no package export)"; exit 1; }
grep -q "A12_PROBE_PASS" "$R/p147/outA12/legA12_summary.txt" \
  || { echo "R169_NO_A12_EVIDENCE (run r166 first)"; exit 1; }
grep -q "A13_PROBE_PASS" "$R/p147/outA13/legA13_summary.txt" \
  || { echo "R169_NO_A13_EVIDENCE (run r167 first)"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/Dockerfile.cert25z" ] \
  || { echo "R169_MISSING Dockerfile.cert25z"; exit 1; }

# the decode path must stay cast-free (native q)
docker run --rm --entrypoint /bin/bash -v "$W":/work:ro \
  vllm-arc:v0.30.0-b70-cert25y -c \
  '! grep -q "query.float()" /work/vllm_gaudi/xpu/patches/mx_kv/engine.py' \
  || { echo "R169_STALE_TREE (engine still upcasts query)"; exit 1; }

rm -rf "$CTX"; mkdir -p "$CTX"
cp "$SP/mx_kv/engine.py" "$SP/mx_kv/triton_mx_decode.py" \
   "$SP/mx_kv/__init__.py" "$CTX/"
cp "$W/tools/ar/p147_mxkv/Dockerfile.cert25z" "$CTX/Dockerfile"

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r169] building vllm-arc:v0.30.0-b70-cert25z FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert25z \
  "$CTX"
echo "R169_CERT25Z_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert25z -c \
  'tail -5 /etc/vllm-arc/IMAGE_MANIFEST'
