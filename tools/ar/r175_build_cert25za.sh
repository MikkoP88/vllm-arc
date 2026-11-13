#!/bin/bash
# r175_build_cert25za.sh -- P6.47: build vllm-arc:v0.30.0-b70-cert25za
# (cert25z + the A14 v6 deep-split kernel line + the BN=32 sweep-winner
# tune) from the shipped ctx tree. Refuses unless the shipped
# triton_mx_decode.py carries the v6 content markers AND the A14 sweep
# evidence PASSed (r173). FORK_SHA must be passed explicitly (the
# content commit the tree was shipped from; cert-release-sequence law:
# manifest fork_sha = content commit).
set -euo pipefail
R=/root/build
CTX=$R/cert25zactx
FORK_SHA=${1:?usage: r175_build_cert25za.sh <FORK_SHA>}

SP=$CTX
grep -q "_mx_decode_combine" "$SP/triton_mx_decode.py" \
  || { echo "R175_STALE_TREE (no v6 combine kernel)"; exit 1; }
grep -q "_V5_SPLIT_DEFAULT = 256" "$SP/triton_mx_decode.py" \
  || { echo "R175_STALE_TREE (no v6 split cap)"; exit 1; }
grep -q "ARC_MXKV_V5_SPLITS" "$SP/triton_mx_decode.py" \
  || { echo "R175_STALE_TREE (no sweep env knob)"; exit 1; }
grep -q "_V5_TUNE = {" "$SP/triton_mx_decode.py" \
  || { echo "R175_STALE_TREE (no v5 tune table)"; exit 1; }
grep -A3 "^_V5_TUNE = {" "$SP/triton_mx_decode.py" | \
  grep -q '"mxfp8": (32, 4, 2, False)' \
  || { echo "R175_STALE_TREE (BN=64 pre-flip tune)"; exit 1; }
grep -q "_V5_DEFAULT" "$SP/engine.py" \
  || { echo "R175_STALE_TREE (no v5 default flip)"; exit 1; }
grep -q "mx_decode_rows_dot" "$SP/__init__.py" \
  || { echo "R175_STALE_TREE (no package export)"; exit 1; }
[ -f "$CTX/Dockerfile" ] \
  || { echo "R175_MISSING Dockerfile (cert25za)"; exit 1; }
grep -q "A14_V6_SWEEP_PASS" "$R/p147/outA14/legA14_summary.txt" \
  || { echo "R175_NO_A14_EVIDENCE (run r173 first)"; exit 1; }

# the decode path must stay cast-free (native q)
docker run --rm --entrypoint /bin/bash -v "$CTX":/work:ro \
  vllm-arc:v0.30.0-b70-cert25z -c \
  '! grep -q "query.float()" /work/engine.py' \
  || { echo "R175_STALE_TREE (engine still upcasts query)"; exit 1; }

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r175] building vllm-arc:v0.30.0-b70-cert25za FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert25za \
  "$CTX"
echo "R175_CERT25ZA_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert25za -c \
  'tail -5 /etc/vllm-arc/IMAGE_MANIFEST'
