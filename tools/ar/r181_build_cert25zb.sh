#!/bin/bash
# r181_build_cert25zb.sh -- P6.47: build vllm-arc:v0.30.0-b70-cert25zb
# (cert25za + the K2/K3 packed-fp16 dequant lanes default-on + the HEAD
# re-overlay of engine.py/__init__.py, docs-sweep phrasing only) from
# the shipped ctx tree. Refuses unless the shipped triton_mx_decode.py
# carries the K2/K3 content markers AND default-on reads AND the cycle
# evidence PASSed (r178/r179/r180 banks). FORK_SHA must be passed
# explicitly (cert-release-sequence law: manifest fork_sha = content
# commit).
set -euo pipefail
R=/root/build
CTX=$R/cert25zbctx
FORK_SHA=${1:?usage: r181_build_cert25zb.sh <FORK_SHA>}

SP=$CTX
grep -q "_f16_scale" "$SP/triton_mx_decode.py" \
  || { echo "R181_STALE_TREE (no K2 helper)"; exit 1; }
grep -q "_e2m1_byte" "$SP/triton_mx_decode.py" \
  || { echo "R181_STALE_TREE (no K3 helper)"; exit 1; }
grep -q 'ARC_MXKV_K2", "1") != "0"' "$SP/triton_mx_decode.py" \
  || { echo "R181_STALE_TREE (K2 not default-on)"; exit 1; }
grep -q 'ARC_MXKV_K3", "1") != "0"' "$SP/triton_mx_decode.py" \
  || { echo "R181_STALE_TREE (K3 not default-on)"; exit 1; }
grep -q "P6.47, 2026-11-11" "$SP/engine.py" \
  || { echo "R181_STALE_TREE (engine.py not the HEAD docs-sweep overlay)"; exit 1; }
grep -q "mx_decode_rows_dot" "$SP/__init__.py" \
  || { echo "R181_STALE_TREE (no package export)"; exit 1; }
[ -f "$CTX/Dockerfile" ] \
  || { echo "R181_MISSING Dockerfile (cert25zb)"; exit 1; }
grep -q "K2_AB_PASS" "$R/p647/k2cycleA17/k2ab.txt" \
  || { echo "R181_NO_K2_EVIDENCE (r178 bank)"; exit 1; }
grep -q "K3_AB_PASS" "$R/p647/k3cycleA18/k3ab.txt" \
  || { echo "R181_NO_K3_EVIDENCE (r179 bank)"; exit 1; }
grep -q "K2_TUNE_PASS" "$R/p647/k2tuneA19/k2tune.txt" \
  || { echo "R181_NO_TUNE_EVIDENCE (r180 bank)"; exit 1; }

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r181] building vllm-arc:v0.30.0-b70-cert25zb FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert25zb \
  "$CTX"
echo "R181_CERT25ZB_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert25zb -c \
  'tail -4 /etc/vllm-arc/IMAGE_MANIFEST'
