#!/bin/bash
# build_image.sh — stage the rig build context and build the vllm-arc image.
# Runs ON THE RIG (host), invoked over ssh from the repo side:
#   repo -> git archive <branch> tarball + tools/ + Dockerfile + pins
#   rig  -> /root/build/arc_build/{Dockerfile,wheels/,pins.md5,tools/,src/}
#           docker build --target vllm-arc -t vllm-arc:v0.30.0-b70
#
# Usage: build_image.sh <fork-archive.tar> <fork-sha> [tag]
set -euo pipefail

archive=${1:?usage: build_image.sh <fork-archive.tar> <fork-sha> [tag]}
fork_sha=${2:?usage: build_image.sh <fork-archive.tar> <fork-sha> [tag]}
tag=${3:-vllm-arc:v0.30.0-b70}

CTX=/root/build/arc_build
rm -rf "$CTX"
mkdir -p "$CTX/wheels" "$CTX/tools" "$CTX/src"

# Ground-truth wheels: the exact files inside the validated radiance image
# (md5s pinned in external_patches/wheels.pins.md5; wheel_guard enforces).
cp /root/build/radiance_wheels/vllm-0.30.0+xpu-cp38-abi3-manylinux_2_34_x86_64.whl "$CTX/wheels/"
cp /root/build/radiance_wheels/triton-3.7.2+xpu-py3-none-any.whl "$CTX/wheels/"
# §6 contingency wheel (p13): int64-fixed vllm-xpu-kernels, built INSIDE the
# arc runtime image with the staged DPC++ 7.2.0 toolchain (arc2 — arc1, built
# in the omix devel image, linked libsycl.so.8 and ImportError-d against the
# venv so.9 runtime). Explicit filename: the build dir also holds the arc1
# reject, which must never enter an image.
cp /root/build/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc2-cp312-cp312-linux_x86_64.whl "$CTX/wheels/"

tar -xf "$archive" -C "$CTX/src"
cp "$CTX/src/Dockerfile" "$CTX/Dockerfile"
cp "$CTX/src/external_patches/wheels.pins.md5" "$CTX/pins.md5"
cp "$CTX/src/tools/wheel_guard.sh" "$CTX/tools/wheel_guard.sh"

echo "BUILD context staged: $(du -sh "$CTX" | cut -f1), fork_sha=$fork_sha"
docker build --target vllm-arc -t "$tag" \
  --build-arg FORK_SHA="$fork_sha" \
  --build-arg BUILD_HOST="$(hostname -s)" \
  "$CTX"
echo "BUILD done: $tag (fork $fork_sha)"
