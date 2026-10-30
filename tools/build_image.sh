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
# §6 contingency wheel (p13): int64-fixed vllm-xpu-kernels, built in
# omix 0.2.0-devel (oneAPI 2026.0.0 = the venv runtime generation) against
# torch 2.13.0+xpu; device-gated CLEAN (results/2026-10-27-p13-vxk-wheel/).
# Explicit filename: the build dir also holds the arc1/arc2 ABI rejects,
# which must never enter an image.
cp /root/build/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc3-cp312-cp312-linux_x86_64.whl "$CTX/wheels/"
# P6.2: internal lmcache wheel (upstream 0.5.6 sdist + the 4-patch arc kit
# [e-adapter-vllm-056-graft, e01-xpu-platform-ops, e02e03, e04] + upstream
# SYCL XPU build, generation-matched per external_patches/lmcache).
# Explicit filename: rejects/rebuilds must never enter an image silently.
cp /root/build/wheels-arc-lmc/lmcache-0.5.6+arc1-cp312-cp312-manylinux_2_35_x86_64.whl "$CTX/wheels/"

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
