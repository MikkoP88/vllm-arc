#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# build_oneccl_arc.sh — P6.24: same-version oneCCL 2022.0.0 rebuild on the
# rig (mode stock = phase-0 fidelity build; mode arc = fp8 patch applied).
# Driven from the RIG HOST:  bash tools/build_oneccl_arc.sh stock|arc
#
# Pairing law (torch 2.13.0+xpu pins oneccl==2022.0.0 EXACTLY; same-version
# rebuild ONLY — identical tag/C-API/sonames/sycl-rt generation; never
# cross-version swap under torch). Builder image = DPC++ 2026.0.0 = the
# cert9 venv's sycl-rt generation (toolchain law, vxk arc3 lineage).
#
# Overlay target on the certified image: /opt/venv/lib/libccl.so.1(.0) —
# the 210 MB fat library where the SYCL kernels live. Thin libccl.so.2
# (C-API shim) is byte-identical and untouched. Provenance + evidence:
# external_patches/oneccl/README.md, results/2026-11-05-p624-fp8-ar/.
set -eo pipefail

MODE="${1:-}"
[ "$MODE" = "stock" ] || [ "$MODE" = "arc" ] || { echo "usage: $0 stock|arc"; exit 1; }

KIT=/root/build/p624-kit            # this kit pushed to the rig
SRC=/root/build/p624/oneccl-src     # pinned clone (persisted across modes)
STAGE=/root/build/p624/stage-$MODE
BUILDER=p624-builder
BASE=intel/deep-learning-essentials:2026.0.0-devel-ubuntu22.04
SUPER_PIN=1318c3aaf8a67536ff538331c418c8cea3114f33   # tag 2022.0.0
LIBCCL_PIN=4ceafd15c03ce46f11eeaf91781a92afebd3cecf  # deps/libccl gitlink

[ -f "$KIT/patches/arc-fp8-allreduce.patch" ] || { echo "FATAL: kit missing at $KIT (push external_patches/oneccl + tools/ first)"; exit 1; }

# --- 1. pinned source (idempotent; hash-verified) -------------------------
if [ ! -d "$SRC/.git" ]; then
  echo "(fetching pinned oneCCL 2022.0.0 + submodule — once)"
  mkdir -p "$(dirname "$SRC")"
  git clone -b 2022.0.0 --depth 1 --recurse-submodules --shallow-submodules \
      https://github.com/oneapi-src/oneCCL "$SRC" 2>&1 | tail -2
fi
cd "$SRC"
got_super=$(git rev-parse HEAD)
got_lib=$(git -C deps/libccl rev-parse HEAD)
[ "$got_super" = "$SUPER_PIN" ] || { echo "FATAL: superproject $got_super != pin"; exit 1; }
[ "$got_lib" = "$LIBCCL_PIN" ] || { echo "FATAL: deps/libccl $got_lib != pin"; exit 1; }
echo "PIN_OK super=$got_super libccl=$got_lib"

# --- 2. patch state per mode (marker-proven, E-law) -----------------------
if [ "$MODE" = "arc" ]; then
  if ! grep -q allreduce_large_fp8_ring deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl.cpp; then
    patch -p1 --forward < "$KIT/patches/arc-fp8-allreduce.patch"
  fi
  grep -q "allreduce_large_fp8_ring" deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl.cpp \
    || { echo "FATAL: dispatch-hook marker missing"; exit 1; }
  grep -q "CCL_SYCL_ALLREDUCE_FP8" deps/libccl/src/common/env/env.cpp \
    || { echo "FATAL: env registration marker missing"; exit 1; }
  grep -q "fp8ar_float_to_e4m3" deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl_ring_fp8.hpp \
    || { echo "FATAL: fp8 header marker missing"; exit 1; }
  echo "PATCH_MARKERS_OK (mode arc)"
else
  [ -z "$(git status --porcelain deps/libccl)" ] \
    || { echo "FATAL: mode stock but deps/libccl is dirty — re-clone $SRC"; exit 1; }
  echo "PRISTINE_OK (mode stock)"
fi

# --- 3. builder container (reused when present) ---------------------------
mkdir -p "$STAGE"
if ! docker inspect "$BUILDER" >/dev/null 2>&1; then
  echo "(creating fresh builder)"
  docker run -d --name "$BUILDER" -v "$SRC":/src -v "$STAGE":/stage "$BASE" sleep infinity >/dev/null
fi
docker start "$BUILDER" >/dev/null 2>&1 || true
# mounts refresh in case $STAGE was created after container birth
docker restart "$BUILDER" >/dev/null

# --- 4. build inside the builder ------------------------------------------
docker exec "$BUILDER" bash -eux -c '
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
  cd /src
  cmake -S /src -B /src/build -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
        -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF 2>&1 | tail -5
  cmake --build /src/build --parallel $(nproc) 2>&1 | tail -5
'

# --- 5. inventory + verify + stage ----------------------------------------
docker exec "$BUILDER" bash -eu -c '
  cd /src/build
  echo "=== built ccl libraries ==="
  find . -name "libccl*" -type f | sort
  FAT=$(find . -name "libccl.so.1.0" -o -name "libccl.so.1" | head -1)
  THIN=$(find . -name "libccl.so.2.0" -o -name "libccl.so.2" | head -1)
  [ -n "$FAT" ] || { echo "FATAL: fat libccl.so.1 not produced (layout mismatch — inventory above is the evidence)"; exit 1; }
  echo "=== SONAMEs ==="
  readelf -d "$FAT" | grep SONAME
  if [ -n "$THIN" ]; then readelf -d "$THIN" | grep SONAME; fi
  echo "=== C-API symbols in thin (sample) ==="
  if [ -n "$THIN" ]; then nm -D "$THIN" | grep -c oneccl; fi
  echo "=== fp8 markers in fat ==="
  strings "$FAT" | grep -c "CCL_SYCL_ALLREDUCE_FP8" || true
  nm -DC "$FAT" 2>/dev/null | c++filt | grep -c fp8_ring || true
'

# stage every libccl artifact the build produced + manifest
docker exec "$BUILDER" bash -eu -c "
  cd /src/build
  find . -name 'libccl*' -type f -exec cp -v {} /stage/ \; 2>/dev/null || true
"
{
  echo "mode=$MODE date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "superproject=$(git rev-parse HEAD) tag=2022.0.0"
  echo "libccl=$(git -C deps/libccl rev-parse HEAD)"
  echo "builder=$BASE (DPC++ 2026.0.0)"
  [ "$MODE" = "arc" ] && echo "patch=arc-fp8-allreduce.patch (sha256 $(sha256sum "$KIT/patches/arc-fp8-allreduce.patch" | cut -d" " -f1))"
  echo "--- staged artifacts ---"
  ( cd "$STAGE" && sha256sum * 2>/dev/null )
} > "$STAGE/BUILD-MANIFEST.txt"
cat "$STAGE/BUILD-MANIFEST.txt"
echo "BUILD_ONECCL_ARC_DONE mode=$MODE stage=$STAGE"
echo "next: overlay = cp $STAGE/libccl.so.1.0 /opt/venv/lib/ (backup first); keep .so.2 untouched"
