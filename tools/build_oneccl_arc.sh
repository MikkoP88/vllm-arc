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
[ "$MODE" = "stock" ] || [ "$MODE" = "arc" ] || [ "$MODE" = "probe" ] \
  || { echo "usage: $0 stock|arc|probe"; exit 1; }

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
if [ "$MODE" = "arc" ] || [ "$MODE" = "probe" ]; then
  if ! grep -q allreduce_large_fp8_ring deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl.cpp; then
    patch -p1 --forward < "$KIT/patches/arc-fp8-allreduce.patch"
  fi
  grep -q "allreduce_large_fp8_ring" deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl.cpp \
    || { echo "FATAL: dispatch-hook marker missing"; exit 1; }
  grep -q "CCL_SYCL_ALLREDUCE_FP8" deps/libccl/src/common/env/env.cpp \
    || { echo "FATAL: env registration marker missing"; exit 1; }
  grep -q "fp8ar_float_to_e4m3" deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl_ring_fp8.hpp \
    || { echo "FATAL: fp8 header marker missing"; exit 1; }
  if [ "$MODE" = "probe" ]; then
    # r100 bottleneck scan: v4 + one-shot IPC wire-rate probe (additive,
    # env-gated CCL_SYCL_ALLREDUCE_FP8_PROBE=1; never enters a cert image).
    # State machine: the probe patch touches 4 files (3 tracked env files +
    # the v4-added hpp); arc mode reverse-applies it whole.
    if ! grep -q fp8ar_run_probe deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl_ring_fp8.hpp; then
      patch -p1 --forward < "$KIT/patches/arc-fp8-probe.patch"
    fi
    grep -q fp8ar_run_probe deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl_ring_fp8.hpp \
      || { echo "FATAL: probe marker missing"; exit 1; }
    grep -q "CCL_SYCL_ALLREDUCE_FP8_PROBE" deps/libccl/src/common/env/env.cpp \
      || { echo "FATAL: probe env registration marker missing"; exit 1; }
    echo "PATCH_MARKERS_OK (mode probe: v4 + wire-rate probe)"
  else
    # back to plain v4 if the probe patch is present (oneCCL_SRC shared)
    if grep -q fp8ar_run_probe deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl_ring_fp8.hpp; then
      patch -p1 -R --forward < "$KIT/patches/arc-fp8-probe.patch"
    fi
    ! grep -q fp8ar_run_probe deps/libccl/src/coll/algorithms/allreduce/sycl/allreduce_large_sycl_ring_fp8.hpp \
      || { echo "FATAL: probe marker still present in arc mode"; exit 1; }
    ! grep -q "CCL_SYCL_ALLREDUCE_FP8_PROBE" deps/libccl/src/common/env/env.cpp \
      || { echo "FATAL: probe env marker still present in arc mode"; exit 1; }
    echo "PATCH_MARKERS_OK (mode arc)"
  fi
else
  # tracked-tree pristine check (the build generates an ignored
  # include/oneapi/ccl/config.h inside the submodule — untracked by design)
  git -C deps/libccl diff --quiet HEAD -- \
    || { echo "FATAL: mode stock but deps/libccl tracked tree is dirty — re-clone $SRC"; exit 1; }
  echo "PRISTINE_OK (mode stock, tracked tree)"
fi

# --- 3. builder container (reused when present) ---------------------------
# Mount the STABLE p624 root (not $STAGE itself): container mounts are frozen
# at creation, and the first-run builder was created with stage-stock:/stage —
# the arc run then staged into stage-stock while stage-arc stayed empty and
# the r97 legs silently raced the stock lib (banked 2026-11-05). A stable
# root mount makes /p624/stage-<mode> always correct.
mkdir -p "$STAGE"
if ! docker inspect "$BUILDER" >/dev/null 2>&1; then
  echo "(creating fresh builder)"
  docker run -d --name "$BUILDER" -v "$(dirname "$SRC")":/p624 "$BASE" sleep infinity >/dev/null
fi
docker start "$BUILDER" >/dev/null 2>&1 || true

# --- 4. build inside the builder ------------------------------------------
# DLE image cmake is 3.22 and has no pip; oneCCL 2022.0.0 requires >=3.24 —
# bootstrap the pinned Kitware binary (~3.31: cmake 4 removes old policies).
docker exec "$BUILDER" bash -eux -c '
  [ -x /opt/cmake/bin/cmake ] || {
    curl -sL https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
      | tar xz -C /opt
    mv /opt/cmake-3.31.6-linux-x86_64 /opt/cmake
  }
  /opt/cmake/bin/cmake --version | head -1
'
# In-container paths ride the STABLE root mount: /p624/oneccl-src (the
# first /p624-mount revision still said /src — the stale builder's mount —
# and cmake "source directory does not exist" hid behind `| tail` because
# the inner bash had no pipefail; banked 2026-11-05).
docker exec "$BUILDER" bash -euxo pipefail -c '
  source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
  export PATH=/opt/cmake/bin:$PATH
  rm -rf /p624/oneccl-src/build
  cmake -S /p624/oneccl-src -B /p624/oneccl-src/build \
        -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
        -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=OFF \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF 2>&1 | tail -8
  cmake --build /p624/oneccl-src/build --parallel $(nproc) 2>&1 | tail -8
'

# --- 5. inventory + verify + stage ----------------------------------------
# Canonical artifact paths (the first staging pass let ./libccl_cpu's 6 MB
# CPU-fallback libccl.so.1.0 overwrite the 210 MB SYCL fat via blind
# find|cp — banked 2026-11-05; the SYCL fat is ./libccl/src):
#   ./libccl/src/libccl.so.1.0   fat SYCL library (SONAME libccl.so.1)
#   ./src/libccl.so.2.0          thin C-API shim (SONAME libccl.so.2)
docker exec "$BUILDER" bash -euo pipefail -c '
  cd /p624/oneccl-src/build
  FAT=./libccl/src/libccl.so.1.0
  THIN=./src/libccl.so.2.0
  [ -f "$FAT" ] || { echo "FATAL: SYCL fat $FAT not produced"; exit 1; }
  [ -f "$THIN" ] || { echo "FATAL: thin shim $THIN not produced"; exit 1; }
  echo "=== sizes (cert9 reference: fat ~210 MB, thin ~142 KB) ==="
  ls -la "$FAT" "$THIN"
  echo "=== SONAMEs ==="
  readelf -d "$FAT" | grep SONAME
  readelf -d "$THIN" | grep SONAME
  echo "=== C-API symbols in thin ==="
  nm -D "$THIN" | grep -c oneccl
  echo "=== fp8 markers in fat (strings / demangled) ==="
  strings "$FAT" | grep -c "CCL_SYCL_ALLREDUCE_FP8" || true
  nm -DC "$FAT" 2>/dev/null | c++filt | grep -c fp8_ring || true
'

# stage canonical artifacts + manifest (via the stable root mount)
docker exec "$BUILDER" bash -euo pipefail -c "
  cd /p624/oneccl-src/build
  D=/p624/stage-$MODE
  rm -rf \$D; mkdir -p \$D
  cp -v ./libccl/src/libccl.so.1.0 \$D/
  cp -v ./libccl/src/libccl.a \$D/ 2>/dev/null || true
  cp -v ./src/libccl.so.2.0 \$D/
  cp -v ./plugins/legacy_cpu/libccl_legacy_cpu.so \$D/ 2>/dev/null || true
  cp -v ./libccl_cpu/src/ext/openmp/libccl_openmp.so.0 \$D/ 2>/dev/null || true
"
{
  echo "mode=$MODE date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "superproject=$(git rev-parse HEAD) tag=2022.0.0"
  echo "libccl=$(git -C deps/libccl rev-parse HEAD)"
  echo "builder=$BASE (DPC++ 2026.0.0)"
  [ "$MODE" = "arc" ] && echo "patch=arc-fp8-allreduce.patch (sha256 $(sha256sum "$KIT/patches/arc-fp8-allreduce.patch" | cut -d" " -f1))"
  [ "$MODE" = "probe" ] && echo "patch=arc-fp8-allreduce.patch + arc-fp8-probe.patch (probe sha256 $(sha256sum "$KIT/patches/arc-fp8-probe.patch" | cut -d" " -f1))"
  echo "--- staged artifacts ---"
  ( cd "$STAGE" && sha256sum * 2>/dev/null )
} > "$STAGE/BUILD-MANIFEST.txt"
cat "$STAGE/BUILD-MANIFEST.txt"
echo "BUILD_ONECCL_ARC_DONE mode=$MODE stage=$STAGE"
echo "next: overlay = cp $STAGE/libccl.so.1.0 /opt/venv/lib/ (backup first); keep .so.2 untouched"
