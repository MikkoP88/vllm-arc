#!/bin/bash
# vxk_v017_inner.sh — Phase-2 feasibility probe (P6.32 item 8): build
# upstream vllm-xpu-kernels v0.1.17 against the arc stack (torch
# 2.13.0+xpu, oneAPI 2026.0.0) inside vxk-arc3-builder. Build-only:
# compile + import + op-registry smoke. Green -> Phase 3 (arc-tree
# rebase onto v0.1.17 as its own patch item); red -> the Phase-1 arc6
# backport kit is the steady state until the upstream lockstep vehicle.
# Flags per plan §P6.32 item 8: XE2=ON XE3P=OFF MOE=OFF ONEDNN=ON GDN=ON.
#
# Runs at /src/vxk-v017 (docker-cp'd source + .deps copied from the arc6
# tree — same rls-v3.13 oneDNN pin 0e2a5bf, so the cache transfers). The
# certified arc6 tree at /src/vllm-xpu-kernels is NEVER touched.
set -uo pipefail  # no -e: every stage captures its own rc

echo "=== VXK_V017 probe $(date) ==="
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

export VIRTUAL_ENV=/opt/arcvenv PATH=/opt/arcvenv/bin:$PATH
icpx --version | head -1
echo "CMPLR_ROOT=${CMPLR_ROOT:-unset}"
/opt/arcvenv/bin/python -c "import torch; print('ENV_READY torch', torch.__version__)"

SRC=/src/vxk-v017
cd "$SRC" || { echo "VXK_V017_FATAL no source at $SRC"; exit 1; }

# torch pin: pyproject build-requires pins 2.15.0+xpu; we build
# --no-build-isolation so the venv torch (2.13.0+xpu) is what compiles
# and links the TUs. Patch the pin for the record; CMake's
# TORCH_SUPPORTED_VERSION_XPU (2.15.0) is a WARNING-level check with an
# upstream TODO — if it hard-fails here the probe verdict is RED for
# exactly the documented reason (compiler/pin isms, plan item 8).
echo "--- pyproject torch pin before:"
grep -n 'torch ==' pyproject.toml | head -3
sed -i 's/torch == 2\.15\.0+xpu/torch == 2.13.0+xpu/' pyproject.toml
echo "--- pyproject torch pin after:"
grep -n 'torch ==' pyproject.toml | head -3
echo "VXK_V017_PIN_PATCHED"

ls -d .deps/onednn-src >/dev/null 2>&1 && \
  echo "ONEDNN_DEPS_PRESENT (copied from arc6 tree — same rls-v3.13 pin)" || \
  echo "ONEDNN_DEPS_MISSING (FetchContent will refetch — rig github reachable)"

export SYCL_HOME="${SYCL_HOME:-$CMPLR_ROOT}"
export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=24 KERNELS_MAX_JOBS=24
# probe flags (setup.py reads identically-named env vars -> -D to cmake)
export VLLM_XPU_ENABLE_XE2=ON
export VLLM_XPU_ENABLE_XE3P=OFF
export MOE_KERNELS_ENABLED=OFF
export GDN_KERNELS_ENABLED=ON
export VLLM_XPU_ENABLE_ONEDNN=ON
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.17+probe

rm -rf build ./*.egg-info /tmp/v017wheel /tmp/v017smoke
mkdir -p /tmp/v017wheel

echo "VXK_V017_BUILD_START $(date)"
set +e
/opt/arcvenv/bin/pip wheel --no-build-isolation --no-deps . -w /tmp/v017wheel 2>&1
rc=$?
echo "VXK_V017_BUILD_EXIT rc=$rc $(date)"

if [ "$rc" -ne 0 ]; then
  echo "VXK_V017_VERDICT RED (wheel build failed — see log above)"
  echo "--- last CMake/build errors, if any:"
  find build -name 'CMakeError.log' -exec tail -40 {} \; 2>/dev/null || true
  tail -25 /tmp/v017pip_err* 2>/dev/null || true
  echo "VXK_V017_DONE $(date)"
  exit "$rc"
fi

ls -la /tmp/v017wheel/
WHL=$(ls /tmp/v017wheel/*.whl)
echo "VXK_V017_WHEEL $WHL"

# smoke: import + op-registry against the arc venv WITHOUT touching the
# certified venv (pip --target + PYTHONPATH)
mkdir -p /tmp/v017smoke
/opt/arcvenv/bin/pip install -q --target /tmp/v017smoke --no-deps "$WHL" || {
  echo "VXK_V017_VERDICT RED (wheel unpack failed)"; exit 3; }

set +e
/opt/arcvenv/bin/python - <<'EOF'
import sys
sys.path.insert(0, "/tmp/v017smoke")
import importlib.metadata as md
import torch
import vllm_xpu_kernels
import vllm_xpu_kernels._xpu_C            # noqa: F401  onednn-backed gemms
import vllm_xpu_kernels._C                # noqa: F401  basic kernels
import vllm_xpu_kernels._vllm_fa2_C       # noqa: F401  FA2
print("VXK_V017_SMOKE version", md.version("vllm-xpu-kernels"))
print("VXK_V017_SMOKE torch", torch.__version__)
ops = sorted(n for n in dir(torch.ops._xpu_C) if not n.startswith("_"))
print("VXK_V017_SMOKE _xpu_C op_count", len(ops))
print("VXK_V017_SMOKE _xpu_C ops", ",".join(ops))
need = ["fp8_gemm_w8a16", "fp4_gemm", "int4_gemm_w4a16", "fp8_gemm"]
have = {n: hasattr(torch.ops._xpu_C, n) for n in need}
print("VXK_V017_SMOKE probe_ops", have)
# GDN TLA kernel lib must ship in the wheel (qwen3.8 hybrid-GDN needs it)
import pathlib, vllm_xpu_kernels as vk
pkg = pathlib.Path(vk.__file__).parent
gdn = sorted(str(p.name) for p in pkg.rglob("gdn_attn_kernels_xe_2*"))
attn = sorted(str(p.name) for p in pkg.rglob("attn_kernels_xe_2*"))
print("VXK_V017_SMOKE gdn_libs", gdn)
print("VXK_V017_SMOKE attn_libs", attn)
missing = [n for n, v in have.items() if not v]
if missing:
    print("VXK_V017_VERDICT AMBER (built+imported but missing ops:", missing, ")")
elif not gdn:
    print("VXK_V017_VERDICT AMBER (no gdn_attn lib in wheel)")
else:
    print("VXK_V017_VERDICT GREEN (build+import+ops+gdn)")
EOF
sm_rc=$?
echo "VXK_V017_SMOKE_EXIT rc=$sm_rc"
if [ "$sm_rc" -ne 0 ]; then echo "VXK_V017_VERDICT RED (smoke import crashed)"; fi
echo "VXK_V017_DONE $(date)"
exit "$sm_rc"
