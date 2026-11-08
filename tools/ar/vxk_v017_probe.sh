#!/bin/bash
# vxk_v017_probe.sh -- P6.33 Phase-2 feasibility probe (build-only, no GPU
# legs; plan §P6.32 item 8 / §P6.33 item 4). Question: does v0.1.17 source
# build against OUR stack (torch 2.13.0+xpu, oneAPI 2026.0.0) inside the
# carried builder venv? Their 0.1.16+ base assumes omix 2026.1.2 --
# compiler isms are the open risk. Green -> Phase-3 rebase as its own
# patch item; red -> the Phase-1 backport kit is the steady state.
#
# Stages: (1) fetch source (github tag v0.1.17, shallow); (2) torch pin
# report + patch -> 2.13.0+xpu; (3) cmake option surface report (XE2/XE3P/
# MOE/ONEDNN/GDN flags); (4) pip wheel build with the pre-registered flags;
# (5) import + op-registry smoke against the arc6test image (the probe
# wheel is NOT installed anywhere persistent). Everything banks to
# /root/build/p633/v017_probe/.
set -uo pipefail
R=/root/build
P=$R/p633/v017_probe
SRC=$R/vxk-v017-src
mkdir -p "$P"

echo "=== [v017 probe] stage 1: source fetch $(date +%H:%M:%S) ==="
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch v0.1.17 \
    https://github.com/vllm-project/vllm-xpu-kernels.git "$SRC" \
    > "$P/clone.log" 2>&1 || { echo "CLONE_FAILED"; tail -5 "$P/clone.log"; exit 1; }
fi
git -C "$SRC" rev-parse HEAD > "$P/head.sha"
git -C "$SRC" describe --tags >> "$P/head.sha" 2>/dev/null || true
cat "$P/head.sha"

echo "=== stage 2: torch pin report + patch ==="
grep -n -i "torch" "$SRC/pyproject.toml" > "$P/pyproject_torch_before.txt" || true
cat "$P/pyproject_torch_before.txt"
python3 - "$SRC/pyproject.toml" <<'PYEOF'
import re, sys
p = sys.argv[1]
t = open(p).read()
# any torch version spec -> our stack pin (--no-deps build makes this
# belt-and-suspenders, but the file must not lie about the ABI basis)
t2 = re.sub(r'torch[<>=~!]+[0-9][^,;\]"\\]*', 'torch==2.13.0+xpu', t)
if t2 != t:
    open(p, 'w').write(t2)
    print('[ok] pyproject torch pin patched -> 2.13.0+xpu')
else:
    print('[warn] no torch spec matched (inspect pyproject_torch_before.txt)')
PYEOF
grep -n -i "torch" "$SRC/pyproject.toml" > "$P/pyproject_torch_after.txt" || true

echo "=== stage 3: cmake option surface ==="
grep -rn "option(\|XE3P\|XE2\|ENABLE_MOE\|ENABLE_ONEDNN\|ENABLE_GDN\|VLLM_XPU" \
  "$SRC/CMakeLists.txt" > "$P/cmake_options.txt" 2>/dev/null || true
head -40 "$P/cmake_options.txt"

echo "=== stage 4: wheel build (flags: XE2=ON XE3P=OFF MOE=OFF ONEDNN=ON GDN=ON) ==="
# Ship the tree into the builder at a probe path (the certified arc6 bind
# at /src/vllm-xpu-kernels is NOT touched).
B=vxk-arc3-builder
docker exec "$B" mkdir -p /src/vxk-v017
docker cp "$SRC"/. "$B":/src/vxk-v017/ > "$P/cp.log" 2>&1
docker exec "$B" bash -c 'cat > /src/v017_inner.sh' <<'INNER'
set -euo pipefail
echo "=== v017 probe toolchain probe $(date) ==="
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u
export VIRTUAL_ENV=/opt/arcvenv PATH=/opt/arcvenv/bin:$PATH
icpx --version | head -1
/opt/arcvenv/bin/python -c "import torch; print('ENV_READY torch', torch.__version__)"
cd /src/vxk-v017
rm -rf build ./*.egg-info
export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=24 KERNELS_MAX_JOBS=24
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.17+probe
# pre-registered flag surface; unknown-to-cmake -D flags are harmless
CMAKE_ARGS="-DVLLM_XPU_ENABLE_XE2=ON -DVLLM_XPU_ENABLE_XE3P=OFF -DENABLE_MOE=OFF -DENABLE_ONEDNN=ON -DENABLE_GDN=ON"
export CMAKE_ARGS
echo "V017_PROBE_BUILD_START $(date)"
set +e
/opt/arcvenv/bin/pip wheel --no-build-isolation --no-deps . -w /wheels-v017 2>&1
rc=$?
set -e
echo "V017_PROBE_BUILD_EXIT rc=$rc $(date)"
ls -la /wheels-v017/ 2>/dev/null || true
exit "$rc"
INNER
docker exec "$B" bash /src/v017_inner.sh 2>&1 | tee "$P/build.log"
rc=${PIPESTATUS[0]}
echo "=== [v017 probe] build rc=$rc ==="

if [ "$rc" -ne 0 ]; then
  echo "V017_PROBE: RED (build) -- Phase-1 kit is the steady state; see $P/build.log"
  exit 0
fi

echo "=== stage 5: import + op-registry smoke (arc6test container, throwaway pip --target) ==="
# copy the probe wheel out to the host p633 dir (evidence; NEVER installed persistent)
mkdir -p "$P/wheel"
docker cp "$B":/wheels-v017/. "$P/wheel/" 2>/dev/null || true
ls -la "$P/wheel/"
docker run --rm --device /dev/dri --group-add video \
  -v "$P":/work \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  vllm-arc:v0.30.0-b70-arc6test \
  /opt/venv/bin/python - <<'SMOKE' 2>&1 | tee "$P/smoke.log"
import glob, importlib.metadata as md, sys, tempfile, os
whl = glob.glob('/work/wheel/*.whl')
assert whl, 'no probe wheel found'
import zipfile
z = zipfile.ZipFile(whl[0])
names = z.namelist()
has_ext = any('_xpu_C' in n and n.endswith('.so') for n in names)
print('probe wheel:', os.path.basename(whl[0]), 'ext .so present:', has_ext)
# registry smoke WITHOUT installing: read the TORCH_LIBRARY registrations
# is not possible offline; instead pip --target install to a temp dir and
# import with the arc6test torch (import registers ops; no kernel launch)
t = tempfile.mkdtemp()
import subprocess
r = subprocess.run(['/opt/venv/bin/pip', 'install', '--target', t, '--no-deps',
                    whl[0]], capture_output=True, text=True)
print('pip --target rc=', r.returncode, r.stderr[-300:] if r.returncode else '')
sys.path.insert(0, t)
os.environ.setdefault('PATH', '/opt/venv/bin:' + os.environ.get('PATH', ''))
import torch
try:
    import vllm_xpu_kernels._xpu_C as C
    ops = torch.ops._xpu_C
    n = sum(1 for _ in dir(ops))
    print('V017_PROBE_SMOKE: import OK, _xpu_C op-table entries visible')
    for probe in ('fp8_gemm', 'fp4_gemm_w4a16', 'gdn_attn'):
        print(f'  op {probe}:', hasattr(ops, probe))
    print('V017_PROBE: GREEN (compile + import + registry)')
except Exception as e:
    print('V017_PROBE_SMOKE: import FAILED:', repr(e))
    print('V017_PROBE: AMBER (compile green, import red -- compiler-ABI risk confirmed)')
SMOKE
echo "=== [v017 probe] done $(date +%H:%M:%S) ==="
