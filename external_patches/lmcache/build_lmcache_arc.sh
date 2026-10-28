#!/bin/bash
# build_lmcache_arc.sh — build lmcache-0.5.5+arcN (SYCL XPU variant) on the
# rig, from the pinned upstream sdist + the e-patches in this directory.
# Driven from the RIG HOST:  bash external_patches/lmcache/build_lmcache_arc.sh
#
# Toolchain law (vxk arc3 lineage, results/2026-10-27-p13-vxk-wheel): the
# COMPILER GENERATION must equal the arc venv RUNTIME GENERATION — build in
# intel/omix:0.2.0-devel-ubuntu24.04 (oneAPI 2026.0.0) against a venv pinned
# to torch 2.13.0+xpu. The persistent vxk-arc3-builder container is REUSED
# when present (warm /opt/arcvenv + toolchain); created otherwise.
#
# Build recipe = upstream's own .github/scripts/build_xpu_wheel.sh (sdist
# ships it): BUILD_WITH_SYCL=1 setup.py bdist_wheel + auditwheel repair
# excluding torch and the oneAPI/SYCL runtime libs (wheel binds to the host
# image at runtime — same policy as the upstream XPU artifacts and our vxk
# wheel). Patches applied before build per the fork law (files, provenance
# in README.md).
set -eo pipefail

KIT=/root/build/lmc-arc-kit          # this directory pushed to the rig
SDIST=/root/build/p40/lmcache-0.5.5.tar.gz
SRC=/root/build/lmc-arc-src
WHEELS=/root/build/wheels-arc-lmc
INNER=/root/build/lmc_arc1_inner.sh
BUILDER=vxk-arc3-builder
BASE=intel/omix:0.2.0-devel-ubuntu24.04
ARC=${ARC_TAG:-arc1}

[ -f "$KIT/patches/e01-xpu-torch-ops.patch" ] || { echo "FATAL: kit missing at $KIT"; exit 1; }
[ -f "$SDIST" ] || { echo "FATAL: sdist missing at $SDIST"; exit 1; }

# --- 1. fresh source tree + patches --------------------------------------
rm -rf "$SRC"; mkdir -p "$SRC"
tar xzf "$SDIST" -C "$SRC"
cd "$SRC/lmcache-0.5.5"
# sdist/wheel file-identity guard (patch applies only on pristine base)
for f in lmcache/v1/cache_engine.py lmcache/v1/platform/torch_ops.py \
         lmcache/integration/vllm/vllm_v1_adapter.py; do
  want=$(md5sum "/root/build/p40/wheel_tree/$f" | cut -d' ' -f1)
  got=$(md5sum "$f" | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "FATAL: sdist $f drift vs wheel tree"; exit 1; }
done
echo "BASE_IDENTITY_OK (3 patched files pristine vs upstream wheel)"
for p in "$KIT"/patches/*.patch; do
  patch -p1 --forward < "$p"
done
# version tag (PEP 440 local): 0.5.5 -> 0.5.5+arcN in the sdist's static
# vcs file (setuptools_scm is inert inside the sdist).
sed -i "s/__version__ = version = '0.5.5'/__version__ = version = '0.5.5+$ARC'/" lmcache/_version.py
grep -q "0.5.5+$ARC" lmcache/_version.py || { echo "FATAL: version tag failed"; exit 1; }

# --- 2. builder container (reuse warm vxk builder when present) ----------
mkdir -p "$WHEELS"
if ! docker inspect "$BUILDER" >/dev/null 2>&1; then
  echo "(creating fresh builder — venv bootstrap ~5 min)"
  docker run -d --name "$BUILDER" -v /root/build/vxk-arc-src:/src/vllm-xpu-kernels \
    -v /root/build/wheels-arc-vxk:/wheels-vxk "$BASE" sleep infinity >/dev/null
fi
docker start "$BUILDER" >/dev/null 2>&1 || true
docker exec "$BUILDER" mkdir -p /src/lmcache /wheels
docker cp "$SRC/lmcache-0.5.5/." "$BUILDER":/src/lmcache/
docker cp "$INNER" "$BUILDER":/inner-lmc.sh

# --- 3. build -------------------------------------------------------------
docker exec "$BUILDER" bash /inner-lmc.sh
docker cp "$BUILDER":/src/lmcache/dist/. "$WHEELS/" 2>/dev/null || true

# --- 4. retag 0.5.5 -> 0.5.5+arcN -----------------------------------------
# setuptools_scm owns the version (pyproject: dynamic + version_file +
# local_scheme="no-local-version" — local tags are STRIPPED by design), so
# the arc tag is applied post-build, deterministically: rename wheel,
# rename dist-info, bump METADATA Version, fix RECORD head. (vxk +arc3
# precedent; pip performs no RECORD verification on install.)
cd "$WHEELS"
OLD=lmcache-0.5.5-cp312-cp312-manylinux_2_35_x86_64.whl
NEW=lmcache-0.5.5+$ARC-cp312-cp312-manylinux_2_35_x86_64.whl
[ -f "$NEW" ] || { [ -f "$OLD" ] || { echo "FATAL: built wheel missing"; exit 1; }
  python3 - "$OLD" "$NEW" "$ARC" <<'PYEOF'
import os, sys, zipfile
old, new, arc = sys.argv[1], sys.argv[2], sys.argv[3]
zin = zipfile.ZipFile(old)
with zipfile.ZipFile(new + ".tmp", "w", zipfile.ZIP_DEFLATED) as zout:
    for it in zin.infolist():
        data = zin.read(it.filename)
        if it.filename == "lmcache-0.5.5.dist-info/METADATA":
            data = data.replace(b"Version: 0.5.5\n", f"Version: 0.5.5+{arc}\n".encode())
            it.filename = f"lmcache-0.5.5+{arc}.dist-info/METADATA"
        elif it.filename.startswith("lmcache-0.5.5.dist-info/"):
            it.filename = f"lmcache-0.5.5+{arc}.dist-info/" + it.filename.split("/", 1)[1]
            if it.filename.endswith("RECORD"):
                data = data.replace(b"lmcache-0.5.5.dist-info/", f"lmcache-0.5.5+{arc}.dist-info/".encode())
        zout.writestr(it, data)
os.replace(new + ".tmp", new)
os.remove(old)
print("RETAG_OK", new)
PYEOF
}
python3 -c "
import zipfile, glob
w = glob.glob('lmcache-0.5.5+$ARC-*.whl')[0]
z = zipfile.ZipFile(w)
meta = z.read('lmcache-0.5.5+$ARC.dist-info/METADATA').decode()
assert 'Version: 0.5.5+$ARC' in meta, meta[:200]
assert any('xpu_ops' in n for n in z.namelist())
print('RETAG_VERIFIED', w)
"
echo "=== wheels in $WHEELS ==="
ls -la "$WHEELS/"
md5sum "$WHEELS"/*.whl
echo "BUILD_LMC_ARC_DONE"
