#!/bin/bash
# stage_arc9test.sh -- P6.34: assemble /root/build/arc9kit + build the
# arc9test image from the finished builder wheel (build9 dual-variant:
# ARC_JM_VAR 14 = SLM-V24 A-direct + B-in-SLM, 5 = V5 byte-unchanged
# control). Run ON THE RIG after cut-10 gate PASS. Files skinny_m.py /
# envs.py / arc9test.Dockerfile are shipped from the repo (tools/ar/);
# the wheel comes from the builder container. platform.py is NOT
# touched (unchanged cert22).
set -euo pipefail
R=/root/build
KIT=$R/arc9kit
WHL=vllm_xpu_kernels-0.1.14.1+arc9-cp312-cp312-linux_x86_64.whl

mkdir -p $KIT
docker exec vxk-arc3-builder cat /wheels/$WHL > $KIT/$WHL
sha256sum $KIT/$WHL | tee $KIT/wheel.sha256
cp $R/r128a/skinny_m_route.py $KIT/skinny_m.py
# envs.py shipped from repo (ARC_SKINNY_M doc gains "jm")
cp /tmp/envs_ship.py $KIT/envs.py
diff -q $KIT/skinny_m.py $R/r128a/skinny_m_route.py

# wheel name sanity before the build (version tag from setuptools-scm)
docker exec vxk-arc3-builder ls -la /wheels/

W=$(sha256sum $KIT/$WHL | cut -d" " -f1)
docker build -f $KIT/arc9test.Dockerfile --build-arg BUILT="$(date -u +%Y-%m-%dT%H:%M:%SZ)" --build-arg WHEEL_SHA256=$W -t vllm-arc:v0.30.0-b70-arc9test $KIT
docker images vllm-arc:v0.30.0-b70-arc9test
echo "ARC9TEST_STAGED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
