#!/bin/bash
# stage_arc8test.sh -- P6.34: assemble /root/build/arc8kit + build the
# arc8test image from the finished builder wheel (build8 dual-variant:
# ARC_JM_VAR 14 = V14 fixed-pointer unrolled ping-pong, 5 = V5 control).
# Run ON THE RIG after cut-9 gate PASS. Files skinny_m.py / envs.py /
# arc8test.Dockerfile are shipped from the repo (tools/ar/); the wheel
# comes from the builder container. platform.py is NOT touched
# (unchanged cert22).
set -euo pipefail
R=/root/build
KIT=$R/arc8kit
WHL=vllm_xpu_kernels-0.1.14.1+arc8-cp312-cp312-linux_x86_64.whl

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
docker build -f $KIT/arc8test.Dockerfile --build-arg BUILT="$(date -u +%Y-%m-%dT%H:%M:%SZ)" --build-arg WHEEL_SHA256=$W -t vllm-arc:v0.30.0-b70-arc8test $KIT
docker images vllm-arc:v0.30.0-b70-arc8test
echo "ARC8TEST_STAGED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
