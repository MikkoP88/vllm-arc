#!/bin/bash
# r208_chain.sh -- arc7 Phase-8 r208: synth2 faces x3 + r198 offline
# faces + probes 0-8 on the r207/r208 wheel (fork reduce, splits-64).
# Same posture as r204_chain.sh; final probes step runs WITHOUT
# KEEP_DOWN so its EXIT trap restores W14.
set -uo pipefail
R=/root/build
W=$R/p199/work
P=$R/p199
IMG=vllm-arc:v0.30.0-b70-cert25zc
WHL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
BN=$(basename $WHL)
mkdir -p $R/wheels-arc-vxk/fix
cp -f $WHL $R/wheels-arc-vxk/fix/vllm_xpu_kernels-0.1.14.1+arc7mx-r208.whl
md5sum $WHL $R/wheels-arc-vxk/fix/vllm_xpu_kernels-0.1.14.1+arc7mx-r208.whl
for FMT in mxfp8 mxfp4 mxk8v4; do
  docker rm -f arc-r208s >/dev/null 2>&1
  echo "=== [r208] synth2 $FMT $(date +%H:%M:%S) ==="
  docker run --rm --name arc-r208s \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P200C_SYNTH=1 -e P200C_FMT=$FMT \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$BN \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p200c_vlane_debug.py /out/p201_synth2_$FMT.json"
  echo "=== [r208] synth2 $FMT rc=$? $(date +%H:%M:%S) ==="
done
echo "=== [r208] faces $(date +%H:%M:%S) ==="
R198_KEEP_DOWN=1 bash /root/build/p198/r198_native_offline.sh
echo "=== [r208] faces rc=$? $(date +%H:%M:%S) ==="
echo "=== [r208] probes 0-8 $(date +%H:%M:%S) ==="
bash /root/build/p199/work/tools/ar/p147_mxkv/r200c_probe.sh
echo "=== [r208] probes rc=$? $(date +%H:%M:%S) ==="
echo "R208_CHAIN_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
