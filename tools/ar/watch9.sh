#!/bin/bash
# watch8.sh -- P6.34 build8 watcher (rig-side). Polls the vxk-arc3-builder
# build8.log until the VXK_ARC9_BUILD_EXIT end marker or the arc9 wheel
# lands, then prints the tail + wheel list. Exit 0 done / 1 failed /
# 2 timeout (~6.7 h cap).
W=vllm_xpu_kernels-0.1.14.1+arc9-cp312-cp312-linux_x86_64.whl
for i in $(seq 1 200); do
  if docker exec vxk-arc3-builder grep -q "VXK_ARC9_BUILD_EXIT" \
      /tmp/build8.log 2>/dev/null; then
    echo "BUILD8_EXITED iter=$i $(date)"
    docker exec vxk-arc3-builder tail -30 /tmp/build8.log
    docker exec vxk-arc3-builder ls -la /wheels/ | tail -4
    if docker exec vxk-arc3-builder test -f /wheels/$W; then
      exit 0
    fi
    exit 1
  fi
  if docker exec vxk-arc3-builder test -f /wheels/$W 2>/dev/null; then
    echo "BUILD8_WHEEL_PRESENT iter=$i $(date)"
    docker exec vxk-arc3-builder ls -la /wheels/$W
    exit 0
  fi
  sleep 120
done
echo "BUILD8_TIMEOUT $(date)"
docker exec vxk-arc3-builder tail -15 /tmp/build8.log
exit 2
