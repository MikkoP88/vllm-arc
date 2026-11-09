#!/bin/bash
# r126_skinny_kbench.sh -- P6.33 arc6 skinny-M kernel gate window.
# Boots the arc6test image standalone (W14 stopped: the kbench owns the
# GPUs for its legs), runs r126_kbench.py (correctness + identical-input
# A/B vs stock + W4A8 dispatch answer), banks the raw to
# /root/build/p633/, restores W14 (trap). The cert22 lane decision reads
# the GATE-1/2/3 lines from r126_kbench.out.
set -uo pipefail
R=/root/build
P=$R/p633
IMG6=vllm-arc:v0.30.0-b70-arc6test

mkdir -p "$P"
[ -f "$P/r126_kbench.py" ] || { echo "FATAL: $P/r126_kbench.py missing"; exit 1; }
docker image inspect $IMG6 >/dev/null 2>&1 || {
  echo "FATAL: $IMG6 not built (build it before this window)"; exit 1; }

w14_start() {
  docker rm -f arc-kbench >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p633r126.log 2>&1 < /dev/null &
}
trap w14_start EXIT

WH=""
for i in $(seq 1 40); do
  WH=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health 2>/dev/null || true)
  [ "$WH" = "200" ] && break
  sleep 60
done
echo "W14 health before kbench: $WH ($(date +%H:%M:%S))"

docker stop radiance-test >/dev/null 2>&1 || true
sleep 10
docker rm -f arc-kbench >/dev/null 2>&1 || true

echo "=== [r126] kbench boot $(date +%H:%M:%S) ==="
# Env law (serve_arc_b70.sh exec block precedent): python in this image
# needs LD_LIBRARY_PATH=venv/lib:torch/lib. Two earlier attempts died at
# torch import (libsycl.so.9 undefined urDeviceWaitExp): bare interpreter
# with NO env, and the oneAPI setvars chain (setvars' own libsycl/UR pair
# does not satisfy torch's bundled sycl). The cert22 build assert proves
# this exact chain imports torch + the skinny ops.
docker run --rm --name arc-kbench \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$P":/work \
  --entrypoint /opt/venv/bin/python \
  $IMG6 /work/r126_kbench.py \
  > "$P/r126_kbench.out" 2>&1
rc=$?
echo "=== [r126] kbench exit rc=$rc $(date +%H:%M:%S) ==="

grep -m1 "GATE-1 correctness" "$P/r126_kbench.out" || true
grep -m1 "GATE-2 skinny_w16pc" "$P/r126_kbench.out" || true
grep -m1 "GATE-3 w4a8 dispatch" "$P/r126_kbench.out" || true
tail -5 "$P/r126_kbench.out"
echo "=== [r126] window done; W14 restoring ==="
exit $rc
