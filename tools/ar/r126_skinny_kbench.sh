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
# The image ENTRYPOINT is what establishes the oneAPI runtime env (setvars
# + oneCCL vars) before any python; overriding it with a bare interpreter
# died at torch import on the first run (libsycl.so.9 vs stale UR loader:
# urDeviceWaitExp). Mirror the entrypoint's env chain, then exec kbench.
docker run --rm --name arc-kbench \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -v "$P":/work \
  --entrypoint bash \
  $IMG6 -c 'source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force >/dev/null 2>&1; exec /opt/venv/bin/python /work/r126_kbench.py' \
  > "$P/r126_kbench.out" 2>&1
rc=$?
echo "=== [r126] kbench exit rc=$rc $(date +%H:%M:%S) ==="

grep -m1 "GATE-1 correctness" "$P/r126_kbench.out" || true
grep -m1 "GATE-2 skinny_w16pc" "$P/r126_kbench.out" || true
grep -m1 "GATE-3 w4a8 dispatch" "$P/r126_kbench.out" || true
tail -5 "$P/r126_kbench.out"
echo "=== [r126] window done; W14 restoring ==="
exit $rc
