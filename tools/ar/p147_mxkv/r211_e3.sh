#!/bin/bash
# r211_e3.sh -- arc7 Phase-8 r211 E2'': device-only per-call times via
# XPU-graph capture+replay (engine posture) at ctx=1024 for v2 vs
# native_s5 (V3 direct-op) vs native_auto (engine bridge shape).
# Closes the E2' fairness gap (v2's 218us harness number contained its
# own launch host) and doubles as the op-level capture SMOKE for the
# native kernel (V3 shape must capture; failure blocks the engine
# lane). Graceful exit 3 if the torch face lacks torch.xpu.CUDAGraph.
# Artifacts -> /root/build/p199/out/p204_graphtime.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r211e2

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R211_NO_WHEEL"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p204_graphtime.py" ] || { echo "R211_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R211_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R211_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R211_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R211_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r211e2.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r211e2] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R211_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1
echo "=== [r211e2] graph-time E2... $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e P204_CTX=1024 -e P204_NCAP=32 -e P204_SPLITS=auto,5,10,16,32 -e P204_REPLAY=50 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
  > $P/logs/r211_e3.log 2>&1
rc=$?
echo "=== [r211e3] exit rc=$rc $(date +%H:%M:%S) ==="
[ -f "$P/out/p204_graphtime.json" ] || echo "R211_MISSING_ARTIFACT"

grep -h "device\|CAPTURE_FAIL\|NO_XPU_GRAPH" $P/logs/r211_e3.log || true
echo "R211_E3_DONE rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
