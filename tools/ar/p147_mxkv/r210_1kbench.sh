#!/bin/bash
# r210_1kbench.sh -- arc7 Phase-8 r210 calibration: forced-splits sweep
# at ctx=1024 on the live wheel. The r209 chain showed 40->4 splits at
# 1K telemetry helped 12% while the probe floor still sits ~0.18-0.69;
# this sweep pins the 1K optimum (and whether s=1 single-launch wins)
# before the r210 auto-heuristic lands (knee back to 4x, workload bound
# kv_tiles/12 vs /16 decided by this table). W14 quiesce + auto-restore
# (r207sb pattern).
# Artifact -> /root/build/p199/out/p210_1k_splits.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r210sb

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R210SB_NO_WHEEL"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p200h_splitbench.py" ] || { echo "R210SB_NO_BENCH"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R210SB_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R210SB_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R210SB_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R210SB_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r210sb.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r210sb] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R210SB_WARN_PORT8000_STILL_HELD"
sleep 5

mkdir -p $P/logs $P/out
docker rm -f $CNAME >/dev/null 2>&1
echo "=== [r210sb] 1K splitbench on live wheel $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e ARC_MXKV_WGTARGET=64 \
    -e P200H_CTX=1024 \
    -e P200H_SPLITS=1,2,3,4,5,6,8,10,12,16 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p200h_splitbench.py /out" \
    > $P/logs/r210_1kbench.log 2>&1
rc=$?
echo "=== [r210sb] splitbench rc=$rc $(date +%H:%M:%S) ==="
NINV=$(grep -c 'Invalid Problem Size' $P/logs/r210_1kbench.log || true)
echo "invalid_launch_count=$NINV (must be 0)"
[ -f $P/out/p200h_splits.json ] && cp $P/out/p200h_splits.json $P/out/p210_1k_splits.json
md5sum "$WHEEL"
echo "R210SB_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
