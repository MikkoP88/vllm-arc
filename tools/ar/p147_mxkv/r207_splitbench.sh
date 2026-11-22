#!/bin/bash
# r207_splitbench.sh -- arc7 Phase-8 r207: re-run the p200h splits sweep
# on the r207 wheel (max_num_kv_splits 64, block-16 policy cap 64,
# ARC_MXKV_WGTARGET plan knob; NaN-poisoned canary in the bench). The
# r207 artifact law killed the previous extended sweep (s>=24 never
# launched); this one measures the TRUE plan-splits curve to 64.
# W14 quiesce + auto-restore (r200c pattern).
# Artifact -> /root/build/p199/out/p207_splits_legal.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r207sb

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R207SB_NO_WHEEL"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p200h_splitbench.py" ] || { echo "R207SB_NO_BENCH"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R207SB_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R207SB_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R207SB_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R207SB_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r207sb.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r207sb] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R207SB_WARN_PORT8000_STILL_HELD"
sleep 5

mkdir -p $P/logs $P/out
docker rm -f $CNAME >/dev/null 2>&1
echo "=== [r207sb] splitbench on r207 wheel $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e ARC_MXKV_WGTARGET=64 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p200h_splitbench.py /out" \
    > $P/logs/r207_splitbench.log 2>&1
rc=$?
echo "=== [r207sb] splitbench rc=$rc $(date +%H:%M:%S) ==="
NINV=$(grep -c 'Invalid Problem Size' $P/logs/r207_splitbench.log || true)
echo "invalid_launch_count=$NINV (must be 0)"
[ -f $P/out/p200h_splits.json ] && cp $P/out/p200h_splits.json $P/out/p207_splits_legal.json
md5sum "$WHEEL"
echo "R207SB_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
