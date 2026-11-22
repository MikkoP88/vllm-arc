#!/bin/bash
# r206_splitbench_r204.sh -- arc7 Phase-8 r206: re-run the p200h splits
# sweep on the CONVERGED r204 wheel (2fb3b721). The r200h law (B ~ 1.7 ms
# constant split overhead) was fit on the 4.76 ms pre-lane-fix wheel;
# with mode0 now 1.278/1.037/0.971 the (A, B) decomposition must be
# re-measured before any walk-base lever is chosen. Zero-build: sweeps
# the kernel's EXISTING num_splits_kv path. W14 quiesce + auto-restore
# (r200c pattern). Artifact -> /root/build/p199/out/p206_splits_r204.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r206sb

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R206SB_NO_WHEEL"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p200h_splitbench.py" ] || { echo "R206SB_NO_BENCH"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R206SB_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R206SB_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R206SB_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R206SB_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r206sb.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r206sb] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R206SB_WARN_PORT8000_STILL_HELD"
sleep 5

mkdir -p $P/logs $P/out
docker rm -f $CNAME >/dev/null 2>&1
echo "=== [r206sb] splitbench on r204 wheel $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p200h_splitbench.py /out" \
    > $P/logs/r206_splitbench.log 2>&1
rc=$?
echo "=== [r206sb] splitbench rc=$rc $(date +%H:%M:%S) ==="
[ -f $P/out/p200h_splits.json ] && cp $P/out/p200h_splits.json $P/out/p206_splits_r204.json
md5sum "$WHEEL"
echo "R206SB_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
