#!/bin/bash
# r200h_splitbench.sh -- arc7 Phase-8 r200h: num_kv_splits sweep (1/2/4/8/16)
# on the live arc7mx wheel via p200h_splitbench.py. Context: the p198/p199
# telemetry gap vs Triton v2 is measured at the kernel default
# num_kv_splits=1 (16 WGs total at HQ=8 x batch=2 -- a serial 2048-k-tile
# walk at 32K) while v2 launches its own block-split grid; the forced8 face
# already gates the kernel's split path for correctness. Artifacts ->
# /root/build/p199/out/p200h_splits.json. Prereqs: wheel at
# /root/build/wheels-arc-vxk + p199 work tree. W14 quiesce before,
# KEEP_DOWN batch mode like r199/r200c.
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r200h

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R200H_NO_WHEEL"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p200h_splitbench.py" ] || { echo "R200H_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R200H_NO_IMAGE $IMG"; exit 1; }
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R200H_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R200H_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r200h.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r200h] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R200H_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r200h] wheel $(md5sum $WHEEL) $(date +%H:%M:%S) ==="
docker rm -f $CNAME >/dev/null 2>&1
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p200h_splitbench.py /out" \
  > $P/logs/r200h_splitbench.log 2>&1
rc=$?
echo "=== [r200h] bench exit rc=$rc $(date +%H:%M:%S) ==="
grep -E "ctx=" $P/logs/r200h_splitbench.log || true
echo "R200H_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
