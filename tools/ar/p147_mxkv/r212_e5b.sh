#!/bin/bash
# r212_e5b.sh -- arc7 Phase-8 r212 E5b: crossover map in the ENGINE
# VERIFY SHAPE (P204_ESHAPE=1: B x r_tile=4 draft rows, the spec-k3
# batch). E5 mapped v2-vs-native on independent rows and set thresholds
# against v2 -- but the engine's short-ctx lanes for mxfp8/mxk8v4 are
# V5 (mx_decode_rows_dot), and both kernels walk R=B*r rows. This leg
# adds v2_es/v5_es/native_es cells per ctx; the honest crossing for the
# r212a defaults is native_es vs the per-format ENGINE lane
# (v5_es for mxfp8/mxk8v4, v2_es/g-adaptive for mxfp4).
# Artifacts -> /root/build/p199/out/p204_graphtime_es_c<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212e5b

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212_NO_HARNESS"; exit 1; }
grep -q "ESHAPE" "$HARNESS" || { echo "R212_STALE_HARNESS (no ESHAPE mode)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212e5b.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212e5b] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R212_WARN_PORT8000_STILL_HELD"
sleep 5

for CTXV in 1024 2048 4096 8192 32768; do
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212e5b] engine-shape crossover ctx $CTXV $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTXV -e P204_NCAP=32 -e P204_SPLITS=auto -e P204_REPLAY=50 \
    -e P204_ESHAPE=1 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212_e5b_c$CTXV.log 2>&1
  rc=$?
  echo "=== [r212e5b] ctx $CTXV exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime.json" ]; then
    mv "$P/out/p204_graphtime.json" "$P/out/p204_graphtime_es_c$CTXV.json"
  else
    echo "R212_MISSING_ARTIFACT_es_c$CTXV"
  fi
done

grep -h -E "device .*us/call" $P/logs/r212_e5b_c*.log | sort -t_ -k3 -n || true
echo "R212_E5B_DONE rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
