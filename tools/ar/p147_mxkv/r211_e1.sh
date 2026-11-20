#!/bin/bash
# r211_e1.sh -- arc7 Phase-8 r211 E1': 1K timeline decomposition per
# format (mxfp8 vs mxfp4 vs mxk8v4) at ctx=1024, splits 5 (r210 optimum
# posture) and 1 (single-split floor posture), for probe modes 0 (full),
# 3 (floor -- the 0.87 ms pathology cell) and 8 (gemm floor). One docker
# process per mode (ARC_MXKV_VPROBE is read at launch). Decomposition:
# per-call kernel time (dev) vs device-timeline inter-call gap vs host
# enqueue cost, plus the classic bracketed cell the r208/r210 numbers
# came from. Verdict logic in p202_timeline.py docstring (H1' vs H2').
# Prereqs: r210 keeper wheel at /root/build/wheels-arc-vxk + p199 work
# tree with p202_timeline.py shipped. W14 quiesce before, trap restore.
# Artifacts -> /root/build/p199/out/p202_timeline_m<n>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r211e1

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R211_NO_WHEEL"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p202_timeline.py" ] || { echo "R211_NO_HARNESS"; exit 1; }
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
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r211e1.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r211e1] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R211_WARN_PORT8000_STILL_HELD"
sleep 5

for PROBE in 0 3 8; do
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r211e1] timeline probe mode $PROBE $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e ARC_MXKV_VPROBE=$PROBE \
    -e P202_CTX=1024 -e P202_SPLITS=5,1 -e P202_REPS=60 -e P202_WARM=5 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p202_timeline.py /out" \
    > $P/logs/r211_e1_m$PROBE.log 2>&1
  rc=$?
  echo "=== [r211e1] mode $PROBE exit rc=$rc $(date +%H:%M:%S) ==="
  [ -f "$P/out/p202_timeline_m$PROBE.json" ] \
    || echo "R211_MISSING_ARTIFACT_m$PROBE"
done

grep -h "splits=" $P/logs/r211_e1_m*.log | sort -t_ -k3 -n || true
echo "R211_E1_DONE rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
