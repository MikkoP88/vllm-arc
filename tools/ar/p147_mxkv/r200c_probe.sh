#!/bin/bash
# r200c_probe.sh -- arc7 Phase-8 r200c: V-lane stage attribution via
# the in-kernel probe knob (ARC_MXKV_VPROBE; wheel mx_paged_decode.hpp
# args comment has the mode table). Runs the r199 attribution harness
# once per probe mode (0=full, 1=skip staging, 2=skip gather,
# 3=floor, 4=no barrier, 5=no scales) -- stage costs are the diffs
# against mode 0 on the SAME wheel. Const-fill law: probe floors are
# structural lower bounds, not targets.
# Prereqs: r200c wheel at /root/build/wheels-arc-vxk (probe fields
# compiled in) + p199 work tree. W14 quiesce before, KEEP_DOWN batch
# mode like r199. Artifacts -> /root/build/p199/out/p200c_probe_<n>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r200c

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R200C_NO_WHEEL"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p199_ablation.py" ] || { echo "R200C_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R200C_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R200C_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R200C_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R200C_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r200c.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r200c] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R200C_WARN_PORT8000_STILL_HELD"
sleep 5

for PROBE in 0 1 2 3 4 5; do
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r200c] probe mode $PROBE $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e ARC_MXKV_VPROBE=$PROBE \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p199_ablation.py /out" \
    > $P/logs/r200c_probe_$PROBE.log 2>&1
  rc=$?
  echo "=== [r200c] probe $PROBE exit rc=$rc $(date +%H:%M:%S) ==="
  [ -f $P/out/p199_ablation.json ] && cp $P/out/p199_ablation.json $P/out/p200c_probe_$PROBE.json
done

grep -hE "ctx=32768" $P/logs/r200c_probe_*.log | sort -t_ -k3 -n | head -60 || true
echo "R200C_DONE rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
