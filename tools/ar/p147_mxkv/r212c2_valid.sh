#!/bin/bash
# r212c2_valid.sh -- arc7 Phase-8 r212c2: validation rerun of the SG-law
# fix (build20): page-64 pools now dispatch the _16 policy (16-row
# sub-tiles, 4 per page via the stock page-table indirection) after the
# vendor _64 policy's (1,4,1) multi-SG layout broke the fork's
# single-SG staging laws (r212c eng256p64 canary 0.07-0.72, all native
# cells). Two ladders:
#   eng256p64  the ENGINE cell (D=256/BLOCK=64 pools, _16 tiles) --
#              canary gate + the engine-pool-true times the r212a
#              thresholds cite
#   reg128p16  certified-cell spot (1024/32768) -- rebuild regression
# Artifacts -> /root/build/p199/out/p204_<TAG>_c<CTX>.json (v2 suffix
# in logs; json name unchanged by the harness)
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212c2

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212C2_NO_WHEEL"; exit 1; }
# build20 must have landed (mtime newer than build19's wheel stamp)
if [ "$(stat -c %Y "$WHEEL")" -lt "$(stat -c %Y "$R/arc7mx_build19.log")" ]; then
  echo "R212C2_WHEEL_STALE (build20 not landed)"; exit 1
fi
grep -q "256/64-via-_16" "$R/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp" \
  || { echo "R212C2_STALE_SRC (no SG-law route)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212C2_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212C2_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212C2_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212C2_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212C2_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212c2.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212c2] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_ladder() {  # tag d hrep block [ctxs...]
  local TAG=$1 DVAL=$2 HREP=$3 BLK=$4; shift 4
  for CTXV in "$@"; do
    docker rm -f $CNAME >/dev/null 2>&1
    echo "=== [r212c2:$TAG] D=$DVAL HREP=$HREP BLOCK=$BLK ctx $CTXV $(date +%H:%M:%S) ==="
    docker run --rm --name $CNAME \
      --device /dev/dri --group-add video \
      -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
      -e P204_CTX=$CTXV -e P204_NCAP=32 -e P204_SPLITS=auto -e P204_REPLAY=50 \
      -e P204_ESHAPE=1 -e P204_D=$DVAL -e P204_HREP=$HREP -e P204_BLOCK=$BLK \
      -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
      -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
      --entrypoint /bin/bash \
      $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
         && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
      > $P/logs/r212c2_${TAG}_c$CTXV.log 2>&1
    rc=$?
    echo "=== [r212c2:$TAG] ctx $CTXV exit rc=$rc $(date +%H:%M:%S) ==="
    if [ -f "$P/out/p204_graphtime.json" ]; then
      mv "$P/out/p204_graphtime.json" "$P/out/p204_${TAG}_c$CTXV.json"
    else
      echo "R212C2_MISSING_ARTIFACT_${TAG}_c$CTXV"
    fi
  done
}

run_ladder eng256p64 256 6 64 1024 2048 4096 8192 32768
run_ladder reg128p16 128 4 16 1024 32768

echo "=== [r212c2] numeric gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
fail = 0
for f in sorted(glob.glob(os.path.join(out, "p204_eng256p64_c*.json"))
                + glob.glob(os.path.join(out, "p204_reg128p16_c*.json"))):
    tag = os.path.basename(f).split("_c")[0][4:]
    ctx = int(f.split("_c")[-1].split(".")[0])
    d = json.load(open(f))
    for fmt, cells in d.items():
        for name, c in cells.items():
            if not isinstance(c, dict):
                continue
            if "capture_error" in c:
                print(f"GATE_FAIL {tag} c{ctx} {fmt} {name}: {c['capture_error']}")
                fail = 1
                continue
            if "canary" in c and c["canary"] > 2e-3:
                print(f"GATE_FAIL {tag} c{ctx} {fmt} {name}: canary {c['canary']}")
                fail = 1
        nat = (cells.get("native_es") or {}).get("dev_us")
        lane = ((cells.get("v5_es") or {}).get("dev_us") if fmt == "mxfp8"
                else (cells.get("v2_es") or {}).get("dev_us"))
        if nat and lane:
            print(f"XOVER {tag} c{ctx} {fmt}: lane {lane:.1f} native {nat:.1f} n/lane {nat/lane:.2f}x")
print("R212C2_GATE_PASS" if not fail else "R212C2_GATE_FAIL")
sys.exit(fail)
PY
grc=$?
echo "R212C2_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
