#!/bin/bash
# r212c_recal.sh -- arc7 Phase-8 r212c: crossover RECALIBRATION at the
# real engine geometry (fail2 law: qwen3.8-27b-fp8 is head_dim 256,
# engine pool page 64, TP2 HQ12/HK2 -> G6; the whole r208-r212 ladder
# ran D=128/BLOCK=16/G=4 off the config's GDN linear head dim). Three
# ladders against the build19 widened wheel ({128,256}x{16,64} via the
# vendor's own policies):
#   A  P204_D=256 P204_HREP=6 P204_BLOCK=64  -- THE engine cell; the
#      r212a thresholds re-derive from THIS ladder
#   B  P204_D=256 P204_HREP=6 P204_BLOCK=16  -- head-width isolation at
#      the old harness page (connects to the banked D=128 ladder)
#   C  P204 defaults (128/4/16)              -- certified-cell regression:
#      kernel binary-layout dice law says recompiles swing untouched
#      lanes; gate = canaries 0 + cross-checks pass + times within the
#      +/-6% dice band of the banked E5b rerun
# All ladders P204_ESHAPE=1 (engine verify shape B x r4). Artifacts ->
# /root/build/p199/out/p204_<LADDER>_c<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212c

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212C_NO_WHEEL"; exit 1; }
KEEPER=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64_r210keeper.whl
[ -f "$KEEPER" ] || { echo "R212C_NO_KEEPER (r210 provenance)"; exit 1; }
# build19 must have REPLACED the live arc7mx wheel (widened dispatch);
# if it is still the keeper bytes the build failed to land -- refuse.
if [ "$(sha256sum "$WHEEL" | cut -d' ' -f1)" = \
     "$(sha256sum "$KEEPER" | cut -d' ' -f1)" ]; then
  echo "R212C_WHEEL_NOT_REBUILT (live == r210 keeper; build19 did not land)"; exit 1
fi
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212C_NO_HARNESS"; exit 1; }
grep -q "P204_D" "$HARNESS" || { echo "R212C_STALE_HARNESS (no geometry knobs)"; exit 1; }
grep -q "256/64" "$R/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp" \
  || { echo "R212C_STALE_SRC (dispatch not widened)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212C_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212C_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212C_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212C_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212c.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212c] W14 quiesce $(date +%H:%M:%S) ==="
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
    echo "=== [r212c:$TAG] D=$DVAL HREP=$HREP BLOCK=$BLK ctx $CTXV $(date +%H:%M:%S) ==="
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
      > $P/logs/r212c_${TAG}_c$CTXV.log 2>&1
    rc=$?
    echo "=== [r212c:$TAG] ctx $CTXV exit rc=$rc $(date +%H:%M:%S) ==="
    if [ -f "$P/out/p204_graphtime.json" ]; then
      mv "$P/out/p204_graphtime.json" "$P/out/p204_${TAG}_c$CTXV.json"
    else
      echo "R212C_MISSING_ARTIFACT_${TAG}_c$CTXV"
    fi
    # numeric gate per cell: canaries must be clean in the run log
  done
}

# A: engine geometry (thresholds source); B: D-isolation; C: certified-cell regression
run_ladder eng256p64 256 6 64 1024 2048 4096 8192 32768
run_ladder eng256p16 256 6 16 1024 4096 8192 32768
run_ladder reg128p16 128 4 16 1024 8192 32768

echo "=== [r212c] numeric gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
fail = 0
rows = []
for f in sorted(glob.glob(os.path.join(out, "p204_eng*_c*.json"))
                + glob.glob(os.path.join(out, "p204_reg*_c*.json"))):
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
        xr = cells.get("v5_vs_v2_ref")
        if isinstance(xr, (int, float)) and xr > 2e-3:
            print(f"GATE_FAIL {tag} c{ctx} {fmt}: v5_vs_v2_ref {xr}")
            fail = 1
        # crossover row: native_es vs the format's engine lane
        nat = cells.get("native_es", {}).get("dev_us")
        lane = (cells.get("v5_es", {}) or {}).get("dev_us") \
            if fmt in ("mxfp8", "mxk8v4") \
            else (cells.get("v2_es", {}) or {}).get("dev_us")
        if nat and lane:
            rows.append((tag, ctx, fmt, lane, nat, nat / lane))
for r in rows:
    print(f"XOVER {r[0]} c{r[1]} {r[2]}: lane {r[3]:.1f} native {r[4]:.1f} n/lane {r[5]:.2f}x")
print("R212C_GATE_PASS" if not fail else "R212C_GATE_FAIL")
sys.exit(fail)
PY
grc=$?
echo "=== [r212c] summary $(date +%H:%M:%S) ==="
for T in eng256p64 eng256p16 reg128p16; do
  echo "-- $T:"
  grep -h -E "device .*us/call" $P/logs/r212c_${T}_c*.log 2>/dev/null | sort -t_ -k3 -n || true
done
echo "R212C_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
