#!/bin/bash
# r212j_tileunit.sh -- arc7 Phase-8 r212j: validate the plan TILE-UNIT
# fix. Root cause source-proven (r212i/i2/i3 wrong-values law): the MX
# decode kernel walks 16-row tiles at EVERY page (mx_paged_decode_xe2.cpp
# routes page-64 pools through the _16 policy, r212c SG law), but the
# python plan builder mirrored the STOCK unit (_kv_tile_from_block_size
# = 16/32/64 by page) -- at BLOCK=64 plans covered 1/4 of the KV ->
# deterministic wrong values; BLOCK=16 coincided, which is why r210
# passed. Harness fix: kt=16 (P204_KVT). This leg re-runs the forced
# cells at the fixed unit: canaries must go 0 at every ctx/split, and
# the TRUE s16-class cost curve vs auto re-prices the headroom (the
# r212i "1.5-2.1x" was largely the 4x under-walk -- invalid bound).
# Cells: ctx {1024,8192,32768}, SPLITS=auto,5,16,64, MAXSK=cap,
# PLANLEN=cap, mxfp4, engine geometry, graph 32x50.
# Artifacts -> /root/build/p199/out/p212j_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212j

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212J_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212J_NO_HARNESS"; exit 1; }
grep -q "P204_KVT" "$HARNESS" || { echo "R212J_STALE_HARNESS (no tile-unit fix)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212J_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212J_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212J_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212J_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212j.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212j] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_ctx() {  # ctx
  local CTX=$1
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212j:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto,5,16,64 -e P204_MAXSK=cap -e P204_PLANLEN=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212j_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212j:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime.json" ]; then
    mv "$P/out/p204_graphtime.json" "$P/out/p212j_$CTX.json"
  else
    echo "R212J_MISSING_ARTIFACT_$CTX"
  fi
}

run_ctx 1024
run_ctx 8192
run_ctx 32768

echo "=== [r212j] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
print("r212j TRUE split-cost curve (tile-unit 16) -- us/call, mxfp4, MAXSK=cap, graph 32x50")
print(f" {'ctx':>6} | " + " | ".join(f"{s:>9}" for s in ("auto", "s5", "s16", "s64")) + " |  nat_es")
best_ratio = 0.0
for ctx in (1024, 8192, 32768):
    f = os.path.join(out, f"p212j_{ctx}.json")
    if not os.path.exists(f):
        print(f" {ctx:>6} | MISSING"); fail = 1; continue
    d = json.load(open(f)).get("mxfp4", {})
    row = []
    vals = {}
    for cell in ("native_auto", "native_5", "native_16", "native_64"):
        c = d.get(cell) or {}
        if "capture_error" in c:
            row.append("       CAP"); fail = 1
        elif c.get("canary") is None or c["canary"] > 2e-3:
            row.append(f"{c['dev_us']:7.1f}!"); fail = 1
        else:
            row.append(f"{c['dev_us']:9.1f}")
            vals[cell] = c["dev_us"]
    es = (d.get("native_es") or {}).get("dev_us", -1)
    print(f" {ctx:>6} | " + " | ".join(row) + f" | {es:7.1f}")
    if "native_auto" in vals and vals.get("native_16"):
        r = vals["native_auto"] / min(vals["native_16"], vals.get("native_64", 1e18))
        best_ratio = max(best_ratio, r)
        print(f"          auto/best-forced = {r:.2f}x")
if not fail:
    print(f"TILE_UNIT_FIX_VALIDATED (all forced canaries 0); best auto/forced ratio {best_ratio:.2f}x")
print("R212J_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212J_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
