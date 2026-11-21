#!/bin/bash
# r212n_probedec.sh -- arc7 Phase-8 r212n: fresh FULL decomposition
# INCLUDING PROBE 9 on the converged r212m3 wheel (the telemetry law:
# never fulls alone; and the r212m3 lane table predates nothing but
# probe 9 was never run on THIS binary). Same wheel md5 e50f1c3a --
# same-binary re-runs reproduce lane layouts (run noise only, +-5 pct;
# layout dice is a cross-wheel phenomenon), so the banked r212m3
# table is the baseline and this leg adds:
#   - decode residual (m0 - m9): the assembly/scale/convert cost that
#     REMAINS after direct-bf16 (r212m measured 736.8/772.7 us of it
#     on the PRE-m3 wheel -- how much is left?)
#   - walk+softmax+P pool (m9 - c8): the tile-width lever's budget
#     (r212k estimated ~147 us from cross-wheel arithmetic)
#   - reproduction check of the m0/m2/m3/m6/m8 lanes vs r212m3_val
# Artifacts -> /root/build/p199/out/p212n_m{0,2,3,6,8,9}_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212n

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=e50f1c3aa38729931c0d6ccc9c57a370
[ -f "$WHEEL" ] || { echo "R212N_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R212N_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"; exit 1; }
grep -q "r212m3 E4M3 REVERT" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212N_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212N_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212N_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212N_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212N_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212N_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212n.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212n] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # tag ctx probe dbf
  local TAG=$1
  local CTX=$2
  local PROBE=$3
  local DBF=$4
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212n:$TAG:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4,mxfp8 \
    -e ARC_MXKV_VPROBE=$PROBE -e ARC_MXKV_DIRECTBF16=$DBF \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212n_${TAG}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212n:$TAG:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m$PROBE.json" ]; then
    mv "$P/out/p204_graphtime_m$PROBE.json" "$P/out/p212n_${TAG}_$CTX.json"
  else
    echo "R212N_MISSING_ARTIFACT_${TAG}_$CTX"
  fi
}

# full decomposition incl. probe 9 on the converged wheel (fast=on)
for PROBE in 0 2 3 6 8 9; do
  run_cell m$PROBE 32768 $PROBE 1
done

echo "=== [r212n] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(tag, ctx, fmt):
    f = os.path.join(out, f"p212n_{tag}_{ctx}.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}
# banked r212m3 same-wheel baselines (commits cc7488f3 / e1cf2b02)
BASE_FULL = {"mxfp4": 749.3, "mxfp8": 1319.7}
BASE_LANE = {"mxfp4": {"K": 178.2, "Vg": 196.9, "Vs": 17.1, "fl": 246.9},
             "mxfp8": {"K": 397.1, "Vg": 338.8, "Vs": 359.4, "fl": 171.5}}
# cross-wheel reference only (r212m pre-flight, different binary)
M9_PRE = {"mxfp4": 291.0, "mxfp8": 589.0}
BW_FLOOR = 298.0
print("r212n full reproduction vs banked r212m3 (same wheel, +-6% band)")
for fmt in ("mxfp4", "mxfp8"):
    c = cell("m0", 32768, fmt)
    if not c or c.get("dev_us") is None:
        print(f" {fmt:>6} | MISSING m0"); fail = 1; continue
    b = BASE_FULL[fmt]
    d = 100.0 * (c["dev_us"] - b) / b
    print(f" {fmt:>6} | new {c['dev_us']:8.1f} | base {b:8.1f} | {d:+5.1f}% | canary {c.get('canary')}")
    if abs(d) > 6.0:
        print(f"          REPRO_FAIL {fmt} (same-binary drift {d:+.1f}%)"); fail = 1
    if c.get("canary") is None or c["canary"] > 2e-3:
        print(f"          CANARY_FAIL {fmt} m0"); fail = 1
print("r212n lane table at 32K (vs banked r212m3 slices) + probe-9 pricing")
print(f" {'fmt':>6} | {'K':>13} | {'Vg':>13} | {'Vs':>12} | {'floor':>13} | {'m9':>7} | {'dec resid':>9} | {'m9-c8 pool':>10}")
for fmt in ("mxfp4", "mxfp8"):
    cs = {}
    ok = True
    for probe in ("0", "2", "3", "6", "8", "9"):
        c = cell(f"m{probe}", 32768, fmt)
        if not c or c.get("dev_us") is None:
            print(f" {fmt}: MISSING m{probe}"); fail = 1; ok = False; break
        cs[probe] = c["dev_us"]
    if not ok:
        continue
    full, c2, c3, c6, c8, m9 = (cs[p] for p in ("0", "2", "3", "6", "8", "9"))
    b = BASE_LANE[fmt]
    k, vg, vs = full - c6, full - c2, c2 - c3
    print(f" {fmt:>6} | {k:6.1f} ({100.0*(k-b['K'])/b['K']:+5.1f}%) | {vg:6.1f} ({100.0*(vg-b['Vg'])/b['Vg']:+5.1f}%)"
          f" | {vs:5.1f} ({vs-b['Vs']:+6.1f}) | {c8:6.1f} ({100.0*(c8-b['fl'])/b['fl']:+5.1f}%)"
          f" | {m9:7.1f} | {full-m9:9.1f} | {m9-c8:10.1f}")
    if abs(100.0 * (k - b["K"]) / b["K"]) > 15.0 or abs(100.0 * (vg - b["Vg"]) / b["Vg"]) > 15.0:
        print(f"          LANE_DRIFT {fmt} (>15% -- investigate before the tile decision)")
print(f"reference: pre-m3 m9 was {M9_PRE['mxfp4']}/{M9_PRE['mxfp8']} us (cross-wheel);"
      f" BW-floor est {BW_FLOOR:.0f} us")
print("note: probe-cell canaries are vacuous (harness does not compute them"
      " in probe mode) -- correctness gates on m0 only")
print("R212N_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212N_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
