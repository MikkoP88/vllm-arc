#!/bin/bash
# r212d_diag.sh -- arc7 Phase-8 r212d-diag: offline reproduction of
# the ENGINE call shape after r212b/r212c falsified the split and
# walk-extent theories (max_seqlen_k 9 vs 262400 = byte-identical
# e2e ITLs; the walk is not max_seqlen_k-bounded). The e2e native
# cost runs ~40x the p204 per-call number at the same nominal shape
# (1 seq, 32K ctx, 4 verify rows: ~53 ms/layer e2e vs 1.27 ms/call
# offline). Remaining engine-vs-harness deltas, one knob at a time
# at c32768 (mxfp4, eng geometry D=256/HREP=6/BLOCK=64, eshape,
# splits=auto):
#   base    knobs off (sanity: reproduce the banked 1266us class)
#   pool    POOLBLKS=11281 (engine pool = 721994 tokens)
#   btw0    BTWIDTH=4100, zero pad
#   btwS    BTWIDTH=4100, stale random-valid pad (engine reality)
#   both    btwS + pool
#   eng     both + MAXSK=cap (the full r212c bridge posture)
# The canary doubles as a walk-bound probe: stale-pad corruption
# would mean the kernel reads beyond-fill bt entries.
# Artifacts -> /root/build/p199/out/p204_diag_<TAG>_c32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212d

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212D_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212D_NO_HARNESS"; exit 1; }
grep -q "r212d-diag engine-shape knobs" "$HARNESS" \
  || { echo "R212D_STALE_HARNESS (knobs not shipped)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212D_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212D_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212D_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212D_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212d.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212d] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # tag extra-env...
  local TAG=$1; shift
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212d:$TAG] $* $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=32768 -e P204_NCAP=32 -e P204_SPLITS=auto -e P204_REPLAY=50 \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4 "$@" \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212d_${TAG}.log 2>&1
  local rc=$?
  echo "=== [r212d:$TAG] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime.json" ]; then
    mv "$P/out/p204_graphtime.json" "$P/out/p204_diag_${TAG}_c32768.json"
  else
    echo "R212D_MISSING_ARTIFACT_$TAG"
  fi
}

run_cell base
run_cell pool  -e P204_POOLBLKS=11281
run_cell btw0  -e P204_BTWIDTH=4100
run_cell btwS  -e P204_BTWIDTH=4100 -e P204_BTPAD=stale
run_cell both  -e P204_BTWIDTH=4100 -e P204_BTPAD=stale -e P204_POOLBLKS=11281
run_cell eng   -e P204_BTWIDTH=4100 -e P204_BTPAD=stale -e P204_POOLBLKS=11281 -e P204_MAXSK=cap

echo "=== [r212d] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, glob, os
out = sys.argv[1]
fail = 0
base = None
for tag in ("base", "pool", "btw0", "btwS", "both", "eng"):
    f = os.path.join(out, f"p204_diag_{tag}_c32768.json")
    if not os.path.exists(f):
        print(f"DIAG {tag}: MISSING"); fail = 1; continue
    d = json.load(open(f))
    cells = d.get("mxfp4", {})
    nat = cells.get("native_es") or {}
    v2 = (cells.get("v2_es") or {}).get("dev_us")
    if "capture_error" in nat:
        print(f"DIAG {tag}: CAPTURE_ERROR {nat['capture_error']}"); fail = 1; continue
    can = nat.get("canary")
    us = nat.get("dev_us")
    if tag == "base":
        base = us
    rel = f" ({us/base:.2f}x base)" if (base and tag != "base") else ""
    print(f"DIAG {tag}: native {us:.1f} us  v2 {v2:.1f} us  canary {can:.2e}{rel}")
    if can is None or can > 2e-3:
        print(f"  CANARY_DIRTY {tag}: {can}"); fail = 1
print("R212D_GATE_PASS" if not fail else "R212D_GATE_FAIL")
sys.exit(fail)
PY
grc=$?
echo "R212D_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
