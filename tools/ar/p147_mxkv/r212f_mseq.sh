#!/bin/bash
# r212f_mseq.sh -- arc7 Phase-8 r212f: the NO-OP-FALSIFICATION audit
# follow-up. git-history audit (2026-11-21): the r212c "capacity-
# priced walk fix" only changed the bridge's TELEMETRY line — the
# kernel call still passed max_seqlen_k=int(max_seq_len) (the
# DUMMY-diluted metadata value), so r212c_min's byte-identical ITLs
# disproved NOTHING, and the r212d cells (ctx/cap) never tested the
# value the engine actually freezes (9/13/18/5). This ladder closes
# the hole at the kernel level, offline, c32768 eshape (the shape
# where e2e ran 8.6x over triton):
#   ctx  control (32768 — prices sane splits; expect ~1272 us)
#   m9   the observed DUMMY-diluted freeze (9 -> kv_tiles=1 ->
#        splits=1: the split-starvation prediction is a blow-up to
#        the ~tens-of-ms class; the e2e gap is then FULLY explained)
#   cap  262400 (the r212f bridge posture; expect ~ctx)
# Verdict rule: m9/ctx >= 5x => SPLIT_STARVATION_CONFIRMED (kernel
# arg causal); m9 ~ ctx => genuinely split-insensitive (the 40x
# really is in the invocation; r212e's in-situ trace then leads).
# Artifacts -> /root/build/p199/out/p204_mseqf_{ctx,m9,cap}_c32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212f

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212F_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212F_NO_HARNESS"; exit 1; }
grep -q "r212f: literal" "$HARNESS" \
  || { echo "R212F_STALE_HARNESS (MAXSK int parse not shipped)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212F_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212F_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212F_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212F_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212f.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212f] W14 quiesce $(date +%H:%M:%S) ==="
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
  echo "=== [r212f:$TAG] $* $(date +%H:%M:%S) ==="
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
    > $P/logs/r212f_${TAG}.log 2>&1
  local rc=$?
  echo "=== [r212f:$TAG] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime.json" ]; then
    mv "$P/out/p204_graphtime.json" "$P/out/p204_mseqf_${TAG}_c32768.json"
  else
    echo "R212F_MISSING_ARTIFACT_$TAG"
  fi
}

run_cell ctx  -e P204_MAXSK=ctx
run_cell m9   -e P204_MAXSK=9
run_cell cap  -e P204_MAXSK=cap

echo "=== [r212f] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
vals = {}
for tag in ("ctx", "m9", "cap"):
    f = os.path.join(out, f"p204_mseqf_{tag}_c32768.json")
    if not os.path.exists(f):
        print(f"MSEQ {tag}: MISSING"); fail = 1; continue
    d = json.load(open(f))
    nat = (d.get("mxfp4", {}).get("native_es") or {})
    if "capture_error" in nat:
        print(f"MSEQ {tag}: CAPTURE_ERROR {nat['capture_error']}"); fail = 1; continue
    can = nat.get("canary")
    us = nat.get("dev_us")
    vals[tag] = us
    if can is None or can > 2e-3:
        print(f"  CANARY_DIRTY {tag}: {can}"); fail = 1
    print(f"MSEQ {tag}: native_es {us:.1f} us  canary {can:.2e}")
if "ctx" in vals and "m9" in vals:
    r = vals["m9"] / vals["ctx"]
    print(f"RATIO m9/ctx = {r:.2f}x")
    if r >= 5:
        print("VERDICT: SPLIT_STARVATION_CONFIRMED (kernel arg causal)")
    elif r < 1.5:
        print("VERDICT: SPLIT_INSENSITIVE (40x is in the invocation; r212e trace leads)")
    else:
        print("VERDICT: PARTIAL (m9 matters but does not explain 40x alone)")
print("R212F_GATE_PASS" if not fail else "R212F_GATE_FAIL")
sys.exit(fail)
PY
grc=$?
echo "R212F_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
