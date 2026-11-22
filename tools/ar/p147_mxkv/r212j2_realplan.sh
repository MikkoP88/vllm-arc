#!/bin/bash
# r212j2_realplan.sh -- arc7 Phase-8 r212j2: the valid plan combination.
# r212j taught the unit law (kt=16) but ran it with PLANLEN=cap -- the
# plan-driven walk branch has NO clamp to the row's real k_blocks
# (auto clamps via windowed_k_blocks; wl branch walks [start,+count)
# verbatim), so capacity plans walked 8x past the real 32K context
# into stale/zero pages -> ref-scale canaries. The VALID combination:
# kt=16 + PLANLEN=real -- the plan partitions exactly the real tiles
# (auto-equivalent coverage, different split boundaries). Cells per
# ctx {1024,8192,32768}: SPLITS=auto,1,5,16,64 at PLANLEN=real.
# s1 isolates walk-vs-reduce: pure plan walk, direct O write, no
# reduce kernel (need_reduce = num_kv_splits>1). If s1 canary 0 and
# multi-split fails -> reduce/partial-slot defect; if s1 fails -> the
# wl walk itself. Note for the ENGINE lane: plans built at capture
# time see the DUMMY-diluted lens, so engine wiring needs a device-
# side clamp (end_blk <= k_block0+windowed) + empty-split-safe
# partials -- a kernel iteration, not today's question.
# Artifacts -> /root/build/p199/out/p212j2_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212j2

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212J2_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212J2_NO_HARNESS"; exit 1; }
grep -q "P204_KVT" "$HARNESS" || { echo "R212J2_STALE_HARNESS (no tile-unit fix)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212J2_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212J2_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212J2_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212J2_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212j2.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212j2] W14 quiesce $(date +%H:%M:%S) ==="
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
  echo "=== [r212j2:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto,1,5,16,64 -e P204_MAXSK=cap -e P204_PLANLEN=real \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212j2_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212j2:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime.json" ]; then
    mv "$P/out/p204_graphtime.json" "$P/out/p212j2_$CTX.json"
  else
    echo "R212J2_MISSING_ARTIFACT_$CTX"
  fi
}

run_ctx 1024
run_ctx 8192
run_ctx 32768

echo "=== [r212j2] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
print("r212j2 REAL-plan split curve (kt=16, PLANLEN=real) -- us/call, mxfp4, MAXSK=cap, graph 32x50")
print(f" {'ctx':>6} | " + " | ".join(f"{s:>9}" for s in ("auto", "s1", "s5", "s16", "s64")))
clean = {"s1": True, "multi": True}
for ctx in (1024, 8192, 32768):
    f = os.path.join(out, f"p212j2_{ctx}.json")
    if not os.path.exists(f):
        print(f" {ctx:>6} | MISSING"); fail = 1; continue
    d = json.load(open(f)).get("mxfp4", {})
    row = []
    for cell in ("native_auto", "native_1", "native_5", "native_16", "native_64"):
        c = d.get(cell) or {}
        if "capture_error" in c:
            row.append("       CAP"); fail = 1
        elif c.get("canary") is None or c["canary"] > 2e-3:
            row.append(f"{c['dev_us']:7.1f}!")
            if cell == "native_1":
                clean["s1"] = False
            else:
                clean["multi"] = False
            fail = 1
        else:
            row.append(f"{c['dev_us']:9.1f}")
    print(f" {ctx:>6} | " + " | ".join(row))
print(f"s1 (plan walk, direct write, no reduce): {'CLEAN' if clean['s1'] else 'WRONG'}")
print(f"multi-split (partials + reduce):         {'CLEAN' if clean['multi'] else 'WRONG'}")
if clean["s1"] and not clean["multi"]:
    print("-> defect is in the partial/reduce path, not the wl walk")
elif not clean["s1"]:
    print("-> defect is in the wl walk itself (start/count semantics)")
print("R212J2_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212J2_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
