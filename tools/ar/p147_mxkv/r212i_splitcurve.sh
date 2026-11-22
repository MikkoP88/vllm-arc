#!/bin/bash
# r212i_splitcurve.sh -- arc7 Phase-8 r212i: the split-count cost
# curve at engine geometry. Post-r212g arithmetic: the MX-vs-fp8
# e2e excess (1.4-2.0x, prop ctx) decomposes offline as a ~600 us
# FIXED floor (native_es 1K ~= 600 us) + ~20 us/Kctx variable
# (32K = 1272 us) -- the fixed term dominates at engine ctxs and
# IS the next kernel-iteration target. This leg prices the floor's
# split-dependence with the harness's existing forced-split cells
# (prebuilt plans): if the floor grows with num_splits it is the
# plan/partial-write/reduction machinery; if flat, launch/indirect-
# ion (_16-policy 16-row sub-tiles at page 64). Cells per ctx
# {1024, 8192, 32768}: auto (capacity-priced engine posture,
# MAXSK=cap) + forced {1,5,16,64}. mxfp4, D=256/HREP=6/BLOCK=64,
# eshape, graph 32x50. v2_es rides free per run as the Triton
# reference.
# Artifacts -> /root/build/p199/out/p212i_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212i

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212I_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212I_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212I_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212I_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212I_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212I_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212i.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212i] W14 quiesce $(date +%H:%M:%S) ==="
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
  echo "=== [r212i:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto,1,5,16,64 -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212i_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212i:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime.json" ]; then
    mv "$P/out/p204_graphtime.json" "$P/out/p212i_$CTX.json"
  else
    echo "R212I_MISSING_ARTIFACT_$CTX"
  fi
}

run_ctx 1024
run_ctx 8192
run_ctx 32768

echo "=== [r212i] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
print("r212i split-count cost curve -- us/call (graph 32x50, mxfp4, MAXSK=cap)")
print(" forced cells at plain shape; auto = heuristic plain; *_es = engine r_tile=4")
print(f" {'ctx':>6} | " + " | ".join(f"{s:>8}" for s in ("auto", "s1", "s5", "s16", "s64"))
      + " | " + " | ".join(f"{s:>8}" for s in ("nat_es", "v2_es")))
for ctx in (1024, 8192, 32768):
    f = os.path.join(out, f"p212i_{ctx}.json")
    if not os.path.exists(f):
        print(f" {ctx:>6} | MISSING"); fail = 1; continue
    d = json.load(open(f)).get("mxfp4", {})
    row = []
    forced = []
    for cell in ("native_auto", "native_1", "native_5", "native_16", "native_64"):
        c = d.get(cell) or {}
        if "capture_error" in c:
            row.append("CAPTURE!"); fail = 1
        elif c.get("canary") is None or c["canary"] > 2e-3:
            row.append(f"CANARY!{c.get('canary')}"); fail = 1
        else:
            row.append(f"{c['dev_us']:8.1f}")
            if cell != "native_auto":
                forced.append(c["dev_us"])
    for cell in ("native_es", "v2_es"):
        c = d.get(cell) or {}
        row.append(f"{c.get('dev_us', -1):8.1f}" if "capture_error" not in c else "CAPTURE!")
    if len(forced) == 4:
        spread = max(forced) - min(forced)
        print(f" {ctx:>6} | " + " | ".join(row)
              + f"   [forced spread {spread:.1f} us]")
    else:
        print(f" {ctx:>6} | " + " | ".join(row))
print("R212I_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212I_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
