#!/bin/bash
# r212k_probedecomp.sh -- arc7 Phase-8 r212k: price the walk chains on the
# CURRENT build21 wheel before the kernel iteration (deep-analysis first).
# The r200h decomposition (K load chain 58-66% of the walk) predates
# r201b/r202/r203/r212g; the r212j2 honest curve (auto 598/423/1040 us at
# 1K/8K/32K mxfp4) leaves the excess attribution open. The kernel ships
# ablation probes (ARC_MXKV_VPROBE via mx_paged_decode.hpp): 0 full,
# 2 const-V (gather severed, staging on), 3 no-staging+const-V,
# 6 const-K (K load chain severed, V real), 8 floor (const-K + no stage +
# const-P). Const-fill ablation law: severed chains UNDERSTATE -- floors
# are lower bounds; triangulate across cells, never adopt as targets.
# Cells: probe {0,2,3,6,8} x ctx {1024,8192,32768}, FMTS=mxfp4,mxfp8,
# SPLITS=auto (the only valid instrument, r212j), engine geometry,
# graph 32x50. Probe cells canary wrong BY DESIGN -- gate is probe-0
# canary + timing presence.
# Artifacts -> /root/build/p199/out/p212k_m<PROBE>_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212k

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212K_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212K_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212K_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212K_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212K_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212K_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212k.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212k] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # probe ctx
  local PROBE=$1
  local CTX=$2
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212k:m$PROBE:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4,mxfp8 \
    -e ARC_MXKV_VPROBE=$PROBE \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212k_m${PROBE}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212k:m$PROBE:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m$PROBE.json" ]; then
    mv "$P/out/p204_graphtime_m$PROBE.json" "$P/out/p212k_m${PROBE}_$CTX.json"
  else
    echo "R212K_MISSING_ARTIFACT_m${PROBE}_$CTX"
  fi
}

for CTX in 1024 8192 32768; do
  for PROBE in 0 2 3 6 8; do
    run_cell $PROBE $CTX
  done
done

echo "=== [r212k] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
PROBES = ("0", "2", "3", "6", "8")
for fmt in ("mxfp4", "mxfp8"):
    print(f"r212k chain decomposition -- {fmt}, auto splits, us/call (graph 32x50)")
    print(f" {'ctx':>6} | " + " | ".join(f"{p:>8}" for p in ("full", "noVgath", "noVlane", "noK", "floor")))
    rows = {}
    for ctx in (1024, 8192, 32768):
        cells = {}
        for probe in PROBES:
            f = os.path.join(out, f"p212k_m{probe}_{ctx}.json")
            if not os.path.exists(f):
                print(f" {ctx:>6} | MISSING"); fail = 1; cells = None; break
            c = (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}
            if "capture_error" in c or c.get("dev_us") is None:
                print(f" {ctx:>6} | CELL_FAIL m{probe}"); fail = 1; cells = None; break
            cells[probe] = c
        if not cells:
            continue
        rows[ctx] = cells
        line = f" {ctx:>6} | " + " | ".join(f"{cells[p]['dev_us']:8.1f}" for p in PROBES)
        print(line)
        c0 = cells["0"]
        if c0.get("canary") is None or c0["canary"] > 2e-3:
            print(f"          FULL_CELL_CANARY_FAIL {c0.get('canary')}")
            fail = 1
    for ctx, cells in sorted(rows.items()):
        full, c2, c3, c6, c8 = (cells[p]["dev_us"] for p in PROBES)
        print(f"  ctx {ctx}: K_chain {full-c6:7.1f}  V_gather {full-c2:7.1f}"
              f"  V_stage {c2-c3:7.1f}  floor {c8:7.1f}"
              f"  (Vlane+softmax+mma above floor {c6-c8:7.1f})")
print("R212K_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212K_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
