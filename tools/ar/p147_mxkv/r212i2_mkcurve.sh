#!/bin/bash
# r212i2_mkcurve.sh -- arc7 Phase-8 r212i2: discriminate the forced-plan
# canary failures (r212i) and re-price the split-cost curve via the mk
# ladder. r212i finding: every forced-split cell canaried at |0-ref|max
# (0.85/0.18/0.11, shrinking with ctx exactly as ref magnitude does)
# while native_auto/native_es stayed clean — the plan path left outputs
# unwritten. Hypothesis: a prebuilt plan must COVER max_seqlen_k;
# plans built from real lens under mk>max_real_len (r212i ran
# MAXSK=cap = bt_width*64 > max len by up to 63) leave the tail of the
# mk-priced partition unassigned -> rows never written. r210 passed
# only because MAXSK=ctx made plan lens == mk exactly.
# Four cells at ctx=32768, SPLITS=auto,5,16 (P204_PLANLEN new):
#   A MAXSK=ctx  PLANLEN=real — r210 conditions at engine geometry:
#     forced cells should pass -> plan path healthy in this wheel
#   B MAXSK=cap  PLANLEN=cap  — capacity plans: should pass if the
#     coverage rule holds -> forced-split instrument rehabilitated
#   C MAXSK=80000 PLANLEN=cap — mid ladder: auto cost vs mk point +
#     coverage-short direction probe (plan 21888 < mk 80000)
#   D MAXSK=262400 PLANLEN=cap — engine-cap mk point at default pool
# The auto cells form the mk->cost curve at fixed real ctx (canary
# must stay 0 — per-row device limits, the r212f law), which is the
# split-dependence decomposition of the ~600 us fixed floor WITHOUT
# trusting the plan path.
# Artifacts -> /root/build/p199/out/p212i2_<TAG>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212i2

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212I2_NO_WHEEL"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212I2_NO_HARNESS"; exit 1; }
grep -q "P204_PLANLEN" "$HARNESS" || { echo "R212I2_STALE_HARNESS (no PLANLEN knob)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212I2_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212I2_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212I2_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212I2_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212i2.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212i2] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # tag maxsk planlen
  local TAG=$1 MAXSKV=$2 PLANLENV=$3
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212i2:$TAG] maxsk=$MAXSKV planlen=$PLANLENV $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=32768 -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto,5,16 -e P204_MAXSK=$MAXSKV -e P204_PLANLEN=$PLANLENV \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212i2_$TAG.log 2>&1
  local rc=$?
  echo "=== [r212i2:$TAG] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime.json" ]; then
    mv "$P/out/p204_graphtime.json" "$P/out/p212i2_$TAG.json"
  else
    echo "R212I2_MISSING_ARTIFACT_$TAG"
  fi
}

run_cell A_mkctx    ctx    real
run_cell B_mkcap    cap    cap
run_cell C_mk80k    80000  cap
run_cell D_mk262k   262400 cap

echo "=== [r212i2] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
auto_ok = True
forced_clean = {"A_mkctx": None, "B_mkcap": None, "C_mk80k": None, "D_mk262k": None}
print("r212i2 -- us/call, ctx=32768 real, graph 32x50, mxfp4, D=256/HREP=6/BLOCK=64")
print(" cells: auto | s5 | s16  ([!] = canary>2e-3, CAP = capture fail)")
print(f" {'cell':>8} {'mk':>7} | " + " | ".join(f"{s:>9}" for s in ("auto", "s5", "s16")))
mk_of = {"A_mkctx": "21846", "B_mkcap": "21888", "C_mk80k": "80000", "D_mk262k": "262400"}
for tag in ("A_mkctx", "B_mkcap", "C_mk80k", "D_mk262k"):
    f = os.path.join(out, f"p212i2_{tag}.json")
    if not os.path.exists(f):
        print(f" {tag:>8} {mk_of[tag]:>7} | MISSING"); fail = 1; continue
    d = json.load(open(f)).get("mxfp4", {})
    row = []
    fc = True
    for cell in ("native_auto", "native_5", "native_16"):
        c = d.get(cell) or {}
        if "capture_error" in c:
            row.append("      CAP"); fc = False
        elif c.get("canary") is None or c["canary"] > 2e-3:
            row.append(f"{c['dev_us']:7.1f}!")
            if cell == "native_auto":
                auto_ok = False
            else:
                fc = False
        else:
            row.append(f"{c['dev_us']:8.1f}")
    forced_clean[tag] = fc
    print(f" {tag:>8} {mk_of[tag]:>7} | " + " | ".join(row))
if not auto_ok:
    print("AUTO_CANARY_FAIL (per-row law violated — do not trust this run)")
    fail = 1
if forced_clean["A_mkctx"]:
    print("A: forced plans VALID at mk=real-max-lens — plan path healthy (r210 conditions reproduce)")
if forced_clean["A_mkctx"] and not forced_clean["B_mkcap"]:
    print("B: capacity plans STILL fail — coverage rule is not the (whole) story")
elif forced_clean["B_mkcap"]:
    print("B: capacity plans CLEAN — coverage law CONFIRMED (plan must cover max_seqlen_k)")
if forced_clean["C_mk80k"] is False and forced_clean["B_mkcap"]:
    print("C: mk=80000 with plan@21888 fails as predicted (shortfall direction)")
print("R212I2_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212I2_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
