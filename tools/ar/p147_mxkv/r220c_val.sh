#!/bin/bash
# r220c_val.sh -- arc7 Phase-8 r220c: VALUE validation of the
# ARC_MXKV_P64NATIVE multi-SG route. The r220a finding: p204's native
# canary is graph-vs-eager SELF-reference (a deterministic value bug
# canaries at 0.0) -- r220c_verify.py closes the gap with cross-policy
# (_64 vs certified _16, same wheel, env-flipped per call), determinism
# (two _64 runs), cross-impl (vs Triton v2) and NaN-prefill unwritten-
# row counts. Eager only -- minutes, no graph capture.
# Gate per format: v2-excess band |d_64_v2 - d_16_v2| <= 2e-3 (the
#   _64 route must track the CERTIFIED _16 route's own v2 relationship)
#   AND d_16_64 <= 2e-3 AND d_det == 0 AND nan64 == 0.
# Band history: v1 gated d_16_v2 <= 2e-3 absolutely -- e4m3-calibrated;
#   the e2m1 lanes' native-vs-v2 floor is quant-noise-scale (0.05 @1K,
#   0.004-0.006 @32K, PRESENT IDENTICALLY on the certified _16 route,
#   measured 2026-11-22 run2: d_64_v2 - d_16_v2 <= 1.2e-4) -- the
#   excess form keeps the same semantics format-agnostically.
# Artifacts -> /root/build/p199/out/r220c_verify_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r220c

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R220C_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
echo "wheel md5 $GOTMD5"
if [ -n "${WHEELMD5:-}" ] && [ "$GOTMD5" != "$WHEELMD5" ]; then
  echo "R220C_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"
  exit 1
fi
grep -q "ARC_MXKV_P64NATIVE" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp \
  || { echo "R220C_TREE_MISSING_ROUTE"; exit 1; }
VER=$W/tools/ar/p147_mxkv/r220c_verify.py
[ -f "$VER" ] || { echo "R220C_NO_VERIFIER"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R220C_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R220C_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R220C_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R220C_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r220c.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r220c] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # ctx
  local CTX=$1
  echo "=== [r220c:c$CTX] $(date +%H:%M:%S) ==="
  docker rm -f $CNAME >/dev/null 2>&1
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e ARC_MXKV_DIRECTBF16=1 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/r220c_verify.py /out" \
    > $P/logs/r220c_c$CTX.log 2>&1
  echo "=== [r220c:c$CTX] exit rc=$? $(date +%H:%M:%S) ==="
  [ -f "$P/out/r220c_verify_$CTX.json" ] \
    || echo "R220C_MISSING_ARTIFACT_$CTX"
}

run_cell 1024
run_cell 32768

echo "=== [r220c] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, os, sys
out = sys.argv[1]
fail = 0
print("r220c value gate -- _64 vs certified _16, same wheel, eager")
print(f" {'fmt':>7} {'ctx':>6} | {'16-64':>9} | {'det':>9} | {'64-v2':>9} | {'16-v2':>9} | {'nan16':>6} {'nan64':>6} | verdict")
for ctx in (1024, 32768):
    f = os.path.join(out, f"r220c_verify_{ctx}.json")
    if not os.path.exists(f):
        print(f" {'-':>7} {ctx:>6} | MISSING"); fail = 1; continue
    j = json.load(open(f))
    for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
        c = j.get(fmt)
        if not c:
            print(f" {fmt:>7} {ctx:>6} | MISSING"); fail = 1; continue
        band = abs(c["d_64_v2"] - c["d_16_v2"]) <= 2e-3
        ok = (band and c["d_16_64"] <= 2e-3 and c["d_det"] == 0
              and c["nan64"] == 0 and c["nan16"] == 0)
        if not ok:
            fail = 1
        print(f" {fmt:>7} {ctx:>6} | {c['d_16_64']:9.6f} | {c['d_det']:9.6f} | "
              f"{c['d_64_v2']:9.6f} | {c['d_16_v2']:9.6f} | {c['nan16']:6d} {c['nan64']:6d} | "
              f"{'OK' if ok else ('CELL_BAD' if not band else 'FAIL')}")
print("R220C_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R220C_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
