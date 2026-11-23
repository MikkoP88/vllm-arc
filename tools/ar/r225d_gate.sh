#!/bin/bash
# r225d_gate.sh — P6.47 r225 leg d: replay gate of the FIXED wheel
# (fix 1: e2m1 subnormal mantissa drop in mx_dequant.hpp; fix 2:
# precise exp2 in the split-KV combine, mx_reduce_split.hpp — the
# native::exp2 rescale carried ~1e-3 relative error that only a FLAT
# decode softmax exposes; r225l: nsplit1 exact, auto wrong; fix 3 /
# THE dominant one: bf16 split partials — temp_out was allocated bf16
# (binding mirrored stock flash_api), quantizing every split's
# unnormalized fp32 accumulator to 2^-9 relative BEFORE the combine
# divides by the softmax sum, which a flat deep-layer softmax (sum
# ~5e3) amplifies to hundreds-thousands of output ulp; fix = fp32
# partials end-to-end, the reduce write is now the single bf16
# quantization point — r225 delta (4) in three fork-owned files)
# against the banked r224a mxk8v4 payloads.
#
# THE GATE (plan r225 step c): with the rebuilt wheel force-installed
# into a fresh cert26 container, replay the same three arbitration
# payloads and require
#
#   1. base(new wheel) lands AT_TRITON_LEVEL vs the float64 reference
#      (r225b arbitration) — the native refusal is gone;
#   2. base == dbf0 (r225d_compare) — the direct-bf16 fast path now
#      equals the float path (all e2m1 values exactly representable in
#      bf16 once the subnormal implicit-bit bug is gone);
#   3. base MOVES off the banked broken out_native — proof the new
#      kernel bytes are actually live (stale-.so canary).
#
# Also re-runs r225c_sim.py with output redirected (byte-true evidence
# for the bank) — the CPU nibble simulation that reproduced the banked
# error to 0.5% pre-fix.
#
# Rig laws honored: W14 down for the window, guarded EXIT-trap restore
# with health-200 poll; warm triton cache mounted; kill-by-container.
# Refuses: cert26 image + manifest fork_sha == $1; build24 log rc=0;
# new wheel md5 != cert26's 62f234c8... AND != fix1-only d66e9255...
# AND != fix1+2 ebc5845b... (the fp32-partials fix must be in the
# gated bytes); payloads + tools + cache present.
#
# Usage: r225d_gate.sh <CERT26_FORK_SHA>
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert26
CNAME=arc-r225d
BANK=$R/p647/r224a
OUT=$R/p647/r225d
TOOLS=$R/r225_tools
WHEELS=$R/wheels-arc-vxk
OLD_WHEEL_MD5=62f234c8d0aa89c4aec3e8d2c6fa22ad
FIX1_WHEEL_MD5=d66e92556a3f44cc69dafc137f872609
FIX2_WHEEL_MD5=ebc5845b88e1b4b9a83eaa8060db454e
FORK_SHA="${1:?usage: r225d_gate.sh <CERT26_FORK_SHA>}"
PAYLOADS="mxkv_ab.2447.18.pt mxkv_ab.2447.30.pt mxkv_ab.2547.18.pt"

mkdir -p "$OUT"
cd "$OUT" || exit 2

fail() { echo "R225D_ABORT: $*"; }

# ---- refuses (staged-ctx law) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert26_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
grep -q "VXK_ARC7MX_BUILD_EXIT rc=0" "$R/arc7mx_build24_r225.log" \
  || { fail "build24 log has no rc=0 exit line"; exit 1; }
WHEEL=$(ls -t "$WHEELS"/vllm_xpu_kernels-0.1.14.1+arc7mx*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { fail "no arc7mx wheel in $WHEELS"; exit 1; }
WHEEL_MD5=$(md5sum "$WHEEL" | cut -d' ' -f1)
[ "$WHEEL_MD5" != "$OLD_WHEEL_MD5" ] && [ "$WHEEL_MD5" != "$FIX1_WHEEL_MD5" ] \
  && [ "$WHEEL_MD5" != "$FIX2_WHEEL_MD5" ] \
  || { fail "wheel md5 == cert26 / fix1 / fix1+2 (rebuild did not take)"; exit 1; }
for f in $PAYLOADS; do
  [ -f "$BANK/$f" ] || { fail "banked payload missing ($BANK/$f)"; exit 1; }
done
for t in r225a_replay.py r225b_arbitrate.py r225c_sim.py r225d_compare.py; do
  [ -f "$TOOLS/$t" ] || { fail "$t not shipped"; exit 1; }
done
[ -d "$R/triton_cache" ] || { fail "warm triton cache missing"; exit 1; }
echo "R225D refuses passed (fork_sha=$FORK_SHA wheel=$WHEEL md5=$WHEEL_MD5)"

RESTORE_DONE=0
finish() {
  # guarded fail-safe: even a mid-flight death must not leave the rig on
  # an arc image (overnight law) — restore W14 exactly once.
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R225D: restoring W14 ($(date +%H:%M:%S))"
  nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R225D: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R225D: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

# ---- arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run -d --name "$CNAME" \
  --device /dev/dri --ipc=host \
  -v "$R/triton_cache:/root/.triton_cache" \
  -v "$BANK:/bank:ro" \
  -v "$OUT:/out" \
  --entrypoint /bin/bash "$IMG" -c "sleep infinity" \
  || { fail "container boot failed"; exit 1; }
for t in r225a_replay.py r225b_arbitrate.py r225c_sim.py r225d_compare.py; do
  docker cp "$TOOLS/$t" "$CNAME:/tmp/$t"
done
WHEEL_BASE=$(basename "$WHEEL")
docker cp "$WHEEL" "$CNAME:/tmp/$WHEEL_BASE"

echo "=== R225D wheel install ($(date +%H:%M:%S)) ==="
docker exec "$CNAME" /opt/venv/bin/pip install --force-reinstall --no-deps \
  "/tmp/$WHEEL_BASE" > "$OUT/wheel_install.log" 2>&1 \
  || { fail "pip install rc=$?"; exit 1; }
tail -3 "$OUT/wheel_install.log"
docker exec "$CNAME" /opt/venv/bin/python -c \
  "import vllm_xpu_kernels as m, os, time; f=m.__file__;
print('VKPKG', f); print('VKMTIME', time.strftime('%Y-%m-%dT%H:%M', time.localtime(os.path.getmtime(f))))" \
  || { fail "import vllm_xpu_kernels failed"; exit 1; }

PARGS=""
for f in $PAYLOADS; do PARGS="$PARGS /bank/$f"; done

cell() {  # cell <name> [ENV=VAL...] — one replay on the fixed wheel
  local name=$1; shift
  local envargs=()
  for e in "$@"; do envargs+=(-e "$e"); done
  echo "=== R225D cell $name ($(date +%H:%M:%S)) ==="
  docker exec "${envargs[@]}" "$CNAME" \
    env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
    /tmp/r225a_replay.py "$name" "/out/r225d_${name}.pt" $PARGS \
    > "$OUT/cell_${name}.log" 2>&1
  local rc=$?
  tail -20 "$OUT/cell_${name}.log"
  [ $rc -eq 0 ] || { fail "cell $name rc=$rc (see $OUT/cell_${name}.log)"; exit 1; }
  grep -q "R225A_DONE" "$OUT/cell_${name}.log" \
    || { fail "cell $name produced no DONE line"; exit 1; }
}

cell base
cell dbf0 ARC_MXKV_DIRECTBF16=0

# arbitration adapter: r225b reads r225a_<factor>.pt names
cp "$OUT/r225d_base.pt" "$OUT/r225a_base.pt"
cp "$OUT/r225d_dbf0.pt" "$OUT/r225a_dbf0.pt"

echo "=== R225D arbitration (float64, CPU) ($(date +%H:%M:%S)) ==="
STEMS=""
for f in $PAYLOADS; do STEMS="$STEMS ${f%.pt}"; done
docker exec "$CNAME" env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
  /tmp/r225b_arbitrate.py /bank /out $STEMS \
  > "$OUT/arbitration.log" 2>&1
rc=$?
cat "$OUT/arbitration.log"
[ $rc -eq 0 ] || { fail "arbitration rc=$rc"; exit 1; }
grep -q "R225B_VERDICT" "$OUT/arbitration.log" \
  || { fail "arbitration produced no verdict"; exit 1; }

echo "=== R225D compare (fix canaries) ($(date +%H:%M:%S)) ==="
docker exec "$CNAME" env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
  /tmp/r225d_compare.py /bank /out $STEMS \
  > "$OUT/compare.log" 2>&1
rc=$?
cat "$OUT/compare.log"
[ $rc -eq 0 ] || { fail "compare rc=$rc"; exit 1; }

echo "=== R225D nibble sim (byte-true evidence) ($(date +%H:%M:%S)) ==="
docker exec "$CNAME" env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
  /tmp/r225c_sim.py $PARGS \
  > "$OUT/r225c_sim.log" 2>&1
rc=$?
tail -12 "$OUT/r225c_sim.log"
[ $rc -eq 0 ] || { fail "sim rc=$rc"; exit 1; }

# ---- final gate ----
GATE=PASS
grep -q "R225B_VERDICT: base" "$OUT/arbitration.log" \
  || { echo "gate: base NOT at triton level"; GATE=FAIL; }
grep -q "R225D_COMPARE_VERDICT: PASS" "$OUT/compare.log" \
  || { echo "gate: compare canaries failed"; GATE=FAIL; }
echo "R225D_GATE: $GATE"
[ "$GATE" = PASS ] || exit 1

# teardown + W14 restore via the EXIT trap
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "R225D: gate complete; restoring W14"
