#!/bin/bash
# r225t_gate.sh — P6.47 r225 FINAL gate: q-cast-matched validation of
# the build25 wheel (fix1-5) on the banked r224a payloads, inside a
# W14-down arc window. One fresh container process per cell.
#
# Criterion (r225r-corrected): each lane vs the fp64 walk of its OWN
# input (native eats bf16 q; the fp16-q Triton oracle is a different
# function — the old r225d criterion was unattainable by construction).
#
# Usage: r225t_gate.sh <CERT26_FORK_SHA>
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert26
CNAME=arc-r225t
BANK=$R/p647/r224a
OUT=$R/p647/r225t
TOOLS=$R/r225_tools
WHEELS=$R/wheels-arc-vxk
FORK_SHA="${1:?usage: r225t_gate.sh <CERT26_FORK_SHA>}"
PAYLOADS="mxkv_ab.2447.18.pt mxkv_ab.2447.30.pt mxkv_ab.2547.18.pt"

mkdir -p "$OUT"
cd "$OUT" || exit 2

fail() { echo "R225T_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert26_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
grep -q "VXK_ARC7MX_BUILD_EXIT rc=0" "$R/arc7mx_build25_r225.log" \
  || { fail "build25 log has no rc=0 exit line"; exit 1; }
WHEEL=$(ls -t "$WHEELS"/vllm_xpu_kernels-0.1.14.1+arc7mx*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { fail "no arc7mx wheel in $WHEELS"; exit 1; }
WHEEL_MD5=$(md5sum "$WHEEL" | cut -d' ' -f1)
for stale in 62f234c8d0aa89c4aec3e8d2c6fa22ad d66e92556a3f44cc69dafc137f872609 \
             ebc5845b88e1b4b9a83eaa8060db454e 42d5920127a0c0244810a3f787e5f157; do
  [ "$WHEEL_MD5" != "$stale" ] \
    || { fail "wheel md5 == a pre-build25 lineage (cert26/fix1/fix2/fix3)"; exit 1; }
done
for f in r225t_replay.py r225j_iso.py; do
  [ -f "$TOOLS/$f" ] || { fail "$f not shipped"; exit 1; }
done
for f in $PAYLOADS; do
  [ -f "$BANK/$f" ] || { fail "banked payload missing ($BANK/$f)"; exit 1; }
done
[ -d "$R/triton_cache" ] || { fail "warm triton cache missing"; exit 1; }
echo "R225T refuses passed (fork_sha=$FORK_SHA wheel_md5=$WHEEL_MD5)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R225T: restoring W14 ($(date +%H:%M:%S))"
  nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R225T: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R225T: WARN W14 not 200 after restore poll (last=$c)"
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
docker cp "$TOOLS/r225t_replay.py" "$CNAME:/tmp/r225t_replay.py"
docker cp "$TOOLS/r225j_iso.py" "$CNAME:/tmp/r225j_iso.py"
WHEEL_BASE=$(basename "$WHEEL")
docker cp "$WHEEL" "$CNAME:/tmp/$WHEEL_BASE"

echo "=== R225T wheel install ($(date +%H:%M:%S)) ==="
docker exec "$CNAME" /opt/venv/bin/pip install --force-reinstall --no-deps \
  "/tmp/$WHEEL_BASE" > "$OUT/wheel_install.log" 2>&1 \
  || { fail "pip install rc=$?"; exit 1; }
tail -2 "$OUT/wheel_install.log"

LOG=$OUT/gate.log
: > "$LOG"
FAILED=0
for factor in base nsplit1 nsplit2 nsplit8; do
  for pf in $PAYLOADS; do
    echo "=== R225T cell $factor $pf ($(date +%H:%M:%S)) ===" | tee -a "$LOG"
    docker exec "$CNAME" env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
      /tmp/r225t_replay.py "$factor" "/bank/$pf" >> "$LOG" 2>&1
    rc=$?
    [ $rc -eq 0 ] || { FAILED=1; echo "R225T: cell $factor $pf rc=$rc"; }
  done
done
cat "$LOG"

total=$(grep -c "^R225T_CELL" "$LOG" || true)
passed=$(grep -c "^R225T_CELL .* PASS$" "$LOG" || true)
bitex=$(grep -c "TRITON_BITEXACT dmax=0.000e+00" "$LOG" || true)
stables=$(grep -c "NATIVE_STABLE bitwise=True nan=0" "$LOG" || true)
[ "$total" -eq 12 ] || { fail "expected 12 cells, got $total"; exit 1; }
[ "$bitex" -eq 12 ] || { fail "triton bit-exact refuses: $bitex/12"; exit 1; }
[ "$stables" -eq 12 ] || { fail "native stability: $stables/12"; exit 1; }
if [ "$FAILED" -eq 0 ] && [ "$passed" -eq 12 ]; then
  echo "R225T_GATE: PASS (12/12 cells, wheel_md5=$WHEEL_MD5)"
else
  echo "R225T_GATE: FAIL ($passed/12 cells)"
  exit 1
fi

docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "R225T: gate complete; restoring W14"
