#!/bin/bash
# r225j_iso.sh — P6.47 r225 leg j: row/surface isolation bisect of the
# real-payload residual on the FIXED wheel (r225j_iso.py inside a
# W14-down arc window; wheel force-installed from the r225 rebuild).
#
# Usage: r225j_iso.sh <CERT26_FORK_SHA>
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert26
CNAME=arc-r225j
BANK=$R/p647/r224a
OUT=$R/p647/r225j
TOOLS=$R/r225_tools
WHEELS=$R/wheels-arc-vxk
OLD_WHEEL_MD5=62f234c8d0aa89c4aec3e8d2c6fa22ad
FORK_SHA="${1:?usage: r225j_iso.sh <CERT26_FORK_SHA>}"
PAYLOADS="mxkv_ab.2447.18.pt mxkv_ab.2447.30.pt mxkv_ab.2547.18.pt"

mkdir -p "$OUT"
cd "$OUT" || exit 2

fail() { echo "R225J_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert26_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
grep -q "VXK_ARC7MX_BUILD_EXIT rc=0" "$R/arc7mx_build22_r225.log" \
  || { fail "build log has no rc=0 exit line"; exit 1; }
WHEEL=$(ls -t "$WHEELS"/vllm_xpu_kernels-0.1.14.1+arc7mx*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { fail "no arc7mx wheel in $WHEELS"; exit 1; }
WHEEL_MD5=$(md5sum "$WHEEL" | cut -d' ' -f1)
[ "$WHEEL_MD5" != "$OLD_WHEEL_MD5" ] \
  || { fail "wheel md5 == cert26 shipped md5 (rebuild did not take)"; exit 1; }
[ -f "$TOOLS/r225j_iso.py" ] || { fail "r225j_iso.py not shipped"; exit 1; }
for f in $PAYLOADS; do
  [ -f "$BANK/$f" ] || { fail "banked payload missing ($BANK/$f)"; exit 1; }
done
[ -d "$R/triton_cache" ] || { fail "warm triton cache missing"; exit 1; }
echo "R225J refuses passed (fork_sha=$FORK_SHA wheel_md5=$WHEEL_MD5)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R225J: restoring W14 ($(date +%H:%M:%S))"
  nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R225J: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R225J: WARN W14 not 200 after restore poll (last=$c)"
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
docker cp "$TOOLS/r225j_iso.py" "$CNAME:/tmp/r225j_iso.py"
WHEEL_BASE=$(basename "$WHEEL")
docker cp "$WHEEL" "$CNAME:/tmp/$WHEEL_BASE"

echo "=== R225J wheel install ($(date +%H:%M:%S)) ==="
docker exec "$CNAME" /opt/venv/bin/pip install --force-reinstall --no-deps \
  "/tmp/$WHEEL_BASE" > "$OUT/wheel_install.log" 2>&1 \
  || { fail "pip install rc=$?"; exit 1; }
tail -2 "$OUT/wheel_install.log"

echo "=== R225J isolation bisect ($(date +%H:%M:%S)) ==="
PARGS=""
for f in $PAYLOADS; do PARGS="$PARGS /bank/$f"; done
docker exec "$CNAME" env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
  /tmp/r225j_iso.py $PARGS \
  > "$OUT/iso.log" 2>&1
rc=$?
cat "$OUT/iso.log"
[ $rc -eq 0 ] || { fail "iso rc=$rc"; exit 1; }
grep -q "R225J_ISO_DONE" "$OUT/iso.log" \
  || { fail "iso produced no DONE line"; exit 1; }

docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "R225J: iso complete; restoring W14"
