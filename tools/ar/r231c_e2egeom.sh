#!/bin/bash
# r231c_e2egeom.sh -- P6.47 r231c step 2: e2e-geometry fixed-term probe
# (L2-defeating scattered pool, engine-capacity bt width arm) on the
# certified build25 wheel. Bare cert27 container, idle GPU. Bank ->
# /root/build/p647/r231c/.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert27
FORK_SHA=d39ab5a26bf737d006d47a3f3125a404dca6e928
SOMD5_25=2d73ff7a3c2aa2caab86179288ca3711
P=$R/p647/r231c
CNAME=arc-r231c2
PROBE=$P/r231c_e2egeom.py

fail() { echo "R231C2_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert27_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
SO_MD5=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels*.so'); print(hashlib.md5(open(ps[0],'rb').read()).hexdigest())\"" 2>/dev/null) || SO_MD5=""
[ "$SO_MD5" = "$SOMD5_25" ] || { fail "image .so md5 $SO_MD5 != build25 pin"; exit 1; }
[ -f "$PROBE" ] || { fail "probe not shipped ($PROBE)"; exit 1; }
PICKS=()
for bank in r224a r229a r229b; do
  f=$(ls "$R/p647/$bank"/mxkv_ab.*.pt 2>/dev/null | head -1)
  [ -n "$f" ] || { fail "payload bank $bank empty"; exit 1; }
  PICKS+=("$f")
done
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R231C2 refuses passed (fork_sha=$FORK_SHA so_md5=$SO_MD5 picks=${#PICKS[@]})"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "R231C2: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  echo "R231C2: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r231c2.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R231C2: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R231C2: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

# ---- 1. arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5
mkdir -p "$P"

# ---- 2. bare container ----
docker run -d --name "$CNAME" \
  --privileged --network host --ipc host \
  --shm-size 16g \
  -v /root/build/neo_cache:/root/.neo_cache \
  --entrypoint /bin/bash "$IMG" -c 'sleep infinity' \
  || { fail "bare container boot failed"; exit 1; }
sleep 5
docker cp "$PROBE" "$CNAME:/tmp/r231c_e2egeom.py"
docker cp "${PICKS[0]}" "$CNAME:/tmp/r231c_p0.pt"
docker cp "${PICKS[1]}" "$CNAME:/tmp/r231c_p1.pt"
docker cp "${PICKS[2]}" "$CNAME:/tmp/r231c_p2.pt"

# ---- 3. probe: e2e-geometry decomposition per format ----
RC=0
for j in 0 1 2; do
  docker exec "$CNAME" /opt/venv/bin/python /tmp/r231c_e2egeom.py \
    /tmp/r231c_p$j.pt >> "$P/e2egeom.log" 2>&1 || RC=1
done
grep -E "R231C_E2E" "$P/e2egeom.log" || true
echo "R231C2_PROBE_RC=$RC"
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "R231C2_DONE $(date +%H:%M:%S)"
exit "$RC"
