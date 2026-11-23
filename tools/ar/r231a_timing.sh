#!/bin/bash
# r231a_timing.sh -- P6.47 r231 Phase A: the split-overprovisioning
# timing decomposition on the cert27 wheel (bare container, no serve
# boot -- the kernel timing needs only the wheel + an idle GPU).
#
# Hypothesis under test (r230 verdict): the MX-vs-ctrl ITL constant
# (~6.4-6.8 ms/step, format-independent) is the frozen auto-split
# over-provisioning -- every graph bucket prices max_seqlen_k at the
# bt-width capacity (4114 x 64 = 262,336), so ~123 of the auto-128
# splits early-exit per row at 8-32K real ctx, plus the 128-partial
# combine. The probe (tools/ar/r231_splittime.py) times
# mx_paged_decode at a ctx x nsplit ladder on banked payload
# geometries: const(ns), slope(ns), recoverable(sl) = auto - best_ns.
#
# One payload per format bank: r224a=mxk8v4 r229a=mxfp8 r229b=mxfp4.
# Bank -> /root/build/p647/r231a/
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert27
FORK_SHA=d39ab5a26bf737d006d47a3f3125a404dca6e928
SOMD5=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r231a
CNAME=arc-r231a
PROBE=$R/r231_splittime.py

fail() { echo "R231A_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert27_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
SO_MD5=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libmqa_logits*.so'); print(hashlib.md5(open(ps[0],'rb').read()).hexdigest())\"" 2>/dev/null) || SO_MD5=""
[ "$SO_MD5" = "$SOMD5" ] || { fail "image .so md5 $SO_MD5 != build25 pin"; exit 1; }
[ -f "$PROBE" ] || { fail "probe missing ($PROBE)"; exit 1; }
PICKS=()
for bank in r224a r229a r229b; do
  f=$(ls "$R/p647/$bank"/mxkv_ab.*.pt 2>/dev/null | head -1)
  [ -n "$f" ] || { fail "payload bank $bank empty"; exit 1; }
  PICKS+=("$f")
done
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R231A refuses passed (fork_sha=$FORK_SHA so_md5=$SO_MD5 picks=${PICKS[*]})"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "R231A: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R231A: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r231a.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R231A: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R231A: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

# ---- 1. arc window (idle GPU for clean device timing) ----
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5
mkdir -p "$P"

# ---- 2. bare container: wheel + GPU, no serve ----
docker run -d --name "$CNAME" \
  --privileged --network host --ipc host \
  --shm-size 16g \
  -v /root/build/neo_cache:/root/.neo_cache \
  --entrypoint /bin/bash "$IMG" -c 'sleep infinity' \
  || { fail "bare container boot failed"; exit 1; }
sleep 5
docker cp "$PROBE" "$CNAME:/tmp/r231_splittime.py"
i=0
for f in "${PICKS[@]}"; do
  # distinct names: the banks share basenames (mxkv_ab.1030.10.pt in
  # both r229a and r229b) and a basename cp silently overwrites
  docker cp "$f" "$CNAME:/tmp/pick$i.pt"
  i=$((i+1))
done

# ---- 3. run the ladder (payload fmt field wins; r224a=mxk8v4
# r229a=mxfp8 r229b=mxfp4) ----
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231_splittime.py \
  /tmp/r231_out.txt \
  /tmp/pick0.pt /tmp/pick1.pt /tmp/pick2.pt \
  > "$P/probe.log" 2>&1
RC=$?
tail -40 "$P/probe.log"
docker cp "$CNAME:/tmp/r231_out.txt" "$P/r231_splittime.txt" >/dev/null 2>&1 || true
docker rm -f "$CNAME" >/dev/null 2>&1 || true
cp "$P/probe.log" "$P/probe_copy.log" 2>/dev/null || true
echo "R231A_PROBE_RC=$RC"
echo "R231A_DONE $(date +%H:%M:%S)"
exit 0
