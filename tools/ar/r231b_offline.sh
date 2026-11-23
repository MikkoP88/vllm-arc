#!/bin/bash
# r231b_offline.sh -- P6.47 r231b offline gate on the build27 wheel
# (Q pre-stage), bare cert27 container (r231a pattern -- kernel timing
# and replay need only the wheel + an idle GPU, no serve boot).
#
# Gates, in order:
#   1. ORACLE  build25-vs-build27 bitwise: the patch is value-neutral
#      by construction; any bit flip is a defect (behavior-
#      preservation law). 4 payloads per format bank (12 cells).
#   2. BANDS   r229 calibrated replay cells (r229_tools/r229_replay_fmt.py)
#      on the SAME 12 payloads with the build27 .so installed -- every
#      cell must PASS (BAND_DMAX 2.0e-1, BAND_MISM 16, BAND_ULP 64.0,
#      BAND_TRITON 1e-2, BAND_CROSS 4.0e-1).
#   3. LADDER  tools r231_splittime.py ctx x nsplit ladder with the
#      build27 .so, same payload picks as r231a (one per format bank)
#      -- direct table comparison vs the banked r231a build25 ladder.
#
# Bank -> /root/build/p647/r231b/
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert27
FORK_SHA=d39ab5a26bf737d006d47a3f3125a404dca6e928
# build25 pin = the cert27 image's libattn_kernels_xe_2.so (the .so that
# actually carries the MX paged-decode kernels -- 37 mx_paged_decode
# dynsyms; libmqa_logits carries none, the r231b-1 misextraction lesson)
SOMD5_25=2d73ff7a3c2aa2caab86179288ca3711
P=$R/p647/r231b
CNAME=arc-r231b
ORACLE=$P/r231b_oracle.py
SCORER=$R/r229_tools/r229_replay_fmt.py
ISOLIB=$R/r225_tools/r225j_iso.py
SPLITTIME=$R/r231_splittime.py
SO27="$P/libattn_kernels_xe_2.so"
FA27="$P/_vllm_fa2_C.abi3.so"
# walk-fit law (r231b-2): banks carry idx 1-17 at sl~8198 over a
# 96-block window (6144-token capacity) -> the fp64 walk in r225j_iso
# IndexErrors; idx 18+ sit at sl~5462-5473 (86 blocks, FIT). Probe:
# tools/ar/r231b_pickprobe.py. The LADDER keeps r231a-identical inputs
# (first file per bank -- splittime synthesizes its own pools, so the
# OVER property is irrelevant there and the table stays comparable).
IDX="18 24 36 48"

fail() { echo "R231B_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert27_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
SO_MD5=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels*.so'); print(hashlib.md5(open(ps[0],'rb').read()).hexdigest())\"" 2>/dev/null) || SO_MD5=""
[ "$SO_MD5" = "$SOMD5_25" ] || { fail "image .so md5 $SO_MD5 != build25 pin"; exit 1; }
[ -f "$SO27" ] || { fail "build27 .so missing: $SO27 (run r231b_build.sh)"; exit 1; }
[ -f "$FA27" ] || { fail "build27 fa2 stub missing: $FA27 (run r231b_build.sh)"; exit 1; }
SO27_MD5=$(md5sum "$SO27" | cut -d' ' -f1)
[ "$SO27_MD5" != "$SOMD5_25" ] || { fail "build27 .so md5 == build25 (stale build?)"; exit 1; }
for f in "$ORACLE" "$SCORER" "$ISOLIB" "$SPLITTIME"; do
  [ -f "$f" ] || { fail "tool not shipped ($f)"; exit 1; }
done
PICKS=()
for bank in r224a:2447:mxk8v4 r229a:1030:mxfp8 r229b:1030:mxfp4; do
  b=${bank%%:*}; rest=${bank#*:}; pre=${rest%%:*}; fmt=${rest##*:}
  for i in $IDX; do
    f=$R/p647/$b/mxkv_ab.$pre.$i.pt
    [ -f "$f" ] || { fail "payload $f missing"; exit 1; }
    PICKS+=("$f")
  done
done
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R231B refuses passed (fork_sha=$FORK_SHA so25=$SO_MD5 so27=$SO27_MD5 picks=${#PICKS[@]})"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  # own container dies FIRST -- the in-flight guard must not match it
  # (the r231b-2 abort left W14 down when the grep saw its own corpse)
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "R231B: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  echo "R231B: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r231b.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R231B: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R231B: WARN W14 not 200 after restore poll (last=$c)"
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
SO_PATH=$(docker exec "$CNAME" /opt/venv/bin/python -c \
  "import glob; print(glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels*.so')[0])")
FA_PATH=$(docker exec "$CNAME" /opt/venv/bin/python -c \
  "import glob; print(glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_vllm_fa2_C.abi3.so')[0])")
docker exec "$CNAME" md5sum "$SO_PATH" "$FA_PATH"
docker cp "$ORACLE" "$CNAME:/tmp/r231b_oracle.py"
docker cp "$SCORER" "$CNAME:/tmp/r229_replay_fmt.py"
docker cp "$ISOLIB" "$CNAME:/tmp/r225j_iso.py"
docker cp "$SPLITTIME" "$CNAME:/tmp/r231_splittime.py"
i=0
for f in "${PICKS[@]}"; do
  # distinct indexed names: banks share basenames (r231a5 law)
  docker cp "$f" "$CNAME:/tmp/r231b_pick$i.pt"
  i=$((i+1))
done
# ladder inputs = r231a-identical first-file-per-bank (table law)
LADD=()
for bank in r224a r229a r229b; do
  f=$(ls "$R/p647/$bank"/mxkv_ab.*.pt 2>/dev/null | head -1)
  [ -n "$f" ] || { fail "ladder pick missing for $bank"; exit 1; }
  LADD+=("$f")
done
docker cp "${LADD[0]}" "$CNAME:/tmp/r231b_l0.pt"
docker cp "${LADD[1]}" "$CNAME:/tmp/r231b_l1.pt"
docker cp "${LADD[2]}" "$CNAME:/tmp/r231b_l2.pt"

# ---- 3. oracle A: build25 (image .so) ----
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231b_oracle.py \
  save /tmp/outA $(for j in $(seq 0 11); do echo -n "/tmp/r231b_pick$j.pt "; done) \
  > "$P/oracleA.log" 2>&1
grep -c "R231B_SAVED" "$P/oracleA.log" || true

# ---- 4. swap in build27 (kernel lib + pybind stub) ----
docker cp "$SO27" "$CNAME:$SO_PATH"
docker cp "$FA27" "$CNAME:$FA_PATH"
SW_MD5=$(docker exec "$CNAME" md5sum "$SO_PATH" | cut -d' ' -f1)
[ "$SW_MD5" = "$SO27_MD5" ] || { fail "in-container swap md5 $SW_MD5 != $SO27_MD5"; exit 1; }
FA27_MD5=$(md5sum "$FA27" | cut -d' ' -f1)
FA_SW=$(docker exec "$CNAME" md5sum "$FA_PATH" | cut -d' ' -f1)
[ "$FA_SW" = "$FA27_MD5" ] || { fail "in-container fa2 swap md5 $FA_SW != $FA27_MD5"; exit 1; }
echo "R231B swap verified (build27 md5=$SW_MD5 fa2=$FA_SW)"

# ---- 5. oracle B + bitwise compare ----
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231b_oracle.py \
  save /tmp/outB $(for j in $(seq 0 11); do echo -n "/tmp/r231b_pick$j.pt "; done) \
  > "$P/oracleB.log" 2>&1
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231b_oracle.py \
  cmp /tmp/outA /tmp/outB | tee "$P/oracle_cmp.txt"
grep -q "R231B_ORACLE PASS" "$P/oracle_cmp.txt" \
  || { fail "bitwise oracle FAILED (see $P/oracle_cmp.txt)"; exit 1; }

# ---- 6. r229 calibrated band cells on build27 ----
: > "$P/r229_cells.log"
CELL_RC=0
for j in $(seq 0 11); do
  docker exec "$CNAME" /opt/venv/bin/python /tmp/r229_replay_fmt.py \
    /tmp/r231b_pick$j.pt >> "$P/r229_cells.log" 2>&1 || CELL_RC=1
done
NPASS=$(grep -c "R229F_CELL .* PASS" "$P/r229_cells.log" || true)
NFAIL=$(grep -c "R229F_CELL .* FAIL" "$P/r229_cells.log" || true)
echo "R231B_R229CELLS pass=$NPASS fail=$NFAIL"
[ "$NFAIL" = "0" ] && [ "$NPASS" -ge 12 ] \
  || { fail "r229 band cells failed ($NPASS/$((NPASS+NFAIL)))"; exit 1; }
grep -E "NATIVE_VS_WALK|NATIVE_VS_QMATCHED" "$P/r229_cells.log" \
  | sort -t= -k2 -g | tail -4

# ---- 7. split ladder on build27 (r231a-identical inputs: first file
# per bank = l0 r224a mxk8v4, l1 r229a mxfp8, l2 r229b mxfp4) ----
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231_splittime.py \
  /tmp/r231b_ladder.txt \
  /tmp/r231b_l0.pt /tmp/r231b_l1.pt /tmp/r231b_l2.pt \
  > "$P/ladder.log" 2>&1
RC=$?
tail -25 "$P/ladder.log"
docker cp "$CNAME:/tmp/r231b_ladder.txt" "$P/r231b_ladder.txt" >/dev/null 2>&1 || true
docker cp "$CNAME:/tmp/outA" "$P/outA" >/dev/null 2>&1 || true
docker cp "$CNAME:/tmp/outB" "$P/outB" >/dev/null 2>&1 || true
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "R231B_LADDER_RC=$RC"
echo "R231B_DONE $(date +%H:%M:%S)"
exit 0
