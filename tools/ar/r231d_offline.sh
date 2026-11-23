#!/bin/bash
# r231d_offline.sh -- P6.47 r231d offline gate on the build28 wheel
# (seqused-tightened splits), bare cert27 container (r231b pattern --
# kernel timing and replay need only the wheel + an idle GPU, no serve
# boot).
#
# Gates, in order:
#   1. ORACLE  the clamp == the env pin, at an identical partition:
#      G1 build25[ns=C] == build28[auto]   bitwise (the clamp selects
#         exactly C and the execution at C is byte-identical)
#      G2 build25[ns=C] == build28[ns=C]   bitwise (pin path unchanged)
#      G3 build25[auto] == build28[auto]   bitwise on capacity-path
#         picks (sl>=8192: blocks/4>=128, the clamp is a no-op --
#         record-path byte identity)
#      C is derived per pick by the oracle (min(128, max(1, blocks/4)))
#      and must be uniform across the 12 walk-fit picks.
#   2. BANDS   r229 calibrated replay cells on the SAME 12 walk-fit
#      payloads with the build28 .so installed -- every cell must PASS
#      (BAND_DMAX 2.0e-1, BAND_MISM 16, BAND_ULP 64.0, BAND_TRITON
#      1e-2, BAND_CROSS 4.0e-1).
#   3. LADDER  r231_splittime.py ctx x nsplit ladder with the build28
#      .so, r231a-identical inputs (first file per bank) -- the AUTO
#      rows below 8K ctx carry the tightening (2K-class: expect the
#      ns32-class time, not the capacity 428-448); 8K/32K rows must
#      hold the banked r231a values within dice.
#
# Bank -> /root/build/p647/r231d/
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert27
FORK_SHA=d39ab5a26bf737d006d47a3f3125a404dca6e928
# build25 pin = the cert27 image's libattn_kernels_xe_2.so (the .so that
# actually carries the MX paged-decode kernels -- 37 mx_paged_decode
# dynsyms; libmqa_logits carries none, the r231b-1 misextraction lesson)
SOMD5_25=2d73ff7a3c2aa2caab86179288ca3711
P=$R/p647/r231d
CNAME=arc-r231d
ORACLE=$P/r231d_oracle.py
SCORER=$R/r229_tools/r229_replay_fmt.py
ISOLIB=$R/r225_tools/r225j_iso.py
SPLITTIME=$R/r231_splittime.py
SO28="$P/libattn_kernels_xe_2.so"
FA28="$P/_vllm_fa2_C.abi3.so"
# walk-fit law (r231b-2): idx 18+ sit at sl~5462-5473 (86 blocks, FIT
# for the fp64 walk); idx 1 sits at sl~8198 (513 blocks = the CAPACITY
# path, blocks/4 >= 128 -> clamp no-op) and walks zero-filled storage
# past the 96-block window -- deterministic on both wheels (G3 is a
# bitwise gate; the walked bytes need only be identical, not real).
IDX="18 24 36 48"
IDX_CAP="1"

fail() { echo "R231D_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert27_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
SO_MD5=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels*.so'); print(hashlib.md5(open(ps[0],'rb').read()).hexdigest())\"" 2>/dev/null) || SO_MD5=""
[ "$SO_MD5" = "$SOMD5_25" ] || { fail "image .so md5 $SO_MD5 != build25 pin"; exit 1; }
[ -f "$SO28" ] || { fail "build28 .so missing: $SO28 (run r231d_build.sh)"; exit 1; }
[ -f "$FA28" ] || { fail "build28 fa2 stub missing: $FA28 (run r231d_build.sh)"; exit 1; }
SO28_MD5=$(md5sum "$SO28" | cut -d' ' -f1)
[ "$SO28_MD5" != "$SOMD5_25" ] || { fail "build28 .so md5 == build25 (stale build?)"; exit 1; }
for f in "$ORACLE" "$SCORER" "$ISOLIB" "$SPLITTIME"; do
  [ -f "$f" ] || { fail "tool not shipped ($f)"; exit 1; }
done
PICKS=()
# walk-fit picks first (indices 0-11), capacity picks after (12-14) --
# the WF_ARGS/CAP_ARGS split below indexes by that order
for bank in r224a:2447:mxk8v4 r229a:1030:mxfp8 r229b:1030:mxfp4; do
  b=${bank%%:*}; rest=${bank#*:}; pre=${rest%%:*}; fmt=${rest##*:}
  for i in $IDX; do
    f=$R/p647/$b/mxkv_ab.$pre.$i.pt
    [ -f "$f" ] || { fail "payload $f missing"; exit 1; }
    PICKS+=("$f")
  done
done
for bank in r224a:2447:mxk8v4 r229a:1030:mxfp8 r229b:1030:mxfp4; do
  b=${bank%%:*}; rest=${bank#*:}; pre=${rest%%:*}; fmt=${rest##*:}
  for i in $IDX_CAP; do
    f=$R/p647/$b/mxkv_ab.$pre.$i.pt
    [ -f "$f" ] || { fail "payload $f missing"; exit 1; }
    PICKS+=("$f")
  done
done
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R231D refuses passed (fork_sha=$FORK_SHA so25=$SO_MD5 so28=$SO28_MD5 picks=${#PICKS[@]})"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  # own container dies FIRST -- the in-flight guard must not match it
  # (the r231b-2 abort left W14 down when the grep saw its own corpse)
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "R231D: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  echo "R231D: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r231d.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R231D: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R231D: WARN W14 not 200 after restore poll (last=$c)"
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
docker cp "$ORACLE" "$CNAME:/tmp/r231d_oracle.py"
docker cp "$SCORER" "$CNAME:/tmp/r229_replay_fmt.py"
docker cp "$ISOLIB" "$CNAME:/tmp/r225j_iso.py"
docker cp "$SPLITTIME" "$CNAME:/tmp/r231_splittime.py"
i=0
for f in "${PICKS[@]}"; do
  # distinct indexed names: banks share basenames (r231a5 law)
  docker cp "$f" "$CNAME:/tmp/r231d_pick$i.pt"
  i=$((i+1))
done
# picks 0-11 = walk-fit (G1/G2/bands); picks 12-14 = capacity (G3)
WF_ARGS=""; CAP_ARGS=""
for j in $(seq 0 11); do WF_ARGS="$WF_ARGS /tmp/r231d_pick$j.pt"; done
for j in $(seq 12 14); do CAP_ARGS="$CAP_ARGS /tmp/r231d_pick$j.pt"; done
# ladder inputs = r231a-identical first-file-per-bank (table law)
LADD=()
for bank in r224a r229a r229b; do
  f=$(ls "$R/p647/$bank"/mxkv_ab.*.pt 2>/dev/null | head -1)
  [ -n "$f" ] || { fail "ladder pick missing for $bank"; exit 1; }
  LADD+=("$f")
done
docker cp "${LADD[0]}" "$CNAME:/tmp/r231d_l0.pt"
docker cp "${LADD[1]}" "$CNAME:/tmp/r231d_l1.pt"
docker cp "${LADD[2]}" "$CNAME:/tmp/r231d_l2.pt"

# ---- 3. slinfo: derive C, refuse if the 12 walk-fit picks disagree ----
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  slinfo $(for j in $(seq 0 14); do echo -n "/tmp/r231d_pick$j.pt "; done) \
  | tee "$P/slinfo.log"
# C field = the row-uniform clamp value (MIXED{..} marks per-row
# disagreement -- a single ns pin could not reproduce auto there)
C_VALS=$(grep "R231D_INFO" "$P/slinfo.log" | head -12 \
  | sed 's/.* C=//;s/ .*//' | sort -u)
[ "$(echo "$C_VALS" | wc -l)" = "1" ] \
  || { fail "walk-fit picks disagree on C: $C_VALS"; exit 1; }
case "$C_VALS" in
  *[!0-9]*) fail "walk-fit C is not a single number: $C_VALS"; exit 1 ;;
esac
C=$C_VALS
CAP_OK=$(grep "R231D_INFO" "$P/slinfo.log" | tail -3 | grep -c "capacity=yes" || true)
[ "$CAP_OK" = "3" ] \
  || { fail "capacity picks are not on the capacity path (C != 128)"; exit 1; }
echo "R231D clamp C=$C (uniform across 12 walk-fit picks; 3 capacity picks confirmed)"

# ---- 4. oracle A: build25 (image .so) at ns=C and auto --------------
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  save /tmp/outA_pin ns:$C $WF_ARGS > "$P/oracleA_pin.log" 2>&1
grep -c "R231D_SAVED" "$P/oracleA_pin.log" || true
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  save /tmp/outA_cap auto $CAP_ARGS > "$P/oracleA_cap.log" 2>&1
grep -c "R231D_SAVED" "$P/oracleA_cap.log" || true

# ---- 5. swap in build28 (kernel lib + pybind stub) ------------------
docker cp "$SO28" "$CNAME:$SO_PATH"
docker cp "$FA28" "$CNAME:$FA_PATH"
SW_MD5=$(docker exec "$CNAME" md5sum "$SO_PATH" | cut -d' ' -f1)
[ "$SW_MD5" = "$SO28_MD5" ] || { fail "in-container swap md5 $SW_MD5 != $SO28_MD5"; exit 1; }
FA28_MD5=$(md5sum "$FA28" | cut -d' ' -f1)
FA_SW=$(docker exec "$CNAME" md5sum "$FA_PATH" | cut -d' ' -f1)
[ "$FA_SW" = "$FA28_MD5" ] || { fail "in-container fa2 swap md5 $FA_SW != $FA28_MD5"; exit 1; }
echo "R231D swap verified (build28 md5=$SW_MD5 fa2=$FA_SW)"

# ---- 6. oracle B: build28 auto (both classes) + ns:C pin ------------
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  save /tmp/outB_auto auto $WF_ARGS > "$P/oracleB_auto.log" 2>&1
grep -c "R231D_SAVED" "$P/oracleB_auto.log" || true
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  save /tmp/outB_cap auto $CAP_ARGS > "$P/oracleB_cap.log" 2>&1
grep -c "R231D_SAVED" "$P/oracleB_cap.log" || true
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  save /tmp/outB_pin ns:$C $WF_ARGS > "$P/oracleB_pin.log" 2>&1
grep -c "R231D_SAVED" "$P/oracleB_pin.log" || true

# ---- 7. the three bitwise gates -------------------------------------
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  cmp /tmp/outA_pin /tmp/outB_auto | tee "$P/oracle_cmp_G1.txt"
grep -q "R231D_ORACLE PASS" "$P/oracle_cmp_G1.txt" \
  || { fail "G1 clamp==pin equivalence FAILED (see $P/oracle_cmp_G1.txt)"; exit 1; }
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  cmp /tmp/outA_pin /tmp/outB_pin | tee "$P/oracle_cmp_G2.txt"
grep -q "R231D_ORACLE PASS" "$P/oracle_cmp_G2.txt" \
  || { fail "G2 pin-path identity FAILED (see $P/oracle_cmp_G2.txt)"; exit 1; }
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231d_oracle.py \
  cmp /tmp/outA_cap /tmp/outB_cap | tee "$P/oracle_cmp_G3.txt"
grep -q "R231D_ORACLE PASS" "$P/oracle_cmp_G3.txt" \
  || { fail "G3 capacity-path identity FAILED (see $P/oracle_cmp_G3.txt)"; exit 1; }
echo "R231D_ORACLE_GATES PASS (G1 clamp==pin, G2 pin-path, G3 capacity)"

# ---- 8. r229 calibrated band cells on build28 ------------------------
: > "$P/r229_cells.log"
CELL_RC=0
for j in $(seq 0 11); do
  docker exec "$CNAME" /opt/venv/bin/python /tmp/r229_replay_fmt.py \
    /tmp/r231d_pick$j.pt >> "$P/r229_cells.log" 2>&1 || CELL_RC=1
done
NPASS=$(grep -c "R229F_CELL .* PASS" "$P/r229_cells.log" || true)
NFAIL=$(grep -c "R229F_CELL .* FAIL" "$P/r229_cells.log" || true)
echo "R231D_R229CELLS pass=$NPASS fail=$NFAIL"
[ "$NFAIL" = "0" ] && [ "$NPASS" -ge 12 ] \
  || { fail "r229 band cells failed ($NPASS/$((NPASS+NFAIL)))"; exit 1; }
grep -E "NATIVE_VS_WALK|NATIVE_VS_QMATCHED" "$P/r229_cells.log" \
  | sort -t= -k2 -g | tail -4

# ---- 9. split ladder on build28 (r231a-identical inputs: first file
# per bank = l0 r224a mxk8v4, l1 r229a mxfp8, l2 r229b mxfp4) ----------
docker exec "$CNAME" /opt/venv/bin/python /tmp/r231_splittime.py \
  /tmp/r231d_ladder.txt \
  /tmp/r231d_l0.pt /tmp/r231d_l1.pt /tmp/r231d_l2.pt \
  > "$P/ladder.log" 2>&1
RC=$?
tail -25 "$P/ladder.log"
docker cp "$CNAME:/tmp/r231d_ladder.txt" "$P/r231d_ladder.txt" >/dev/null 2>&1 || true
for d in outA_pin outA_cap outB_auto outB_cap outB_pin; do
  docker cp "$CNAME:/tmp/$d" "$P/$d" >/dev/null 2>&1 || true
done
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "R231D_LADDER_RC=$RC"
echo "R231D_DONE $(date +%H:%M:%S)"
exit 0
