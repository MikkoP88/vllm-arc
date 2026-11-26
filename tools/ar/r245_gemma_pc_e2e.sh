#!/bin/bash
# r245_gemma_pc_e2e.sh -- P6.47 P-C e2e: the gemma-4-31b MX-KV grid
# (r240b gkv_m4/gkv_m8/gkv_mk shape) on cert28 + the build6 (r245a)
# arc7mx wheel + the P-C2 engine overlay. The r240b cells booted,
# prefilled, and died at the FIRST NATIVE DECODE (hd512
# kernel-instantiation refusal on the 10 full-attention layers);
# P-C1's dispatch widening + r242b V-slice fix instantiate D=512, and
# P-C2 routes the 50 sliding layers (D=256, W=1024) onto the masked
# twin with the window pair (W-1,0) -- prefill rides the stock varlen
# window via _flash_attn_varlen. This leg is the e2e authority for
# the whole P-C increment (offline oracles: r242/r243/r244 banks).
# RE-RUN AUTHORITY (r245a): the first run (build5 wheel, 2026-11-26)
# booted all three cells healthy (pc2_windowed_impl=100, KV pools
# 105669/54426/71856, window=1024 native calls at capture) and then
# DIED AT THE FIRST REAL REQUEST -- the gemma hybrid KV manager banks
# the D=512 full_attention group a PAGE-128 pool, and the first real
# decode hit "mx_paged_decode: block_size 128 not yet compiled"
# (serve traceback engine.py:1261 -> native_decode.py:135). All 98
# faces banked n_ok=0 (Connection refused) behind GREEN CONC_OK
# lines -- the existence-only-green-gate lesson, now fixed here: the
# warmup curl is the earliest real-request probe (its failure = the
# engine death), and every banked face JSON gates on n_ok>0
# semantically (n_ok=0 banks CONC_DEAD + serve tail and sets
# CELL_DEAD for the rest of the cell). The failure evidence is
# preserved at /root/build/p647/r245_pre_r245a_failrun; the re-run
# caller CLEARS /root/build/p647/r245 first (the resume law skips
# only genuinely-passed faces now, but a clean bank is the honest
# matrix). Build6 = build5 + the page-128 admission (dispatch branch
# head_size==512 && block_size==128 riding the same _512/_16
# sub-tile frame; mainloop page indirection is runtime-generic).
# Layering (the r213 e2e law): ARC_MXKV_WHEEL_DIR mounts the wheels,
# ARC_MXKV_WHEEL pip-installs build6 over the image venv at boot,
# ARC_MXKV_OVERLAY mounts the P-C2 mx_kv package over site-packages
# (the r243-staged tree, md5-verified against the committed PR #148
# bytes at ship time). Refuses: image cert28 + build6 wheel md5
# (sidecar .r245a.md5 written by r245a_build.sh) + overlay
# engine/native md5s + p139_conc.py presence. Posture per banked
# r241 law: ARC_LMC=0 and ANY LMCache line in a serve log aborts the
# leg (exit 3). Faces: the r240b gkv grid byte-exact (1K/8K/32K x
# conc {1,2,4,8} + t1.0 solos at maxlen 40960); infeasible points
# bank as KV_CAP rows; a boot refusal banks as BOOT_FAIL with the
# serve log, a warmup death banks as WARMUP_DEAD, a mid-cell death
# banks CONC_DEAD faces (vmatrix law: no abort, matrix continues).
# Telemetry banks the P-C2 windowed-impl boot line + the MXKV native
# call lines (window= rides them) per cell. W14 quiesce before,
# EXIT-trap fullres restore after (r24x law). Artifacts ->
# /root/build/p647/r245/<cell>/. Exit 0 = all three cells completed
# (faces may individually bank KV_CAP/TIMEOUT/DEAD).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
P=$R/p647/r245
MT=600
CNAME=arc-r245
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEEL=$WHEELDIR/$WHEELNAME
OVERLAY=$R/p243/work/vllm_gaudi/xpu/patches/mx_kv
ENGINE_MD5_EXPECT=789b229582c257f213d51157ace31650
NATIVE_MD5_EXPECT=8a858aa052a960d9030ec6fa665a64e1
mkdir -p "$P"
FAILED=""
CELL_DEAD=""

fail() { echo "R245_ABORT: $*"; }

face_nok() {  # <json path> -> n_ok (0 on any parse trouble)
  python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$1" 2>/dev/null || echo 0
}

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
[ -f "$WHEEL" ] || { fail "build6 wheel missing ($WHEEL)"; exit 1; }
WMD5=$(md5sum "$WHEEL" | cut -d" " -f1)
WMD5_EXPECT=$(cat "$WHEEL.r245a.md5" 2>/dev/null | cut -d" " -f1)
[ -n "$WMD5_EXPECT" ] || { fail "build6 md5 sidecar missing (r245a_build.sh not completed)"; exit 1; }
[ "$WMD5" = "$WMD5_EXPECT" ] || { fail "wheel md5 $WMD5 != build6 $WMD5_EXPECT"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { fail "overlay engine.py missing ($OVERLAY)"; exit 1; }
EMD5=$(md5sum "$OVERLAY/engine.py" | cut -d" " -f1)
NMD5=$(md5sum "$OVERLAY/native_decode.py" | cut -d" " -f1)
[ "$EMD5" = "$ENGINE_MD5_EXPECT" ] || { fail "overlay engine.py md5 $EMD5 != PR-148 $ENGINE_MD5_EXPECT"; exit 1; }
[ "$NMD5" = "$NATIVE_MD5_EXPECT" ] || { fail "overlay native_decode.py md5 $NMD5 != PR-148 $NATIVE_MD5_EXPECT"; exit 1; }
[ -f "$R/p139_conc.py" ] || { fail "p139_conc.py missing"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R245 refuses passed (img=$IMG wheel=$WMD5 overlay_engine=$EMD5)"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R245_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (docker start + serve_only can never
  # restore the actor-recreated radiance-test -- r185b/r186 law)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r245.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    docker rm -f "$1" >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "R245_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec $CNAME grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r245:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r245:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r245:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {  # parse "KV cache size: N tokens" from the booted engine
  local v
  v=$(docker exec $CNAME grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
  v=${v#KV cache size: }; v=${v% tokens}; v=${v//,/}
  echo "${v:-0}"
}

group() {  # ctx conc tag [temp] -- runs one concurrency group, banks the JSON
  local CTX=$1 NC=$2 TAG=$3 TP=${4:-0.0}
  if [ -n "$CELL_DEAD" ]; then
    echo "KV_SKIP_ENGINE_DEAD $CELLN $TAG" | tee -a "$CD/conc.log"
    return 0
  fi
  # resume law: only a GENUINELY-PASSED face skips (n_ok>0 in the
  # banked JSON -- an n_ok=0 JSON from a dead-engine run is dead
  # evidence, not a completed face; r245 first-run lesson)
  if [ -f "$CD/p139_conc_$TAG.json" ]; then
    local BNOK
    BNOK=$(face_nok "$CD/p139_conc_$TAG.json")
    if [ "${BNOK:-0}" -gt 0 ]; then
      echo "CONC_SKIP_$TAG (banked n_ok=$BNOK)" | tee -a "$CD/conc.log"
      return 0
    fi
  fi
  local need=$(( NC * (CTX + MT + 64) ))
  if [ "$POOL" -gt 0 ] && [ "$need" -gt "$POOL" ]; then
    echo "KV_CAP $CELLN c$CTX n$NC POOL=$POOL NEED=$need" | tee -a "$CD/conc.log"
    return 0
  fi
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" "$TP"
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec $CNAME test -f /root/p139_conc_$TAG.json && break
    docker inspect $CNAME >/dev/null 2>&1 || { CELL_DEAD=1; break; }
    sleep 4; i=$((i+1))
  done
  if ! docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s dead=$CELL_DEAD)" | tee -a "$CD/conc.log"
    [ -n "$CELL_DEAD" ] && docker exec $CNAME tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp $CNAME:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  # n_ok semantic gate (existence-only-green-gate law, r245 lesson):
  # a banked JSON with n_ok=0 is a DEAD face, not a pass -- the first
  # run banked 98 such JSONs behind green CONC_OK lines while the
  # engine had died at its first real request (Connection refused per
  # slot). A dead face banks the serve tail and kills the cell's rest.
  local NOK NFAIL
  NOK=$(face_nok "$CD/p139_conc_$TAG.json")
  NFAIL=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_fail'])" "$CD/p139_conc_$TAG.json" 2>/dev/null || echo "?")
  if [ "${NOK:-0}" -le 0 ]; then
    echo "CONC_DEAD_$TAG (n_ok=${NOK:-?} n_fail=${NFAIL:-?})" | tee -a "$CD/conc.log"
    CELL_DEAD=1
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  echo "CONC_OK_$TAG n_ok=$NOK" | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2=served $3="kvface:<ctx:conc[:temp] ..." $4..=env
  CELLN=$1; SERVED=$2; local SPEC=$3; shift 3
  local KVFACES=${SPEC#*:}
  CD=$P/$CELLN
  # resume law: only a COMPLETED cell (outcome CONC_OK) skips whole;
  # BOOT_FAIL/WARMUP_DEAD/CELL_DEAD outcomes re-run (their cells are
  # not completed faces -- the matrix wants them measured)
  if [ -f "$CD/outcome.txt" ] && [ "$(cat "$CD/outcome.txt")" = "CONC_OK" ]; then
    echo "=== [r245:$CELLN] CELL_SKIP (outcome CONC_OK banked) ==="
    SERVED=""
    return 0
  fi
  mkdir -p "$CD"
  echo "=== [r245:$CELLN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R245_BOOT_FAIL_$CELLN -- banking log, matrix continues"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3; SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r245_$CELLN $(date +%H:%M:%S)"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
  echo "lmc_lines=$(gcount 'LMCache')" >> "$CD/posture.txt"
  echo "pc2_windowed_impl=$(gcount 'MXKV P-C2 windowed impl')" >> "$CD/posture.txt"
  POOL=$(kv_pool)
  echo "kv_pool_tokens=$POOL" >> "$CD/posture.txt"
  [ "$POOL" = "0" ] && echo "POOL_PARSE_FAIL -- sweep runs ungated" >> "$CD/posture.txt"
  cat "$CD/posture.txt"
  # r241 posture law: ANY LMCache line on an MX lane aborts the leg
  LMCN=$(gcount "LMCache")
  if [ "${LMCN:-0}" -gt 0 ]; then
    echo "R245_POSTURE_DEFECT (LMCache lines=$LMCN on MX lane) -- aborting"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
    sleep 8; docker rm -f $CNAME >/dev/null 2>&1
    exit 3
  fi
  sleep 5
  # warmup-death detector (r245 first-run lesson): the engine died at
  # its FIRST REAL REQUEST (page-128 refusal) AFTER boot_wait's
  # healthy 200 -- /health never runs decode, so the warmup
  # completion IS the earliest real-request probe. A non-200 warmup
  # is the engine death with the traceback fresh in the serve log.
  local WRC
  WRC=$(curl -s -m 120 -o "$CD/warmup.json" -w '%{http_code}' \
    http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}")
  if [ "$WRC" != "200" ]; then
    echo "WARMUP_DEAD_$CELLN (http=$WRC) -- banking log, matrix continues"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
    sleep 8; docker rm -f $CNAME >/dev/null 2>&1
    verify_removed $CNAME || true
    FAILED="$FAILED $CELLN"
    echo "WARMUP_DEAD" > "$CD/outcome.txt"
    sleep 3; SERVED=""
    return 0
  fi
  sleep 3
  docker cp $R/p139_conc.py $CNAME:/root/p139_conc.py
  CELL_DEAD=""

  for f in $KVFACES; do
    # local-multiassign law: one binding per command
    local fx rest fn ft tsuf
    fx=${f%%:*}; rest=${f#*:}; fn=${rest%%:*}; ft=${rest#*:}
    [ "$ft" = "$rest" ] && ft=0.0
    tsuf="_t${ft}"
    [ "$ft" = "0.0" ] && tsuf=""
    group "$fx" "$fn" "${CELLN}_c${fx}_n${fn}${tsuf}" "$ft"
  done

  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1
  verify_removed $CNAME || return 1
  # telemetry: the native-call lines carry window= (P-C2) per fmt
  grep -h "MXKV native call" "$CD/serve.log" 2>/dev/null | tail -20 > "$CD/native_telemetry.txt" || true
  if [ -n "$CELL_DEAD" ]; then
    echo "CELL_DEAD" > "$CD/outcome.txt"
    FAILED="$FAILED $CELLN"
  else
    echo "CONC_OK" > "$CD/outcome.txt"
  fi
  sleep 3
  echo "=== [r245:$CELLN] done $(date +%H:%M:%S) outcome=$(cat "$CD/outcome.txt") ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R245_WARN_PORT8000_STILL_HELD"
sleep 10

GM="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"
GKVFACES="kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0"

mkdir -p $P
cp -f $R/p139_conc.py $P/p139_conc.py 2>/dev/null || true

run_cell gkv_m4 gemma4-31b "$GKVFACES" $GM ARC_QUANT=fp8 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1
run_cell gkv_m8 gemma4-31b "$GKVFACES" $GM ARC_QUANT=fp8 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1
run_cell gkv_mk gemma4-31b "$GKVFACES" $GM ARC_QUANT=fp8 ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1

echo "R245_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
