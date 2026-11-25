#!/bin/bash
# r241_mxkv_lmcfree_cert28.sh -- P6.47 posture-correction leg: the
# r240b qwen MX-KV cells (kvm4/kvm8/kvmk) re-run on cert28 with the
# CERTIFIED LMC-free MX posture (ARC_LMC=0 explicit).
#
# WHY THIS LEG EXISTS (2026-11-25 root-cause, banked evidence):
#   r240b_conc_cert28.sh:249 defines QF8 with ARC_LMC=1 and the kvm
#   cells inherit it -- forcing LMCache ON over the MX lanes against
#   the certified posture (serve_arc_b70.sh auto-off) and the boot's
#   own WARN (r168 banked essay-rehit REP-LOOP garble on e2m1-V
#   lanes). All three banked r240b boots carry the WARN. The r240b
#   verdict then read the resulting 32K-n4 engine deaths as an
#   intrinsic packed-KV stability boundary. The banked evidence
#   contradicts that reading:
#     - r237 m4q/m8q/mkq (same three MX lanes, LMC-free): 0 LMCache
#       lines in serve.log, clean sustained n8-ss.
#     - r240b kvm8 (e4m3-V, LMC=1): 1206 store lines, survived 22/22
#       (the r176 store-tax lane).
#     - r240b kvm4/kvmk (e2m1-V, LMC=1): 274/306 store lines, both
#       storing seconds before their wedges (kvm4 RPC watchdog
#       TimeoutError; kvmk UR_RESULT_ERROR_DEVICE_LOST) -- the
#       r168 e2m1-V x LMCache interaction class, escalated.
#   This leg re-runs the EXACT r240b kvm grids (same image, same
#   MT=600 sustained-decode protocol, same face list) under ARC_LMC=0
#   to discriminate: if the lanes survive LMC-free, the r240b
#   "stability boundary" finding is retracted as a harness posture
#   defect and the packed-KV capacity lanes (mxfp4-KV 1.81x, mxk8v4
#   1.26x pool) reopen for >=32K deployment.
#
# GATE LAW FIX (existence-only-green-gate memory): r240b's group()
# polled test -f and printed CONC_OK over dead engines. r241's gate
# parses n_ok/n_fail from the banked JSON and health-discriminates:
#   n_fail==0 && n_ok>0        -> CONC_OK_<tag> (real)
#   n_fail>0 && health alive   -> CONC_PARTIAL_<tag> (recorded, cell continues)
#   n_fail>0 && health dead    -> CONC_ENGINE_DEAD_<tag> + CELL_DEAD
#   (rest of the cell banks as KV_SKIP_ENGINE_DEAD, no vacuous faces)
# The completion line carries the recomputed census (real vs dead).
#
# POSTURE REFUSE: this leg's purpose IS the posture -- after each
# boot, serve_full.log must contain ZERO "LMCache" lines (the r237
# clean-cell signature). Any LMCache line aborts the leg loudly
# (exit 3; W14 restored by the EXIT trap).
#
# 3 cells; ETA ~2-3.5 h. Artifacts -> /root/build/p647/r241/<cell>/.
# p139_conc.py v2, KV_CAP gate from the booted pool, resume law,
# W14 entry quiesce + EXIT-trap restore (r240b shapes byte-carried).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_MQA=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r241
MT=600
mkdir -p "$P"
FAILED=""
CELL_DEAD=""

fail() { echo "R241_ABORT: $*"; }

# ---- refuses (r239/r240b pattern) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert28_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
read -r SO_A SO_M <<< "$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; vp=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/')[0]; print(hashlib.md5(open(vp+'libattn_kernels_xe_2.so','rb').read()).hexdigest(), hashlib.md5(open(glob.glob(vp+'libmqa_logits*.so')[0],'rb').read()).hexdigest())\"" 2>/dev/null)" || true
[ "${SO_A:-}" = "$SO_ATTN" ] || { fail "image libattn md5 ${SO_A:-?} != build28 pin"; exit 1; }
[ "${SO_M:-}" = "$SO_MQA" ] || { fail "image libmqa md5 ${SO_M:-?} != pin"; exit 1; }
[ -f "$R/p139_conc.py" ] || { fail "p139_conc.py missing"; exit 1; }
[ -f "$R/serve_arc_b70.sh" ] || { fail "serve_arc_b70.sh missing"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
[ -d /models/qwen3.8-27b-fp8 ] || { fail "model dir /models/qwen3.8-27b-fp8 missing (docker -v empty-dir trap)"; exit 1; }
echo "R241 refuses passed (fork_sha=$FORK_SHA libattn=$SO_A libmqa=$SO_M)"

w14_restore() {
  docker rm -f arc-r241 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r241.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R241: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R241: WARN W14 not 200 after restore poll (last=$c)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    docker rm -f "$1" >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "R241_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-r241 grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-r241 >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r241:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r241 pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r241:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r241:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {
  local v
  v=$(docker exec arc-r241 grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
  v=${v#KV cache size: }; v=${v% tokens}; v=${v//,/}
  echo "${v:-0}"
}

group() {  # ctx conc tag [temp] -- one face; semantic gate (n_ok/n_fail)
  local CTX=$1 NC=$2 TAG=$3 TP=${4:-0.0}
  if [ -n "$CELL_DEAD" ]; then
    echo "KV_SKIP_ENGINE_DEAD $CELLN $TAG" | tee -a "$CD/conc.log"
    return 0
  fi
  # resume law: a banked face JSON is done (idempotent restart)
  if [ -f "$CD/p139_conc_$TAG.json" ]; then
    echo "CONC_SKIP_$TAG (banked)" | tee -a "$CD/conc.log"
    return 0
  fi
  local need=$(( NC * (CTX + MT + 64) ))
  if [ "$POOL" -gt 0 ] && [ "$need" -gt "$POOL" ]; then
    echo "KV_CAP $CELLN c$CTX n$NC POOL=$POOL NEED=$need" | tee -a "$CD/conc.log"
    return 0
  fi
  docker exec arc-r241 rm -f /root/p139_conc_$TAG.json
  docker exec -d arc-r241 /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" "$TP"
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec arc-r241 test -f /root/p139_conc_$TAG.json && break
    docker inspect arc-r241 >/dev/null 2>&1 || { CELL_DEAD=1; break; }
    sleep 4; i=$((i+1))
  done
  if ! docker exec arc-r241 test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s dead=$CELL_DEAD)" | tee -a "$CD/conc.log"
    [ -n "$CELL_DEAD" ] && docker exec arc-r241 tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp arc-r241:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  # semantic gate: parse n_ok/n_fail; health-discriminate failures
  local NOK NFAIL
  read -r NOK NFAIL <<< "$(python3 -c "import json; d=json.load(open('$CD/p139_conc_$TAG.json')); print(d.get('n_ok',0), d.get('n_fail',0))" 2>/dev/null || echo 0 1)"
  if [ "${NFAIL:-1}" = "0" ] && [ "${NOK:-0}" -gt 0 ]; then
    echo "CONC_OK_$TAG n_ok=$NOK" | tee -a "$CD/conc.log"
    return 0
  fi
  if curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "CONC_PARTIAL_$TAG n_ok=${NOK:-0} n_fail=${NFAIL:-?} (engine alive -- recorded, cell continues)" | tee -a "$CD/conc.log"
    return 0
  fi
  echo "CONC_ENGINE_DEAD_$TAG n_ok=${NOK:-0} n_fail=${NFAIL:-?} -- CELL_DEAD" | tee -a "$CD/conc.log"
  docker exec arc-r241 tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
  CELL_DEAD=1
}

run_cell() {
  # $1=cell $2=served $3=family(kvface) $4..=env
  # family kvface: $3="kvface:<ctx:conc[:temp] ..." -- explicit face list.
  CELLN=$1; SERVED=$2; local SPEC=$3; shift 3
  local FAM=${SPEC%%:*} KVFACES=${SPEC#*:}
  CD=$P/$CELLN
  # resume law: a completed cell (outcome.txt banked) is skipped whole
  if [ -f "$CD/outcome.txt" ]; then
    echo "=== [r241:$CELLN] CELL_SKIP (outcome banked) ==="
    SERVED=""
    return 0
  fi
  mkdir -p "$CD"
  echo "=== [r241:$CELLN] boot $(date +%H:%M:%S) fam=$FAM env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r241 "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R241_BOOT_FAIL_$CELLN -- banking log, matrix continues"
    docker cp arc-r241:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f arc-r241 >/dev/null 2>&1; verify_removed arc-r241 || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3; SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r241_$CELLN $(date +%H:%M:%S)"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
  # POSTURE REFUSE: zero LMCache lines in serve_full.log (the r237
  # clean-cell signature; any line = ARC_LMC leaked through = the leg
  # measures nothing). Abort the whole leg loudly.
  LMC_LINES=$(gcount 'LMCache')
  echo "lmc_lines=$LMC_LINES" >> "$CD/posture.txt"
  echo "kv_pool_tokens=$(kv_pool)" >> "$CD/posture.txt"
  cat "$CD/posture.txt"
  if [ "${LMC_LINES:-1}" -gt 0 ]; then
    echo "R241_POSTURE_DEFECT_$CELLN lmc_lines=$LMC_LINES -- ARC_LMC leaked through, ABORTING LEG"
    docker cp arc-r241:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f arc-r241 >/dev/null 2>&1
    echo "POSTURE_DEFECT" > "$CD/outcome.txt"
    exit 3
  fi
  sleep 5
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  docker cp $R/p139_conc.py arc-r241:/root/p139_conc.py
  POOL=$(kv_pool)
  [ "$POOL" = "0" ] && echo "POOL_PARSE_FAIL -- sweep runs ungated" >> "$CD/posture.txt"
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

  docker cp arc-r241:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  echo "lmc_lines_end=$(grep -c 'LMCache' "$CD/serve.log" 2>/dev/null || true)" >> "$CD/posture.txt"
  docker exec arc-r241 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-r241 >/dev/null 2>&1
  verify_removed arc-r241 || return 1
  echo "CONC_OK" > "$CD/outcome.txt"
  sleep 3
  echo "=== [r241:$CELLN] done $(date +%H:%M:%S) ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R241_WARN_PORT8000_STILL_HELD"
sleep 10

# THE FIX: ARC_LMC=0 explicit (the certified MX posture; r240b's QF8
# carried ARC_LMC=1 -- the defect this leg corrects). Everything else
# byte-parity with the r240b kvm cells.
QF8L0="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=0 ARC_FP8=1"

mkdir -p $P
cp -f $R/p139_conc.py $P/p139_conc.py 2>/dev/null || true

# -- the three MX KV cells: the EXACT r240b kvm grids, LMC-free --
run_cell kvm4 qwen3.8-27b-fp8 "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 131072:1:0.0 131072:2:0.0 131072:4:0.0 131072:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0 65536:1:1.0 131072:1:1.0" $QF8L0 ARC_SPEC=3 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1
run_cell kvm8 qwen3.8-27b-fp8 "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 131072:1:0.0 131072:2:0.0 131072:4:0.0 131072:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0 65536:1:1.0 131072:1:1.0" $QF8L0 ARC_SPEC=3 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1
run_cell kvmk qwen3.8-27b-fp8 "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 131072:1:0.0 131072:2:0.0 131072:4:0.0 131072:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0 65536:1:1.0 131072:1:1.0" $QF8L0 ARC_SPEC=3 ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1

# ---- close-out: checksums + the semantic census (gate line = truth) ----
cd "$P"
find . -name "p139_conc_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
REAL=0; DEADF=0
for j in $(find . -name "p139_conc_*.json"); do
  if python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('n_fail',1)==0 and d.get('n_ok',0)>0 else 1)" "$j"; then
    REAL=$((REAL+1)); else DEADF=$((DEADF+1)); fi
done
echo "BOOT_FAILS:${FAILED:-none}"
echo "R241_CENSUS real_faces=$REAL dead_or_partial_faces=$DEADF"
echo "R241_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + verdict + doc corrections next"
