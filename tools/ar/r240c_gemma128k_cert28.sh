#!/bin/bash
# r240c_gemma128k_cert28.sh -- P6.47 deep-sweep: the gemma 128K annex
# re-run on cert28 (the 2026-11-24 directive: ALL report battery tests
# on the latest certified image). Derived from r140_gemma128k.sh (the
# cert23 original) with:
#   - IMG=cert28 + the full refuse set (fork_sha + .so pins)
#   - bank -> /root/build/p647/r240c/, container arc-r240c
#   - the r140c boundary probes (131000/130300) folded into the gm4x
#     cell BEFORE the sweep: p139_conc.py v2 budgets server-side
#     (ctx-mt-64) so the 131K faces fit by construction; the probes
#     bank the server's own stated rule as evidence.
# Two boots: gm8x (the 8-bit control at 131072 -- refusal banks the
# "128K needs MXFP4" boundary, boot proves servable) + gm4x (MXFP4
# W4A16 @ 131072: full rung ladder 1K..128K x conc {1..8} + t1.0 solo).
# PLUS the MX KV annex (2026-11-24 directive, every ctx step on the MX
# lanes): gkv_m4x + gkv_mkx (online-fp8 weights + mxfp4/mxk8v4 KV @
# 131072, same full ladder) -- the 64K/128K rungs the maxlen-40960
# r240b cells structurally cannot run.
# PLUS the 64K closure (2026-11-24 directive #4: every analyzed test on
# every ctx step 1K/8K/32K/64K/128K): gf8x64 / gm8x64 / gkv_m8x64 (the
# 8-bit lanes at maxlen 65536, kvface grid n1/t1.0 + pool-gated n2/4/8
# KV_CAP rows) + gkv_m8x131 (the mxfp8-KV lane's OWN 128K refusal --
# fp8-class bytes, same boundary class as gm8x).
# Feasibility from the booted KV pool; infeasible points bank as KV_CAP
# rows. Boot failure does NOT abort (vmatrix law). W14 entry quiesce +
# trap restore. Artifacts -> /root/build/p647/r240c/<cell>/. ETA ~7-9 h.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_MQA=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r240c
MT=600
mkdir -p "$P"
FAILED=""
CELL_DEAD=""

fail() { echo "R240C_ABORT: $*"; }

# ---- refuses (r239 pattern) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert28_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
read -r SO_A SO_M <<< "$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; vp=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/')[0]; print(hashlib.md5(open(vp+'libattn_kernels_xe_2.so','rb').read()).hexdigest(), hashlib.md5(open(glob.glob(vp+'libmqa_logits*.so')[0],'rb').read()).hexdigest())\"" 2>/dev/null)" || true
[ "${SO_A:-}" = "$SO_ATTN" ] || { fail "image libattn md5 ${SO_A:-?} != build28 pin"; exit 1; }
[ "${SO_M:-}" = "$SO_MQA" ] || { fail "image libmqa md5 ${SO_M:-?} != pin"; exit 1; }
[ -f "$R/p139_conc.py" ] || { fail "p139_conc.py missing"; exit 1; }
[ -f "$R/p140_probe.py" ] || { fail "p140_probe.py missing"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R240C refuses passed (fork_sha=$FORK_SHA libattn=$SO_A libmqa=$SO_M)"

w14_restore() {
  docker rm -f arc-r240c >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r240c.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R240C: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R240C: WARN W14 not 200 after restore poll (last=$c)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    docker rm -f "$1" >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "R240C_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-r240c grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-r240c >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r240c:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r240c pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r240c:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r240c:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {
  local v
  v=$(docker exec arc-r240c grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
  v=${v#KV cache size: }; v=${v% tokens}; v=${v//,/}
  echo "${v:-0}"
}

probe() {  # budget tag -- one boundary probe, banks status+body json
  local B=$1 TAG=$2
  docker exec arc-r240c rm -f /root/p140_probe_$TAG.json
  docker exec -d arc-r240c /opt/venv/bin/python /root/p140_probe.py "$B" 600 "$TAG"
  local i=0
  while [ $i -lt 270 ]; do
    docker exec arc-r240c test -f /root/p140_probe_$TAG.json && break
    docker inspect arc-r240c >/dev/null 2>&1 || break
    sleep 4; i=$((i+1))
  done
  if docker exec arc-r240c test -f /root/p140_probe_$TAG.json 2>/dev/null; then
    docker cp arc-r240c:/root/p140_probe_$TAG.json "$CD/" >/dev/null 2>&1
    cat "$CD/p140_probe_$TAG.json" | tee -a "$CD/probe.log"
    echo "" >> "$CD/probe.log"
    echo "PROBE_OK_$TAG" | tee -a "$CD/probe.log"
  else
    echo "PROBE_TIMEOUT_$TAG" | tee -a "$CD/probe.log"
  fi
}

group() {  # ctx conc tag [temp] -- runs one concurrency group, banks the JSON
  local CTX=$1 NC=$2 TAG=$3 TP=${4:-0.0}
  if [ -n "$CELL_DEAD" ]; then
    echo "KV_SKIP_ENGINE_DEAD $CELLN $TAG" | tee -a "$CD/conc.log"
    return 0
  fi
  # resume law: a banked face JSON is done -- skip (idempotent restart)
  if [ -f "$CD/p139_conc_$TAG.json" ]; then
    echo "CONC_SKIP_$TAG (banked)" | tee -a "$CD/conc.log"
    return 0
  fi
  local need=$(( NC * (CTX + MT + 64) ))
  if [ "$POOL" -gt 0 ] && [ "$need" -gt "$POOL" ]; then
    echo "KV_CAP $CELLN c$CTX n$NC POOL=$POOL NEED=$need" | tee -a "$CD/conc.log"
    return 0
  fi
  docker exec arc-r240c rm -f /root/p139_conc_$TAG.json
  docker exec -d arc-r240c /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" "$TP"
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec arc-r240c test -f /root/p139_conc_$TAG.json && break
    docker inspect arc-r240c >/dev/null 2>&1 || { CELL_DEAD=1; break; }
    sleep 4; i=$((i+1))
  done
  if ! docker exec arc-r240c test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s dead=$CELL_DEAD)" | tee -a "$CD/conc.log"
    [ -n "$CELL_DEAD" ] && docker exec arc-r240c tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp arc-r240c:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2=served $3=family(bootprobe|gemma128|kvface:<list>) $4..=env
  # family kvface: $3="kvface:<ctx:conc[:temp] ..." -- explicit face list
  # (r240b's SPEC-split design: FAM=prefix, KVFACES=the list).
  CELLN=$1; SERVED=$2; local SPEC=$3; shift 3
  local FAM=${SPEC%%:*} KVFACES=${SPEC#*:}
  CD=$P/$CELLN
  # resume law: a completed cell (outcome.txt banked) is skipped whole
  if [ -f "$CD/outcome.txt" ]; then
    echo "=== [r240c:$CELLN] CELL_SKIP (outcome banked) ==="
    SERVED=""
    return 0
  fi
  mkdir -p "$CD"
  echo "=== [r240c:$CELLN] boot $(date +%H:%M:%S) fam=$FAM env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r240c "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R240C_BOOT_FAIL_$CELLN -- banking log, gate continues"
    docker cp arc-r240c:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f arc-r240c >/dev/null 2>&1; verify_removed arc-r240c || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3; SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r240c_$CELLN $(date +%H:%M:%S)"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
  POOL=$(kv_pool)
  echo "kv_pool_tokens=$POOL" >> "$CD/posture.txt"
  [ "$POOL" = "0" ] && echo "POOL_PARSE_FAIL -- sweep runs ungated" >> "$CD/posture.txt"
  cat "$CD/posture.txt"
  sleep 5
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  docker cp $R/p139_conc.py arc-r240c:/root/p139_conc.py
  docker cp $R/p140_probe.py arc-r240c:/root/p140_probe.py
  CELL_DEAD=""

  if [ "$FAM" = "bootprobe" ]; then
    # the 8-bit control: prove servable at 8K + the biggest ctx that fits
    group 8192 1 "${CELLN}_c8192_n1"
    group 131072 1 "${CELLN}_c131072_n1"
    group 131072 1 "${CELLN}_c131072_n1_t1.0" 1.0
  elif [ "$FAM" = "kvface" ]; then
    # explicit face list "ctx:conc[:temp] ..." (the 64K-closure + refusal
    # cells). Tag law (the r139 kvface gap): t0.0 faces carry NO _t
    # suffix -- p139_report.load_conc builds
    # p139_conc_<cell>_c<ctx>_n<nc>.json for temp 0.0.
    # local-multiassign law: one binding per command (rig bash 5.2.21).
    local f
    for f in $KVFACES; do
      local fx rest fn ft tsuf
      fx=${f%%:*}; rest=${f#*:}; fn=${rest%%:*}; ft=${rest#*:}
      [ "$ft" = "$rest" ] && ft=0.0
      tsuf="_t${ft}"
      [ "$ft" = "0.0" ] && tsuf=""
      group "$fx" "$fn" "${CELLN}_c${fx}_n${fn}${tsuf}" "$ft"
    done
  else
    # the boundary probes FIRST (bank the server's own stated rule),
    # then the canonical rung ladder swept to the wall
    probe 131000 "${CELLN}_probe_b131000"
    probe 130300 "${CELLN}_probe_b130300"
    local CTX
    for CTX in 1024 8192 32768 65536 131072; do
      local ran="" c
      local maxf=0
      for c in 1 2 3 4 5 6 7 8; do
        [ $(( c * (CTX + MT + 64) )) -le "$POOL" ] && maxf=$c
      done
      for c in 1 2 3 4 8 $maxf; do
        [ "$c" = "0" ] && continue
        case " $ran " in *" $c "*) continue ;; esac
        ran="$ran $c"
        group "$CTX" "$c" "${CELLN}_c${CTX}_n${c}"
      done
      group "$CTX" 1 "${CELLN}_c${CTX}_n1_t1.0" 1.0
    done
  fi

  docker cp arc-r240c:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-r240c bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-r240c >/dev/null 2>&1
  verify_removed arc-r240c || return 1
  echo "CONC_OK" > "$CD/outcome.txt"
  sleep 3
  echo "=== [r240c:$CELLN] done $(date +%H:%M:%S) ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R240C_WARN_PORT8000_STILL_HELD"
sleep 10

GM128="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=131072 ARC_FP8=0"
GM64="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=65536 ARC_FP8=0"

mkdir -p $P
cp -f $R/p139_conc.py $P/p139_conc.py 2>/dev/null || true

# -- the 8-bit control first (short cell either way), then the MXFP4 sweep --
run_cell gm8x gemma4-31b bootprobe $GM128 ARC_QUANT=mxfp8
run_cell gm4x gemma4-31b gemma128 $GM128 ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16
# -- the MX KV annex (2026-11-24 directive: every ctx step on the MX
# lanes): online-fp8 weights (constant with the r240b gkv_m* cells) +
# packed KV at maxlen 131072 -- mxfp4/mxk8v4 KV buy the 128K capacity
# fp8 KV cannot (r239 law: fp8-KV gemma est max 120704; mxfp8 KV = fp8
# bytes, same boundary, covered by the gm8x control's refusal). Full
# rung ladder 1K..128K x conc {1..8} + t1.0, pool-gated with KV_CAP rows. --
run_cell gkv_m4x gemma4-31b gemma128 $GM128 ARC_QUANT=fp8 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1
run_cell gkv_mkx gemma4-31b gemma128 $GM128 ARC_QUANT=fp8 ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1
# -- the 64K closure (2026-11-24 directive #4: the 8-bit lanes' missing
# 64K rung -- one 64K session fits the fp8-class pool, n2+ banks KV_CAP
# rows) + the mxfp8-KV lane's own 128K refusal (fp8 bytes, the same
# boundary class gm8x banks for the weights side). --
run_cell gf8x64 gemma4-31b "kvface:65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 65536:1:1.0" $GM64 ARC_QUANT=fp8
run_cell gm8x64 gemma4-31b "kvface:65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 65536:1:1.0" $GM64 ARC_QUANT=mxfp8
run_cell gkv_m8x64 gemma4-31b "kvface:65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 65536:1:1.0" $GM64 ARC_QUANT=fp8 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1
run_cell gkv_m8x131 gemma4-31b "kvface:131072:1:0.0" $GM128 ARC_QUANT=fp8 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1

cd "$P"
find . -name "p139_conc_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
echo "BOOT_FAILS:${FAILED:-none}"
echo "R240C_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + report gen next"
