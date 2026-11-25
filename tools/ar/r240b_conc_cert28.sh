#!/bin/bash
# r240b_conc_cert28.sh -- P6.47 deep-sweep: the concurrency-sweep
# battery re-run on cert28 (the 2026-11-24 directive: ALL report
# battery tests on the latest certified image). Derived byte-shape
# from r139_conc.sh (the cert23 original) with:
#   - IMG=cert28 + the full refuse set (fork_sha + .so pins)
#   - bank -> /root/build/p647/r240b/, container arc-r240b
#   - NEW: the three MX KV cells (kvm4/kvm8/kvmk) ride the kvface
#     plan at the kv-comparison grid (8K/32K x conc {1,4,8} + t1.0
#     solo) -- the KV report's conc dimension gains the cert28 MX
#     lanes; the 9 config cells + 4 turboquant cells + 2 gemma kv
#     cells keep the full r139 rung ladder byte-exact.
#   - OWNER DIRECTIVE (2026-11-24, mid-campaign): MX KV lanes must
#     cover BOTH models + concurrency {1,2,4,8} -> the kvm grids gain
#     the n2 faces, and 4 gemma KV cells join (gkv_m4/gkv_m8/gkv_mk
#     at the full grid + gkv_e5m2 refusal probe).
#   - OWNER DIRECTIVE 2 (2026-11-24, mid-campaign): every performance
#     test captured at every ctx step 1K/8K/32K/64K/128K -> the kvm
#     grids extend to the full 5-rung ladder (conc {1,2,4,8} + t1.0
#     solos per rung; KV_CAP points bank as rows), the gemma MX grids
#     gain the 1K rung (64K/128K run in the r240c MX-KV annex cells).
# 22 cells; ETA ~14-17 h (overnight-class chain, the r232 precedent:
# W14 stays restored by the EXIT trap at completion, session monitors).
# p139_conc.py v2 (server-side prompt budget ctx-mt-64; per-slot unique
# prompts, common submit barrier, per-request ttft). Feasibility from
# the booted KV pool; infeasible points bank as KV_CAP rows. Boot
# failure does NOT abort (vmatrix law). W14 entry quiesce + trap
# restore. Artifacts -> /root/build/p647/r240b/<cell>/.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_MQA=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r240b
MT=600
mkdir -p "$P"
FAILED=""
CELL_DEAD=""

fail() { echo "R240B_ABORT: $*"; }

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
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R240B refuses passed (fork_sha=$FORK_SHA libattn=$SO_A libmqa=$SO_M)"

w14_restore() {
  docker rm -f arc-r240b >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r240b.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R240B: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R240B: WARN W14 not 200 after restore poll (last=$c)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    docker rm -f "$1" >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "R240B_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-r240b grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-r240b >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r240b:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r240b pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r240b:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r240b:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {  # parse "KV cache size: N tokens" from the booted engine
  local v
  v=$(docker exec arc-r240b grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
  v=${v#KV cache size: }; v=${v% tokens}; v=${v//,/}
  echo "${v:-0}"
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
  docker exec arc-r240b rm -f /root/p139_conc_$TAG.json
  docker exec -d arc-r240b /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" "$TP"
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec arc-r240b test -f /root/p139_conc_$TAG.json && break
    docker inspect arc-r240b >/dev/null 2>&1 || { CELL_DEAD=1; break; }
    sleep 4; i=$((i+1))
  done
  if ! docker exec arc-r240b test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s dead=$CELL_DEAD)" | tee -a "$CD/conc.log"
    [ -n "$CELL_DEAD" ] && docker exec arc-r240b tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp arc-r240b:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2=served $3=family(qwen|gemma|kvface) $4..=env (byte-exact r139)
  # family kvface: $3="kvface:<ctx:conc[:temp] ..." -- explicit face list, no sweep.
  CELLN=$1; SERVED=$2; local SPEC=$3; shift 3
  local FAM=${SPEC%%:*} KVFACES=${SPEC#*:}
  CD=$P/$CELLN
  # resume law: a completed cell (outcome.txt banked) is skipped whole
  if [ -f "$CD/outcome.txt" ]; then
    echo "=== [r240b:$CELLN] CELL_SKIP (outcome banked) ==="
    SERVED=""
    return 0
  fi
  mkdir -p "$CD"
  echo "=== [r240b:$CELLN] boot $(date +%H:%M:%S) fam=$FAM env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r240b "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R240B_BOOT_FAIL_$CELLN -- banking log, matrix continues"
    docker cp arc-r240b:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f arc-r240b >/dev/null 2>&1; verify_removed arc-r240b || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3; SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r240b_$CELLN $(date +%H:%M:%S)"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
  echo "tq_hook_lines=$(gcount 'tq_spec_fix')" >> "$CD/posture.txt"
  POOL=$(kv_pool)
  echo "kv_pool_tokens=$POOL" >> "$CD/posture.txt"
  [ "$POOL" = "0" ] && echo "POOL_PARSE_FAIL -- sweep runs ungated" >> "$CD/posture.txt"
  cat "$CD/posture.txt"
  sleep 5
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  docker cp $R/p139_conc.py arc-r240b:/root/p139_conc.py
  CELL_DEAD=""

  if [ "$FAM" = "kvface" ]; then
    # explicit face list "ctx:conc[:temp] ..." (gemma KV + MX KV cells).
    # Tag law (the r139 kvface gap): t0.0 faces carry NO _t suffix --
    # p139_report.load_conc builds p139_conc_<cell>_c<ctx>_n<nc>.json
    # for temp 0.0 (r139's gkv cells boot-failed so the always-suffixed
    # form never bit; the r240b kvm cells WILL have data).
    for f in $KVFACES; do
      # local-multiassign law: one binding per command -- a later word
      # referencing a var assigned in the SAME local statement expands
      # it unset under set -u (rig bash 5.2.21, proven 2026-11-24).
      local fx rest fn ft tsuf
      fx=${f%%:*}; rest=${f#*:}; fn=${rest%%:*}; ft=${rest#*:}
      [ "$ft" = "$rest" ] && ft=0.0
      tsuf="_t${ft}"
      [ "$ft" = "0.0" ] && tsuf=""
      group "$fx" "$fn" "${CELLN}_c${fx}_n${fn}${tsuf}" "$ft"
    done
  else
    # standing directive: the canonical rung ladder -- every rung the
    # boot's maxlen/KV pool permits, swept to the wall. gemma main
    # boots (maxlen 40960) stop at 32K; long gemma boots use the annex.
    local CTXS="1024 8192 32768 65536 131072"
    [ "$FAM" = "gemma" ] && CTXS="1024 8192 32768"
    for CTX in $CTXS; do
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
      # the t1.0 solo face: the temp 0.0/1.0 pair per ctx row
      group "$CTX" 1 "${CELLN}_c${CTX}_n1_t1.0" 1.0
    done
  fi

  docker cp arc-r240b:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-r240b bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-r240b >/dev/null 2>&1
  verify_removed arc-r240b || return 1
  echo "CONC_OK" > "$CD/outcome.txt"
  sleep 3
  echo "=== [r240b:$CELLN] done $(date +%H:%M:%S) ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R240B_WARN_PORT8000_STILL_HELD"
sleep 10

QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
QF8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1"
QMX="ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b ARC_QUANTCFG=$QCFG ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0"
GM="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

mkdir -p $P
cp -f $R/p139_conc.py $P/p139_conc.py 2>/dev/null || true

# -- the 9 report cells (order = report priority; qwen spec first) --
run_cell qf3 qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3
run_cell qf0 qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=0
run_cell q83 qwen3.8-27b      qwen $QMX ARC_QUANT=mxfp8 ARC_SPEC=3
run_cell q80 qwen3.8-27b      qwen $QMX ARC_QUANT=mxfp8 ARC_SPEC=0
run_cell q43 qwen3.8-27b      qwen $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=3
run_cell q40 qwen3.8-27b      qwen $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=0
run_cell gf8 gemma4-31b gemma $GM ARC_QUANT=fp8
run_cell gm8 gemma4-31b gemma $GM ARC_QUANT=mxfp8
run_cell gm4 gemma4-31b gemma $GM ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16

# -- the 4 turboquant cells (qwen-fp8 certified posture, KV preset knob) --
run_cell tq_k8v4  qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3 ARC_KVDTYPE=turboquant_k8v4
run_cell tq_4bit  qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3 ARC_KVDTYPE=turboquant_4bit_nc
run_cell tq_k3v4  qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3 ARC_KVDTYPE=turboquant_k3v4_nc
run_cell tq_3bit  qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3 ARC_KVDTYPE=turboquant_3bit_nc

# -- gemma KV-cache-dtype comparison (auto = fp16 KV; expect 40960 refusal,
#    16384 working -- the same bank-the-cliff pattern as qwen kvauto) --
run_cell gkv_auto   gemma4-31b "kvface:8192:1:0.0" $GM ARC_QUANT=fp8 ARC_KVDTYPE=auto
run_cell gkv_auto16 gemma4-31b "kvface:8192:1:0.0 16384:1:0.0" $GM ARC_QUANT=fp8 ARC_KVDTYPE=auto ARC_MAXLEN=16384

# -- NEW: the three MX KV cells (cert28's certified MX postures: native
#    decode, LMC auto-off) at the kv-comparison grid --
# the 5-rung directive grid (2026-11-24: every ctx step 1K..128K on the MX
# lanes, conc {1,2,4,8} + t1.0 solos): infeasible (n,ctx) points hit the
# group() POOL gate and bank as KV_CAP rows -- the table stays complete.
run_cell kvm4 qwen3.8-27b-fp8 "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 131072:1:0.0 131072:2:0.0 131072:4:0.0 131072:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0 65536:1:1.0 131072:1:1.0" $QF8 ARC_SPEC=3 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1
run_cell kvm8 qwen3.8-27b-fp8 "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 131072:1:0.0 131072:2:0.0 131072:4:0.0 131072:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0 65536:1:1.0 131072:1:1.0" $QF8 ARC_SPEC=3 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1
run_cell kvmk qwen3.8-27b-fp8 "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 65536:1:0.0 65536:2:0.0 65536:4:0.0 65536:8:0.0 131072:1:0.0 131072:2:0.0 131072:4:0.0 131072:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0 65536:1:1.0 131072:1:1.0" $QF8 ARC_SPEC=3 ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1

# -- NEW (owner directive 2026-11-24): gemma MX KV lanes — the same
#    kv-comparison grid as the qwen MX cells (n {1,2,4,8} at 8K/32K +
#    the t1.0 solo pair), gemma certified posture (online-fp8 weights
#    to match the gkv_auto family; ARC_LMC=0 + RPC 1200 + maxlen
#    40960). Weights lane constant across the gemma KV table
#    (ctrl gf8 = online fp8 + fp8_e4m3 KV). Infeasible points bank as
#    KV_CAP rows (gemma pools are ~an order below qwen's); a boot
#    refusal banks as BOOT_FAIL with the serve log — either outcome
#    is the comprehensive answer (vmatrix law: no abort). --
# gemma MX KV: the servable rungs at maxlen 40960 (1K/8K/32K, conc
# {1,2,4,8} + t1.0 solos); the 64K/128K steps of the ladder run in the
# r240c MX-KV annex cells (gkv_m4x/gkv_mkx at maxlen 131072 -- packed KV
# buys the capacity fp8 KV cannot, r239 law: fp8-KV gemma est max 120704).
run_cell gkv_m4 gemma4-31b "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0" $GM ARC_QUANT=fp8 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1
run_cell gkv_m8 gemma4-31b "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0" $GM ARC_QUANT=fp8 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1
run_cell gkv_mk gemma4-31b "kvface:1024:1:0.0 1024:2:0.0 1024:4:0.0 1024:8:0.0 8192:1:0.0 8192:2:0.0 8192:4:0.0 8192:8:0.0 32768:1:0.0 32768:2:0.0 32768:4:0.0 32768:8:0.0 1024:1:1.0 8192:1:1.0 32768:1:1.0" $GM ARC_QUANT=fp8 ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1
#    e5m2 on gemma: the refusal is an engine-level law (no e5m2 KV
#    dequant path in esimd FA2 on Xe2, banked on qwen kve5m2) — this
#    probe banks the per-model answer cheaply (~a failed boot).
run_cell gkv_e5m2 gemma4-31b "kvface:8192:1:0.0" $GM ARC_QUANT=fp8 ARC_KVDTYPE=fp8_e5m2

cd "$P"
find . -name "p139_conc_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
echo "BOOT_FAILS:${FAILED:-none}"
echo "R240B_CONC_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + report gen next"
