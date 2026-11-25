#!/bin/bash
# r240_reports_cert28.sh -- P6.47 deep-sweep: the per-config performance
# report matrix re-run on cert28 (the 2026-11-24 directive: re-run ALL
# report battery tests on the latest certified image). Derived
# byte-shape from r138_reports.sh (the cert23 original) with:
#   - IMG=cert28 + the full refuse set (fork_sha + .so pins)
#   - bank -> /root/build/p647/r240/, container arc-r240
#   - the r138b kvauto131 supplement folded in as one cell
#   - NEW: the three MX KV dtype cells (kvm4/kvm8/kvmk) -- the kv
#     comparison gains the current image's certified MX lanes
#     (ARC_KVDTYPE=mxfp4/mxfp8/mxk8v4 + ARC_MXKV_NATIVE=1, qwen fp8 k3)
# 16 cells (~3.5-4 h): E qf3/qf0/q83/q80/q43/q40 gf8/gm8/gm4,
# G kvauto/kvfp8/kve5m2/kvauto131/kvm4/kvm8/kvmk.
# Faces: p138_face.py (per-tag prompt seeds). Boot failure does NOT
# abort (vmatrix law -- a refusal is a banked outcome). W14 entry
# quiesce + trap restore. SHA256SUMS at end.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_MQA=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r240
mkdir -p "$P"
FAILED=""

fail() { echo "R240_ABORT: $*"; }

# ---- refuses (r239 pattern) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert28_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
read -r SO_A SO_M <<< "$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; vp=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/')[0]; print(hashlib.md5(open(vp+'libattn_kernels_xe_2.so','rb').read()).hexdigest(), hashlib.md5(open(glob.glob(vp+'libmqa_logits*.so')[0],'rb').read()).hexdigest())\"" 2>/dev/null)" || true
[ "${SO_A:-}" = "$SO_ATTN" ] || { fail "image libattn md5 ${SO_A:-?} != build28 pin"; exit 1; }
[ "${SO_M:-}" = "$SO_MQA" ] || { fail "image libmqa md5 ${SO_M:-?} != pin"; exit 1; }
[ -f "$R/p138_face.py" ] || { fail "p138_face.py missing"; exit 1; }
[ -f "$R/serve_arc_b70.sh" ] || { fail "serve_arc_b70.sh missing"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R240 refuses passed (fork_sha=$FORK_SHA libattn=$SO_A libmqa=$SO_M)"

w14_restore() {
  docker rm -f arc-r240 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r240.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R240: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R240: WARN W14 not 200 after restore poll (last=$c)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    docker rm -f "$1" >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "R240_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-r240 grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-r240 >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r240:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r240 pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r240:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r240:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

# face <ctx> <maxtok> <tag> <temp>  (model = $SERVED, always passed --
# the p135 default-model 404 bit twice; r137b law)
face() {
  local CTX=$1 MT=$2 TAG=$3 TP=$4
  docker exec arc-r240 rm -f /root/p138_face_$TAG.json
  docker exec -d arc-r240 /opt/venv/bin/python /root/p138_face.py \
    "$CTX" "$MT" "$TAG" "$SERVED" "$TP"
  local i=0
  while [ $i -lt 250 ]; do
    docker exec arc-r240 test -f /root/p138_face_$TAG.json && break
    sleep 2; i=$((i+1))
  done
  if [ $i -ge 250 ]; then
    echo "FACE_TIMEOUT_$TAG"
    echo "FACE_TIMEOUT_$TAG" >> "$CD/faces.log"
    return 0
  fi
  docker cp arc-r240:/root/p138_face_$TAG.json "$CD/" >/dev/null 2>&1
  echo "FACE_OK_$TAG waited=$((i*2))s" | tee -a "$CD/faces.log"
}

run_cell() {
  # $1=cellname $2=served-model $3=plan(qwen3|qwen0|gemma|kv) $4..=env
  local CELLN=$1; SERVED=$2; local PLAN=$3; shift 3
  CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r240:$CELLN] boot $(date +%H:%M:%S) plan=$PLAN env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r240 "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R240_BOOT_FAIL_$CELLN -- banking log, matrix continues"
    docker cp arc-r240:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec arc-r240 tail -30 /root/serve_full.log 2>/dev/null
    docker rm -f arc-r240 >/dev/null 2>&1; verify_removed arc-r240 || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3
    SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r240_$CELLN $(date +%H:%M:%S)"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
  cat "$CD/posture.txt"
  sleep 5
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  docker cp $P/p138_face.py arc-r240:/root/p138_face.py

  case "$PLAN" in
  qwen3)
    for C in 1024 8192 32768 65536 131072; do
      face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
    done
    for T in 0.0 0.2 0.4 0.6 0.8 1.0; do
      face 8192 600 "${CELLN}_s8192_t${T}" "$T"
    done ;;
  qwen0)
    for C in 1024 8192 32768 65536 131072; do
      face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
    done ;;
  gemma)
    for C in 1024 8192 32768; do
      face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
    done ;;
  kv)
    for C in 8192 32768; do
      face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
    done ;;
  esac

  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker cp arc-r240:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-r240 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-r240 >/dev/null 2>&1
  verify_removed arc-r240 || return 1
  echo "FACES_OK" > "$CD/outcome.txt"
  sleep 3
  echo "=== [r240:$CELLN] done $(date +%H:%M:%S) ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law -- a live W14 holds the B70s and
#      answers :8000 with bogus HEALTH_OK). ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R240_WARN_PORT8000_STILL_HELD"
sleep 10

QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
QF8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1"
QMX="ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b ARC_QUANTCFG=$QCFG ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0"
GM="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

cp -f $R/p138_face.py $P/p138_face.py 2>/dev/null || true

# ---- E cells: qwen 3 quants x {k3, k0} ----
run_cell qf3 qwen3.8-27b-fp8   qwen3 $QF8 ARC_SPEC=3
run_cell qf0 qwen3.8-27b-fp8   qwen0 $QF8 ARC_SPEC=0
run_cell q83 qwen3.8-27b       qwen3 $QMX ARC_QUANT=mxfp8 ARC_SPEC=3
run_cell q80 qwen3.8-27b       qwen0 $QMX ARC_QUANT=mxfp8 ARC_SPEC=0
run_cell q43 qwen3.8-27b       qwen3 $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=3
run_cell q40 qwen3.8-27b       qwen0 $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=0

# ---- E cells: gemma 3 quants, spec0 ----
run_cell gf8 gemma4-31b gemma $GM ARC_QUANT=fp8
run_cell gm8 gemma4-31b gemma $GM ARC_QUANT=mxfp8
run_cell gm4 gemma4-31b gemma $GM ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16

# ---- G cells: kv-cache-dtype sweep on qwen fp8 k3 ----
run_cell kvauto qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=auto
run_cell kvfp8  qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=fp8
run_cell kve5m2 qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=fp8_e5m2

# ---- G cells: the r138b supplement (auto @ 131072) + the MX KV lanes
#      (cert28's certified MX postures: native decode, LMC auto-off) ----
run_cell kvauto131 qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=auto ARC_MAXLEN=131072
run_cell kvm4 qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1
run_cell kvm8 qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1
run_cell kvmk qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1

cd "$P"
find . -name "p138_face_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
echo "BOOT_FAILS:${FAILED:-none}"
echo "R240_REPORTS_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + report gen next"
