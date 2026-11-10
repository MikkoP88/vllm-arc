#!/bin/bash
# p635_gate3.sh -- P6.35-P posture sweep (stock cert22, no probe patches).
# The db-fix analysis (legs J/K) put solo decode at the fp8 pipeline floor
# (~40 ms step = ~30 ms device incl. 23 ms weight-stream + ~8-10 exposed
# host). This sweep measures the POSTURE levers on stock cert22:
#   spec-k (tokens per weight read) x online quant (weight bytes):
#   fp8 k4/k5, mxfp4-W4A16 k3/k4/k5, mxfp8-W8A16 k3.
# Cells mirror r125 cert21 validated postures (QCFG ignore set, GMU 0.85,
# BASE checkpoint for MX lanes). Face = U8k solo 600 (p135_face.py, model
# arg added). trap restores W14.
set -uo pipefail
R=/root/build
CT=$R/p635
IMG=vllm-arc:v0.30.0-b70-cert22
LEG=$CT/sweep-$(date +%Y%m%d-%H%M%S)
QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
mkdir -p "$LEG"

w14_restore() {
  docker rm -f arc-psw >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    sleep 2
    i=$((i + 1))
  done
  echo "GATE_FAIL_STALE_CONTAINER $1"
  return 1
}

gcount() {  # count pattern in the engine log (inside container)
  local c
  c=$(docker exec arc-psw grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

run_cell() {
  # $1=cellname $2=served-model $3=modelpath $4=spec  rest via env set by caller
  local CELL=$1 SERVED=$2 MPATH=$3 SPEC=$4
  local CD=$LEG/$CELL
  mkdir -p "$CD"
  echo "=== [$CELL] boot $(date +%H:%M:%S) spec=$SPEC quant=${ARC_QUANT:-fp8(native)} mode=${ARC_MXFP4_MODE:-${ARC_MXFP8_MODE:--}} ==="
  echo "cell=$CELL served=$SERVED spec=$SPEC quant=${ARC_QUANT:-native-fp8} qcfg=${ARC_QUANTCFG:--} gmu=${ARC_GMU:-0.8}" > "$CD/cell_config.txt"
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-psw \
    ARC_MODEL="$MPATH" ARC_SERVED="$SERVED" \
    ARC_GRAPH=1 ARC_SPEC=$SPEC ARC_LMC=1 ARC_FP8=${ARC_FP8:-1} \
    ${ARC_QUANT:+ARC_QUANT=$ARC_QUANT} ${ARC_QUANTCFG:+ARC_QUANTCFG=$ARC_QUANTCFG} \
    ${ARC_MXFP4_MODE:+ARC_MXFP4_MODE=$ARC_MXFP4_MODE} \
    ${ARC_GMU:+ARC_GMU=$ARC_GMU} \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  local i=0 ok=""
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; echo "HEALTH_OK_$CELL after ${i}0s"; break; }
    local tb
    tb=$(gcount "Traceback (most recent call last)")
    [ "${tb:-0}" -gt 0 ] && { echo "GATE_FAIL_BOOT_TB_$CELL"; docker exec arc-psw tail -40 /root/serve_full.log 2>/dev/null; break; }
    docker exec arc-psw pgrep -f EngineCore >/dev/null 2>&1 || {
      [ $i -gt 12 ] && { echo "GATE_FAIL_ENGINE_DEAD_$CELL poll=$i"; docker exec arc-psw tail -30 /root/serve_full.log 2>/dev/null; break; }
    }
    i=$((i + 1)); sleep 10
  done
  if [ "$ok" != "1" ]; then
    echo "GATE_FAIL_HEALTH_TIMEOUT_$CELL"
    docker cp arc-psw:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f arc-psw >/dev/null 2>&1; verify_removed arc-psw || true
    return 1
  fi
  sleep 5
  # MX dispatch evidence
  docker exec arc-psw grep -m2 "reroute ACTIVE\|Quantized .* layers of types\|Using .* GEMM" /root/serve_full.log > "$CD/dispatch.txt" 2>/dev/null || true
  cat "$CD/dispatch.txt" 2>/dev/null
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  docker cp $CT/p135_face.py arc-psw:/root/p135_face.py
  docker exec -d arc-psw /opt/venv/bin/python /root/p135_face.py 8192 600 "$CELL" "$SERVED"
  local i2=0
  while [ $i2 -lt 250 ]; do docker exec arc-psw test -f /root/p135_face_$CELL.json && break; sleep 2; i2=$((i2+1)); done
  echo "FACE_DONE_$CELL waited=$((i2*2))s"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker cp arc-psw:/root/p135_face_$CELL.json "$CD/" >/dev/null 2>&1 || echo "face MISSING $CELL"
  docker cp arc-psw:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-psw bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-psw >/dev/null 2>&1
  verify_removed arc-psw || return 1
  sleep 3
  echo "=== [$CELL] done $(date +%H:%M:%S) ==="
}

docker stop radiance-test >/dev/null 2>&1
sleep 5

# --- fp8 cells (native checkpoint, default GMU, cert22 baseline posture) ---
unset ARC_QUANT ARC_QUANTCFG ARC_MXFP4_MODE ARC_MXFP8_MODE ARC_GMU
export ARC_FP8=1
run_cell fp8k4 qwen3.8-27b-fp8 /models/qwen3.8-27b-fp8 4 || exit 1
run_cell fp8k5 qwen3.8-27b-fp8 /models/qwen3.8-27b-fp8 5 || exit 1

# --- mxfp4 W4A16 cells (BASE checkpoint, QCFG ignore set, GMU 0.85) ---
export ARC_QUANT=mxfp4 ARC_QUANTCFG="$QCFG" ARC_MXFP4_MODE=w4a16 ARC_GMU=0.85 ARC_FP8=0
unset ARC_MXFP8_MODE
run_cell mx4k3 qwen3.8-27b /models/qwen3.8-27b 3 || exit 1
run_cell mx4k4 qwen3.8-27b /models/qwen3.8-27b 4 || exit 1
run_cell mx4k5 qwen3.8-27b /models/qwen3.8-27b 5 || exit 1

# --- mxfp8 W8A16 cell (cert21 lane) ---
unset ARC_MXFP4_MODE
export ARC_QUANT=mxfp8
run_cell mx8k3 qwen3.8-27b /models/qwen3.8-27b 3 || exit 1

cd "$LEG" && find . -name "p135_face_*.json" -exec sha256sum {} \; > SHA256SUMS
ls -la "$LEG"
echo "LEG_ARTIFACTS $LEG"
echo "P635_GATE3_SWEEP_COMPLETE -- analysis next"
