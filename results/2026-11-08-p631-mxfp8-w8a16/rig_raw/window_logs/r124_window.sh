#!/bin/bash
# r124_window.sh -- P6.31 e2e A/B battery on the arc5test image:
# online-mxfp8 W8A16 (the new ARC_MXFP8_MODE route) vs W8A8 stock control,
# qwen + gemma, full battery per leg (G1 + faces U8k/U32k + decode + spec
# gate on qwen). Reads against banked refs: qwen champion checkpoint-fp8
# (U8k 3.76 / U32k 16.93 / solo 84.5, r105-corrected) and the r120c/g
# online rows. The r123 kernel matrix predicts: w8a16 ~1.7x faster than
# w8a8 stock (GEMM medians 0.327 vs 0.569 decode / 5.944 vs 15.911
# prefill) but NOT superior to checkpoint-fp8 (kernel 1.50x/1.06x slower
# than the w16pc class) — this battery banks the model-level truth.
# Postures: qwen GMU=0.85 graph=1 spec=3 lmc=1 (certified, QCFG ignore
# list); gemma GMU=0.8 graph=1 spec=0 lmc=0 rpc=1200 (P6.28 law; no MTP
# head -> no spec gate on gemma legs).
# Flow: serve-script passthrough gate -> W14 health -> stop W14 ->
#   leg qwen_mxfp8_w8a16 -> leg qwen_mxfp8_w8a8 ->
#   leg gemma_mxfp8_w8a16 -> leg gemma_mxfp8_w8a8 -> trap restores W14.
set -uo pipefail
R=/root/build
CT=$R/p631
IMG5=vllm-arc:v0.30.0-b70-arc5test
QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'

grep -q "ARC_MXFP8_MODE" $R/serve_arc_b70.sh || {
  echo "FATAL: serve_arc_b70.sh lacks ARC_MXFP8_MODE passthrough"; exit 1; }
docker image inspect $IMG5 >/dev/null 2>&1 || {
  echo "FATAL: $IMG5 not built"; exit 1; }

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p631r124.log 2>&1 < /dev/null &
}
trap w14_start EXIT

kill_arc() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 5
}

gcount() {
  local c
  c=$(docker exec arc-test grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {  # $1 = tag -- EngineCore seen-then-gone + Traceback detect
  LAST_WARM=0; ok=""; SEEN=0; ABSENT=0
  for i in $(seq 1 1500); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
    # r124a fix: fail fast when the serve launch itself died (no container
    # ever appears) instead of burning the full 1500-poll loop.
    if ! docker inspect arc-test >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1))
      if [ "$ABSENT" -gt 8 ]; then
        echo "[$1 boot] container ABSENT $ABSENT polls -- serve launch failed"; break
      fi
    else
      ABSENT=0
    fi
    if docker exec arc-test pgrep -f "EngineCore" >/dev/null 2>&1; then
      SEEN=1
    else
      local TB
      TB=$(gcount "Traceback (most recent call last)")
      if [ "$TB" -ge 1 ] 2>/dev/null; then
        echo "[$1 boot] TRACEBACK in serve log at poll $i"; break
      fi
      if [ "$SEEN" = 1 ] && [ "$i" -gt 24 ]; then
        echo "[$1 boot] ENGINECORE seen-then-DIED at poll $i"; break
      fi
    fi
    local NW
    NW=$(gcount "ARC XPU tune warmup: load")
    if [ "$NW" != "$LAST_WARM" ]; then
      echo "[$1 boot] warm progress: $NW/32 load-lines ($(date +%H:%M:%S))"
      LAST_WARM=$NW
    fi
    sleep 5
  done
  [ "$ok" = "1" ]
}

m816_proofs() {  # $1 legdir, $2 mode -- route marker + stock kernel select
  local LEG=$1 MODE=$2
  if [ "$MODE" = w8a8 ]; then
    docker exec arc-test grep -m1 "mxfp8 w8a16 reroute OFF" /root/serve_full.log > $LEG/mode_proof.txt 2>/dev/null || true
  else
    docker exec arc-test grep -m1 "mxfp8 w8a16 reroute ACTIVE" /root/serve_full.log > $LEG/mode_proof.txt 2>/dev/null || true
  fi
  cat $LEG/mode_proof.txt 2>/dev/null
  docker exec arc-test grep -m1 "Quantized .* layers of types" /root/serve_full.log > $LEG/online_dispatch.txt 2>/dev/null || true
  docker exec arc-test grep -m1 "Using XPUMxFp8LinearKernel for MXFP8 GEMM" /root/serve_full.log > $LEG/kernel_select.txt 2>/dev/null || true
  cat $LEG/online_dispatch.txt $LEG/kernel_select.txt 2>/dev/null
  local QA KB
  QA=$(gcount "Quantized .* layers of types")
  KB=$(gcount "Using XPUMxFp8LinearKernel for MXFP8 GEMM")
  echo "dispatch proofs: online-summary=$QA kernel-select=$KB (want >=1 and >=1)"
  [ "$QA" -ge 1 ] 2>/dev/null && [ "$KB" -ge 1 ] 2>/dev/null
}

leg_evidence() {  # $1 legdir
  local LEG=$1
  docker exec arc-test bash -c "grep -c ERROR /root/serve_full.log; grep -c Traceback /root/serve_full.log" > $LEG/erraudit.txt 2>/dev/null || true
  cat $LEG/erraudit.txt
  docker exec arc-test grep -m4 "Available KV cache memory" /root/serve_full.log > $LEG/kv_lines.txt 2>/dev/null || true
  cat $LEG/kv_lines.txt 2>/dev/null
  docker cp arc-test:/root/serve_full.log $LEG/serve.log >/dev/null 2>&1 || true
}

# leg <sfx> <model qwen|gemma> <mode w8a16|w8a8>
leg() {
  local SFX=$1 MK=$2 MODE=$3
  local LEG=$CT/leg_${SFX}
  mkdir -p $LEG
  local MODEL_DIR=/models/qwen3.8-27b MNAME=qwen3.8-27b GMU=0.85 SPEC=3 LMC=1 RPC="" QC="$QCFG"
  if [ "$MK" = gemma ]; then
    MODEL_DIR=/models/gemma4-31b; MNAME=gemma-4-31b; GMU=0.8; SPEC=0; LMC=0; RPC=1200; QC=""
  fi
  if [ -n "$QC" ]; then export ARC_QUANTCFG="$QC"; else unset ARC_QUANTCFG; fi
  if [ "$MODE" = w8a8 ]; then export ARC_MXFP8_MODE=w8a8; else unset ARC_MXFP8_MODE; fi
  # r124a fix: ARC_RPC_TIMEOUT via export/unset, NOT ${RPC:+...} in the
  # env-assignment prefix (a non-lexical-assignment word there demotes every
  # following ARC_*=... word to command-args -> "ARC_GRAPH=1: command not
  # found", leg 1 r124).
  if [ -n "$RPC" ]; then export ARC_RPC_TIMEOUT="$RPC"; else unset ARC_RPC_TIMEOUT; fi
  echo "=== [leg $SFX] boot $(date +%H:%M:%S) (model=$MNAME quant=mxfp8 mode=$MODE gmu=$GMU spec=$SPEC lmc=$LMC rpc=${RPC:-def}) ==="
  echo "mode=$MODE quant=mxfp8 model=$MNAME gmu=$GMU spec=$SPEC lmc=$LMC rpc=${RPC:-def} qcfg=${QC:-none} img=$IMG5" > $LEG/leg_config.txt

  ARC_IMAGE=$IMG5 ARC_NAME=arc-test \
  ARC_MODEL=$MODEL_DIR ARC_SERVED=$MNAME ARC_QUANT=mxfp8 \
  ARC_GMU=$GMU \
  ARC_GRAPH=1 ARC_SPEC=$SPEC ARC_LMC=$LMC ARC_FP8=0 \
    bash $R/serve_arc_b70.sh
  if ! boot_wait "leg_${SFX}"; then
    echo "BOOT_FAILED leg_${SFX} -- serve traceback:"
    docker cp arc-test:/root/serve_full.log $LEG/serve.log >/dev/null 2>&1 || true
    grep -n -B2 -A25 "Traceback (most recent call last" $LEG/serve.log 2>/dev/null | tail -90 || true
    kill_arc; return 1
  fi
  echo "--- leg_${SFX} HEALTH UP $(date +%H:%M:%S); warm lines: $(gcount 'ARC XPU tune warmup: load') ---"
  sleep 90
  curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MNAME\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
    > $LEG/probe.out 2>&1
  head -c 300 $LEG/probe.out; echo
  if ! m816_proofs $LEG $MODE; then
    echo "DISPATCH_PROOF_FAILED leg_${SFX} (P-1 class) -- evidence kept, continuing"
  fi
  if grep -q '"content"' $LEG/probe.out; then
    python3 $R/r113_battery.py g1 $MNAME > $LEG/g1_probes.txt 2>&1 || true
    cat $LEG/g1_probes.txt
    ( cd $CT && P66_FACES="U8k,U32k" timeout 1200 python3 $R/p66/p66_solo_ttft.py $LEG/${SFX}_u8k32k.json > $LEG/faces.txt 2>&1 ) || true
    cat $LEG/faces.txt
    python3 $R/r113_battery.py decode $MNAME > $LEG/decode.txt 2>&1 || true
    cat $LEG/decode.txt
    if [ "$MK" = qwen ]; then
      echo "--- spec gate (MTP k3) ---"
      python3 $R/arc_spec_gate.py --model $MNAME > $LEG/spec_gate.txt 2>&1
      echo "spec_gate exit=$?"
      cat $LEG/spec_gate.txt
    fi
  else
    echo "PROBE_STALLED leg_${SFX}"
  fi
  leg_evidence $LEG
  kill_arc
}

WH=""
for i in $(seq 1 40); do
  WH=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health 2>/dev/null || true)
  [ "$WH" = "200" ] && break
  sleep 60
done
echo "W14 health before e2e legs: $WH ($(date +%H:%M:%S))"

docker stop radiance-test >/dev/null 2>&1 || true
sleep 10
docker rm -f arc-test >/dev/null 2>&1 || true

leg qwen_mxfp8_w8a16  qwen  w8a16
leg qwen_mxfp8_w8a8   qwen  w8a8
leg gemma_mxfp8_w8a16 gemma w8a16
leg gemma_mxfp8_w8a8  gemma w8a8

echo "=== [r124] window done $(date +%H:%M:%S); W14 restoring ==="
