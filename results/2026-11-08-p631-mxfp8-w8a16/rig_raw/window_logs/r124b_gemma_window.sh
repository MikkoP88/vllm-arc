#!/bin/bash
# r124b_gemma_window.sh -- P6.31 gemma-only re-run of the r124 e2e battery.
# r124 verdict: both qwen legs GREEN (w8a16 U8k 3.61 / U32k 17.07 / solo
# 53.8-55.6 / agg8 265.9 / spec pos1=0.9804 PASS; w8a8 stock U8k 5.59 /
# U32k 25.05 / solo 35.6-36.3 / agg8 194.2 / spec PASS) -- adjudication
# settled: w8a16 decisively beats stock w8a8 online mxfp8 (~1.5x faces/
# solo/agg), stays behind checkpoint-fp8 champion (solo 84.5) -> default
# adjudication. Both gemma legs BOOT_FAILED on the KV-CAPACITY gate, not a
# lane defect: online mxfp8 = ~29 GiB TP2 weights on the 31B model (2x the
# mxfp4 legs that fit) -> only 5.31 GiB KV at gmu 0.8 < 8.33 GiB needed
# for one 262144 seq ("estimated maximum model length 103296"). Fix:
# ARC_MAXLEN=40960 for the gemma-mxfp8 legs -- the battery's longest probe
# is the U32k face (32768 prompt + 32 out = 32800 <= 40960); g1/decode
# probes are ~200 tokens; max_model_len gates admission only, so faces/
# decode stay comparable to the banked 262144 refs (r111 cert19 fp8 faces
# 4.9/17.2 decode 28.4/195; r120g mxfp4 w4a16 faces 3.94/21.16 decode
# 40.5/262). Leg dirs get the _ml40k suffix -- the r124 BOOT_FAILED
# evidence stays untouched.
# Postures: gemma GMU=0.8 graph=1 spec=0 lmc=0 rpc=1200 maxlen=40960
# (P6.28 law; no MTP head -> no spec gate on gemma legs).
# Flow: W14 health -> stop W14 ->
#   leg gemma_mxfp8_w8a16_ml40k -> leg gemma_mxfp8_w8a8_ml40k ->
#   trap restores W14.
set -uo pipefail
R=/root/build
CT=$R/p631
IMG5=vllm-arc:v0.30.0-b70-arc5test
MLEN=40960

grep -q "ARC_MXFP8_MODE" $R/serve_arc_b70.sh || {
  echo "FATAL: serve_arc_b70.sh lacks ARC_MXFP8_MODE passthrough"; exit 1; }
grep -q 'ARC_MAXLEN' $R/serve_arc_b70.sh || {
  echo "FATAL: serve_arc_b70.sh lacks ARC_MAXLEN support"; exit 1; }
docker image inspect $IMG5 >/dev/null 2>&1 || {
  echo "FATAL: $IMG5 not built"; exit 1; }

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p631r124b.log 2>&1 < /dev/null &
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
    # fail fast when the serve launch itself died (no container ever
    # appears) instead of burning the full 1500-poll loop (r124a lesson).
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

# leg <sfx> <mode w8a16|w8a8>  (gemma-only: model/posture hardcoded)
leg() {
  local SFX=$1 MODE=$2
  local LEG=$CT/leg_${SFX}
  mkdir -p $LEG
  local MODEL_DIR=/models/gemma4-31b MNAME=gemma-4-31b GMU=0.8 SPEC=0 LMC=0 RPC=1200
  unset ARC_QUANTCFG
  if [ "$MODE" = w8a8 ]; then export ARC_MXFP8_MODE=w8a8; else unset ARC_MXFP8_MODE; fi
  export ARC_RPC_TIMEOUT="$RPC"
  export ARC_MAXLEN="$MLEN"
  echo "=== [leg $SFX] boot $(date +%H:%M:%S) (model=$MNAME quant=mxfp8 mode=$MODE gmu=$GMU spec=$SPEC lmc=$LMC rpc=$RPC maxlen=$MLEN) ==="
  echo "mode=$MODE quant=mxfp8 model=$MNAME gmu=$GMU spec=$SPEC lmc=$LMC rpc=$RPC maxlen=$MLEN qcfg=none img=$IMG5" > $LEG/leg_config.txt

  ARC_IMAGE=$IMG5 ARC_NAME=arc-test \
  ARC_MODEL=$MODEL_DIR ARC_SERVED=$MNAME ARC_QUANT=mxfp8 \
  ARC_GMU=$GMU \
  ARC_GRAPH=1 ARC_SPEC=$SPEC ARC_LMC=$LMC ARC_FP8=0 \
    bash $R/serve_arc_b70.sh
  if ! boot_wait "leg_${SFX}"; then
    echo "BOOT_FAILED leg_${SFX} -- serve traceback:"
    docker cp arc-test:/root/serve_full.log $LEG/serve.log >/dev/null 2>&1 || true
    grep -n -B2 -A25 "Traceback (most recent call last)" $LEG/serve.log 2>/dev/null | tail -90 || true
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

leg gemma_mxfp8_w8a16_ml40k w8a16
leg gemma_mxfp8_w8a8_ml40k  w8a8

echo "=== [r124b] window done $(date +%H:%M:%S); W14 restoring ==="
