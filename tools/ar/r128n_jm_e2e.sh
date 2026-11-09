#!/bin/bash
# r128n_jm_e2e.sh -- P6.34 arc9 JM skinny-M e2e battery (arc9test image).
#
# Five jm legs + two causality legs, all on ONE image (arc9test = cert22
# + arc9 wheel + jm skinny_m.py), each read against its banked control
# (identical inputs/postures to r127; the only delta is
# ARC_SKINNY_M=jm -> the joint-matrix route):
#   leg A qwen fp8      -> jm w16 champion lane   (cert22 leg A ref)
#   leg B qwen mxfp8    -> jm m816 lane           (cert21 leg B ref)
#   leg C qwen mxfp4    -> jm g16 lane            (cert21 leg C ref)
#   leg D gemma mxfp8   -> jm m816 lane           (r124b ml40k ref)
#   leg E gemma mxfp4   -> jm g16 lane            (r120g ref)
#   leg F qwen fp8 ARC_SKINNY_M=off  -- causality: stock routes, no wrap
#   leg G qwen fp8 ARC_SKINNY_M=on   -- causality: arc6 scalar skinny
# (jm-vs-on-vs-off on the same image isolates the route flip; F/G reuse
# leg A's shape = the champion lane.)
# Per-leg proof markers: "skinny-M wrap installed" + the lane's
# "jm route FIRST <lane>" line + absence of "route FAILED" (F instead
# proves "reroute OFF"). NO gates beyond ERROR=0/Traceback=0 + spec PASS
# (qwen legs) -- the numbers bank against the refs in the docs tables.
set -uo pipefail
R=/root/build
P=$R/p634
IMG9=vllm-arc:v0.30.0-b70-arc9test
G=$P/e2e_jm
GMU_Q=0.85
GMU_G=0.8
QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
mkdir -p $G

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p634r128n.log 2>&1 < /dev/null &
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

boot_wait() {
  LAST_WARM=0; ok=""; SEEN=0; ABSENT=0
  for i in $(seq 1 1500); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
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
        echo "[$1 boot] TRACEBACK at poll $i"; break
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

battery() {  # battery <legdir> <served> <spec: qwen|none>
  local L=$1 S=$2 SP=$3
  python3 $R/r113_battery.py g1 $S > $L/g1_probes.txt 2>&1 || true
  cat $L/g1_probes.txt
  ( cd $P && P66_FACES="U8k,U32k" timeout 1200 python3 $R/p66/p66_solo_ttft.py $L/faces.json > $L/faces.txt 2>&1 ) || true
  cat $L/faces.txt
  python3 $R/r113_battery.py decode $S > $L/decode.txt 2>&1 || true
  cat $L/decode.txt
  if [ "$SP" = "qwen" ]; then
    echo "--- spec gate (MTP k3) ---"
    python3 $R/arc_spec_gate.py --model $S > $L/spec_gate.txt 2>&1
    echo "spec_gate exit=$?"
    tail -2 $L/spec_gate.txt
  else
    echo "--- spec gate skipped (gemma: no MTP head) ---"
  fi
}

erraudit() {  # $1 = leg dir -- inner || true law (r125 defect, zero-match grep -c)
  docker exec arc-test bash -c "grep -c ERROR /root/serve_full.log || true; grep -c Traceback /root/serve_full.log || true" \
    > "$1/erraudit.txt" 2>/dev/null || echo "?" > "$1/erraudit.txt"
  cat "$1/erraudit.txt"
}

route_audit() {  # $1 = leg dir $2 = expected lane tag $3 = mode (jm|on|off)
  case "$3" in
    off) { echo "reroute_off=$(gcount 'reroute OFF')";
           echo "wrap_installed=$(gcount 'skinny-M wrap installed')";
           echo "jm_first=$(gcount 'jm route FIRST')";
           echo "skinny_first=$(gcount 'skinny-M route FIRST')";
           echo "route_failed=$(gcount 'route FAILED')"; } ;;
    on)  { echo "wrap_installed=$(gcount 'skinny-M wrap installed')";
           echo "skinny_first_$2=$(gcount "skinny-M route FIRST $2")";
           echo "jm_first=$(gcount 'jm route FIRST')";
           echo "route_failed=$(gcount 'route FAILED')"; } ;;
    *)   { echo "wrap_installed=$(gcount 'skinny-M wrap installed')";
           echo "jm_first_$2=$(gcount "jm route FIRST $2")";
           echo "skinny_first=$(gcount 'skinny-M route FIRST')";
           echo "route_failed=$(gcount 'route FAILED')"; } ;;
  esac > "$1/route_audit.txt"
  cat "$1/route_audit.txt"
}

leg_done() {  # $1 = leg dir $2 = lane tag $3 = mode -- serve.log + markers before kill
  route_audit "$1" "$2" "$3"
  erraudit "$1"
  docker exec arc-test grep -m4 "Available KV cache memory" /root/serve_full.log > "$1/kv_lines.txt" 2>/dev/null || true
  docker cp arc-test:/root/serve_full.log "$1/serve.log" >/dev/null 2>&1 || true
  kill_arc
}

boot_leg() {  # boot_leg <legdir> <legname> <lane-tag> <mode: jm|on|off>
  # remaining env via globals; mirrors the r127 leg body
  local L=$1 NAME=$2 LANE=$3 MODE=$4
  mkdir -p $L
  export ARC_SKINNY_M=$MODE
  echo "=== [$NAME boot $(date +%H:%M:%S)] skinny=$MODE ==="
  echo "leg=$NAME img=$IMG9 gmu=$ARC_GMU graph=$ARC_SPEC Graph=1 lmc=$ARC_LMC quant=$ARC_QUANT${ARC_MXFP4_MODE:+ mode=$ARC_MXFP4_MODE} maxlen=${ARC_MAXLEN:-default} skinny=$MODE" > $L/leg_config.txt
  # conditional vars go through `env` -- bash does not re-parse expanded
  # name=value words as assignment prefixes (they would become args)
  ARC_IMAGE=$IMG9 ARC_NAME=arc-test \
  ARC_MODEL=$ARC_MODEL_PATH ARC_SERVED=$ARC_SERVED_NAME \
  ARC_GMU=$ARC_GMU \
  ARC_GRAPH=1 ARC_SPEC=$ARC_SPEC ARC_LMC=$ARC_LMC \
  ARC_FP8=$ARC_FP8 ARC_SKINNY_M=$MODE \
    env "${ARC_QUANT:+ARC_QUANT=$ARC_QUANT}" \
        "${ARC_MXFP4_MODE:+ARC_MXFP4_MODE=$ARC_MXFP4_MODE}" \
        "${ARC_MAXLEN:+ARC_MAXLEN=$ARC_MAXLEN}" \
        "${ARC_RPC:+ARC_RPC=$ARC_RPC}" \
        "${ARC_QUANTCFG:+ARC_QUANTCFG=$ARC_QUANTCFG}" \
        bash $R/serve_arc_b70.sh
  if ! boot_wait $NAME; then
    echo "BOOT_FAILED $NAME -- serve traceback:"
    docker cp arc-test:/root/serve_full.log $L/serve.log >/dev/null 2>&1 || true
    grep -n -B2 -A25 "Traceback (most recent call last" $L/serve.log 2>/dev/null | tail -90 || true
    kill_arc; return 1
  fi
  echo "--- $NAME HEALTH UP $(date +%H:%M:%S) ---"
  sleep 90
  curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$ARC_SERVED_NAME\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
    > $L/probe.out 2>&1
  head -c 300 $L/probe.out; echo
  if grep -q '"content"' $L/probe.out; then
    battery $L $ARC_SERVED_NAME $ARC_SPEC_GATE
    if [ -n "${ARC_REROUTE_GREP:-}" ]; then
      docker exec arc-test grep -m2 "$ARC_REROUTE_GREP" /root/serve_full.log > $L/reroute_marker.txt 2>/dev/null || true
    fi
    docker exec arc-test grep -m1 "Using .* GEMM" /root/serve_full.log > $L/kernel_select.txt 2>/dev/null || true
    docker exec arc-test grep -m1 "Quantized .* layers of types" /root/serve_full.log > $L/online_dispatch.txt 2>/dev/null || true
  else
    echo "PROBE_STALLED $NAME"
  fi
  leg_done $L $LANE $MODE
  echo "=== [$NAME] done $(date +%H:%M:%S) ==="
}

# --- window prep: the arc dev containers must be gone (e2e owns both GPUs)
docker rm -f arc-jm7 >/dev/null 2>&1 || true
docker stop radiance-test >/dev/null 2>&1 || true
sleep 10
docker rm -f arc-test >/dev/null 2>&1 || true

# ---------- Leg A: qwen fp8 (jm w16 champion lane) ----------
unset ARC_MXFP8_MODE ARC_MXFP4_MODE ARC_QUANTCFG ARC_QUANT ARC_MAXLEN ARC_RPC
ARC_MODEL_PATH=/models/qwen3.8-27b-fp8 ARC_SERVED_NAME=qwen3.8-27b-fp8 \
ARC_GMU=$GMU_Q ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1 ARC_SPEC_GATE=qwen \
boot_leg $G/leg_qwen_fp8_jm qwen_fp8_jm w16 jm || exit 1

# ---------- Leg B: qwen mxfp8 w8a16 (jm m816 lane) ----------
unset ARC_MXFP4_MODE
export ARC_QUANTCFG="$QCFG"
ARC_MODEL_PATH=/models/qwen3.8-27b ARC_SERVED_NAME=qwen3.8-27b \
ARC_GMU=$GMU_Q ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 ARC_QUANT=mxfp8 \
ARC_SPEC_GATE=qwen ARC_REROUTE_GREP="mxfp8 w8a16 reroute ACTIVE" \
boot_leg $G/leg_qwen_mxfp8_jm qwen_mxfp8_jm m816 jm || exit 1

# ---------- Leg C: qwen mxfp4 w4a16 (jm g16 lane) ----------
unset ARC_MXFP8_MODE
ARC_MODEL_PATH=/models/qwen3.8-27b ARC_SERVED_NAME=qwen3.8-27b \
ARC_GMU=$GMU_Q ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 \
ARC_QUANTCFG="$QCFG" ARC_SPEC_GATE=qwen ARC_REROUTE_GREP="reroute ACTIVE" \
boot_leg $G/leg_qwen_mxfp4_jm qwen_mxfp4_jm g16 jm || exit 1

# ---------- Leg D: gemma mxfp8 w8a16 (jm m816, ARC_MAXLEN=40960 law) ----------
unset ARC_MXFP4_MODE ARC_QUANTCFG
ARC_MODEL_PATH=/models/gemma4-31b ARC_SERVED_NAME=gemma4-31b \
ARC_GMU=$GMU_G ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0 \
ARC_QUANT=mxfp8 ARC_SPEC_GATE=none ARC_REROUTE_GREP="mxfp8 w8a16 reroute ACTIVE" \
boot_leg $G/leg_gemma_mxfp8_jm gemma_mxfp8_jm m816 jm || exit 1

# ---------- Leg E: gemma mxfp4 w4a16 (jm g16 lane) ----------
unset ARC_MXFP8_MODE
ARC_MODEL_PATH=/models/gemma4-31b ARC_SERVED_NAME=gemma4-31b \
ARC_GMU=$GMU_G ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_FP8=0 \
ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC_GATE=none ARC_REROUTE_GREP="reroute ACTIVE" \
boot_leg $G/leg_gemma_mxfp4_jm gemma_mxfp4_jm g16 jm || exit 1

# ---------- Legs F/G: causality qwen fp8, same image, off/on ----------
unset ARC_MXFP8_MODE ARC_MXFP4_MODE ARC_QUANTCFG ARC_QUANT ARC_MAXLEN ARC_RPC
ARC_MODEL_PATH=/models/qwen3.8-27b-fp8 ARC_SERVED_NAME=qwen3.8-27b-fp8 \
ARC_GMU=$GMU_Q ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1 ARC_SPEC_GATE=qwen \
boot_leg $G/leg_qwen_fp8_off qwen_fp8_off off off || exit 1

ARC_MODEL_PATH=/models/qwen3.8-27b-fp8 ARC_SERVED_NAME=qwen3.8-27b-fp8 \
ARC_GMU=$GMU_Q ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1 ARC_SPEC_GATE=qwen \
boot_leg $G/leg_qwen_fp8_on qwen_fp8_on w16 on || exit 1

echo "=== [r128n] all legs done $(date +%H:%M:%S); W14 restoring ==="
exit 0
