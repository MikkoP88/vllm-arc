#!/bin/bash
# r127_skinny_e2e.sh -- P6.33 arc6 skinny-M e2e battery (arc6test image).
#
# Five legs, all ARC_SKINNY_M unset (= on; the M<=8 route installs when
# the arc6 wheel's ops are present), each read against its banked control
# (identical inputs/postures; the M-split only adds the M<=8 branch, and
# r126's big-M stock cells already banked backport neutrality at kernel
# level):
#   leg A qwen fp8      (cert21 leg A ref)   -> skinny w16 champion lane
#   leg B qwen mxfp8    (cert21 leg B ref)   -> skinny m816 lane
#   leg C qwen mxfp4    (cert21 leg C ref)   -> skinny g16 lane
#   leg D gemma mxfp8   (r124b ml40k ref)    -> skinny m816 lane
#   leg E gemma mxfp4   (r120g ref)          -> skinny g16 lane
# Per-leg proof markers: "skinny-M wrap installed" + the lane's
# "skinny-M route FIRST <lane>" line + absence of "skinny.*route FAILED".
# NO gates beyond ERROR=0/Traceback=0 + spec PASS (qwen legs) -- the
# numbers bank against the refs in the docs tables, they do not gate.
set -uo pipefail
R=/root/build
P=$R/p633
IMG6=vllm-arc:v0.30.0-b70-arc6test
G=$P/r127_l1
GMU_Q=0.85
GMU_G=0.8
QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
mkdir -p $G

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p633r127.log 2>&1 < /dev/null &
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

w14_health() {
  docker exec radiance-test curl -s -o /dev/null -w "%{http_code}" -m 5 \
    http://127.0.0.1:8000/health 2>/dev/null || true
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

skinny_audit() {  # $1 = leg dir $2 = expected lane tag
  { echo "wrap_installed=$(gcount 'skinny-M wrap installed')";
    echo "route_first_$2=$(gcount "skinny-M route FIRST $2")";
    echo "route_failed=$(gcount 'route FAILED')"; } > "$1/skinny_audit.txt"
  cat "$1/skinny_audit.txt"
}

leg_done() {  # $1 = leg dir $2 = lane tag -- serve.log + markers before kill
  skinny_audit "$1" "$2"
  erraudit "$1"
  docker exec arc-test grep -m4 "Available KV cache memory" /root/serve_full.log > "$1/kv_lines.txt" 2>/dev/null || true
  docker cp arc-test:/root/serve_full.log "$1/serve.log" >/dev/null 2>&1 || true
  kill_arc
}

echo "=== [r127] waiting for r126 window to exit ($(date +%H:%M:%S)) ==="
while pgrep -f "bash .*r126_skinny_kbench" >/dev/null 2>&1; do sleep 60; done
WH=""
for i in $(seq 1 40); do
  WH=$(w14_health)
  [ "$WH" = "200" ] && break
  sleep 60
done
echo "W14 health before r127: $WH ($(date +%H:%M:%S))"

docker stop radiance-test >/dev/null 2>&1 || true
sleep 10
docker rm -f arc-test >/dev/null 2>&1 || true

# ---------- Leg A: qwen fp8 (skinny w16 champion lane) ----------
LA=$G/leg_qwen_fp8
mkdir -p $LA
unset ARC_MXFP8_MODE ARC_MXFP4_MODE ARC_QUANTCFG ARC_QUANT ARC_MAXLEN
echo "=== [leg A qwen fp8] boot $(date +%H:%M:%S) (default posture, skinny on) ==="
echo "leg=A model=qwen3.8-27b-fp8 img=$IMG6 gmu=$GMU_Q graph=1 spec=3 lmc=1 skinny=on(default)" > $LA/leg_config.txt
ARC_IMAGE=$IMG6 ARC_NAME=arc-test \
ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
ARC_GMU=$GMU_Q \
ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_arc_b70.sh
if ! boot_wait qwen_fp8; then
  echo "BOOT_FAILED leg_qwen_fp8 -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $LA/serve.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $LA/serve.log 2>/dev/null | tail -90 || true
  kill_arc; exit 1
fi
echo "--- leg_qwen_fp8 HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-27b-fp8\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > $LA/probe.out 2>&1
head -c 300 $LA/probe.out; echo
grep -q '"content"' $LA/probe.out && battery $LA qwen3.8-27b-fp8 qwen || echo "PROBE_STALLED leg_qwen_fp8"
leg_done $LA w16
echo "=== [leg A] done $(date +%H:%M:%S) ==="

# ---------- Leg B: qwen mxfp8 w8a16 (skinny m816 lane) ----------
LB=$G/leg_qwen_mxfp8_w8a16
mkdir -p $LB
export ARC_QUANTCFG="$QCFG"
unset ARC_MXFP8_MODE
echo "=== [leg B qwen mxfp8] boot $(date +%H:%M:%S) (quant=mxfp8-online w8a16 route) ==="
echo "leg=B model=qwen3.8-27b quant=mxfp8-w8a16 img=$IMG6 gmu=$GMU_Q graph=1 spec=3 lmc=1 qcfg=$QCFG skinny=on(default)" > $LB/leg_config.txt
ARC_IMAGE=$IMG6 ARC_NAME=arc-test \
ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b \
ARC_QUANT=mxfp8 \
ARC_GMU=$GMU_Q \
ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh
if ! boot_wait qwen_m816; then
  echo "BOOT_FAILED leg_qwen_mxfp8_w8a16 -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $LB/serve.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $LB/serve.log 2>/dev/null | tail -90 || true
  kill_arc; exit 1
fi
echo "--- leg_qwen_mxfp8_w8a16 HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-27b\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > $LB/probe.out 2>&1
head -c 300 $LB/probe.out; echo
grep -q '"content"' $LB/probe.out && battery $LB qwen3.8-27b qwen || echo "PROBE_STALLED leg_qwen_mxfp8_w8a16"
docker exec arc-test grep -m2 "mxfp8 w8a16 reroute ACTIVE" /root/serve_full.log > $LB/reroute_marker.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Using XPUMxFp8LinearKernel for MXFP8 GEMM" /root/serve_full.log > $LB/kernel_select.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Quantized .* layers of types" /root/serve_full.log > $LB/online_dispatch.txt 2>/dev/null || true
leg_done $LB m816
echo "=== [leg B] done $(date +%H:%M:%S) ==="

# ---------- Leg C: qwen mxfp4 w4a16 (skinny g16 lane) ----------
LC=$G/leg_qwen_mxfp4_w4a16
mkdir -p $LC
unset ARC_MXFP8_MODE
echo "=== [leg C qwen mxfp4] boot $(date +%H:%M:%S) (quant=mxfp4-online w4a16 route) ==="
echo "leg=C model=qwen3.8-27b quant=mxfp4-w4a16 img=$IMG6 gmu=$GMU_Q graph=1 spec=3 lmc=1 qcfg=$QCFG skinny=on(default)" > $LC/leg_config.txt
ARC_IMAGE=$IMG6 ARC_NAME=arc-test \
ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b \
ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 \
ARC_GMU=$GMU_Q \
ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh
if ! boot_wait qwen_g16; then
  echo "BOOT_FAILED leg_qwen_mxfp4_w4a16 -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $LC/serve.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $LC/serve.log 2>/dev/null | tail -90 || true
  kill_arc; exit 1
fi
echo "--- leg_qwen_mxfp4_w4a16 HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-27b\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > $LC/probe.out 2>&1
head -c 300 $LC/probe.out; echo
grep -q '"content"' $LC/probe.out && battery $LC qwen3.8-27b qwen || echo "PROBE_STALLED leg_qwen_mxfp4_w4a16"
docker exec arc-test grep -m2 "reroute ACTIVE" /root/serve_full.log > $LC/reroute_marker.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Using .* GEMM" /root/serve_full.log > $LC/kernel_select.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Quantized .* layers of types" /root/serve_full.log > $LC/online_dispatch.txt 2>/dev/null || true
leg_done $LC g16
echo "=== [leg C] done $(date +%H:%M:%S) ==="

# ---------- Leg D: gemma mxfp8 w8a16 (skinny m816, ARC_MAXLEN=40960 law) ----------
LD=$G/leg_gemma_mxfp8_w8a16_ml40k
mkdir -p $LD
unset ARC_MXFP8_MODE ARC_MXFP4_MODE ARC_QUANTCFG
echo "=== [leg D gemma mxfp8] boot $(date +%H:%M:%S) (quant=mxfp8 w8a16, maxlen 40960) ==="
echo "leg=D model=gemma4-31b quant=mxfp8-w8a16 img=$IMG6 gmu=$GMU_G graph=1 spec=0 lmc=0 rpc=1200 maxlen=40960 skinny=on(default)" > $LD/leg_config.txt
ARC_IMAGE=$IMG6 ARC_NAME=arc-test \
ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
ARC_QUANT=mxfp8 \
ARC_GMU=$GMU_G \
ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 \
ARC_MAXLEN=40960 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh
if ! boot_wait gemma_m816; then
  echo "BOOT_FAILED leg_gemma_mxfp8_w8a16_ml40k -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $LD/serve.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $LD/serve.log 2>/dev/null | tail -90 || true
  kill_arc; exit 1
fi
echo "--- leg_gemma_mxfp8_w8a16_ml40k HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gemma4-31b\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > $LD/probe.out 2>&1
head -c 300 $LD/probe.out; echo
grep -q '"content"' $LD/probe.out && battery $LD gemma4-31b none || echo "PROBE_STALLED leg_gemma_mxfp8_w8a16_ml40k"
docker exec arc-test grep -m2 "mxfp8 w8a16 reroute ACTIVE" /root/serve_full.log > $LD/reroute_marker.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Using XPUMxFp8LinearKernel for MXFP8 GEMM" /root/serve_full.log > $LD/kernel_select.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Quantized .* layers of types" /root/serve_full.log > $LD/online_dispatch.txt 2>/dev/null || true
leg_done $LD m816
echo "=== [leg D] done $(date +%H:%M:%S) ==="

# ---------- Leg E: gemma mxfp4 w4a16 (skinny g16 lane) ----------
LE=$G/leg_gemma_mxfp4_w4a16
mkdir -p $LE
unset ARC_MXFP8_MODE
echo "=== [leg E gemma mxfp4] boot $(date +%H:%M:%S) (quant=mxfp4 w4a16) ==="
echo "leg=E model=gemma4-31b quant=mxfp4-w4a16 img=$IMG6 gmu=$GMU_G graph=1 spec=0 lmc=0 rpc=1200 skinny=on(default)" > $LE/leg_config.txt
ARC_IMAGE=$IMG6 ARC_NAME=arc-test \
ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 \
ARC_GMU=$GMU_G \
ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 \
ARC_FP8=0 \
  bash $R/serve_arc_b70.sh
if ! boot_wait gemma_g16; then
  echo "BOOT_FAILED leg_gemma_mxfp4_w4a16 -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $LE/serve.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $LE/serve.log 2>/dev/null | tail -90 || true
  kill_arc; exit 1
fi
echo "--- leg_gemma_mxfp4_w4a16 HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gemma4-31b\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > $LE/probe.out 2>&1
head -c 300 $LE/probe.out; echo
grep -q '"content"' $LE/probe.out && battery $LE gemma4-31b none || echo "PROBE_STALLED leg_gemma_mxfp4_w4a16"
docker exec arc-test grep -m2 "reroute ACTIVE" /root/serve_full.log > $LE/reroute_marker.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Using .* GEMM" /root/serve_full.log > $LE/kernel_select.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Quantized .* layers of types" /root/serve_full.log > $LE/online_dispatch.txt 2>/dev/null || true
leg_done $LE g16
echo "=== [leg E] done $(date +%H:%M:%S) ==="

echo "=== [r127] all legs done $(date +%H:%M:%S); W14 restoring ==="
exit 0
