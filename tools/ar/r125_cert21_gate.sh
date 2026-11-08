#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r125_cert21_gate.sh -- the cert21 release gate (r121 cert20 precedent,
# three lanes because the cert21 overlay touches three: wheel arc4->arc5
# carries the fp8 champion's fp8_gemm_w8a16 + fp4_gemm_w4a16, adds
# fp8_gemm_w8a16_mxfp8; patches/mxfp4_w4a16.py gains the w4a8 probe
# branch; patches/mxfp8_w8a16.py + platform entry are new).
# cert21 = cert20 + the P6.31 online-MXFP8 W8A16 lane (arc5 wheel +
# mxfp8_w8a16 patch + platform entry, via tools/cert21.Dockerfile).
# Flow: wait r124b exit -> W14 health (docker-exec probe -- radiance-test
# law) -> BUILD cert21 (GPU-free, W14 stays live) -> manifest capture ->
# stop W14 ->
#   leg A: qwen fp8 DISCRIMINATOR (default posture -- proves the champion
#         lane untouched by the wheel swap; bars = cert20 r121 leg A, -5%
#         block law) ->
#   leg B: qwen ARC_QUANT=mxfp8, ARC_MXFP8_MODE unset (= the w8a16 route
#         default; reroute ACTIVE marker + dispatch + battery + spec gate;
#         cross-check band = the r124 arc5test rows) ->
#   leg C: qwen ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 (carry proof of the
#         UPDATED mxfp4 patch on the arc5 wheel; bars = cert20 r121 leg B)
# -> verdict -> trap W14 restore.
# Evidence: /root/build/p631/cert21_l1/.
#
# Pre-registered bars (banked cert20 gate refs, r121 cert20_l1; -5% block):
#   leg A fp8:  U8k <= 3.99 (ref 3.80) / U32k <= 18.43 (ref 17.55)
#               solo min >= 56.5 (ref 59.5) / agg8 min >= 286.7 (ref 301.8)
#               SPEC_GATE PASS
#   leg B m816: reroute ACTIVE + XPUMxFp8LinearKernel select present;
#               U8k <= 3.79 (r124 3.61 +5%) / U32k <= 17.92 (17.07 +5%)
#               solo min >= 51.1 (53.8 -5%) / agg8 min >= 252.6 (265.9 -5%)
#               SPEC_GATE PASS
#   leg C w4a16: reroute ACTIVE; U8k <= 3.81 (ref 3.63) / U32k <= 17.89
#               (ref 17.04) / solo min >= 58.4 (ref 61.5) / agg8 min
#               >= 286.0 (ref 301.1)
#   all legs: ERROR=0 Traceback=0
set -uo pipefail
R=/root/build
CT=$R/p631
C21=$R/p631/cert21ctx
IMG21=vllm-arc:v0.30.0-b70-cert21
G=$CT/cert21_l1
GMU=0.85
WHEEL_SHA=219268cf7d24ebc7a99e5afec5097edd63d88ef26255511c589da71f3615399a
QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
GATE_FAIL=0
mkdir -p $G

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p631r125.log 2>&1 < /dev/null &
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

battery() {  # battery <legdir> <served>
  local L=$1 S=$2
  python3 $R/r113_battery.py g1 $S > $L/g1_probes.txt 2>&1 || true
  cat $L/g1_probes.txt
  ( cd $CT && P66_FACES="U8k,U32k" timeout 1200 python3 $R/p66/p66_solo_ttft.py $L/faces.json > $L/faces.txt 2>&1 ) || true
  cat $L/faces.txt
  python3 $R/r113_battery.py decode $S > $L/decode.txt 2>&1 || true
  cat $L/decode.txt
  echo "--- spec gate (MTP k3) ---"
  python3 $R/arc_spec_gate.py --model $S > $L/spec_gate.txt 2>&1
  echo "spec_gate exit=$?"
  tail -2 $L/spec_gate.txt
}

erraudit() {  # $1 = leg dir
  docker exec arc-test bash -c "grep -c ERROR /root/serve_full.log; grep -c Traceback /root/serve_full.log" \
    > "$1/erraudit.txt" 2>/dev/null || echo "?" > "$1/erraudit.txt"
  cat "$1/erraudit.txt"
  grep -q "^0$" "$1/erraudit.txt" || { echo "ERRAUDIT: FAIL ($1)"; GATE_FAIL=1; }
}

echo "=== [r125] waiting for r124b to exit ($(date +%H:%M:%S)) ==="
while pgrep -f "bash .*r124b_gemma_window" >/dev/null 2>&1; do sleep 60; done
WH=""
for i in $(seq 1 40); do
  WH=$(w14_health)
  [ "$WH" = "200" ] && break
  sleep 60
done
echo "W14 health before cert21 gate: $WH"

FORK_SHA=$(cat $C21/fork_sha.txt 2>/dev/null || echo "")
if [ -z "$FORK_SHA" ]; then
  echo "FATAL: $C21/fork_sha.txt missing -- cannot stamp cert21_fork_sha"
  exit 1
fi
BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r125] BUILD cert21 FORK_SHA=$FORK_SHA BUILT=$BUILT ($(date +%H:%M:%S); GPU-free, W14 stays live) ==="
if ! docker build -f $C21/cert21.Dockerfile \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  --build-arg WHEEL_SHA256=$WHEEL_SHA \
  -t $IMG21 $C21 > $G/build.log 2>&1; then
  echo "BUILD_FAILED cert21 -- tail:"
  tail -40 $G/build.log
  exit 1
fi
grep -c "CERT21_W8A16_LIVE" $G/build.log || true
echo "BUILD OK $(date +%H:%M:%S)"
docker run --rm --entrypoint cat $IMG21 /etc/vllm-arc/IMAGE_MANIFEST > $G/manifest.txt 2>/dev/null || true
grep -E "cert21|fork_sha" $G/manifest.txt || echo "WARN: no cert21 rows in manifest"

docker stop radiance-test >/dev/null 2>&1 || true
sleep 10
docker rm -f arc-test >/dev/null 2>&1 || true

# ---------- Leg A: qwen fp8 DISCRIMINATOR (default certified posture) ----------
LA=$G/leg_fp8_disc
mkdir -p $LA
unset ARC_MXFP8_MODE ARC_MXFP4_MODE ARC_QUANTCFG ARC_QUANT ARC_MAXLEN
echo "=== [leg A fp8 disc] boot $(date +%H:%M:%S) (model=qwen3.8-27b-fp8 native, default posture) ==="
echo "leg=A model=qwen3.8-27b-fp8 img=$IMG21 gmu=$GMU graph=1 spec=3 lmc=1 (default posture, no lane envs)" > $LA/leg_config.txt
ARC_IMAGE=$IMG21 ARC_NAME=arc-test \
ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
ARC_GMU=$GMU \
ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_arc_b70.sh
if ! boot_wait fp8_disc; then
  echo "BOOT_FAILED leg_fp8_disc -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $LA/serve.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $LA/serve.log 2>/dev/null | tail -90 || true
  kill_arc; exit 1
fi
echo "--- leg_fp8_disc HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-27b-fp8\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > $LA/probe.out 2>&1
head -c 300 $LA/probe.out; echo
if grep -q '"content"' $LA/probe.out; then
  battery $LA qwen3.8-27b-fp8
else
  echo "PROBE_STALLED leg_fp8_disc"; GATE_FAIL=1
fi
{ echo "fa2_marker=$(gcount '\[ARC-P628-FA2\]')"; \
  echo "slide_marker=$(gcount '\[ARC-P629-SLIDE\]')"; } > $LA/marker_audit.txt
cat $LA/marker_audit.txt
erraudit $LA
docker exec arc-test grep -m4 "Available KV cache memory" /root/serve_full.log > $LA/kv_lines.txt 2>/dev/null || true
docker cp arc-test:/root/serve_full.log $LA/serve.log >/dev/null 2>&1 || true
kill_arc
echo "=== [leg A] done $(date +%H:%M:%S) ==="

# ---------- Leg B: qwen online mxfp8 W8A16 (the cert21 lane) ----------
LB=$G/leg_w8a16
mkdir -p $LB
export ARC_QUANTCFG="$QCFG"
unset ARC_MXFP8_MODE
echo "=== [leg B w8a16] boot $(date +%H:%M:%S) (model=qwen3.8-27b BASE, quant=mxfp8-online w8a16) ==="
echo "leg=B model=qwen3.8-27b quant=mxfp8-w8a16 img=$IMG21 gmu=$GMU graph=1 spec=3 lmc=1 qcfg=$QCFG mode=w8a16(default-route)" > $LB/leg_config.txt
ARC_IMAGE=$IMG21 ARC_NAME=arc-test \
ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b \
ARC_QUANT=mxfp8 \
ARC_GMU=$GMU \
ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh
if ! boot_wait w8a16; then
  echo "BOOT_FAILED leg_w8a16 -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $LB/serve.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $LB/serve.log 2>/dev/null | tail -90 || true
  kill_arc; exit 1
fi
echo "--- leg_w8a16 HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-27b\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > $LB/probe.out 2>&1
head -c 300 $LB/probe.out; echo
docker exec arc-test grep -m2 "mxfp8 w8a16 reroute ACTIVE" /root/serve_full.log > $LB/reroute_marker.txt 2>/dev/null || true
cat $LB/reroute_marker.txt
docker exec arc-test grep -m1 "Quantized .* layers of types" /root/serve_full.log > $LB/online_dispatch.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Using XPUMxFp8LinearKernel for MXFP8 GEMM" /root/serve_full.log > $LB/kernel_select.txt 2>/dev/null || true
cat $LB/online_dispatch.txt $LB/kernel_select.txt 2>/dev/null
[ -s "$LB/reroute_marker.txt" ] || { echo "REROUTE MARKER: FAIL (leg B)"; GATE_FAIL=1; }
[ -s "$LB/kernel_select.txt" ] || { echo "KERNEL SELECT: FAIL (leg B)"; GATE_FAIL=1; }
if grep -q '"content"' $LB/probe.out; then
  battery $LB qwen3.8-27b
else
  echo "PROBE_STALLED leg_w8a16"; GATE_FAIL=1
fi
erraudit $LB
docker exec arc-test grep -m4 "Available KV cache memory" /root/serve_full.log > $LB/kv_lines.txt 2>/dev/null || true
docker cp arc-test:/root/serve_full.log $LB/serve.log >/dev/null 2>&1 || true
kill_arc
echo "=== [leg B] done $(date +%H:%M:%S) ==="

# ---------- Leg C: qwen online mxfp4 W4A16 (carry proof, updated patch) ----------
LC=$G/leg_w4a16
mkdir -p $LC
export ARC_MXFP4_MODE=w4a16
echo "=== [leg C w4a16] boot $(date +%H:%M:%S) (model=qwen3.8-27b BASE, quant=mxfp4-online w4a16) ==="
echo "leg=C model=qwen3.8-27b quant=mxfp4-w4a16 img=$IMG21 gmu=$GMU graph=1 spec=3 lmc=1 qcfg=$QCFG mode=w4a16" > $LC/leg_config.txt
ARC_IMAGE=$IMG21 ARC_NAME=arc-test \
ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b \
ARC_QUANT=mxfp4 \
ARC_GMU=$GMU \
ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh
if ! boot_wait w4a16; then
  echo "BOOT_FAILED leg_w4a16 -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $LC/serve.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $LC/serve.log 2>/dev/null | tail -90 || true
  kill_arc; exit 1
fi
echo "--- leg_w4a16 HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen3.8-27b\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > $LC/probe.out 2>&1
head -c 300 $LC/probe.out; echo
docker exec arc-test grep -m2 "reroute ACTIVE" /root/serve_full.log > $LC/reroute_marker.txt 2>/dev/null || true
cat $LC/reroute_marker.txt
docker exec arc-test grep -m1 "Quantized .* layers of types" /root/serve_full.log > $LC/online_dispatch.txt 2>/dev/null || true
docker exec arc-test grep -m1 "Using .* GEMM" /root/serve_full.log > $LC/kernel_select.txt 2>/dev/null || true
cat $LC/online_dispatch.txt $LC/kernel_select.txt 2>/dev/null
[ -s "$LC/reroute_marker.txt" ] || { echo "REROUTE MARKER: FAIL (leg C)"; GATE_FAIL=1; }
if grep -q '"content"' $LC/probe.out; then
  battery $LC qwen3.8-27b
else
  echo "PROBE_STALLED leg_w4a16"; GATE_FAIL=1
fi
erraudit $LC
docker exec arc-test grep -m4 "Available KV cache memory" /root/serve_full.log > $LC/kv_lines.txt 2>/dev/null || true
docker cp arc-test:/root/serve_full.log $LC/serve.log >/dev/null 2>&1 || true
kill_arc
echo "=== [leg C] done $(date +%H:%M:%S) ==="

echo "=== [V] verdict vs pre-registered bars $(date +%H:%M:%S) ==="
python3 - > verdict.txt 2>&1 << 'PYEOF' || true
import re

def faces(p):
    d = {}
    try:
        for line in open(p, errors="replace"):
            m = re.search(r"(U8k|U32k): prompt=\d+ cached=\S+ TTFT=\s*([0-9.]+)s", line)
            if m:
                d[m.group(1)] = float(m.group(2))
    except OSError:
        pass
    return d

def decode(p):
    solo, agg = [], []
    try:
        for line in open(p, errors="replace"):
            m = re.search(r"solo conc1 rep\d+ wall=[0-9.]+s ctok=\d+ tps=([0-9.]+)", line)
            if m:
                solo.append(float(m.group(1)))
            m = re.search(r"agg8 conc8 rep\d+ wall=[0-9.]+s ctok=\d+ tps=([0-9.]+)", line)
            if m:
                agg.append(float(m.group(1)))
    except OSError:
        pass
    return solo, agg

def spec_pass(p):
    try:
        return "SPEC_GATE PASS" in open(p, errors="replace").read()
    except OSError:
        return False

fails = []
# leg A: cert20 r121 refs -5%/+5% block law
fA = faces("leg_fp8_disc/faces.txt"); sA, aA = decode("leg_fp8_disc/decode.txt")
bars_A = {"U8k": 3.99, "U32k": 18.43}
for k, bar in bars_A.items():
    if k not in fA:
        fails.append(f"A {k} face missing")
    else:
        print(f"A {k}: TTFT={fA[k]:.2f}s bar<={bar} ({(fA[k]/bar-1)*100:+.1f}%)")
        if fA[k] > bar: fails.append(f"A {k} {fA[k]} > {bar}")
smin = min(sA) if sA else 0; amin = min(aA) if aA else 0
print(f"A solo min={smin:.2f} bar>=56.5; agg8 min={amin:.2f} bar>=286.7")
if smin < 56.5: fails.append(f"A solo {smin} < 56.5")
if amin < 286.7: fails.append(f"A agg8 {amin} < 286.7")
if not spec_pass("leg_fp8_disc/spec_gate.txt"): fails.append("A spec gate")
# leg B: r124 arc5test cross-check band
fB = faces("leg_w8a16/faces.txt"); sB, aB = decode("leg_w8a16/decode.txt")
bars_B = {"U8k": 3.79, "U32k": 17.92}
for k, bar in bars_B.items():
    if k not in fB:
        fails.append(f"B {k} face missing")
    else:
        print(f"B {k}: TTFT={fB[k]:.2f}s bar<={bar} ({(fB[k]/bar-1)*100:+.1f}%)")
        if fB[k] > bar: fails.append(f"B {k} {fB[k]} > {bar}")
smin = min(sB) if sB else 0; amin = min(aB) if aB else 0
print(f"B solo min={smin:.2f} bar>=51.1; agg8 min={amin:.2f} bar>=252.6")
if smin < 51.1: fails.append(f"B solo {smin} < 51.1")
if amin < 252.6: fails.append(f"B agg8 {amin} < 252.6")
if not spec_pass("leg_w8a16/spec_gate.txt"): fails.append("B spec gate")
# leg C: cert20 r121 leg-B refs
fC = faces("leg_w4a16/faces.txt"); sC, aC = decode("leg_w4a16/decode.txt")
bars_C = {"U8k": 3.81, "U32k": 17.89}
for k, bar in bars_C.items():
    if k not in fC:
        fails.append(f"C {k} face missing")
    else:
        print(f"C {k}: TTFT={fC[k]:.2f}s bar<={bar} ({(fC[k]/bar-1)*100:+.1f}%)")
        if fC[k] > bar: fails.append(f"C {k} {fC[k]} > {bar}")
smin = min(sC) if sC else 0; amin = min(aC) if aC else 0
print(f"C solo min={smin:.2f} bar>=58.4; agg8 min={amin:.2f} bar>=286.0")
if smin < 58.4: fails.append(f"C solo {smin} < 58.4")
if amin < 286.0: fails.append(f"C agg8 {amin} < 286.0")
if not spec_pass("leg_w4a16/spec_gate.txt"): fails.append("C spec gate")
print("R125_CERT21:", "PASS" if not fails else f"FAIL {fails}")
PYEOF
cat verdict.txt
grep -q "R125_CERT21: PASS" verdict.txt || GATE_FAIL=1

echo "=== [r125] window done $(date +%F\ %H:%M:%S); W14 restoring; GATE_FAIL=$GATE_FAIL ==="
exit $GATE_FAIL
