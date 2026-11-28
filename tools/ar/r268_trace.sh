#!/bin/bash
# r268_trace.sh -- P6.47 P-D (r268): the G3 trace leg -- ONE c3-posture
# boot (gemma, lawful g3 skin, KV mxfp8 native) with the TRACED mx_kv
# overlay (ARC_MXKV_OVERLAY=/root/build/r268_overlay -- built by
# tools/ar/r268_build_trace_overlay.sh: byte-copy of the image's own
# package -- md5-proven parity 2026-11-28, engine.py d1fa3c17 both
# sides -- plus the decode-side decx trace) and ARC_MXKV_TRACE=1, a
# 2-prompt x 8-token workload, then banks serve.log + the MXKVTRACE
# extract for the store-smap vs bt-walk comparison (the direct engine
# readout IF the r267 canary PASSES: e2 eliminated the graph
# hypothesis -- eager MX reproduces c3 byte-identically, so the
# defect is the eager decode-step store/walk; prefill never reads the
# cache (raw flash first chunk), so a wrong-table/wrong-pool-space
# read lands exactly at token 2 -- this leg sees the addressing).
# Requests: 2 raw completions (text path -- the attractor) + 1 with
# logprobs (the 400-nan reproduction check on this boot). W14 quiesce
# before, clean-recipe trap restore after (r262 law). Artifacts ->
# /root/build/p647/r268_trace/{out,logs}.  Exit 0 = leg banked.
# Launch (ship + launch in SEPARATE ssh calls, md5-verified ship):
#   nohup setsid bash /root/build/r268/r268_trace.sh \
#     > /root/build/p647/r268_trace/logs/launch.log 2>&1 &
set -uo pipefail
R=/root/build
P=$R/p647/r268_trace
S=$R/r268
OVL=$R/r268_overlay
IMG=vllm-arc:v0.30.0-b70-cert29
CNAME=arc-r268

[ -f "$OVL/engine.py" ] || { echo "R268_NO_OVERLAY (ship tools/ar/p268_trace/overlay first)"; exit 1; }
[ -f "$S/r268_trace.sh" ] || true
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R268_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R268_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R268_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R268_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"; return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"; return 0
  fi
  bash $R/r262_w14_restore.sh > $P/logs/w14_restore_r268.log 2>&1
  rc=$?
  grep -E "^W14_" $P/logs/w14_restore_r268.log | tail -3 || true
  echo "W14_RESTORE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$P/logs/w14_restore_r268.log"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out
echo "R268_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) overlay_engine_md5=$(md5sum $OVL/engine.py | cut -d' ' -f1)"

echo "=== [r268] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R268_WARN_PORT8000_STILL_HELD"
sleep 5
docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r268] boot c3 posture EAGER + traced overlay + ARC_MXKV_TRACE=1 $(date +%H:%M:%S) ==="
# EAGER LAW: e2 (04:15Z) proved ARC_GRAPH=0 reproduces the c3 corruption
# -- and under capture the trace python runs ONCE over the dummy batch
# (replays never re-run it; the native_decode.py telemetry law), so the
# live store/walk addressing at real decode steps only prints EAGER.
# The r266 invocation law: env FIRST, backgrounded, boot.out polled --
# serve_arc_b70.sh reads env vars, never positionals
nohup env \
  ARC_IMAGE=$IMG ARC_NAME=$CNAME \
  ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
  ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 \
  ARC_RPC_TIMEOUT=1200 ARC_MAXLEN=40960 ARC_FP8=0 \
  ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja \
  ARC_RPARSER= ARC_TPARSER= \
  ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 \
  ARC_MXKV_OVERLAY=$OVL ARC_MXKV_TRACE=1 \
  bash $R/r266/serve_arc_b70.sh > $P/logs/boot.out 2>&1 &
BOOTPID=$!
B=0
while [ $B -lt 60 ]; do
  grep -q "ARC_SERVE_LAUNCHED" $P/logs/boot.out 2>/dev/null && break
  kill -0 $BOOTPID 2>/dev/null || break
  sleep 2; B=$((B+1))
done
if ! grep -q "ARC_SERVE_LAUNCHED" $P/logs/boot.out 2>/dev/null; then
  echo "R268_BOOT_FAIL (no serve line after $((B*2))s)"; tail -20 $P/logs/boot.out; exit 3
fi
grep -E "ARC_SERVE_LAUNCHED" $P/logs/boot.out | head -1 | cut -c1-220

echo "=== [r268] health poll (identity-verified) ==="
H=0; HEALTH=""
while [ $H -lt 1500 ]; do
  M=$(curl -s --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null || true)
  echo "$M" | grep -q "gemma4-31b" && { HEALTH="$M"; break; }
  sleep 3; H=$((H+1))
done
[ -n "$HEALTH" ] || { echo "R268_HEALTH_FAIL (no gemma4-31b identity after $((H*3))s)"; exit 4; }
echo "HEALTH_OK_r268 $(date -u +%H:%M:%SZ) (identity-verified)"

echo "=== [r268] the 2x8 workload + the nan detector ==="
curl -s --max-time 60 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-31b","prompt":"The capital of France is","max_tokens":8,"temperature":0}' \
  > $P/out/gen1.json || echo "gen1 http_err"
curl -s --max-time 60 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-31b","prompt":"def fibonacci(n):","max_tokens":8,"temperature":0}' \
  > $P/out/gen2.json || echo "gen2 http_err"
curl -s --max-time 60 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-31b","prompt":"The capital of France is","max_tokens":8,"temperature":0,"logprobs":1}' \
  > $P/out/gen3_nanprobe.json || echo "gen3 http_err"
echo "gen1: $(python3 -c "import json;print(json.load(open('$P/out/gen1.json'))['choices'][0]['text'])" 2>/dev/null || head -c 200 $P/out/gen1.json)"
echo "gen2: $(python3 -c "import json;print(json.load(open('$P/out/gen2.json'))['choices'][0]['text'])" 2>/dev/null || head -c 200 $P/out/gen2.json)"
echo "gen3: $(head -c 200 $P/out/gen3_nanprobe.json)"

docker cp $CNAME:/root/serve_full.log $P/out/serve.log >/dev/null 2>&1 || echo "R268_SERVE_LOG_MISSING"
docker rm -f $CNAME >/dev/null 2>&1

grep -c "MXKVTRACE" $P/out/serve.log > $P/out/trace_count.txt 2>/dev/null || true
grep -E "MXKVTRACE (store|decx)" $P/out/serve.log > $P/out/trace_extract.txt 2>/dev/null || true
echo "=== [r268] trace lines: $(cat $P/out/trace_count.txt 2>/dev/null) total, $(wc -l < $P/out/trace_extract.txt) store/decx"
grep -E "MXKVTRACE store" $P/out/serve.log | tail -4 || true
grep -E "MXKVTRACE decx" $P/out/serve.log | tail -4 || true

echo "R268_DONE rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
