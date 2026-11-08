#!/bin/bash
# r103_mxfp4_boot.sh — P6.26 phase-A boot legs (quant-path isolation).
#
# Two pre-quantized mxfp4 checkpoints, first boots on the cert10 image
# (plan §P6.26, gates G0). Isolation posture — eager, MTP off, LMC off —
# so the ONLY new variable vs the certified fp8 incumbent is the weight
# quant path. Full-posture legs (graph+spec3+lmc) run as phase B after a
# green phase A.
#
# Predictions pre-registered in the plan (§P6.26 issue matrix):
#   P-1 qurk: platform/scheme-level rejection of quant_method=quark on XPU
#   P-2 rhmxfp4: boots; spurious supports_mx warning (cosmetic)
# Telemetry capture points per leg (E-law): full serve log; the
# "Using <K> for MXFP4 GEMM" dispatch line(s); quant-config resolution
# lines; any warning/ValueError on the scheme path.
#
# W14 stopped for the window, restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p626
mkdir -p $CT

LEG=${1:?leg required: rhmxfp4 | qurk}
case "$LEG" in
  rhmxfp4) MODEL=/models/qwen3.8-27b-mxfp4xfp8;   SERVED=qwen3.8-27b-mxfp4xfp8 ;;
  qurk)    MODEL=/models/qwen3.8-27b-quark-awq-mxfp4; SERVED=qwen3.8-27b-quark-awq-mxfp4 ;;
  *) echo "unknown leg $LEG"; exit 1 ;;
esac

w14_start() {
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p626.log 2>&1 < /dev/null &
}
trap w14_start EXIT

docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

# ship the P6.26-knobbed serve interface (idempotent; defaults unchanged)
[ -f $R/serve_arc_b70.sh ] || { echo "FATAL: $R/serve_arc_b70.sh missing"; exit 1; }

echo "=== [$LEG] phase-A boot start $(date +%H:%M:%S) ==="
docker rm -f arc-test >/dev/null 2>&1 || true   # model-mount freshness (P6.26 knob)
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert10 \
ARC_NAME=arc-test \
ARC_MODEL=$MODEL ARC_SERVED=$SERVED ARC_QUANT= \
ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh

# health-wait: config errors land in <60s; weight load of 13-22 GB takes
# minutes. Poll 14 min, then probe generation.
UP=0
for i in $(seq 1 168); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then UP=1; break; fi
  # early-exit on hard death of the serve process (config-phase failure)
  if ! docker exec arc-test pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "SERVE PROCESS DIED at poll $i ($(date +%H:%M:%S)) — config/parse-phase failure"; break
  fi
  sleep 5
done
docker cp arc-test:/root/serve_full.log $CT/${LEG}_phaseA.log >/dev/null 2>&1 || true

echo "=== [$LEG] telemetry extraction ==="
grep -n "Using .* for MXFP4 GEMM\|MXFP4\|mxfp4\|quark\|compressed_tensors\|CompressedTensors\|supports_mx\|Simulated\|ValueError\|not supported\|quantization config\|Detected" $CT/${LEG}_phaseA.log 2>/dev/null | head -40

if [ "$UP" = 1 ]; then
  echo "=== [$LEG] HEALTH UP at $(date +%H:%M:%S) — generation probe ==="
  docker exec arc-test curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED\",\"max_tokens\":24,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say the number seven in words, then stop.\"}]}" \
    | head -c 1200
  echo; echo "PROBE_DONE $LEG $(date +%H:%M:%S)"
else
  echo "=== [$LEG] BOOT FAILED — decisive tail ==="
  tail -50 $CT/${LEG}_phaseA.log 2>/dev/null
fi
