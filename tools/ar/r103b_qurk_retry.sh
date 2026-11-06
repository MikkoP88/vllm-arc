#!/bin/bash
# r103b_qurk_retry.sh — P6.26 qurk phase-A retry with the platform patch.
# Deploy = docker cp of the patched vllm_gaudi/xpu/platform.py into the
# EXISTING arc-test container (image untouched — cert discipline; the
# certified-image rebuild happens only after gates pass). Requires the
# arc-test container to already exist from the failed r103 qurk leg
# (same image + same model mount, so serve_arc_b70.sh keeps it).
# W14 stopped for the window, restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p626
MODEL=/models/qwen3.8-27b-quark-awq-mxfp4
SERVED=qwen3.8-27b-quark-awq-mxfp4
SP=/opt/venv/lib/python3.12/site-packages

w14_start() {
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p626.log 2>&1 < /dev/null &
}
trap w14_start EXIT

docker stop radiance-test >/dev/null 2>&1 || true
sleep 2

[ -f $CT/platform_p626.py ] || { echo "FATAL: $CT/platform_p626.py missing (ship it first)"; exit 1; }
docker exec arc-test true 2>/dev/null || { echo "FATAL: arc-test container missing"; exit 1; }

echo "=== [qurk-retry] deploying patched platform.py $(date +%H:%M:%S) ==="
docker exec arc-test pkill -f "vllm serve" 2>/dev/null || true
docker cp $CT/platform_p626.py arc-test:$SP/vllm_gaudi/xpu/platform.py
docker exec arc-test md5sum $SP/vllm_gaudi/xpu/platform.py

echo "=== [qurk-retry] serve start $(date +%H:%M:%S) ==="
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert10 \
ARC_NAME=arc-test \
ARC_MODEL=$MODEL ARC_SERVED=$SERVED ARC_QUANT= \
ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh

UP=0
for i in $(seq 1 200); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then UP=1; break; fi
  if ! docker exec arc-test pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "SERVE PROCESS DIED at poll $i ($(date +%H:%M:%S))"; break
  fi
  sleep 5
done
docker cp arc-test:/root/serve_full.log $CT/qurk_phaseA_retry.log >/dev/null 2>&1 || true

echo "=== [qurk-retry] telemetry extraction ==="
grep -n "Using .* for MXFP4 GEMM\|MXFP4\|quark\|Quark\|supports_mx\|Simulated\|ValueError\|not supported\|quantization=quark\|Loading weights took\|Available KV" $CT/qurk_phaseA_retry.log 2>/dev/null | head -30

if [ "$UP" = 1 ]; then
  echo "=== [qurk-retry] HEALTH UP at $(date +%H:%M:%S) — generation probe ==="
  docker exec arc-test curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED\",\"max_tokens\":24,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say the number seven in words, then stop.\"}]}" \
    | head -c 1200
  echo; echo "PROBE_DONE qurk-retry $(date +%H:%M:%S)"
else
  echo "=== [qurk-retry] BOOT FAILED — decisive tail ==="
  tail -45 $CT/qurk_phaseA_retry.log 2>/dev/null
fi
