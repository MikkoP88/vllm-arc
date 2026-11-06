#!/bin/bash
# r106_mxfp8_boot.sh — P6.27 phase-A boot leg (quant-path isolation).
#
# intel-ai/Qwen3.8-27B-MXFP8-MXFP8MTP-CT-AutoRound, first boot on the
# CERTIFIED cert11 image (plan §P6.27, gate G0). Isolation posture —
# eager, MTP off, LMC off — so the ONLY new variable vs the certified
# fp8 incumbent is the MXFP8 W8A8 weight path (+ the quantized MTP
# drafter weights at load). Full-posture legs run after a green G0.
#
# r106 recon (plan §P6.27) predicts NO platform rejection:
#   - quant_method=compressed-tensors (platform list since P0)
#   - _is_mxfp8 matches all six predicates -> CompressedTensorsW8A8Mxfp8
#   - _POSSIBLE_MXFP8_KERNELS[XPU] -> XPUMxFp8LinearKernel ->
#     torch.ops._xpu_C.fp8_gemm (oneDNN) + xpu_mxfp8_quantize(x)
# Predictions P-1 (param mismatch at load, low) / P-2 (quantized-MTP
# numerics, gated in G1) / P-3 (graph interaction, later phase) — the
# first boot decides P-1.
#
# Telemetry capture points (E-law): full serve log; the
# "Using XPUMxFp8LinearKernel for MXFP8 GEMM" dispatch line
# (kernels/linear/__init__.py:923); CT scheme-resolution lines; MTP
# drafter load lines; any ValueError/param error at weight load.
#
# W14 stopped for the window, restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p627
mkdir -p $CT

MODEL=/models/qwen3.8-27b-mxfp8
SERVED=qwen3.8-27b-mxfp8

# [0] quiescence guard (r105 lesson: never boot/measure through a
# host-side model download)
if pgrep -f "snapshot_download" >/dev/null 2>&1; then
  echo "FATAL: host model download in flight — quiesce the rig before [0]"; exit 1
fi
uptime > $CT/host_load.txt

w14_start() {
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p627.log 2>&1 < /dev/null &
}
trap w14_start EXIT

# stop W14 only after a health sanity read (r105t lesson)
WH=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health 2>/dev/null || true)
echo "W14 health before stop: $WH"
docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

echo "=== [mxfp8] phase-A boot start $(date +%H:%M:%S) (cert11) ==="
docker rm -f arc-test >/dev/null 2>&1 || true   # model-mount freshness
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert11 \
ARC_NAME=arc-test \
ARC_MODEL=$MODEL ARC_SERVED=$SERVED ARC_QUANT= \
ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh

# health-wait: config errors land in <60s; ~15 GiB/rank weight load
# takes minutes. Poll 14 min, then probe generation.
UP=0
for i in $(seq 1 168); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then UP=1; break; fi
  if ! docker exec arc-test pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "SERVE PROCESS DIED at poll $i ($(date +%H:%M:%S)) — config/parse-phase failure"; break
  fi
  sleep 5
done
docker cp arc-test:/root/serve_full.log $CT/mxfp8_phaseA.log >/dev/null 2>&1 || true

echo "=== [mxfp8] telemetry extraction ==="
grep -n "Using .* for MXFP8 GEMM\|MXFP8\|mxfp8\|compressed_tensors\|CompressedTensors\|Mxfp8\|ValueError\|not supported\|quantization config\|Detected\|mtp\|num_speculative" $CT/mxfp8_phaseA.log 2>/dev/null | head -60

if [ "$UP" = 1 ]; then
  echo "=== [mxfp8] HEALTH UP at $(date +%H:%M:%S) — generation probe ==="
  docker exec arc-test curl -s http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED\",\"max_tokens\":24,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say the number seven in words, then stop.\"}]}" \
    | head -c 1200
  echo; echo "PROBE_DONE mxfp8 $(date +%H:%M:%S)"
else
  echo "=== [mxfp8] BOOT FAILED — decisive tail ==="
  tail -60 $CT/mxfp8_phaseA.log 2>/dev/null
fi
