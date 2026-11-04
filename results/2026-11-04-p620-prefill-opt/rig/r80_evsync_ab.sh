#!/bin/bash
# p620 r80 — evsync A/B first-light on cert8 (bind-mount probe, NO image
# build). Phase a: VLLM_ARC_LMC_EVSYNC=1 (the P6.20 L1 fix, default-on),
# phase b: =0 (device-sync control — the certified cert8 behavior). ONE
# variable between phases; both ARC_STEP_TRACE=1 (the P6.20 L2 instrument,
# silent by default, armed here). Traffic = p65_timeline v5.1 full faces
# (U32k/C32k/U8k/U64k + 6 bystander streams) — the P6.15 gap-measurement
# harness, so the A/B lands on the SAME face definitions as the banked
# stall matrix. LMCache CPU pool is per-container (no disk backend on our
# path) — kill_arc between phases keeps U-faces uncached in both.
# Evidence: results/2026-11-04-p620-prefill-opt/ (plan P6.20 lanes L1/L2).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert8
SP=/opt/venv/lib/python3.12/site-packages
CTROOT=$R/p620/r80
P620=$R/p620/files
mkdir -p "$CTROOT" "$P620"
cd "$CTROOT"

kill_arc() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 10
}

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

echo "=================== P620 R80 START $(date +%F\ %H:%M:%S) ==================="

echo "=== [0] py_compile the three staged files inside the image (p618c law) ==="
docker run --rm --entrypoint /opt/venv/bin/python -v "$P620":/p620f "$IMG" \
  -m py_compile /p620f/vllm_v1_adapter.py /p620f/step_trace.py /p620f/platform.py \
  && echo "PY_COMPILE: PASS" || { echo "PY_COMPILE: FAIL"; exit 1; }

phase() {  # $1=tag $2=evsync-value
  local TAG=$1 EV=$2 CT=$CTROOT/$1
  mkdir -p "$CT"
  echo "=================== [phase $TAG] evsync=$EV $(date +%H:%M:%S) ==================="
  kill_arc
  docker run -d --name arc-test --privileged --network host --ipc host \
    --shm-size 64g \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /root/build/v127_stage/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
    -v /root/build/neo_cache:/root/.neo_cache \
    -v /root/build/triton_cache:/root/.triton_cache \
    -v /root/build/inductor_cache:/root/.inductor_cache \
    -v "$P620/vllm_v1_adapter.py":$SP/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py \
    -v "$P620/step_trace.py":$SP/vllm_gaudi/xpu/patches/step_trace.py \
    -v "$P620/platform.py":$SP/vllm_gaudi/xpu/platform.py \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --entrypoint /bin/bash "$IMG" -c 'sleep infinity'
  sleep 2
  cp $R/serve_arc_b70.sh "$CT/serve_p620_$TAG.sh"
  sed -i "s|export VLLM_XPU_ENABLE_XPU_GRAPH=1;|export VLLM_XPU_ENABLE_XPU_GRAPH=1; export ARC_STEP_TRACE=1; export VLLM_ARC_LMC_EVSYNC=$EV;|" "$CT/serve_p620_$TAG.sh"
  grep -q "ARC_STEP_TRACE=1" "$CT/serve_p620_$TAG.sh" || { echo "FATAL: env inject failed"; exit 1; }
  grep -q "VLLM_ARC_LMC_EVSYNC=$EV" "$CT/serve_p620_$TAG.sh" || { echo "FATAL: evsync inject failed"; exit 1; }
  cp "$CT/serve_p620_$TAG.sh" "$CT/serve_used.sh"
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash "$CT/serve_p620_$TAG.sh" > "$CT/boot.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/boot.out" || { echo "FATAL: launch failed"; cat "$CT/boot.out"; exit 1; }
  wait_health | tee "$CT/health.txt"
  docker exec arc-test sh -c "grep -m3 'step_trace ARMED' /root/serve_full.log" > "$CT/posture.txt" 2>&1 || true
  cat "$CT/posture.txt"
  [ -s "$CT/posture.txt" ] || { echo "FATAL: step_trace ARMED line absent — instrument not installed"; exit 1; }
  echo "--- traffic: p65_timeline v5.1 full faces (~4 min) ---"
  python3 $R/p65/p65_timeline.py "$CT/timeline.json" > "$CT/timeline_stdout.txt" 2>&1 || echo "timeline exit=$? (see stdout)"
  tail -5 "$CT/timeline_stdout.txt"
  echo "--- fetch traces + serve log ---"
  docker exec arc-test sh -c 'ls -la /tmp/arc_step_trace_* 2>/dev/null' > "$CT/trace_ls.txt" || true
  cat "$CT/trace_ls.txt"
  for f in $(docker exec arc-test sh -c 'ls /tmp/arc_step_trace_* 2>/dev/null'); do
    docker cp "arc-test:$f" "$CT/$(basename "$f")" >/dev/null 2>&1 || true
  done
  docker cp arc-test:/root/serve_full.log "$CT/serve_full.log" >/dev/null 2>&1 || true
  kill_arc
}

phase a 1
phase b 0

echo "=================== P620 R80 DONE $(date +%F\ %H:%M:%S) ==================="
