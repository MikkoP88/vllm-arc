#!/bin/bash
# p620 r81 — L3 off-thread-store A/B/A/B on cert8 (bind-mount probe, NO
# image build). r80 measured the evsync-only form (L1) NEUTRAL and taught
# two lessons: (1) on the in-order XPU stream a flush-time event sits
# behind the just-submitted chunk forward — the host wait is the same
# lockstep either way; the fix must move the WHOLE flush body (sync +
# gathers + store) off the engine thread (L3, VLLM_ARC_LMC_ASYNC_STORE);
# (2) the step_trace instrument targeted vllm_gaudi hpu_* classes that
# never load on the Arc image (habana_frameworks absent) — r81 wraps CORE
# GPUModelRunner/Scheduler. Phases interleaved l3on/off/l3on/off to
# bracket order effects; ONE variable between postures (async store on/
# off); both ARC_STEP_TRACE=1 (per-pid files now). Traffic = p65_timeline
# v5.1 full faces (same as r80 / the banked P6.15 stall matrix).
# Evidence: results/2026-11-04-p620-prefill-opt/ (plan P6.20 L3 + L2).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert8
SP=/opt/venv/lib/python3.12/site-packages
CTROOT=$R/p620/r81
P620=$R/p620/files
mkdir -p "$CTROOT"
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

phase() {  # $1=tag $2=async-store 0/1
  local TAG=$1 EV=$2 CT=$CTROOT/$1
  mkdir -p "$CT"
  echo "=================== [phase $TAG] async_store=$EV $(date +%H:%M:%S) ==================="
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
  sed -i "s|export VLLM_XPU_ENABLE_XPU_GRAPH=1;|export VLLM_XPU_ENABLE_XPU_GRAPH=1; export ARC_STEP_TRACE=1; export VLLM_ARC_LMC_ASYNC_STORE=$EV;|" "$CT/serve_p620_$TAG.sh"
  grep -q "ARC_STEP_TRACE=1" "$CT/serve_p620_$TAG.sh" || { echo "FATAL: env inject failed"; exit 1; }
  grep -q "VLLM_ARC_LMC_ASYNC_STORE=$EV" "$CT/serve_p620_$TAG.sh" || { echo "FATAL: async-store inject failed"; exit 1; }
  cp "$CT/serve_p620_$TAG.sh" "$CT/serve_used.sh"
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash "$CT/serve_p620_$TAG.sh" > "$CT/boot.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/boot.out" || { echo "FATAL: launch failed"; cat "$CT/boot.out"; exit 1; }
  wait_health | tee "$CT/health.txt"
  docker exec arc-test sh -c "grep -m3 'step_trace INSTALLED' /root/serve_full.log" > "$CT/posture.txt" 2>&1 || true
  cat "$CT/posture.txt"
  [ -s "$CT/posture.txt" ] || { echo "FATAL: step_trace INSTALLED line absent — instrument not installed (r80 lesson)"; exit 1; }
  echo "--- traffic: p65_timeline v5.1 full faces (~4 min) ---"
  python3 $R/p65/p65_timeline.py "$CT/timeline.json" > "$CT/timeline_stdout.txt" 2>&1 || echo "timeline exit=$? (see stdout)"
  tail -12 "$CT/timeline_stdout.txt"
  echo "--- fetch traces + serve log ---"
  # L3 posture gate runs POST-TRAFFIC: the store worker initializes at the
  # FIRST wait_for_save (first traffic), not at health — the r81a relaunch
  # lesson (pre-traffic gate was vacuously fatal).
  docker exec arc-test sh -c "grep -m2 'store worker ACTIVE' /root/serve_full.log" > "$CT/posture_l3.txt" 2>&1 || true
  cat "$CT/posture_l3.txt"
  if [ "$EV" = "1" ]; then
    grep -q "store worker ACTIVE" "$CT/posture_l3.txt" || { echo "FATAL: L3 worker-ACTIVE line absent in async phase (post-traffic)"; exit 1; }
  else
    grep -q "store worker ACTIVE" "$CT/posture_l3.txt" && { echo "FATAL: L3 worker ACTIVE in a=0 phase — posture leak"; exit 1; } || true
  fi
  docker exec arc-test sh -c 'ls -la /tmp/arc_step_trace_* 2>/dev/null' > "$CT/trace_ls.txt" || true
  cat "$CT/trace_ls.txt"
  [ -s "$CT/trace_ls.txt" ] || { echo "FATAL: no trace files written"; exit 1; }
  for f in $(docker exec arc-test sh -c 'ls /tmp/arc_step_trace_* 2>/dev/null'); do
    docker cp "arc-test:$f" "$CT/$(basename "$f")" >/dev/null 2>&1 || true
  done
  docker cp arc-test:/root/serve_full.log "$CT/serve_full.log" >/dev/null 2>&1 || true
  kill_arc
}

phase l3on1 1
phase off1  0
phase l3on2 1
phase off2  0

echo "=================== P620 R81 DONE $(date +%F\ %H:%M:%S) ==================="
