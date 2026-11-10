#!/bin/bash
# p635_gate1.sh -- P6.35 leg A: cert22 decode-step telemetry capture.
# Probe image = cert22 + ARC_P135 wrappers (file-append step walls + XPU
# profiler windows). Certified posture: ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 TP2.
# Faces: U8k solo 600-tok + U32k solo 300-tok + U8k 200-tok post window.
# W14 law: radiance-test stopped for the window, RESTORED by trap.
set -uo pipefail
R=/root/build
CT=$R/p635
SP=/opt/venv/lib/python3.12/site-packages
IMG=vllm-arc:v0.30.0-b70-cert22
PROBE=vllm-arc:v0.30.0-b70-cert22-p135probe
NAME=arc-tel1
LEG=$CT/legA-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LEG"

w14_restore() {
  docker stop "$NAME" >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

docker stop radiance-test >/dev/null 2>&1
docker rm -f "$NAME" p135x >/dev/null 2>&1
sleep 2

# ---- 1. fresh source extraction from cert22 ----
mkdir -p $CT/probe_src $CT/probe_out
docker run -d --name p135x --entrypoint sleep "$IMG" 120 >/dev/null
declare -A REL=(
  [core.py]=vllm/v1/engine/core.py
  [scheduler.py]=vllm/v1/core/sched/scheduler.py
  [model_runner.py]=vllm/v1/worker/gpu/model_runner.py
  [mamba_hybrid.py]=vllm/v1/worker/gpu/model_states/mamba_hybrid.py
  [cudagraph_utils.py]=vllm/v1/worker/gpu/cudagraph_utils.py
  [gpu_worker.py]=vllm/v1/worker/gpu_worker.py
)
for f in "${!REL[@]}"; do
  docker cp "p135x:$SP/${REL[$f]}" "$CT/probe_src/$f" || { echo "EXTRACT_FAIL $f"; exit 1; }
done
docker rm -f p135x >/dev/null

# ---- 2. patch + image build ----
python3 $CT/patch_p135a.py > "$LEG/patch.txt" 2>&1 || { cat "$LEG/patch.txt"; echo GATE_FAIL_PATCH; exit 1; }
cat "$LEG/patch.txt"
docker run -d --name p135x --entrypoint sleep "$IMG" 120 >/dev/null
for f in "${!REL[@]}"; do
  docker cp "$CT/probe_out/$f" "p135x:$SP/${REL[$f]}" || { echo CP_FAIL $f; exit 1; }
done
docker exec p135x find $SP/vllm/v1 -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
SYN=0
for f in "${!REL[@]}"; do
  docker exec p135x /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/${REL[$f]}').read())" || { echo "GATE_FAIL_SYNTAX $f"; SYN=1; }
done
[ $SYN -eq 0 ] || exit 1
echo SYNTAX_OK
docker commit p135x "$PROBE" >/dev/null
docker rm -f p135x >/dev/null
echo "PROBE_IMAGE_OK $PROBE"

# ---- 3. boot certified posture ----
mkdir -p $R/logs/$NAME 2>/dev/null || true
nohup env ARC_IMAGE="$PROBE" ARC_NAME=$NAME ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_arc_b70.sh > "$LEG/boot.txt" 2>&1 &
HP=/tmp/p635_health.$$
DEAD=/tmp/p635_dead.$$
: > "$DEAD"
i=0
while [ $i -lt 150 ]; do
  if curl -sf -m 4 http://127.0.0.1:8000/health > "$HP" 2>/dev/null; then
    echo "HEALTH_OK after ${i}0s"; break
  fi
  d=$(docker logs "$NAME" 2>&1 | grep -cE "EngineCore.*(died|exception|Traceback)|RuntimeError|DEVICE_LOST" || true)
  if [ "${d:-0}" -gt 0 ]; then docker logs "$NAME" 2>&1 | tail -40 | tee -a "$DEAD"; echo GATE_FAIL_ENGINE_DEAD; exit 1; fi
  i=$((i + 1)); sleep 10
done
if [ $i -ge 150 ]; then echo GATE_FAIL_HEALTH_TIMEOUT; docker logs "$NAME" 2>&1 | tail -40; exit 1; fi
sleep 5

# ---- 4. warmup + instrumentation liveness gate ----
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"telemetry warmup","max_tokens":16,"temperature":0}' > /dev/null
sleep 3
docker exec "$NAME" sh -c 'wc -l < /root/p135_steps.log 2>/dev/null' > /tmp/p635_lines.$$ || true
L=$(cat /tmp/p635_lines.$$ 2>/dev/null || echo 0)
if [ "${L:-0}" -lt 5 ]; then echo "GATE_FAIL_NO_TELEMETRY lines=$L"; docker logs "$NAME" 2>&1 | tail -30; exit 1; fi
echo "TELEMETRY_ALIVE lines=$L"
curl -s -m 10 http://127.0.0.1:8000/metrics > "$LEG/metrics_warm.txt" || true

# ---- 5. faces with profiler windows ----
docker cp $CT/p135_face.py "$NAME":/root/p135_face.py

docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py 8192 600 u8k
sleep 8
docker exec "$NAME" sh -c 'echo 32 > /root/PROF_ON'
i=0; while [ $i -lt 200 ]; do docker exec "$NAME" test -f /root/p135_face_u8k.json && break; sleep 2; i=$((i+1)); done
echo "FACE_U8K_DONE waited=$((i*2))s"
curl -s -m 10 http://127.0.0.1:8000/metrics > "$LEG/metrics_u8k.txt" || true

docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py 32768 300 u32k
sleep 30
docker exec "$NAME" sh -c 'echo 32 > /root/PROF_ON'
i=0; while [ $i -lt 200 ]; do docker exec "$NAME" test -f /root/p135_face_u32k.json && break; sleep 2; i=$((i+1)); done
echo "FACE_U32K_DONE waited=$((i*2))s"
curl -s -m 10 http://127.0.0.1:8000/metrics > "$LEG/metrics_u32k.txt" || true

docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py 8192 200 u8kb
i=0; while [ $i -lt 150 ]; do docker exec "$NAME" test -f /root/p135_face_u8kb.json && break; sleep 2; i=$((i+1)); done
echo "FACE_U8KB_DONE waited=$((i*2))s"

# ---- 6. artifacts out ----
sleep 3
docker exec "$NAME" sh -c 'grep -cE "PROF_(START|DONE|ARM_ERR|STOP_ERR)" /root/p135_steps.log' > /tmp/p635_prof.$$ 2>/dev/null || true
echo "prof_lines=$(cat /tmp/p635_prof.$$ 2>/dev/null || echo 0)"
docker cp "$NAME":/root/p135_steps.log "$LEG/" >/dev/null 2>&1 || echo "steps.log MISSING"
for r in 0 1; do docker cp "$NAME":/root/p135_prof_rank$r.json "$LEG/" >/dev/null 2>&1 || echo "prof rank$r missing (ok if one window)"; done
for t in u8k u32k u8kb; do docker cp "$NAME":/root/p135_face_$t.json "$LEG/" >/dev/null 2>&1 || echo "face $t missing"; done
docker logs "$NAME" > "$LEG/docker.log" 2>&1 || true
docker exec "$NAME" sh -c 'grep -E "PROF_" /root/p135_steps.log' > "$LEG/prof_events.txt" 2>/dev/null || true
cd "$LEG" && sha256sum p135_steps.log p135_prof_rank*.json p135_face_*.json > SHA256SUMS 2>/dev/null
ls -la "$LEG"
echo "LEG_ARTIFACTS $LEG"
echo "P635_GATE1_CAPTURE_COMPLETE -- analysis next"
