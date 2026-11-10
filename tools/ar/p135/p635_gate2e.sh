#!/bin/bash
# p635_gate2e.sh -- P6.35 leg K: AsyncOutput waiter-mode A/B on the db image.
# p135kbusy / p135kstream = p135jprobe + async_utils.py waiter patch (mode
# baked). Certified posture (GRAPH=1 SPEC=3 LMC=1). Per boot: health, clean
# telemetry, warmup, face U8k solo 600, steps+face pull. Control = leg J
# (event/sleep mode) numbers, no re-run. trap restores W14.
set -uo pipefail
R=/root/build
CT=$R/p635
SP=/opt/venv/lib/python3.12/site-packages
BASE=vllm-arc:v0.30.0-b70-cert22-p135jprobe
LEG=$CT/legK-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LEG"

w14_restore() {
  docker stop arc-telk1 arc-telk2 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    sleep 2
    i=$((i + 1))
  done
  echo "GATE_FAIL_STALE_CONTAINER $1"
  return 1
}

# ---- build both probe images from the p135j base ----
rm -rf $CT/probe_src $CT/probe_out
mkdir -p $CT/probe_src $CT/probe_out
docker run -d --name p135x --entrypoint sleep "$BASE" 120 >/dev/null
docker cp "p135x:$SP/vllm/v1/worker/gpu/async_utils.py" "$CT/probe_src/async_utils.py" || { echo EXTRACT_FAIL; exit 1; }
docker rm -f p135x >/dev/null
md5sum $CT/patch_p135k.py
for MODE in busy stream; do
  python3 $CT/patch_p135k.py $CT/probe_src/async_utils.py $MODE > "$LEG/patch_$MODE.txt" 2>&1 || { cat "$LEG/patch_$MODE.txt"; echo GATE_FAIL_PATCH; exit 1; }
  cat "$LEG/patch_$MODE.txt"
  docker run -d --name p135x --entrypoint sleep "$BASE" 30 >/dev/null
  docker cp "$CT/probe_out/async_utils.py" "p135x:$SP/vllm/v1/worker/gpu/async_utils.py"
  docker exec p135x find $SP/vllm -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
  docker exec p135x /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/vllm/v1/worker/gpu/async_utils.py').read())" || { echo "GATE_FAIL_SYNTAX $MODE"; exit 1; }
  docker commit p135x "vllm-arc:v0.30.0-b70-cert22-p135k$MODE" >/dev/null
  docker rm -f p135x >/dev/null
  echo "PROBE_IMAGE_OK p135k$MODE"
done
echo SYNTAX_OK

# ---- boot + face per mode ----
run_boot() {
  local MODE=$1 NAME=$2
  docker rm -f "$NAME" >/dev/null 2>&1
  verify_removed "$NAME" || exit 1
  sleep 2
  nohup env ARC_IMAGE="vllm-arc:v0.30.0-b70-cert22-p135k$MODE" ARC_NAME=$NAME ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash $R/serve_arc_b70.sh > "$LEG/boot_$MODE.txt" 2>&1 &
  local i=0
  while [ $i -lt 180 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "HEALTH_OK_$MODE after ${i}0s"; break; }
    d=$(docker logs "$NAME" 2>&1 | grep -cE "EngineCore.*(died|exception)|RuntimeError|DEVICE_LOST|out of memory" || true)
    [ "${d:-0}" -gt 0 ] && { docker logs "$NAME" 2>&1 | tail -40; echo "GATE_FAIL_ENGINE_DEAD_$MODE"; exit 1; }
    i=$((i + 1)); sleep 10
  done
  [ $i -lt 180 ] || { echo "GATE_FAIL_HEALTH_TIMEOUT_$MODE"; docker logs "$NAME" 2>&1 | tail -40; exit 1; }
  sleep 5
  docker exec "$NAME" sh -c 'rm -f /root/p135_steps.log /root/PROF_ON /root/p135_face_*.json'
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  L=$(docker exec "$NAME" sh -c 'wc -l < /root/p135_steps.log 2>/dev/null' || echo 0)
  [ "${L:-0}" -ge 5 ] || { echo "GATE_FAIL_NO_TELEMETRY_$MODE lines=$L"; exit 1; }
  C=$(docker exec "$NAME" sh -c 'grep -c CEVT /root/p135_steps.log' || echo 0)
  echo "TELEMETRY_ALIVE_$MODE lines=$L cevt=${C:-0}"
  docker cp $CT/p135_face.py "$NAME":/root/p135_face.py
  docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py 8192 600 "$MODE"
  local i2=0
  while [ $i2 -lt 200 ]; do docker exec "$NAME" test -f /root/p135_face_$MODE.json && break; sleep 2; i2=$((i2+1)); done
  echo "FACE_DONE_$MODE waited=$((i2*2))s"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$LEG/metrics_$MODE.txt" || true
  docker cp "$NAME":/root/p135_steps.log "$LEG/p135_steps_$MODE.log" >/dev/null 2>&1 || echo "steps MISSING"
  docker cp "$NAME":/root/p135_face_$MODE.json "$LEG/" >/dev/null 2>&1 || echo "face missing"
  docker stop "$NAME" >/dev/null 2>&1
}

docker stop radiance-test >/dev/null 2>&1
run_boot busy arc-telk1
run_boot stream arc-telk2

cd "$LEG" && sha256sum p135_steps_*.log p135_face_*.json > SHA256SUMS 2>/dev/null
ls -la "$LEG"
echo "LEG_ARTIFACTS $LEG"
echo "P635_GATE2E_CAPTURE_COMPLETE -- analysis next"
