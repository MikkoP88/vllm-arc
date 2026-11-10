#!/bin/bash
# p635_gate2c.sh -- P6.35 leg J: D1 fix probe (graph-exec double-buffer).
# Probe image p135jprobe from patch_p135j (leg F telemetry + db fix).
# SINGLE boot, certified posture (GRAPH=1 SPEC=3 LMC=1). Face U8k solo 600
# + PROF window + metrics + docker boot log (ARC_DB_CAPTURED / capture time
# / OOM watch). Compare against leg F spec3 (blocked-replay baseline).
set -uo pipefail
R=/root/build
CT=$R/p635
SP=/opt/venv/lib/python3.12/site-packages
IMG=vllm-arc:v0.30.0-b70-cert22
PROBE=vllm-arc:v0.30.0-b70-cert22-p135jprobe
NAME=arc-telj
LEG=$CT/legJ-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LEG"

w14_restore() {
  docker stop "$NAME" >/dev/null 2>&1
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

docker stop radiance-test >/dev/null 2>&1
docker rm -f "$NAME" >/dev/null 2>&1
verify_removed "$NAME" || exit 1
sleep 2

rm -rf $CT/probe_src $CT/probe_out
mkdir -p $CT/probe_src $CT/probe_out
docker run -d --name p135x --entrypoint sleep "$IMG" 180 >/dev/null
declare -A REL=(
  [core.py]=vllm/v1/engine/core.py
  [scheduler.py]=vllm/v1/core/sched/scheduler.py
  [model_runner.py]=vllm/v1/worker/gpu/model_runner.py
  [mamba_hybrid.py]=vllm/v1/worker/gpu/model_states/mamba_hybrid.py
  [cudagraph_utils.py]=vllm/v1/worker/gpu/cudagraph_utils.py
  [gpu_worker.py]=vllm/v1/worker/gpu_worker.py
  [llm_base_proposer.py]=vllm/v1/spec_decode/llm_base_proposer.py
  [speculator.py]=vllm/v1/worker/gpu/spec_decode/speculator.py
  [sd_utils.py]=vllm/v1/worker/gpu/spec_decode/utils.py
  [ar_speculator.py]=vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
  [off_base.py]=vllm/model_executor/offloader/base.py
  [off_prefetch.py]=vllm/model_executor/offloader/prefetch.py
  [batch_shard.py]=vllm/v1/worker/gpu/sample/batch_shard.py
)
for f in "${!REL[@]}"; do
  docker cp "p135x:$SP/${REL[$f]}" "$CT/probe_src/$f" || { echo "EXTRACT_FAIL $f"; exit 1; }
done
docker rm -f p135x >/dev/null

md5sum $CT/patch_p135j.py
python3 $CT/patch_p135j.py > "$LEG/patch.txt" 2>&1 || { cat "$LEG/patch.txt"; echo GATE_FAIL_PATCH; exit 1; }
cat "$LEG/patch.txt"
docker run -d --name p135x --entrypoint sleep "$IMG" 180 >/dev/null
for f in "${!REL[@]}"; do
  docker cp "$CT/probe_out/$f" "p135x:$SP/${REL[$f]}" || { echo CP_FAIL $f; exit 1; }
done
docker exec p135x find $SP/vllm -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
SYN=0
for f in "${!REL[@]}"; do
  docker exec p135x /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/${REL[$f]}').read())" || { echo "GATE_FAIL_SYNTAX $f"; SYN=1; }
done
[ $SYN -eq 0 ] || exit 1
echo SYNTAX_OK
docker commit p135x "$PROBE" >/dev/null
docker rm -f p135x >/dev/null
echo "PROBE_IMAGE_OK $PROBE"

# ---- single boot, certified posture ----
nohup env ARC_IMAGE="$PROBE" ARC_NAME=$NAME ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_arc_b70.sh > "$LEG/boot_spec3.txt" 2>&1 &
i=0
while [ $i -lt 180 ]; do
  curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "HEALTH_OK_spec3 after ${i}0s"; break; }
  d=$(docker logs "$NAME" 2>&1 | grep -cE "EngineCore.*(died|exception)|RuntimeError|DEVICE_LOST|out of memory" || true)
  [ "${d:-0}" -gt 0 ] && { docker logs "$NAME" 2>&1 | tail -50; echo "GATE_FAIL_ENGINE_DEAD"; exit 1; }
  i=$((i + 1)); sleep 10
done
[ $i -lt 180 ] || { echo "GATE_FAIL_HEALTH_TIMEOUT"; docker logs "$NAME" 2>&1 | tail -50; exit 1; }
docker logs "$NAME" 2>&1 | grep -E "ARC_DB_CAPTURED|Capturing" | tail -8 > "$LEG/capture_log.txt" || true
cat "$LEG/capture_log.txt"
sleep 5
docker exec "$NAME" sh -c 'rm -f /root/p135_steps.log /root/PROF_ON /root/p135_face_*.json /root/p135_prof_rank*.json'
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
sleep 3
L=$(docker exec "$NAME" sh -c 'wc -l < /root/p135_steps.log 2>/dev/null' || echo 0)
[ "${L:-0}" -ge 5 ] || { echo "GATE_FAIL_NO_TELEMETRY lines=$L"; exit 1; }
echo "TELEMETRY_ALIVE lines=$L"
DB=$(docker exec "$NAME" sh -c 'grep -c ARC_DB_CAPTURED /root/p135_steps.log' || echo 0)
echo "arc_db_captured_lines=${DB:-0}"
docker cp $CT/p135_face.py "$NAME":/root/p135_face.py
docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py 8192 600 spec3
sleep 8
docker exec "$NAME" sh -c 'echo 24 > /root/PROF_ON'
i2=0
while [ $i2 -lt 200 ]; do docker exec "$NAME" test -f /root/p135_face_spec3.json && break; sleep 2; i2=$((i2+1)); done
echo "FACE_DONE waited=$((i2*2))s"
sleep 3
PD=$(docker exec "$NAME" sh -c 'grep -c PROF_DONE /root/p135_steps.log' || echo 0)
echo "prof_done_count=${PD:-0}"
curl -s -m 10 http://127.0.0.1:8000/metrics > "$LEG/metrics_spec3.txt" || true
docker cp "$NAME":/root/p135_steps.log "$LEG/p135_steps_spec3.log" >/dev/null 2>&1 || echo "steps MISSING"
for r in 0 1; do
  docker exec "$NAME" test -f /root/p135_prof_rank$r.json && docker cp "$NAME":/root/p135_prof_rank$r.json "$LEG/p135_prof_spec3_rank$r.json" >/dev/null 2>&1
done
docker cp "$NAME":/root/p135_face_spec3.json "$LEG/" >/dev/null 2>&1 || echo "face missing"
docker logs "$NAME" > "$LEG/docker_log.txt" 2>&1 || true

cd "$LEG" && sha256sum p135_steps_*.log p135_prof_*.json p135_face_*.json > SHA256SUMS 2>/dev/null
ls -la "$LEG"
echo "LEG_ARTIFACTS $LEG"
echo "P635_GATE2C_CAPTURE_COMPLETE -- analysis next"
