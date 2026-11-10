#!/bin/bash
# r135b_dboff.sh -- P6.35 D1 cert23 gate RE-RUN of the stock-control leg.
# The first gate pass's dboff leg ran db ON by accident: the scp carrying the
# VLLM_ARC_GRAPH_EXEC_DB passthrough line in serve_arc_b70.sh aborted on a
# missing file arg, so =0 never reached the container (proof: db_info_lines=6
# in that leg's db_markers.txt). That leg stands re-labeled as an accidental
# db-on REPLICATE. This script re-runs the control with the verified script
# (md5 58b4b515). Expect: db_info_lines=0, arcdb_capture_passes=0, face at
# the stock k3 posture (legF: 81.6 tok/s, ITL mean 47.3 >> p50 40.5).
set -uo pipefail
R=/root/build
P=$R/p635/cert23
IMG=vllm-arc:v0.30.0-b70-cert23
LEGN=dboff2
CD=$P/$LEGN
mkdir -p "$CD"

w14_restore() {
  docker rm -f arc-c23 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635c23b.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "GATE_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-c23 grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

echo "=== [$LEGN] boot $(date +%H:%M:%S) VLLM_ARC_GRAPH_EXEC_DB=0 (script md5 $(md5sum $R/serve_arc_b70.sh | cut -c1-8)) ==="
docker stop radiance-test >/dev/null 2>&1; sleep 5
nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-c23 \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1 \
  VLLM_ARC_GRAPH_EXEC_DB=0 \
  bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
ok=""
for i in $(seq 1 210); do
  curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  tb=$(gcount "Traceback (most recent call last)")
  [ "${tb:-0}" -gt 0 ] && { echo "TRACEBACK poll=$i"; break; }
  sleep 10
done
if [ "$ok" != "1" ]; then
  echo "GATE_FAIL_BOOT_$LEGN"
  docker cp arc-c23:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker rm -f arc-c23 >/dev/null 2>&1; exit 1
fi
echo "HEALTH_OK_$LEGN $(date +%H:%M:%S)"
sleep 5
echo "db_info_lines=$(gcount 'graph-exec double-buffer captured')" > "$CD/db_markers.txt"
echo "arcdb_capture_passes=$(gcount 'arc-db')" >> "$CD/db_markers.txt"
echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/db_markers.txt"
cat "$CD/db_markers.txt"
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
sleep 3
python3 $R/r113_battery.py decode qwen3.8-27b-fp8 > "$CD/battery.txt" 2>&1 || echo "BATTERY_NONZERO_EXIT"
tail -5 "$CD/battery.txt"
docker cp $R/p635/p135_face.py arc-c23:/root/p135_face.py
docker exec -d arc-c23 /opt/venv/bin/python /root/p135_face.py 8192 600 "$LEGN" qwen3.8-27b-fp8
i2=0
while [ $i2 -lt 250 ]; do docker exec arc-c23 test -f /root/p135_face_$LEGN.json && break; sleep 2; i2=$((i2+1)); done
echo "FACE_DONE_$LEGN waited=$((i2*2))s"
curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
docker cp arc-c23:/root/p135_face_$LEGN.json "$CD/" >/dev/null 2>&1 || echo "face MISSING"
docker cp arc-c23:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
docker exec arc-c23 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 8
docker rm -f arc-c23 >/dev/null 2>&1
verify_removed arc-c23 || true
echo "R135B_DBOFF_DONE"
