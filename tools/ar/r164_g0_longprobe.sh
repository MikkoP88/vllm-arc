#!/bin/bash
# r164_g0_longprobe.sh -- P6.47: native-KV gemma longprobe (the m8g
# CORRUPTION_SIGNAL disposition baseline, cert25x image, 2026-11-12).
#
# r159's g0 leg is battery-only, but the m8g REP-LOOP disposition needs
# the native lane's LONG-RETURN shape on the SAME image (reploop-
# classifier-baseline law): if native-KV gemma shows the same thinking-
# stream REP-LOOP signature (top-window counts ~4-10 quoting vs 100+
# loops), the m8g flag is baseline model behavior, not MX-KV corruption.
# Boots the gemma champion posture WITHOUT ARC_KVDTYPE (native fp8_e4m3
# KV), runs p647_longprobe v4 full (banks top-window counts + raw heads
# + channel split), UNGATED -- evidence only. Artifacts ->
# /root/build/p647/cert25/g0probe/. W14 quiesce before, trap restore.
set -uo pipefail
R=/root/build
P=$R/p647/cert25/g0probe
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25x}
CNAME=arc-g0p
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r164.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

boot_wait() {
  # serve_arc_b70.sh takes a few seconds to `docker run` — tolerate up to
  # 6 absent polls before calling the boot dead (the r164 first run raced
  # this: CONTAINER_ABSENT at poll=0 killed the wrapper while the detached
  # boot was still about to create the container).
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 6 ] && { echo "[r164] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1 && SEEN=1
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

echo "=== [r164] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r164] gemma native-KV boot $(date +%H:%M:%S) ==="
GEMMA_M8="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $GEMMA_M8 \
  bash $R/serve_arc_b70.sh > "$P/boot.txt" 2>&1 &
if ! boot_wait; then
  echo "R164_BOOT_FAIL -- banking log"
  docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
  docker rm -f $CNAME >/dev/null 2>&1
  exit 1
fi
echo "HEALTH_OK_r164 $(date +%H:%M:%S)"
docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" \
  /root/serve_full.log 2>/dev/null | head -3 > "$P/markers.txt" || true
docker exec $CNAME grep -c "Using MX KV attention backend" /root/serve_full.log \
  > "$P/mx_backend_lines.txt" 2>/dev/null || echo 0 > "$P/mx_backend_lines.txt"
cat "$P/markers.txt" "$P/mx_backend_lines.txt"

# warm, then the v4 longprobe (evidence, ungated)
curl -s -m 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-31b","messages":[{"role":"user","content":"warmup"}],"max_tokens":16,"temperature":0}' > /dev/null
sleep 3
( cd "$P" && ARC_PROBE_MODEL=gemma4-31b \
    python3 $R/p647/p647_longprobe.py c25xg0 full > longprobe_full.out 2>&1 ) || true
grep -E "^\[|LONGPROBE" "$P/longprobe_full.out" || true

docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 8
docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
docker rm -f $CNAME >/dev/null 2>&1
cd "$P"
sha256sum longprobe_full.out *.json *.txt 2>/dev/null > SHA256SUMS || true
echo "R164_G0PROBE_DONE (evidence banked; disposition = compare vs m8g longprobe windows) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
