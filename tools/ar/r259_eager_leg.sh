#!/bin/bash
# r259_eager_leg.sh -- the ARC_GRAPH=0 suite replay (P6.47 r259 rung).
# The r258b triangulation: graph-replay + suite state = CRASH (r257),
# graph + pristine state = clean (r257 minimal), EAGER direct calls +
# the crash tables/garbage bytes = clean (r258b). rows=118 IS a
# captured bucket, so the r257 crash step rode an XPU-graph REPLAY.
# This leg is the decisive discriminator: the r257 replay phase B
# EXACTLY (the ACTUAL p647_longprobe.py full suite on the m8g posture)
# with ONE env delta -- ARC_GRAPH=0 (eager engine; the identical
# native calls run python-side instead of as captured-graph replays).
#   CRASH  -> the hang is engine-level state (graph-independent): the
#             surface is the allocator/queue layer, and the graph
#             mechanics are exonerated.
#   CLEAN  -> the graph replay is a NECESSARY component (or its timing
#             is): the r260 surface = the offline graph-capture
#             synthetic (frozen plan x stale static-tail x command-
#             graph execution).
# Crash outcomes are DATA: exit is always 0; the client polls
# P647_R259_LEG_DONE. trap restores W14 (health-aware).
set -uo pipefail
R=/root/build
P=$R/p647/r259
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert29}
CNAME=arc-r259
mkdir -p "$P" "$P/devcores"

# ---- pre-refuses ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R259_BUSY (arc leg in flight)"; exit 2
fi
GOTSS=$(md5sum $R/serve_arc_b70.sh 2>/dev/null | cut -d" " -f1)
[ "$GOTSS" = "c2db777506e776a86426852b6bb80de4" ] \
  || { echo "R259_SERVE_SCRIPT_STALE ($GOTSS != c2db777506e776a86426852b6bb80de4)"; exit 1; }
[ -f $R/p647/p647_longprobe.py ] || { echo "R259_NO_LONGPROBE"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R259_NO_IMAGE $IMG"; exit 1; }
echo "R259_LONGPROBE_MD5 $(md5sum $R/p647/p647_longprobe.py)"
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "R259_IMAGE $IMG $MF_SHA"

# THE DELTA: ARC_GRAPH=0 (everything else is the r257 m8g posture verbatim)
GEMMA_NAT="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

w14_restore() {
  rm -f "$P/watch_on"
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  if ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    docker start radiance-test >/dev/null 2>&1
    nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r259.log 2>&1 &
    echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "W14_ALREADY_HEALTHY (restore skipped)"
  fi
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "R259_FAIL_STALE_CONTAINER $CNAME"; return 1
}

gcount() {
  local c
  c=$(docker exec $CNAME grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[boot] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[boot] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[boot] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

devcore_preclear() {
  local d
  for d in /sys/class/drm/card*/device/devcoredump; do
    [ -d "$d" ] || continue
    echo 1 > "$d/data" 2>/dev/null || true
    echo "DEVCORE_PRECLEARED $(basename "$(dirname "$(dirname "$d")")")"
  done
  true
}

devcore_watch() {
  local d card nm sz
  while [ -f "$P/watch_on" ]; do
    for d in /sys/class/drm/card*/device/devcoredump; do
      [ -f "$d/data" ] || continue
      card=$(basename "$(dirname "$(dirname "$d")")")
      nm="$P/devcores/${card}_$(date +%H%M%S)"
      if timeout 30 cat "$d/data" > "${nm}.devcore" 2>/dev/null \
         && [ -s "${nm}.devcore" ]; then
        sz=$(stat -c%s "${nm}.devcore" 2>/dev/null || echo 0)
        echo "DEVCORE_CAPTURED $card $(date +%H:%M:%S) bytes=$sz"
        echo 1 > "$d/data" 2>/dev/null || true
        dmesg -T > "${nm}_dmesg.txt" 2>/dev/null || true
      fi
    done
    sleep 2
  done
}

# ---- entry: quiesce W14, arm the watcher, boot eager, run the suite ----
docker stop radiance-test >/dev/null 2>&1
sleep 5
devcore_preclear
touch "$P/watch_on"
devcore_watch > "$P/devcores/watch.log" 2>&1 &

echo "=== [boot] $CNAME on $IMG (gemma m8 posture, ARC_GRAPH=0 EAGER) $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $GEMMA_NAT ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 \
  bash $R/serve_arc_b70.sh > "$P/boot.txt" 2>&1 &
if ! boot_wait; then
  echo "R259_BOOT_FAIL"
  docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "P647_R259_LEG_DONE (boot fail banked)"
  exit 0
fi
echo "R259_HEALTH_OK $(date +%H:%M:%S)"
sleep 5
{
  echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
  echo "native_lines=$(gcount 'MXKV native call')"
  echo "pc2_windowed_impl=$(gcount 'MXKV P-C2 windowed impl')"
  echo "lmcache_lines=$(gcount 'LMCacheConnector')"
  echo "xpu_graph_lines=$(gcount -i 'xpu graph\|XPU graph\|cuda graph\|CUDA graph\|graph capture')"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
  docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" /root/serve_full.log 2>/dev/null | head -4
} > "$P/markers.txt"
cat "$P/markers.txt"
# posture WARNs (not refuses): the m8g posture evidence must be in the
# bank; the graph-mode marker is the DELTA evidence (eager must show NO
# capture activity)
grep -Eq "mx_backend_lines=[1-9]" "$P/markers.txt" || echo "R259_POSTURE_WARN no MX backend line"
grep -Eq "native_lines=[1-9]" "$P/markers.txt" || echo "R259_POSTURE_WARN no native telemetry"
grep -Eq "pc2_windowed_impl=[1-9]" "$P/markers.txt" || echo "R259_POSTURE_WARN no P-C2 window telemetry"
grep -q "lmcache_lines=0" "$P/markers.txt" || echo "R259_POSTURE_WARN LMCache engaged"
grep -Eq "xpu_graph_lines=0" "$P/markers.txt" || echo "R259_POSTURE_WARN graph capture lines present (ARC_GRAPH=0 not clean-eager?)"
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4-31b","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
sleep 3

echo "=== [suite] the ACTUAL p647_longprobe full suite, EAGER engine $(date +%H:%M:%S) ==="
( cd "$P" && ARC_PROBE_MODEL=gemma4-31b timeout 4800 python3 $R/p647/p647_longprobe.py p47r259 full > longprobe_full.out 2>&1 ) \
  || echo "R259_LONGPROBE_NONZERO (engine died mid-suite? the verdict below decides)"

# ---- capture (crash or clean) ----
sleep 10   # the reset precedes the python surface by 13-18 s
dmesg -T > "$P/dmesg_final.txt" 2>/dev/null || true
docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 \
  || echo "serve.log MISSING $P"
docker exec $CNAME sh -c \
  'echo DEVICE_LOST=$(grep -c UR_RESULT_ERROR_DEVICE_LOST /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
  > "$P/erraudit.txt" 2>/dev/null || echo "DEVICE_LOST=? TRACEBACK=?" > "$P/erraudit.txt"
cat "$P/erraudit.txt"
docker exec $CNAME grep -E "prompt_token_ids_len|num_computed_tokens|num_scheduled_tokens" \
  /root/serve_full.log 2>/dev/null | tail -8 > "$P/req_lines.txt" || true
docker exec $CNAME grep -B2 -A14 "UR_RESULT_ERROR_DEVICE_LOST" /root/serve_full.log 2>/dev/null \
  | tail -60 > "$P/devicelost_context.txt" || true
echo "-- longprobe tail:"
tail -20 "$P/longprobe_full.out" 2>/dev/null || true

# ---- verdict ----
DL=$(grep -o "DEVICE_LOST=[0-9]*" "$P/erraudit.txt" 2>/dev/null | head -1 | cut -d= -f2)
DL=${DL:-0}
if [ "$DL" -gt 0 ] || ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "R259_EAGER_CRASHED (the suite reproduces WITHOUT graphs -- engine-level state; the graph mechanics are exonerated)"
else
  echo "R259_EAGER_CLEAN (no crash eager: the graph replay is a necessary component -- r260 = the offline graph-capture synthetic)"
fi

# ---- teardown + bank ----
docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 8
docker rm -f $CNAME >/dev/null 2>&1
verify_removed || true
rm -f "$P/watch_on"
sleep 4   # let the watcher loop observe the off switch
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.out" -o -name "*.json" \
  -o -name "serve.log" -o -name "boot.txt" -o -name "*.devcore" \) \
  -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "R259_BANK $P ($(find . -type f | wc -l) files)"
echo "P647_R259_LEG_DONE -- bank ready for pull"
exit 0
