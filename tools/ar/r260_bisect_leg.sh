#!/bin/bash
# r260_bisect_leg.sh -- the suite-face bisect (P6.47 r260 rung).
# r259 closed the mode question: the suite crashes EAGER (ARC_GRAPH=0,
# same DEVICE_LOST class) -- engine + suite state is the necessary pair,
# graphs exonerated. The r257 minimal pair (multi alone, pristine boot)
# ran CLEAN, so a PRIOR-STATE PREFIX of the suite is required. This
# leg bisects WHICH prefix suffices: four fresh-boot phases, each one
# prior face's FULL class (the actual longprobe v3 semantics: cold ->
# rehit -> rehit2 temp0/1536 + t1a/t1b temp1-seed1234/768, byte-
# identical prompts, enable_thinking OFF) then the multi class (the
# crash site in every rung):
#   V1 essay   -> multi  (shortest long prompt; the suite's own prefix)
#   V2 code    -> multi  (the longest prompt)
#   V3 multi   -> multi  (the SAME class's own debris -- cross-prompt
#                          diversity irrelevant?)
#   V4 simple  -> multi  (78-tok chat traffic -- does ANY traffic do?)
# Every phase boots EAGER (the r259-confirmed crash mode; no capture
# confound, same boot cost). V1 crash = one long class suffices; all
# clean = the combination (multi-class debris interleaving) is
# required. Crash outcomes are DATA: exit is always 0; the client
# polls P647_R260_LEG_DONE. trap restores W14 (health-aware).
set -uo pipefail
R=/root/build
P=$R/p647/r260
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert29}
CNAME=arc-r260
mkdir -p "$P" "$P/devcores"

# ---- pre-refuses ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R260_BUSY (arc leg in flight)"; exit 2
fi
GOTSS=$(md5sum $R/serve_arc_b70.sh 2>/dev/null | cut -d" " -f1)
[ "$GOTSS" = "c2db777506e776a86426852b6bb80de4" ] \
  || { echo "R260_SERVE_SCRIPT_STALE ($GOTSS)"; exit 1; }
[ -f $R/p647/r260_bisect_probe.py ] || { echo "R260_NO_PROBE (ship p647/r260_bisect_probe.py first)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R260_NO_IMAGE $IMG"; exit 1; }
echo "R260_PROBE_MD5 $(md5sum $R/p647/r260_bisect_probe.py)"
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "R260_IMAGE $IMG $MF_SHA"

# r259-confirmed crash mode: EAGER (the m8g posture minus graphs)
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
    nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r260.log 2>&1 &
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
  echo "R260_FAIL_STALE_CONTAINER $CNAME"; return 1
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

teardown_phase() {
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1
  verify_removed || return 1
  sleep 3
}

boot_phase() {  # $1 = phase dir
  local CD=$1
  mkdir -p "$CD"
  devcore_preclear
  echo "=== [boot] $CNAME (gemma m8 posture, EAGER) $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $GEMMA_NAT ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R260_BOOT_FAIL $CD"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    return 1
  fi
  echo "R260_HEALTH_OK $(date +%H:%M:%S)"
  sleep 5
  {
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "native_lines=$(gcount 'MXKV native call')"
    echo "pc2_windowed_impl=$(gcount 'MXKV P-C2 windowed impl')"
    echo "lmcache_lines=$(gcount 'LMCacheConnector')"
    echo "xpu_graph_lines=$(gcount -i 'xpu graph\|XPU graph\|cuda graph\|CUDA graph\|graph capture')"
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
  } > "$CD/markers.txt"
  cat "$CD/markers.txt"
  grep -Eq "mx_backend_lines=[1-9]" "$CD/markers.txt" || echo "R260_POSTURE_WARN no MX backend line"
  grep -Eq "pc2_windowed_impl=[1-9]" "$CD/markers.txt" || echo "R260_POSTURE_WARN no P-C2 window telemetry"
  grep -q "lmcache_lines=0" "$CD/markers.txt" || echo "R260_POSTURE_WARN LMCache engaged"
  grep -Eq "xpu_graph_lines=0" "$CD/markers.txt" || echo "R260_POSTURE_WARN graph capture lines present"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma4-31b","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
}

run_variant() {  # $1 = variant label $2 = phase dir $3.. = probe faces
  local V=$1 CD=$2; shift 2
  local FACES="$*"
  if ! boot_phase "$CD"; then return 1; fi
  echo "=== [V $V] faces: $FACES $(date +%H:%M:%S) ==="
  ( cd "$CD" && ARC_PROBE_MODEL=gemma4-31b timeout 2400 \
      python3 $R/p647/r260_bisect_probe.py "p47r260$V" $FACES > probe.out 2>&1 ) \
    || echo "R260_PROBE_NONZERO $V (bank the partial; the verdict below decides)"
  sleep 10   # the reset precedes the python surface by 13-18 s
  dmesg -T > "$CD/dmesg_final.txt" 2>/dev/null || true
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 \
    || echo "serve.log MISSING $CD"
  docker exec $CNAME sh -c \
    'echo DEVICE_LOST=$(grep -c UR_RESULT_ERROR_DEVICE_LOST /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || echo "DEVICE_LOST=? TRACEBACK=?" > "$CD/erraudit.txt"
  cat "$CD/erraudit.txt"
  docker exec $CNAME grep -E "prompt_token_ids_len|num_computed_tokens|num_scheduled_tokens" \
    /root/serve_full.log 2>/dev/null | tail -8 > "$CD/req_lines.txt" || true
  docker exec $CNAME grep -B2 -A14 "UR_RESULT_ERROR_DEVICE_LOST" /root/serve_full.log 2>/dev/null \
    | tail -60 > "$CD/devicelost_context.txt" || true
  echo "-- [$V] probe tail:"
  tail -12 "$CD/probe.out" 2>/dev/null || true
  local DL
  DL=$(grep -o "DEVICE_LOST=[0-9]*" "$CD/erraudit.txt" 2>/dev/null | head -1 | cut -d= -f2)
  DL=${DL:-0}
  if [ "$DL" -gt 0 ] || ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "R260_V${V} CRASHED (faces: $FACES)"
    return 2
  fi
  echo "R260_V${V} CLEAN (faces: $FACES)"
  return 0
}

# ---- entry: quiesce W14, arm the watcher, run the four variants ----
docker stop radiance-test >/dev/null 2>&1
sleep 5
touch "$P/watch_on"
devcore_watch > "$P/devcores/watch.log" 2>&1 &

S1=1 S2=1 S3=1 S4=1
run_variant 1 "$P/v1_essay_multi" essay multi;    S1=$?
teardown_phase || true
run_variant 2 "$P/v2_code_multi" code multi;     S2=$?
teardown_phase || true
run_variant 3 "$P/v3_multi_multi" multi multi;   S3=$?
teardown_phase || true
run_variant 4 "$P/v4_simple_multi" simple multi; S4=$?
teardown_phase || true

rm -f "$P/watch_on"
sleep 4   # let the watcher loop observe the off switch
cp $R/p647/r260_bisect_probe.py "$P/" >/dev/null 2>&1 || true
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.out" -o -name "*.json" \
  -o -name "serve.log" -o -name "boot.txt" -o -name "*.devcore" \) \
  -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "R260_BANK $P ($(find . -type f | wc -l) files)"
echo "R260_SUMMARY v1_essay=$S1 v2_code=$S2 v3_multi=$S3 v4_simple=$S4 (2=crash 0=clean 1=fail)"
echo "P647_R260_LEG_DONE -- bank ready for pull"
exit 0
