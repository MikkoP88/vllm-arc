#!/bin/bash
# r256_devicelost_repro.sh -- the r255 open defect minimal repro + tail
# bisect (cert29 gate, P6.47 r256 rung). All three gemma MX legs died
# deterministically at the longprobe `multi` face's A/A resend:
# prompt_token_ids_len=3446 / num_computed_tokens=3328 (26 full
# page-128 blocks) / num_scheduled_tokens=118 -> UR_RESULT_ERROR_
# DEVICE_LOST, dmesg ccs (compute) engine reset on BOTH GPUs 13-18 s
# before the python surface. The r255 devcoredumps were eaten by the
# retention window before capture -- this leg banks them within 2 s.
# Phases (one boot each, gemma m8 posture = the r255 m8g crash leg):
#   A repro: the exact r255 request (multi, 3446 tok = 26 blocks + 118
#     tail) fresh (expect clean) -> A/A resend (expect DEVICE_LOST ~4s
#     in). The devcore watcher runs the whole leg at 2 s cadence.
#   B bisect: tails 24,48,63,64,65,96,118 with the 3,328-tok prefix
#     held constant; every rung's fresh send is itself a prefix-hit
#     prefill (shared beginning); stops at the first dead rung -> the
#     page-64 multi-block boundary under test. Fresh boot if A
#     crashed; same boot if A was clean (dirtier cache = closer to
#     the r255 state is the next lever).
#   C control: English-only content at the same 3,446 tok
#     (content-independence), fresh boot.
# Probes run IN-CONTAINER (venv transformers tokenizer; the probe
# feeds via docker exec -i stdin). Bank -> /root/build/p647/r256/.
# trap restores W14. Experimental leg: crash outcomes are DATA, the
# exit is always 0; the client polls P647_R256_REPRO_COMPLETE.
set -uo pipefail
R=/root/build
P=$R/p647/r256
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert29}
CNAME=arc-r256
mkdir -p "$P" "$P/devcores"

# ---- pre-refuses: no second arc leg, serve-script byte pin, probe ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R256_BUSY (arc leg in flight)"; exit 2
fi
GOTSS=$(md5sum $R/serve_arc_b70.sh 2>/dev/null | cut -d" " -f1)
[ "$GOTSS" = "c2db777506e776a86426852b6bb80de4" ] \
  || { echo "R256_SERVE_SCRIPT_STALE ($GOTSS != c2db777506e776a86426852b6bb80de4)"; exit 1; }
[ -f $R/p647/r256_probe.py ] || { echo "R256_NO_PROBE (ship p647/r256_probe.py first)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R256_NO_IMAGE $IMG"; exit 1; }
echo "R256_PROBE_MD5 $(md5sum $R/p647/r256_probe.py)"
# image identity for the bank (log-only: the r255 gate already refused
# on fork_sha + content markers -- this cites what the repro ran on)
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "R256_IMAGE $IMG $MF_SHA"

GEMMA_NAT="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

w14_restore() {
  rm -f "$P/watch_on"
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r256.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "R256_FAIL_STALE_CONTAINER $CNAME"; return 1
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

devcore_preclear() {
  # drop any pre-leg dumps so everything the watcher captures is THIS
  # leg's (writing the data handle is the kernel's documented release)
  local d
  for d in /sys/class/drm/card*/device/devcoredump; do
    [ -d "$d" ] || continue
    echo 1 > "$d/data" 2>/dev/null || true
    echo "DEVCORE_PRECLEARED $(basename "$(dirname "$(dirname "$d")")")"
  done
  true
}

devcore_watch() {
  # 2 s cadence: bank the devcoredump + a dmesg snapshot the MOMENT a
  # GPU reset lands (the r255 retention window ate them ~20 min after
  # the reset -- the reset precedes the python surface by 13-18 s, so
  # this catches the dump while the probe still waits on the 500)
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

boot_leg() {  # $1 = phase dir -> HEALTH or boot-fail capture + teardown
  local CD=$1
  mkdir -p "$CD"
  echo "=== [boot] $(date +%H:%M:%S) (gemma m8 posture: mxfp8 KV + native + P-C2 window) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $GEMMA_NAT ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait r256; then
    echo "R256_BOOT_FAIL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    return 1
  fi
  echo "R256_HEALTH_OK $(date +%H:%M:%S)"
  sleep 5
  {
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "native_lines=$(gcount 'MXKV native call')"
    echo "pc2_windowed_impl=$(gcount 'MXKV P-C2 windowed impl')"
    echo "lmcache_lines=$(gcount 'LMCacheConnector')"
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
    docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" /root/serve_full.log 2>/dev/null | head -4
  } > "$CD/markers.txt"
  cat "$CD/markers.txt"
  # posture WARNs (not refuses): the m8g posture evidence -- MX backend
  # + native + P-C2 window telemetry, LMC off -- must be in the bank
  grep -Eq "mx_backend_lines=[1-9]" "$CD/markers.txt" || echo "R256_POSTURE_WARN no MX backend line"
  grep -Eq "native_lines=[1-9]" "$CD/markers.txt" || echo "R256_POSTURE_WARN no native telemetry"
  grep -Eq "pc2_windowed_impl=[1-9]" "$CD/markers.txt" || echo "R256_POSTURE_WARN no P-C2 window telemetry"
  grep -q "lmcache_lines=0" "$CD/markers.txt" || echo "R256_POSTURE_WARN LMCache engaged"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma4-31b","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
}

run_probe() {  # $1 = output file $2 = ceiling-polls $3.. = probe argv
  local OUT=$1 CEIL=$2; shift 2
  ( docker exec -i $CNAME /opt/venv/bin/python - "$@" \
      < $R/p647/r256_probe.py > "$OUT" 2>&1 ) || true &
  local i=0
  while [ $i -lt $CEIL ]; do
    grep -q "R256_PROBE_DONE" "$OUT" 2>/dev/null && return 0
    sleep 10; i=$((i + 1))
  done
  echo "R256_PROBE_CEILING_HIT $OUT (no DONE line in $((CEIL*10))s)"
  grep -q "R256_PROBE_DONE" "$OUT" 2>/dev/null
}

capture_phase() {  # $1 = phase dir -- post-probe evidence (crash or clean)
  local CD=$1
  sleep 10   # the reset precedes the python surface by 13-18 s; the
  # probe only exits once the 500 lands, so dmesg is complete here
  dmesg -T > "$CD/dmesg_final.txt" 2>/dev/null || true
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 \
    || echo "serve.log MISSING $CD"
  docker exec $CNAME sh -c \
    'echo DEVICE_LOST=$(grep -c UR_RESULT_ERROR_DEVICE_LOST /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || echo "DEVICE_LOST=? TRACEBACK=?" > "$CD/erraudit.txt"
  cat "$CD/erraudit.txt"
  docker exec $CNAME grep -E "prompt_token_ids_len|num_computed_tokens|num_scheduled_tokens" \
    /root/serve_full.log 2>/dev/null | tail -8 > "$CD/req_lines.txt" || true
  # the EngineCore death blob (the r255 crash signature lived here)
  docker exec $CNAME grep -B2 -A14 "UR_RESULT_ERROR_DEVICE_LOST" /root/serve_full.log 2>/dev/null \
    | tail -60 > "$CD/devicelost_context.txt" || true
}

teardown_phase() {
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1
  verify_removed || return 1
  sleep 3
}

phase_repro() {  # returns 2 = crash reproduced, 0 = clean, 1 = fail
  local CD=$P/repro
  devcore_preclear
  if ! boot_leg "$CD"; then return 1; fi
  echo "=== [A repro] the exact r255 request: multi 3446 tok (26 blocks + 118 tail), fresh -> A/A resend $(date +%H:%M:%S) ==="
  if ! run_probe "$CD/probe.txt" 90 --rungs 118; then
    echo "R256_REPRO_PROBE_NO_DONE (probe hung with the engine? bank partial)"
  fi
  capture_phase "$CD"
  grep -E "^FULL_MULTI_N|^RUNG " "$CD/probe.txt" 2>/dev/null || true
  if grep -q '"alive": false' "$CD/probe.txt" 2>/dev/null; then
    echo "R256_REPRO_CONFIRMED (the A/A resend of the exact r255 request reproduced the DEVICE_LOST)"
    return 2
  fi
  echo "R256_REPRO_NO_CRASH (the minimal state did NOT reproduce -- cache-state dependence is a live lead)"
  return 0
}

phase_ladder() {  # $1 = 0 fresh boot | 1 reuse A's live boot; same returns
  local CD=$P/ladder
  if [ "$1" = 0 ]; then
    teardown_phase || true
    devcore_preclear
    if ! boot_leg "$CD"; then return 1; fi
  else
    mkdir -p "$CD"
  fi
  echo "=== [B bisect] tails 24,48,63,64,65,96,118 with the 3,328-tok prefix held constant $(date +%H:%M:%S) ==="
  if ! run_probe "$CD/probe.txt" 150 --rungs 24,48,63,64,65,96,118; then
    echo "R256_LADDER_PROBE_NO_DONE"
  fi
  capture_phase "$CD"
  grep -E "^FULL_MULTI_N|^RUNG |^LADDER_" "$CD/probe.txt" 2>/dev/null || true
  if grep -q "LADDER_DEAD_AT_TAIL" "$CD/probe.txt" 2>/dev/null; then
    echo "R256_LADDER_THRESHOLD_FOUND (first dead rung = the boundary)"
    return 2
  elif grep -q "LADDER_SURVIVED_ALL_RUNGS" "$CD/probe.txt" 2>/dev/null; then
    echo "R256_LADDER_SURVIVED (no rung reproduced -- state/content lead)"
    return 0
  fi
  return 1
}

phase_control() {  # always a fresh boot; same returns
  local CD=$P/control
  teardown_phase || true
  devcore_preclear
  if ! boot_leg "$CD"; then return 1; fi
  echo "=== [C control] English-only content at 3,446 tok, fresh -> A/A resend $(date +%H:%M:%S) ==="
  if ! run_probe "$CD/probe.txt" 90 --control 3446; then
    echo "R256_CONTROL_PROBE_NO_DONE"
  fi
  capture_phase "$CD"
  grep -E "^FULL_ENGLISH_N|^RUNG " "$CD/probe.txt" 2>/dev/null || true
  if grep -q '"alive": false' "$CD/probe.txt" 2>/dev/null; then
    echo "R256_CONTROL_CRASHED (English at the same length/geometry also dies -> content-INdependent)"
    return 2
  fi
  echo "R256_CONTROL_CLEAN (English content survives -> content-dependent, not purely geometric)"
  return 0
}

# ---- entry: quiesce W14, start the devcore watcher, run the phases ----
docker stop radiance-test >/dev/null 2>&1
sleep 5
touch "$P/watch_on"
devcore_watch > "$P/devcores/watch.log" 2>&1 &

RA=1 RB=1 RC=1
phase_repro; RA=$?
if [ "$RA" = 2 ]; then
  phase_ladder 0; RB=$?
else
  phase_ladder 1; RB=$?
fi
phase_control; RC=$?

rm -f "$P/watch_on"
sleep 4   # let the watcher loop observe the off switch
cp $R/serve_arc_b70.sh "$P/" >/dev/null 2>&1 || true
cp $R/p647/r256_probe.py "$P/" >/dev/null 2>&1 || true
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.out" -o -name "*.json" \
  -o -name "serve.log" -o -name "boot.txt" -o -name "*.devcore" \) \
  -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "R256_BANK $P ($(find . -type f | wc -l) files)"
echo "R256_SUMMARY repro=$RA ladder=$RB control=$RC (2=crash 0=clean 1=fail)"
echo "P647_R256_REPRO_COMPLETE -- bank ready for pull"
exit 0
