#!/bin/bash
# r257_faithful_replay.sh -- the r255 gemma MX DEVICE_LOST faithful replay
# (P6.47 r257 rung). r256 proved the repro gap: the r255 crash request
# rode the chat template's OFF branch (chat_template_kwargs
# {"enable_thinking": False}; the qwen38_high template serves on gemma
# -- GEMMA_NAT sets no ARC_TEMPLATE), while every r256 request rode the
# ON branch (+47 template tokens: same lengths, different bytes, the
# r255 tail window never tested).
# This leg replays the r255 sequence FAITHFULLY, devcore watcher live:
#   B replay (boot 1): the ACTUAL /root/build/p647/p647_longprobe.py
#     full suite -- the identical binary probe that crashed r255
#     (essay -> code -> multi, cold/rehit/rehit2 + temp1 x2, the kwarg
#     in every request). Crash expected at the multi second sample.
#     PHASE ORDER NOTE: this supersedes the journal's A-then-B listing
#     -- running the minimal pair first would seed the multi blocks and
#     turn the suite's cold multi into a prefix-hit, which is NOT the
#     r255 sequence. The suite replays first, uncontaminated.
#   A faithful (only if B crashed -- fresh boot): the r255 request
#     byte-exact modulo max_tokens (UNCUT multi + the enable_thinking
#     kwarg) fresh -> A/A resend, in the MINIMAL state. B-crash +
#     A-clean = the state delta is isolated (the suite-state allocator
#     layout is the trigger); B-crash + A-crash = the pair alone
#     suffices (bytes-intrinsic).
#   F fullstate (only if B was clean -- boot 2): the r255 run_leg
#     preamble state (r113 decode battery -> garbage probe) before the
#     longprobe full again -- the state the crash actually rode. (The
#     text_battery + post-suite churn/rehit cycles remain the r258
#     deltas if F is also clean.)
# Crash outcomes are DATA: exit is always 0; the client polls
# P647_R257_REPLAY_COMPLETE. trap restores W14.
set -uo pipefail
R=/root/build
P=$R/p647/r257
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert29}
CNAME=arc-r257
mkdir -p "$P" "$P/devcores"

# ---- pre-refuses: no second arc leg, byte pins, probe binaries ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R257_BUSY (arc leg in flight)"; exit 2
fi
GOTSS=$(md5sum $R/serve_arc_b70.sh 2>/dev/null | cut -d" " -f1)
[ "$GOTSS" = "c2db777506e776a86426852b6bb80de4" ] \
  || { echo "R257_SERVE_SCRIPT_STALE ($GOTSS != c2db777506e776a86426852b6bb80de4)"; exit 1; }
[ -f $R/p647/r257_probe.py ] || { echo "R257_NO_PROBE (ship p647/r257_probe.py first)"; exit 1; }
[ -f $R/p647/p647_longprobe.py ] || { echo "R257_NO_LONGPROBE"; exit 1; }
[ -f $R/r113_battery.py ] || { echo "R257_NO_BATTERY"; exit 1; }
[ -f $R/p647/arc_garbage_probe.py ] || { echo "R257_NO_GARBAGE_PROBE"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R257_NO_IMAGE $IMG"; exit 1; }
echo "R257_PROBE_MD5 $(md5sum $R/p647/r257_probe.py)"
# image identity for the bank (log-only: the r255 gate already refused
# on fork_sha + content markers -- this cites what the replay ran on)
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "R257_IMAGE $IMG $MF_SHA"

GEMMA_NAT="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

w14_restore() {
  rm -f "$P/watch_on"
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r257.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "R257_FAIL_STALE_CONTAINER $CNAME"; return 1
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
  # phase's (writing the data handle is the kernel's documented release)
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
  if ! boot_wait r257; then
    echo "R257_BOOT_FAIL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    return 1
  fi
  echo "R257_HEALTH_OK $(date +%H:%M:%S)"
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
  grep -Eq "mx_backend_lines=[1-9]" "$CD/markers.txt" || echo "R257_POSTURE_WARN no MX backend line"
  grep -Eq "native_lines=[1-9]" "$CD/markers.txt" || echo "R257_POSTURE_WARN no native telemetry"
  grep -Eq "pc2_windowed_impl=[1-9]" "$CD/markers.txt" || echo "R257_POSTURE_WARN no P-C2 window telemetry"
  grep -q "lmcache_lines=0" "$CD/markers.txt" || echo "R257_POSTURE_WARN LMCache engaged"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma4-31b","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
}

run_probe() {  # $1 = output file $2 = ceiling-polls $3.. = probe argv
  local OUT=$1 CEIL=$2; shift 2
  ( docker exec -i $CNAME /opt/venv/bin/python - "$@" \
      < $R/p647/r257_probe.py > "$OUT" 2>&1 ) || true &
  local i=0
  while [ $i -lt $CEIL ]; do
    grep -q "R257_PROBE_DONE" "$OUT" 2>/dev/null && return 0
    sleep 10; i=$((i + 1))
  done
  echo "R257_PROBE_CEILING_HIT $OUT (no DONE line in $((CEIL*10))s)"
  grep -q "R257_PROBE_DONE" "$OUT" 2>/dev/null
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

crash_verdict() {  # $1 = phase dir -> returns 2 = crash, 0 = clean
  local CD=$1
  local DL
  DL=$(grep -o "DEVICE_LOST=[0-9]*" "$CD/erraudit.txt" 2>/dev/null | head -1 | cut -d= -f2)
  DL=${DL:-0}
  if [ "$DL" -gt 0 ]; then
    echo "R257_ENGINE_DEAD $CD (DEVICE_LOST=$DL)"
    return 2
  fi
  if ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "R257_ENGINE_DOWN $CD (health dead, no DEVICE_LOST line)"
    return 2
  fi
  return 0
}

phase_replay() {  # boot 1 + the pure r255 suite replay. 2=crash 0=clean 1=fail
  local CD=$P/replay
  local V
  devcore_preclear
  if ! boot_leg "$CD"; then return 1; fi
  echo "=== [B replay] the ACTUAL p647_longprobe full suite -- the identical binary probe that crashed r255 $(date +%H:%M:%S) ==="
  ( cd "$CD" && ARC_PROBE_MODEL=gemma4-31b timeout 2400 python3 $R/p647/p647_longprobe.py p47r257 full > longprobe_full.out 2>&1 ) \
    || echo "R257_LONGPROBE_NONZERO (engine died mid-suite? the verdict below decides)"
  capture_phase "$CD"
  echo "-- longprobe tail:"
  tail -20 "$CD/longprobe_full.out" 2>/dev/null || true
  crash_verdict "$CD"; V=$?
  if [ "$V" = 2 ]; then
    echo "R257_REPLAY_CRASHED (the r255 sequence REPRODUCED on the faithful bytes -- dumps banked by the watcher)"
    return 2
  fi
  echo "R257_REPLAY_CLEAN (the r255 sequence did NOT crash on this boot)"
  return 0
}

phase_faithful() {  # fresh boot: the OFF-branch minimal (the state question)
  local CD=$P/faithful
  teardown_phase || true
  devcore_preclear
  if ! boot_leg "$CD"; then return 1; fi
  echo "=== [A faithful] the r255 request byte-exact (UNCUT multi + the enable_thinking kwarg): fresh -> A/A resend, MINIMAL state $(date +%H:%M:%S) ==="
  if ! run_probe "$CD/probe.txt" 90; then
    echo "R257_FAITHFUL_PROBE_NO_DONE (hung with the engine? bank partial)"
  fi
  capture_phase "$CD"
  grep -E "^CALIB |^FULL_MULTI|^RUNG " "$CD/probe.txt" 2>/dev/null || true
  if grep -q '"alive": false' "$CD/probe.txt" 2>/dev/null; then
    echo "R257_FAITHFUL_CRASHED (the minimal OFF-branch pair alone reproduces -- bytes-intrinsic)"
    return 2
  fi
  echo "R257_FAITHFUL_CLEAN (B crashed, A clean -> the state delta is ISOLATED: the suite-state allocator layout is the trigger)"
  return 0
}

phase_fullstate() {  # boot 2: the r255 run_leg preamble + the suite again
  local CD=$P/fullstate
  local V
  teardown_phase || true
  devcore_preclear
  if ! boot_leg "$CD"; then return 1; fi
  echo "=== [F preamble] r113 decode battery (the run_leg state) $(date +%H:%M:%S) ==="
  timeout 900 python3 $R/r113_battery.py decode gemma4-31b > "$CD/battery.txt" 2>&1 \
    || echo "R257_BATTERY_NONZERO (bank the partial; the suite still runs)"
  tail -6 "$CD/battery.txt" 2>/dev/null || true
  curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 \
    && echo "R257_BATTERY_PHASE_HEALTH_OK" \
    || echo "R257_BATTERY_PHASE_ENGINE_DOWN (engine died in the battery -- the longprobe still runs)"
  echo "=== [F preamble] garbage probe $(date +%H:%M:%S) ==="
  timeout 300 env ARC_PROBE_MODEL=gemma4-31b python3 $R/p647/arc_garbage_probe.py > "$CD/garbage.out" 2>&1 || true
  tail -4 "$CD/garbage.out" 2>/dev/null || true
  curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 \
    && echo "R257_GARBAGE_PHASE_HEALTH_OK" \
    || echo "R257_GARBAGE_PHASE_ENGINE_DOWN"
  echo "=== [F] the longprobe full suite again, on the preamble state $(date +%H:%M:%S) ==="
  ( cd "$CD" && ARC_PROBE_MODEL=gemma4-31b timeout 2400 python3 $R/p647/p647_longprobe.py p47r257f full > longprobe_full.out 2>&1 ) \
    || echo "R257_LONGPROBE_NONZERO"
  capture_phase "$CD"
  tail -20 "$CD/longprobe_full.out" 2>/dev/null || true
  crash_verdict "$CD"; V=$?
  if [ "$V" = 2 ]; then
    echo "R257_FULLSTATE_CRASHED (the r255 sequence reproduced WITH the run_leg preamble state)"
    return 2
  fi
  echo "R257_FULLSTATE_CLEAN (the preamble state did not reproduce either -- r258: text_battery + churn cycles + cross-leg history)"
  return 0
}

# ---- entry: quiesce W14, start the devcore watcher, run the phases ----
docker stop radiance-test >/dev/null 2>&1
sleep 5
touch "$P/watch_on"
devcore_watch > "$P/devcores/watch.log" 2>&1 &

RB=1 RA=3 RF=3
phase_replay; RB=$?
if [ "$RB" = 2 ]; then
  phase_faithful; RA=$?
elif [ "$RB" = 0 ]; then
  phase_fullstate; RF=$?
fi

rm -f "$P/watch_on"
sleep 4   # let the watcher loop observe the off switch
cp $R/serve_arc_b70.sh "$P/" >/dev/null 2>&1 || true
cp $R/p647/r257_probe.py "$P/" >/dev/null 2>&1 || true
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.out" -o -name "*.json" \
  -o -name "serve.log" -o -name "boot.txt" -o -name "*.devcore" \) \
  -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "R257_BANK $P ($(find . -type f | wc -l) files)"
echo "R257_SUMMARY replay=$RB faithful=$RA fullstate=$RF (2=crash 0=clean 1=fail 3=skipped)"
echo "P647_R257_REPLAY_COMPLETE -- bank ready for pull"
exit 0
