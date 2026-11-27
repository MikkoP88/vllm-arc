#!/bin/bash
# r262_expandview_leg.sh -- the expand-view stride A/B (P6.47 r262).
# Static audit after the r261 all-clean real-walk replay: the windowed
# synth path (engine.py, continuation prefill 8 < q_len <= 128) hands
# the native wheel bt_row.expand(q_len, -1) -- a stride-0 VIEW of one
# persistent block-table row. The Triton lanes multiply by stride(0);
# the wheel does not (mx_paged_decode_xe2.cpp passes data_ptr + size(1);
# the kernel addresses row r at r * bt_width). Rows 1..63 therefore
# read the table's zeroed following rows (block 0) and rows 64.. read
# PAST the [64, 640] table as block ids. Every crash in the rung chain
# is a q_len > 64 synth ride (the multi rehit: 118 rows); the 56/57-row
# essay/code rehits stay inside the table.
#
# Three fresh-boot EAGER phases (gemma m8 posture) on the cert29 image
# with an mx_kv OVERLAY mounted by the wrapper's ARC_MXKV_OVERLAY hook
# (tools/ar/r262_patch.py builds it from the image's own bytes; the
# mode is baked into the bytes and sha256-proven inside the container):
#   a  probe overlay (fix ON + the MXKVBTPROBE row dump): face multi
#      -> the kernel-visible rows, then the fixed ride (expect CLEAN)
#   c  on overlay (the landing-candidate fix): faces essay code multi
#      simple multi -> expect CLEAN through all 25 gens
#   b  off overlay (boot marker ONLY -- the control that the mount
#      alone does not defuse the crash): face multi -> expect CRASH
# Crash outcomes are DATA: exit is always 0; the client polls
# P647_R262_LEG_DONE. W14 is restored (health-aware, one container
# restart if the serve boot hangs) BEFORE the DONE line.
set -uo pipefail
R=/root/build
P=$R/p647/r262
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert29}
CNAME=arc-r262
MXKV=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv
mkdir -p "$P" "$P/devcores"

# ---- pre-refuses ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R262_BUSY (arc leg in flight)"; exit 2
fi
GOTSS=$(md5sum $R/serve_arc_b70.sh 2>/dev/null | cut -d" " -f1)
[ "$GOTSS" = "c2db777506e776a86426852b6bb80de4" ] \
  || { echo "R262_SERVE_SCRIPT_STALE ($GOTSS)"; exit 1; }
[ -f $R/p647/r260_bisect_probe.py ] || { echo "R262_NO_PROBE (ship p647/r260_bisect_probe.py first)"; exit 1; }
GOTPS=$(sha256sum $R/p647/r260_bisect_probe.py | cut -c1-64)
[ "$GOTPS" = "997204d72ec16d35e3ee4aacae4d8cd793491ee61943c04055892dbc26a767db" ] \
  || { echo "R262_PROBE_STALE ($GOTPS)"; exit 1; }
[ -f $R/p647/r262_patch.py ] || { echo "R262_NO_PATCHER (ship p647/r262_patch.py first)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R262_NO_IMAGE $IMG"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "R262_IMAGE $IMG $MF_SHA"

# r259-confirmed crash mode: EAGER (the m8g posture minus graphs)
GEMMA_NAT="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

w14_wait() {  # $1 = max polls (10 s each)
  local i=0
  while [ $i -lt "$1" ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
    i=$((i + 1)); sleep 10
  done
  return 1
}

w14_restore() {
  rm -f "$P/watch_on"
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  if curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "W14_ALREADY_HEALTHY (restore skipped)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r262.log 2>&1 &
  if w14_wait 54; then   # a healthy boot takes ~5 min
    echo "W14_HEALTH_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    return 0
  fi
  # 2026-11-27 (after the r261 leg): start + serve booted to "JIT kernel
  # warmup finished" and hung on shm_broadcast forever; a container
  # restart + relaunch was the remedy. One retry, same recipe.
  echo "W14_BOOT_HANG $(date -u +%Y-%m-%dT%H:%M:%SZ) (no health 9 min after start + serve; one container restart)"
  docker restart radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r262b.log 2>&1 &
  if w14_wait 54; then
    echo "W14_HEALTH_OK_AFTER_RESTART $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    return 0
  fi
  echo "W14_RESTORE_FAIL $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  return 1
}
trap w14_restore EXIT

build_overlay() {  # $1 = mode
  local MODE=$1
  local OV=$R/p647/r262_ov_$MODE
  local RC=0
  if [ -f "$OV/.r262_ready" ]; then
    echo "R262_OVERLAY_READY $MODE $(cd "$OV" && sha256sum engine.py native_decode.py | awk '{print $1}' | tr '\n' ' ')"
    return 0
  fi
  rm -rf "$OV"; mkdir -p "$OV"
  docker create --name r262mk --entrypoint /bin/bash "$IMG" -c "sleep 1" >/dev/null 2>&1 \
    || { echo "R262_OVERLAY_CREATE_FAIL"; return 1; }
  docker cp "r262mk:$MXKV/." "$OV/" >/dev/null 2>&1; RC=$?
  docker rm r262mk >/dev/null 2>&1
  [ "$RC" -eq 0 ] || { echo "R262_OVERLAY_CP_FAIL $MODE"; return 1; }
  rm -rf "$OV/__pycache__"
  python3 $R/p647/r262_patch.py "$OV" --mode "$MODE" > "$P/patch_$MODE.out" 2>&1 \
    || { cat "$P/patch_$MODE.out"; echo "R262_PATCH_FAIL $MODE"; return 1; }
  cat "$P/patch_$MODE.out"
  touch "$OV/.r262_ready"
}

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "R262_FAIL_STALE_CONTAINER $CNAME"; return 1
}

gcount() {  # $.. = grep args (a leading -i is a grep flag, not the pattern)
  local c
  c=$(docker exec $CNAME grep -c "$@" /root/serve_full.log 2>/dev/null)
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

boot_phase() {  # $1 = phase dir  $2 = overlay dir  $3 = mode
  local CD=$1
  local OV=$2
  local MODE=$3
  mkdir -p "$CD"
  devcore_preclear
  echo "=== [boot] $CNAME (gemma m8 posture, EAGER, overlay=$MODE) $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME ARC_MXKV_OVERLAY="$OV" \
    $GEMMA_NAT ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R262_BOOT_FAIL $CD"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    return 1
  fi
  echo "R262_HEALTH_OK $(date +%H:%M:%S)"
  sleep 5
  # Overlay identity = BYTE identity: the two touched files inside the
  # container must hash exactly as the host overlay (positive proof of
  # what the phase ran; the first launch gated on a source-grep marker
  # that does not exist literally -- the mode is a %s argument).
  (cd "$OV" && sha256sum engine.py native_decode.py | awk '{print $1}') > "$CD/ov_sha.txt"
  docker exec $CNAME sha256sum $MXKV/engine.py $MXKV/native_decode.py 2>/dev/null \
    | awk '{print $1}' > "$CD/ctr_sha.txt"
  {
    echo "mode=$MODE overlay=$OV"
    echo "-- host overlay:"
    sed 's/^/   /' "$CD/ov_sha.txt"
    echo "-- inside the container:"
    sed 's/^/   /' "$CD/ctr_sha.txt"
  } > "$CD/overlay_identity.txt"
  cat "$CD/overlay_identity.txt"
  if [ ! -s "$CD/ctr_sha.txt" ] || ! cmp -s "$CD/ctr_sha.txt" "$CD/ov_sha.txt"; then
    echo "R262_OVERLAY_NOT_MOUNTED $CD (mode=$MODE: container bytes != overlay bytes)"
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    return 1
  fi
  echo "R262_OVERLAY_MOUNTED mode=$MODE (byte-identical: $(tr '\n' ' ' < "$CD/ctr_sha.txt"))"
  {
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "native_lines=$(gcount 'MXKV native call')"
    echo "bridge_marker_lines=$(gcount 'MXKV native bridge r262 overlay: mode=')"
    echo "bridge_mode_lines=$(gcount "MXKV native bridge r262 overlay: mode=$MODE")"
    echo "pc2_windowed_impl=$(gcount 'MXKV P-C2 windowed impl')"
    echo "lmcache_lines=$(gcount 'LMCacheConnector')"
    # 2026-11-27 marker audit: the r259/r260 "xpu_graph_lines=0" was
    # VACUOUS (the count helper took its first arg as the pattern, so
    # "-i" was counted and the pattern dropped). Their eager posture is
    # nonetheless proven by the banked serve logs themselves
    # (cudagraph_mode=NONE / "Skipping ... CUDA graph capture"). Here
    # the posture is counted on the lines that actually discriminate.
    echo "graph_capture_lines=$(gcount 'Capturing CUDA graphs\|Capturing XPU graphs\|graph capture beginning\|cudagraph_mode=FULL')"
    echo "eager_posture_lines=$(gcount 'graph posture cudagraph_mode=NONE capture_sizes=None')"
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
  } > "$CD/markers.txt"
  cat "$CD/markers.txt"
  grep -Eq "mx_backend_lines=[1-9]" "$CD/markers.txt" || echo "R262_POSTURE_WARN no MX backend line"
  grep -Eq "pc2_windowed_impl=[1-9]" "$CD/markers.txt" || echo "R262_POSTURE_WARN no P-C2 window telemetry"
  if ! grep -Eq "bridge_mode_lines=[1-9]" "$CD/markers.txt"; then
    echo "R262_RUNTIME_MARKER_ABSENT $CD (mode=$MODE: no runtime bridge marker after warmup)"
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    return 1
  fi
  grep -q "lmcache_lines=0" "$CD/markers.txt" || echo "R262_POSTURE_WARN LMCache engaged"
  grep -q "graph_capture_lines=0" "$CD/markers.txt" || echo "R262_POSTURE_WARN graph capture lines present"
  grep -Eq "eager_posture_lines=[1-9]" "$CD/markers.txt" || echo "R262_POSTURE_WARN no eager posture line"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma4-31b","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
}

run_phase() {  # $1 = label  $2 = phase dir  $3 = mode  $4 = probe cap s  $5.. = faces
  local PH=$1
  local CD=$2
  local MODE=$3
  local CAP=$4
  shift 4
  local FACES="$*"
  local OV=$R/p647/r262_ov_$MODE
  local NG=0
  local NF=0
  local DL=0
  local ST=1
  local WANT=0
  if ! boot_phase "$CD" "$OV" "$MODE"; then return 1; fi
  echo "=== [$PH] overlay=$MODE faces: $FACES $(date +%H:%M:%S) ==="
  ( cd "$CD" && ARC_PROBE_MODEL=gemma4-31b timeout "$CAP" \
      python3 $R/p647/r260_bisect_probe.py "p47r262$PH" $FACES > probe.out 2>&1 ) \
    || echo "R262_PROBE_NONZERO $PH (cap=${CAP}s or nonzero exit; the verdict below decides)"
  sleep 10   # the reset precedes the python surface by 13-18 s
  dmesg -T > "$CD/dmesg_final.txt" 2>/dev/null || true
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 \
    || echo "serve.log MISSING $CD"
  docker exec $CNAME grep -E "MXKVBTPROBE|MXKV native bt materialized|MXKV native bridge r262|MXKV native call" \
    /root/serve_full.log > "$CD/btprobe.txt" 2>/dev/null || true
  docker exec $CNAME sh -c \
    'echo DEVICE_LOST=$(grep -c UR_RESULT_ERROR_DEVICE_LOST /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || echo "DEVICE_LOST=? TRACEBACK=?" > "$CD/erraudit.txt"
  cat "$CD/erraudit.txt"
  docker exec $CNAME grep -E "prompt_token_ids_len|num_computed_tokens|num_scheduled_tokens" \
    /root/serve_full.log 2>/dev/null | tail -8 > "$CD/req_lines.txt" || true
  docker exec $CNAME grep -B2 -A14 "UR_RESULT_ERROR_DEVICE_LOST" /root/serve_full.log 2>/dev/null \
    | tail -60 > "$CD/devicelost_context.txt" || true
  grep "^R260_GEN face=" "$CD/probe.out" > "$CD/gen_summary.txt" 2>/dev/null || true
  echo "-- [$PH] generations:"
  cat "$CD/gen_summary.txt" 2>/dev/null || true
  tail -3 "$CD/probe.out" 2>/dev/null || true
  echo "-- [$PH] materializations=$(grep -c 'MXKV native bt materialized' "$CD/btprobe.txt" 2>/dev/null || true) probe_calls=$(grep -c 'MXKVBTPROBE call' "$CD/btprobe.txt" 2>/dev/null || true) probe_rows=$(grep -c 'MXKVBTPROBE row=' "$CD/btprobe.txt" 2>/dev/null || true) beyond_rows=$(grep -c 'MXKVBTPROBE row=.* BEYOND ' "$CD/btprobe.txt" 2>/dev/null || true)"
  NG=$(grep -c "^R260_GEN face=" "$CD/probe.out" 2>/dev/null || true)
  NG=${NG:-0}
  NF=$(echo "$FACES" | wc -w)
  DL=$(grep -o "DEVICE_LOST=[0-9]*" "$CD/erraudit.txt" 2>/dev/null | head -1 | cut -d= -f2)
  DL=${DL:-0}
  WANT=$((5 * NF))
  if [ "$DL" -gt 0 ] || grep -q "R260_DONE crashed=True" "$CD/probe.out" 2>/dev/null \
     || ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    ST=2
  elif [ "$NG" -eq "$WANT" ] && grep -q "R260_DONE crashed=False" "$CD/probe.out" 2>/dev/null; then
    ST=0
  else
    ST=3
  fi
  echo "R262_PHASE $PH status=$ST gens=$NG/$WANT devlost=$DL (0=clean 2=crash 3=incomplete 1=fail) $(date +%H:%M:%S)"
  return $ST
}

# ---- build the three overlays BEFORE quiescing W14 (CPU-only steps) ----
for M in probe on off; do
  build_overlay "$M" || { echo "R262_ABORT (overlay $M)"; trap - EXIT; exit 1; }
done
docker rm -f $CNAME >/dev/null 2>&1

# ---- entry: quiesce W14, arm the watcher, run the three phases ----
docker stop radiance-test >/dev/null 2>&1
sleep 5
touch "$P/watch_on"
devcore_watch > "$P/devcores/watch.log" 2>&1 &

SA=1 SC=1 SB=1
run_phase a "$P/a_probe_fixon" probe 1500 multi;                        SA=$?
teardown_phase || true
run_phase c "$P/c_suite_fixon" on 4800 essay code multi simple multi;   SC=$?
teardown_phase || true
run_phase b "$P/b_ctrl_fixoff" off 1500 multi;                          SB=$?
teardown_phase || true

rm -f "$P/watch_on"
sleep 4   # let the watcher loop observe the off switch
for M in probe on off; do
  mkdir -p "$P/overlay_$M"
  cp $R/p647/r262_ov_$M/engine.py $R/p647/r262_ov_$M/native_decode.py "$P/overlay_$M/" >/dev/null 2>&1 || true
done
cp $R/p647/r262_patch.py $R/p647/r262_expandview_leg.sh $R/p647/r260_bisect_probe.py "$P/" >/dev/null 2>&1 || true
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.out" -o -name "*.json" \
  -o -name "serve.log" -o -name "boot.txt" -o -name "*.devcore" \
  -o -name "*.py" -o -name "*.sh" \) \
  -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "R262_BANK $P ($(find . -type f | wc -l) files)"
echo "R262_SUMMARY a_probe=$SA c_suite=$SC b_ctrl=$SB (0=clean 2=crash 3=incomplete 1=fail)"

# ---- W14 first (session-end law), then the DONE line ----
W14=1
w14_restore; W14=$?
trap - EXIT
echo "W14_FINAL rc=$W14"
echo "P647_R262_LEG_DONE -- bank ready for pull"
exit 0
