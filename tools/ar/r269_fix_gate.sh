#!/bin/bash
# r269_fix_gate.sh -- P6.47 P-D (r269): the D-B FIX-GATE leg -- ONE
# c3-posture boot (gemma, lawful g3 skin, KV mxfp8 native, GRAPHS ON =
# the production/cert posture) with the r269 FIX overlay mounted over
# the image's installed mx_kv package (ARC_MXKV_OVERLAY=/root/build/
# r269_fix -- built by tools/ar/r269_build_fix_overlay.sh: byte-copy of
# the tree package + the WRAPPER-LEVEL class fix, all three decode
# entry points resolve block_size None -> the cache's OWN page), then
# the FULL p266 matrix probe (chat t0,card + raw + nll + gate cells,
# logprobs on every request -- the strongest A/B against the banked c3
# RED: 324x400-nan + 'our' attractor) + the r264 gate battery (g1).
#
# VERDICT LOGIC: the fix closes the HALF-page walk (engine.py's five
# call sites all omit block_size; wrapper default 64 on the p128
# global pool = wrong rows at seq_len>64).  LEG GREEN (logprobs
# finite, faces clean) -> D-B was the engine corruption: land the
# wrapper fix in the tree with this leg as evidence, then the m8g/
# m4g/mkg re-gates + full battery + cert29 cut per Law 11.  LEG RED
# with the SAME signature -> a residual defect (G2/G3) lives at
# short-ctx: the r268 trace leg (shipped, 0ce0954c) reads the live
# store/bt values next.  Either outcome is decisive evidence.
#
# W14 (radiance-test) is STOPPED for the leg window (never inside its
# first 180s -- stale-baton law) and the EXIT trap restores it with
# the clean recipe (r262_w14_restore.sh).  Artifacts ->
# /root/build/p647/r269_fixgate/f3_fixm8g_29/.  Exit 0 = leg banked.
# Launch (ship + launch in SEPARATE ssh calls, md5-verified ship):
#   nohup setsid bash /root/build/r269/r269_fix_gate.sh \
#     > /root/build/p647/r269_fixgate/logs/launch.log 2>&1 &
set -uo pipefail
R=/root/build
P=$R/p647/r269_fixgate
S=$R/r269
OVL=$R/r269_fix
IMG=vllm-arc:v0.30.0-b70-cert29
CNAME=arc-r269
NAME=f3_fixm8g_29
SERVED=gemma4-31b
T_G3=$R/chat_template_gemma4.jinja
T_G4=$R/r266/chat_template_gemma4_official.jinja
PROBE=$R/r266/p266_matrix_probe.py
BATT=$R/r113_battery.py
PROBE_TIMEOUT=2700

# --- preflight -------------------------------------------------------
[ -f "$OVL/triton_mx_decode.py" ] && [ -f "$OVL/engine.py" ] || {
  echo "R269_NO_OVERLAY (ship tools/ar/p269_fix/overlay first)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R269_NO_IMAGE $IMG"; exit 1; }
[ -f "$PROBE" ] && [ -f "$BATT" ] && [ -f "$T_G3" ] && [ -f "$T_G4" ] || {
  echo "R269_NO_HELPER (probe/battery/templates)"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R269_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R269_PORT_HELD (unknown holder)"; exit 3
  fi
fi

kmark() { cut -d' ' -f1 /proc/uptime | cut -d. -f1; }
khealth() {  # $1 = bank dir $2 = kernel mark
  local T0=$2 N
  dmesg 2>/dev/null | awk -v t0="$T0" \
    '{ s = $1; gsub(/[][]/, "", s); if ((s + 0) >= (t0 + 0)) print }' \
    | grep -E 'Engine reset|exec queue reset|devcoredump|GPU HANG|[Dd]evice lost' \
    > "$1/kernel_faults.txt" || true
  N=$(wc -l < "$1/kernel_faults.txt")
  echo "kernel_fault_lines=$N" >> "$1/markers.txt" 2>/dev/null || true
  echo "KERNEL_HEALTH[f3]: $([ "$N" = "0" ] && echo OK || echo FAIL) fault_lines=$N"
  [ "$N" = "0" ] || head -6 "$1/kernel_faults.txt" | cut -c1-160
}

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R269_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"; return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"; return 0
  fi
  bash $R/r262_w14_restore.sh > $P/logs/w14_restore_r269.log 2>&1
  rc=$?
  grep -E "^W14_" $P/logs/w14_restore_r269.log | tail -3 || true
  echo "W14_RESTORE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$P/logs/w14_restore_r269.log"
}
trap w14_restore EXIT

CD=$P/$NAME
mkdir -p $P/logs "$CD"

echo "R269_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) leg=$NAME overlay: engine.py=$(md5sum $OVL/engine.py | cut -d' ' -f1) triton_mx_decode.py=$(md5sum $OVL/triton_mx_decode.py | cut -d' ' -f1)"
md5sum $R/r266/serve_arc_b70.sh "$PROBE" "$BATT" "$T_G3" "$T_G4"

echo "=== [r269] W14 quiesce $(date +%H:%M:%S) ==="
UP=$(docker inspect -f '{{.State.Running}}' radiance-test 2>/dev/null || echo false)
if [ "$UP" = true ]; then
  AGE=$(( $(date +%s) - $(date -d "$(docker inspect -f '{{.State.StartedAt}}' radiance-test)" +%s) ))
  if [ "$AGE" -lt 180 ]; then
    echo "R269_REFUSE_W14_RESTORE_BOOT (up ${AGE}s < 180s -- stale-baton law)"; exit 3
  fi
  docker stop radiance-test >/dev/null 2>&1
  echo "W14_STOPPED (was up ${AGE}s)"
else
  echo "W14_ALREADY_DOWN"
fi
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R269_WARN_PORT8000_STILL_HELD"
sleep 5
docker rm -f $CNAME >/dev/null 2>&1

K0=$(kmark)
NAT="ARC_MODEL=/models/gemma4-31b ARC_SERVED=$SERVED ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"
echo "leg=$NAME image=$IMG served_fmt=g3 chat_cells=t0,card g1=1 env=[$NAT ARC_TEMPLATE=$T_G3 ARC_RPARSER= ARC_TPARSER= ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 ARC_MXKV_OVERLAY=$OVL]" > "$CD/leg_env.txt"

echo "=== [r269] boot c3 posture + FIX overlay $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE=$IMG ARC_NAME=$CNAME $NAT \
  ARC_TEMPLATE=$T_G3 ARC_RPARSER= ARC_TPARSER= \
  ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 ARC_MXKV_OVERLAY=$OVL \
  bash $R/r266/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
BOOTPID=$!

boot_wait() {  # identity-verified health (the r266 law)
  local i=0 MODELS="" SEEN=0 ABSENT=0 tb
  while [ $i -lt 210 ]; do
    if curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      MODELS=$(curl -s -m 10 http://127.0.0.1:8000/v1/models 2>/dev/null)
      if printf '%s' "$MODELS" | grep -q "gemma4-31b"; then return 0; fi
    fi
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT + 1)); [ "$ABSENT" -gt 5 ] && { echo "[r269] CONTAINER_ABSENT poll=$i"; return 1; }
    else
      ABSENT=0
      if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
        tb=$(docker exec $CNAME grep -c "Traceback (most recent call last)" /root/serve_full.log 2>/dev/null || true)
        tb=${tb:-0}
        [ "$tb" -gt 0 ] 2>/dev/null && { echo "[r269] TRACEBACK poll=$i"; return 1; }
        [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r269] ENGINE_DEAD poll=$i"; return 1; }
      fi
    fi
    i=$((i + 1)); sleep 10
  done
  return 1
}

if ! boot_wait; then
  echo "R269_BOOT_FAIL"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null | cut -c1-300
  docker rm -f $CNAME >/dev/null 2>&1
  exit 3
fi
echo "HEALTH_OK_$NAME $(date +%H:%M:%S) (identity-verified)"
sleep 5
curl -s -m 10 http://127.0.0.1:8000/v1/models | cut -c1-200 > "$CD/models.txt"

cp_in() {  # $1 host -> $2 container (retry 5)
  local i
  for i in 1 2 3 4 5; do
    docker cp "$1" "$CNAME:$2" >/dev/null 2>&1 && return 0
    sleep 3
  done
  echo "CP_FAIL $1 -> $2"; return 1
}
# distinct in-container names: the served template is bind-mounted at
# /root/$(basename) -- a docker cp into that exact path is a cp into a
# mount (the v2 TEMPLATE_CP_FAIL). GEMMA TEMPLATES ONLY (owner law).
cp_in "$T_G3" /root/tpl_g3.jinja
cp_in "$T_G4" /root/tpl_g4.jinja
cp_in "$PROBE" /root/p266_matrix_probe.py

PARGS=(--model $SERVED --tokenizer /models/target --out /root/p266_out --tag "$NAME" --served g3
       --g3 /root/tpl_g3.jinja
       --g4 /root/tpl_g4.jinja --workers 8 --cells chat,raw,nll,gate --chat-cells t0,card)
echo "=== [r269] probe start $(date +%H:%M:%S) args=${PARGS[*]}"
timeout "$PROBE_TIMEOUT" docker exec $CNAME /opt/venv/bin/python /root/p266_matrix_probe.py "${PARGS[@]}" \
  > "$CD/matrix.out" 2>&1
PRC=$?
docker cp $CNAME:/root/p266_out/matrix_$NAME.json "$CD/" >/dev/null 2>&1 || echo "MATRIX_JSON_MISSING $NAME"
echo "probe done $(date +%H:%M:%S) rc=$PRC"
grep -E "^(PROMPT_ORACLE|FORMAT|chat |raw |    ptok_ok|gate264|    \[gate|NLL |P266_MATRIX_DONE)" "$CD/matrix.out" | cut -c1-330
[ "$PRC" = 0 ] || tail -15 "$CD/matrix.out" | cut -c1-300

python3 $BATT g1 $SERVED > "$CD/g1_probes.txt" 2>&1 || echo "G1_NONZERO_EXIT $NAME"
cut -c1-300 "$CD/g1_probes.txt"

{ grep -hE "^ARC_(SERVE_LAUNCHED|SERVE_KNOBS|CONTAINER_FRESH)" "$CD/boot.txt" 2>/dev/null
  docker exec $CNAME grep -n -E "non-default args|Casting torch|Default vLLM sampling|Resolved architecture|Using MX KV attention backend|MXKV native call|ARC-P628-FA2|ARC-P629-SLIDE|Quantized [0-9]+ layers|GPU KV cache size|Maximum concurrency|Model loading took|model weights took|Graph capturing finished|Using Triton backend" \
    /root/serve_full.log 2>/dev/null | cut -c1-700
} > "$CD/engine_config.txt" || true
head -24 "$CD/engine_config.txt" | cut -c1-400

{ echo "post_device_lost_lines=$(docker exec $CNAME grep -c 'UR_RESULT_ERROR_DEVICE_LOST' /root/serve_full.log 2>/dev/null || true)"
  echo "post_tracebacks=$(docker exec $CNAME grep -c 'Traceback (most recent call last)' /root/serve_full.log 2>/dev/null || true)"
  echo "post_errors=$(docker exec $CNAME grep -c ERROR /root/serve_full.log 2>/dev/null || true)"
} >> "$CD/markers.txt" 2>/dev/null || true
cat "$CD/markers.txt"

docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
  > "$CD/erraudit.txt" 2>/dev/null || echo "ERROR=?" > "$CD/erraudit.txt"
cat "$CD/erraudit.txt"

docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || echo "serve.log MISSING $NAME"
docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1 || true
sleep 8
docker rm -f $CNAME >/dev/null 2>&1
sleep 22
khealth "$CD" "$K0"

echo "R269_DONE rc=$PRC $(date -u +%Y-%m-%dT%H:%M:%SZ) bank=$CD"
exit $PRC
