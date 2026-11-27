#!/bin/bash
# r266_gemma_skin.sh -- gemma-4 (google/gemma-4-31B) issue-analysis matrix (r266).
#
# ANALYSIS ONLY. Nothing here changes a serving default or the engine: every
# leg boots one image with the r264 gate's gemma posture and varies exactly
# ONE serving-side value (image, chat-template skin, dtype, attention route,
# weight quant, KV dtype). The capture layer is identical on every boot:
#
#   CP1 request     p266_matrix_probe.py (in the engine container): full text,
#                   finish reason, token counts and the per-generated-token
#                   logprobs of every chat and raw request -> matrix_<leg>.json
#   CP2 prompt      the exact prompt string + token ids per format and the
#                   engine's own /tokenize ids (PROMPT_ORACLE, ptok_ok)
#   CP3 distribution teacher-forced logprob of every prompt token per format
#                   and the first-token top-5 (NLL lines)
#   CP4 engine      non-default args, dtype cast, sampling-default override,
#                   quantization summary, KV backend + route lines, the
#                   ARC_SERVE_* lines -> engine_config.txt
#   CP5 kernel      DEVICE_LOST / traceback / error counts (markers.txt) and
#                   dmesg fault lines since a mark taken before each boot
#   CP6 determinism greedy solo x3, concurrent x4, seeded samples, per format
#   CP7 gate        the r264 gate requests, verbatim, on the same boot
#
# Legs (name_image_skin[_knob]); R266_LEGS selects by comma list:
#   b2_g3_29    cert29  P6.28 template, fp8 weights + KV fp8   (the baseline)
#   c3_m8g_29   cert29  b2 + KV mxfp8 native     (gate lane m8g)
#   c4_m4g_29   cert29  b2 + KV mxfp4 native     (gate lane m4g)
#   c5_mkg_29   cert29  b2 + KV mxk8v4 native    (gate lane mkg)
#   w1_mxfp4kvfp8_29 cert29 b2 skin + MXFP4 W4A16 weights + KV fp8 (the
#                   cert28 gm4 lane shape -- weight-MX with KV FP8)
#   w2_mxfp8kvfp8_29 cert29 b2 skin + MXFP8 W8A16 weights + KV fp8 (cert28 gm8)
#   b3_g4_29    cert29  official google/gemma-4-31B-it template + gemma4 parsers
#   b5_g3_19    cert19  P6.28 skin on the last content-valid era image
#   b4_g3_20    cert20  P6.28 skin on the P6.30 close-out image
#   c1_bf16_29  cert29  b2 + ARC_DTYPE=bfloat16
#   c2_fa2off_29 cert29 b2 + ARC_ATTN_FA2=0 (pure-triton attention)
#
# Laws this runner encodes:
#   - OWNER DIRECTIVE (2026-11-28, raised twice): NO qwen env, env value,
#     template or parser touches ANY gemma test -- not even as a probe input
#     or a shipped file. Mixing the qwen skin into gemma testing is forbidden
#     outright; where a gemma template/parser cannot be found, the env is
#     REMOVED, never substituted. This runner ships and renders GEMMA
#     templates only (P6.28 + official google/gemma-4-31B-it), and the probe
#     refuses (P266_REFUSE_SERVED_TEMPLATE_ABSENT) rather than substitute.
#   - parsers are a measured variable, not a default: every b2-derived leg
#     boots with NO parser flags at all (the cert12-cert19 posture; the owner
#     hypothesis is that most gemma tests need none). Only the optional b3
#     adds the gemma4 parser pair -- never a qwen parser.
#   - W14 (radiance-test) owns host port 8000; it is STOPPED for the leg
#     window and the EXIT trap restores it with the clean recipe
#     (r262_w14_restore.sh) -- the r264 gate pattern. A restore boot inside
#     its first 3 minutes is never stopped (stale-baton law).
#   - boot_wait passes only when /v1/models answers THIS leg's served model
#     (engine-identity gate): the first launch passed health against W14's
#     qwen engine in seconds -- that evidence was discarded and the banks
#     wiped.
R=/root/build/r266
P=/root/build/p647/r266_gemma
R0=/root/build
CNAME=arc-r266
DRY=${R266_DRY:-0}
LEGS=${R266_LEGS:-b2,c3,c4,c5,w1,w2}
PROBE_TIMEOUT=${R266_PROBE_TIMEOUT:-2700}
SERVED=gemma4-31b

IMG29=vllm-arc:v0.30.0-b70-cert29
IMG20=vllm-arc:v0.30.0-b70-cert20
IMG19=vllm-arc:v0.30.0-b70-cert19
T_G3=$R0/chat_template_gemma4.jinja
T_G4=$R/chat_template_gemma4_official.jinja

# the r264 gate's GEMMA_NAT posture, byte for byte (ARC_RPC= kept as the gate
# passes it; the serve script reads ARC_RPC_TIMEOUT, so it is a no-op there)
NAT="ARC_MODEL=/models/gemma4-31b ARC_SERVED=$SERVED ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"
# skins: gemma-only. No SKIN_QWEN exists in this script (owner directive
# 2026-11-28) -- a qwen-skinned gemma boot cannot be constructed from here.
SKIN_G3="ARC_TEMPLATE=$T_G3 ARC_RPARSER= ARC_TPARSER="
SKIN_G4="ARC_TEMPLATE=$T_G4 ARC_RPARSER=gemma4 ARC_TPARSER=gemma4"

LEG_FAIL=0
mkdir -p "$P/logs"

kmark() { cut -d' ' -f1 /proc/uptime | cut -d. -f1; }
khealth() {  # $1 = bank dir $2 = kernel mark $3 = label
  local T0=$2 N=""
  dmesg 2>/dev/null | awk -v t0="$T0" \
    '{ s = $1; gsub(/[][]/, "", s); if ((s + 0) >= (t0 + 0)) print }' \
    | grep -E 'Engine reset|exec queue reset|devcoredump|GPU HANG|[Dd]evice lost' \
    > "$1/kernel_faults.txt" || true
  N=$(wc -l < "$1/kernel_faults.txt")
  echo "kernel_fault_lines=$N" >> "$1/markers.txt" 2>/dev/null || true
  echo "KERNEL_HEALTH[$3]: $([ "$N" = "0" ] && echo OK || echo FAIL) fault_lines=$N (dmesg since uptime ${T0}s)"
  if [ "$N" != "0" ]; then
    LEG_FAIL=$((LEG_FAIL + 1))
    head -6 "$1/kernel_faults.txt" | cut -c1-160
  fi
}

gcount() {
  local c
  c=$(docker exec $CNAME grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {  # $1 = label  $2 = served name -- health AND identity
  local i=0 ok="" SEEN=0 ABSENT=0 MODELS=""
  while [ $i -lt 210 ]; do
    if curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
      MODELS=$(curl -s -m 10 http://127.0.0.1:8000/v1/models 2>/dev/null)
      if printf '%s' "$MODELS" | grep -q "\"$2\""; then ok=1; break; fi
      # a foreign engine holds the port: keep waiting for OUR engine
    fi
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT + 1)); [ "$ABSENT" -gt 5 ] && { echo "[$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  printf '%s' "$MODELS" > "$P/models_last.txt" 2>/dev/null || true
  [ "$ok" = "1" ]
}

cp_in() {  # $1 = host path  $2 = container path (docker cp with retry)
  local i
  for i in 1 2 3 4 5; do
    docker cp "$1" "$CNAME:$2" >/dev/null 2>&1 && return 0
    sleep 3
  done
  echo "CP_FAIL $1 -> $2 (last error:)"; docker cp "$1" "$CNAME:$2" 2>&1 | tail -2
  return 1
}

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "R266_STALE_CONTAINER $CNAME"; return 1
}

w14_stop() {  # W14 owns host port 8000; stop it for the leg window
  local UP AGE
  UP=$(docker inspect -f '{{.State.Running}}' radiance-test 2>/dev/null || echo false)
  [ "$UP" = true ] || { echo "W14_ALREADY_DOWN"; return 0; }
  AGE=$(( $(date +%s) - $(date -d "$(docker inspect -f '{{.State.StartedAt}}' radiance-test)" +%s) ))
  [ "$AGE" -gt 180 ] || refuse "R266_REFUSE_W14_RESTORE_BOOT (up ${AGE}s < 180s -- stale-baton law)"
  docker stop radiance-test >/dev/null 2>&1 || refuse "R266_W14_STOP_FAIL"
  echo "W14_STOPPED (was up ${AGE}s)"
}

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  bash $R0/r262_w14_restore.sh > $P/logs/w14_restore_r266.log 2>&1
  local rc=$?
  grep -hE "^W14_(STALE_BATON_DIRS_REMOVED|SERVE_LAUNCHED|HEALTH_OK|GEN_OK|GEN_FAIL|RESTORE_FAIL|FINAL|ALREADY_HEALTHY)" \
    $P/logs/w14_restore_r266.log || true
  echo "W14_RESTORE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$P/logs/w14_restore_r266.log"
}
trap w14_restore EXIT

refuse() {  # $1 = message  $2 = rc -- a refusal must not fire the W14 restore
  trap - EXIT
  echo "$1"
  exit "${2:-1}"
}

markers() {  # $1 = leg dir $2 = prefix (pre|post)
  {
    echo "${2}_mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "${2}_native_lines=$(gcount 'MXKV native call')"
    echo "${2}_materialized_lines=$(gcount 'MXKV native bt materialized')"
    echo "${2}_device_lost_lines=$(gcount 'UR_RESULT_ERROR_DEVICE_LOST')"
    echo "${2}_lmcache_lines=$(gcount 'LMCacheConnector')"
    echo "${2}_tracebacks=$(gcount 'Traceback (most recent call last)')"
    echo "${2}_errors=$(gcount ERROR)"
  } >> "$1/markers.txt"
}

engine_config() {  # $1 = leg dir
  {
    grep -hE "^ARC_(SERVE_LAUNCHED|SERVE_KNOBS|CONTAINER_FRESH)" "$1/boot.txt" 2>/dev/null
    docker exec $CNAME grep -n -E "non-default args|Casting torch|Default vLLM sampling|Resolved architecture|Using MX KV attention backend|MXKV native call|ARC-P628-FA2|ARC-P629-SLIDE|Quantized [0-9]+ layers|GPU KV cache size|Maximum concurrency|Model loading took|model weights took|Graph capturing finished|Using Triton backend" \
      /root/serve_full.log 2>/dev/null | cut -c1-700
  } > "$1/engine_config.txt" || true
  cut -c1-400 "$1/engine_config.txt" | head -24
}

preflight() {
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    refuse "R266_REFUSE_ARC_LIVE" 2
  fi
  (cd $R && md5sum -c --quiet md5.txt) || refuse "R266_SHIP_MD5_FAIL"
  local f
  for f in "$T_G3:$R/chat_template_gemma4.jinja" "$T_G4:$R/chat_template_gemma4_official.jinja"; do
    cmp -s "${f%%:*}" "${f##*:}" || refuse "R266_TEMPLATE_MISMATCH ${f%%:*} != ${f##*:}"
  done
  [ -f $R0/r113_battery.py ] && [ -f $R0/r262_w14_restore.sh ] || refuse "R266_NO_HELPER"
  local i
  for i in $IMG29 $IMG20 $IMG19; do
    docker image inspect "$i" >/dev/null 2>&1 || refuse "R266_NO_IMAGE $i"
  done
  echo "R266_PREFLIGHT_OK $(date -u +%Y-%m-%dT%H:%M:%SZ) bank=$P legs=[$LEGS]"
  md5sum $R/serve_arc_b70.sh $R/p266_matrix_probe.py "$T_G3" "$T_G4" $R0/r113_battery.py
}

leg() {  # $1=name $2=image $3=served-format $4=chat-cells (''=all) $5=g1(0|1) $6..=extra env
  local NAME=$1 IMG=$2 FMT=$3 CCELLS=$4 DO_G1=$5; shift 5
  local CD=$P/$NAME K0 PRC=0
  mkdir -p "$CD"
  echo "=== [$NAME] boot $(date +%H:%M:%S) image=$IMG served_fmt=$FMT chat_cells=${CCELLS:-all} g1=$DO_G1 env=[$NAT $*] ==="
  echo "leg=$NAME image=$IMG served_fmt=$FMT chat_cells=${CCELLS:-all} g1=$DO_G1 env=[$NAT $*]" > "$CD/leg_env.txt"
  K0=$(kmark)
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $NAT "$@" bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$NAME" "$SERVED"; then
    echo "R266_LEG_FAIL_BOOT $NAME"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    engine_config "$CD"
    markers "$CD" post
    docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null | cut -c1-300
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    LEG_FAIL=$((LEG_FAIL + 1)); khealth "$CD" "$K0" "$NAME"
    echo "R266_LEG $NAME RESULT=BOOT_FAIL"
    return 1
  fi
  echo "HEALTH_OK_$NAME $(date +%H:%M:%S) (identity-verified)"
  sleep 5
  markers "$CD" pre
  cat "$CD/markers.txt"
  curl -s -m 10 http://127.0.0.1:8000/v1/models | cut -c1-200 > "$CD/models.txt"
  # distinct in-container names: the served template is bind-mounted at
  # /root/$(basename) by the serve script -- a docker cp into that exact path
  # is a cp into a mount (the v2 TEMPLATE_CP_FAIL). The probe only READS these
  # copies (local rendering for the prompt oracle); bytes are preflight-cmp'd.
  # GEMMA TEMPLATES ONLY (owner directive 2026-11-28): nothing qwen ships.
  cp_in "$T_G3" /root/tpl_g3.jinja
  cp_in "$T_G4" /root/tpl_g4.jinja
  cp_in $R/p266_matrix_probe.py /root/p266_matrix_probe.py
  local PARGS=(--model $SERVED --tokenizer /models/target --out /root/p266_out --tag "$NAME" --served "$FMT"
               --g3 /root/tpl_g3.jinja
               --g4 /root/tpl_g4.jinja --workers 8 --cells chat,raw,nll,gate)
  if [ -n "$CCELLS" ]; then PARGS+=(--chat-cells "$CCELLS"); fi
  echo "probe start $(date +%H:%M:%S) args=${PARGS[*]}"
  timeout "$PROBE_TIMEOUT" docker exec $CNAME /opt/venv/bin/python /root/p266_matrix_probe.py "${PARGS[@]}" \
    > "$CD/matrix.out" 2>&1
  PRC=$?
  docker cp $CNAME:/root/p266_out/matrix_$NAME.json "$CD/" >/dev/null 2>&1 || echo "MATRIX_JSON_MISSING $NAME"
  echo "probe done $(date +%H:%M:%S) rc=$PRC"
  grep -E "^(PROMPT_ORACLE|FORMAT|chat |raw |    ptok_ok|gate264|    \[gate|NLL |P266_MATRIX_DONE)" "$CD/matrix.out" | cut -c1-330
  if [ "$PRC" != 0 ]; then tail -15 "$CD/matrix.out" | cut -c1-300; fi
  if [ "$DO_G1" = 1 ]; then
    python3 $R0/r113_battery.py g1 $SERVED > "$CD/g1_probes.txt" 2>&1 || echo "G1_NONZERO_EXIT $NAME"
    cut -c1-300 "$CD/g1_probes.txt"
  fi
  engine_config "$CD"
  markers "$CD" post
  tail -8 "$CD/markers.txt"
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || echo "ERROR=?" > "$CD/erraudit.txt"
  cat "$CD/erraudit.txt"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || echo "serve.log MISSING $NAME"
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1 || true
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1 || true
  verify_removed || LEG_FAIL=$((LEG_FAIL + 1))
  sleep 22
  khealth "$CD" "$K0" "$NAME"
  if [ "$PRC" != 0 ]; then LEG_FAIL=$((LEG_FAIL + 1)); fi
  echo "R266_LEG $NAME RESULT=$([ "$PRC" = 0 ] && echo OK || echo PROBE_RC_$PRC) $(date +%H:%M:%S)"
}

run_leg() {  # $1.. = the leg() argument list
  leg "$@" || true
}

echo "R266_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) legs=[$LEGS] dry=$DRY"
preflight
if [ "$DRY" = 1 ]; then
  echo "R266_DRY_RUN (no boot, W14 untouched)"
  trap - EXIT
  exit 0
fi
w14_stop
for sel in ${LEGS//,/ }; do
  case $sel in
    b2) run_leg b2_g3_29 "$IMG29" g3 "" 1 $SKIN_G3 ;;
    c3) run_leg c3_m8g_29 "$IMG29" g3 t0,card 1 $SKIN_G3 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 ;;
    c4) run_leg c4_m4g_29 "$IMG29" g3 t0,card 1 $SKIN_G3 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1 ;;
    c5) run_leg c5_mkg_29 "$IMG29" g3 t0,card 1 $SKIN_G3 ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1 ;;
    w1) run_leg w1_mxfp4kvfp8_29 "$IMG29" g3 t0,card 1 $SKIN_G3 ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ;;
    w2) run_leg w2_mxfp8kvfp8_29 "$IMG29" g3 t0,card 1 $SKIN_G3 ARC_QUANT=mxfp8 ;;
    b3) run_leg b3_g4_29 "$IMG29" g4 "" 1 $SKIN_G4 ;;
    b5) run_leg b5_g3_19 "$IMG19" g3 t0,card 1 $SKIN_G3 ;;
    b4) run_leg b4_g3_20 "$IMG20" g3 t0,card 1 $SKIN_G3 ;;
    c1) run_leg c1_bf16_29 "$IMG29" g3 t0,card 0 $SKIN_G3 ARC_DTYPE=bfloat16 ;;
    c2) run_leg c2_fa2off_29 "$IMG29" g3 t0,card 0 $SKIN_G3 ARC_ATTN_FA2=0 ;;
    *) echo "R266_UNKNOWN_LEG $sel"; LEG_FAIL=$((LEG_FAIL + 1)) ;;
  esac
done
echo "R266_END $(date -u +%Y-%m-%dT%H:%M:%SZ) leg_failures=$LEG_FAIL bank=$P"
exit $LEG_FAIL
