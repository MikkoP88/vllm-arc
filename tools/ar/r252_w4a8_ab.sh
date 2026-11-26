#!/bin/bash
# r252_w4a8_ab.sh -- P6.47 P-I cycle 3 (r252): the mxfp4-W4A8 weights
# lane vs the certified W4A16 incumbent — same-boot A/B on cert28, the
# r240 q43 record lane (online mxfp4 of the BASE qwen3.8-27b checkpoint:
# ARC_QUANTCFG ignore-list, ARC_GMU=0.85, ARC_LMC=1, ARC_FP8=0, KV
# default fp8_e4m3 — the weights lane isolated, spec3, graphs).
# Attempt 1 (banked p252a1) refused at boot: pointing at the fp8 VL
# (--quantization mxfp4 vs config fp8 pydantic refusal) — the record
# lane is the base checkpoint + ARC_QUANTCFG, per r240_reports_cert28.
# The corrected lever map's open candidate #1 (MX-RESEARCH §5.1): the
# fp4_gemm_w4a8 op ships (arc5+ wheel; r247 surface census), per-token
# e4m3 activation quant avoids the p630 per-call tax class, and r122's
# "dominated by w4a16" prediction has never faced an e2e measurement.
# Faces per boot: coherence (fixed mt64 temp-0 completion) + the
# metrics-bracketed p139 groups {c8192 n1 mt192, c8192 n8 mt128,
# c32768 n1 mt192} — decode TPS + ITL at solo / concurrency /
# long-ctx, the lever's named faces. Acceptance canary = the
# spec_decode counter deltas per cell (weights changes move the DRAFT
# too — the acceptance battery is mandatory on weight lanes).
# Gates: the reroute-ACTIVE line per mode (a SKIPPED/absent w4a8 op
# fails the cell rather than silently measuring stock W4A4);
# n_ok full per face; error/traceback audit per boot.
# Cells: w4a16 control first, w4a8 probe second (baseline banked even
# if the probe dies). Artifacts -> /root/build/p252/. W14 quiesced for
# the leg, trap-restored at exit (arc-in-flight guard). Exit 0 = both
# cells banked with every face n_ok full (W4A8_AB_PASS).
set -uo pipefail
R=/root/build
P=$R/p252
IMG=vllm-arc:v0.30.0-b70-cert28
CNAME=arc-r252
PROBE=$R/p139_conc.py   # canonical rig copy (API length-budget fix, r140c law)
FAILED=""

[ -f "$PROBE" ] || { echo "R252_NO_PROBE $PROBE"; exit 1; }
[ -f "$R/serve_arc_b70.sh" ] || { echo "R252_NO_SERVE_SCRIPT"; exit 1; }
md5sum "$R/serve_arc_b70.sh" | grep -q 93ac8477545352886f43fce92e84ac16 \
  || { echo "R252_REFUSE_SERVE_MD5 (rig serve script drifted from the repo-parity bytes)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R252_NO_IMAGE $IMG"; exit 1; }
docker ps --format '{{.Names}}' | grep -E '^arc-' && { echo "R252_REFUSE_ARC_LEG_IN_FLIGHT"; exit 1; }
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r252.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "STALE_CONTAINER $CNAME"; return 1
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

kv_pool() {
  local v
  v=$(docker exec $CNAME grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
  v=${v#KV cache size: }; v=${v% tokens}; v=${v//,/}
  echo "${v:-0}"
}

group() {  # ctx conc -- metrics-bracketed streaming cell -> $CD/c{ctx}_n{nc}/
  local CTX=$1 NC=$2
  local TAG="c${1}_n${2}"
  local GD="$CD/$TAG"
  local MT; [ "$NC" = 1 ] && MT=192 || MT=128
  local need=$(( NC * (CTX + MT + 64) ))
  mkdir -p "$GD"
  if [ "$POOL" -gt 0 ] && [ "$need" -gt "$POOL" ]; then
    echo "KV_CAP $MODE $TAG POOL=$POOL NEED=$need" | tee -a "$P/conc.log"
    echo "CAP" > "$GD/outcome.txt"; FAILED="$FAILED ${MODE}_${TAG}"; return 0
  fi
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_before.txt" || true
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "qwen3.8-27b" 0.0
  local cap=$(( 300 + CTX / 8192 * 90 + 120 + (NC == 8 ? 240 : 0) ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec $CNAME test -f /root/p139_conc_$TAG.json && break
    docker inspect $CNAME >/dev/null 2>&1 || break
    sleep 4; i=$((i+1))
  done
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_after.txt" || true
  if ! docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s)" | tee -a "$P/conc.log"
    echo "TIMEOUT" > "$GD/outcome.txt"; FAILED="$FAILED ${MODE}_${TAG}"; return 0
  fi
  docker cp $CNAME:/root/p139_conc_$TAG.json "$GD/" >/dev/null 2>&1
  local NOK
  NOK=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$GD/p139_conc_$TAG.json" 2>/dev/null || echo 0)
  echo "CONC_${TAG} n_ok=$NOK" | tee -a "$P/conc.log"
  if [ "${NOK:-0}" != "$NC" ]; then
    echo "NOK_SHORT" > "$GD/outcome.txt"; FAILED="$FAILED ${MODE}_${TAG}"
  fi
}

run_cell() {
  # $1 = mode (w4a16 control | w4a8 probe)
  MODE=$1; CD=$P/$MODE
  mkdir -p "$CD"
  echo "=== [r252:$MODE] boot $(date +%H:%M:%S) (ARC_QUANT=mxfp4 mode=$MODE) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b \
    ARC_QUANTCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}' \
    ARC_GMU=0.85 ARC_LMC=1 ARC_FP8=0 \
    ARC_GRAPH=1 ARC_SPEC=3 \
    ARC_QUANT=mxfp4 ARC_MXFP4_MODE=$MODE \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$MODE"; then
    echo "R252_BOOT_FAIL_$MODE -- banking log, continuing"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    FAILED="$FAILED ${MODE}_boot"; echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r252_$MODE $(date +%H:%M:%S)"

  # reroute gate: the ACTIVE line for THIS mode must be present; a
  # SKIPPED warn (op absent) fails the cell -- never measure stock W4A4
  # silently.
  docker exec $CNAME grep -E "mxfp4 .*(${MODE} reroute|${MODE} probe) (ACTIVE|SKIPPED)" \
    /root/serve_full.log 2>/dev/null | head -2 > "$CD/reroute.txt" || true
  cat "$CD/reroute.txt"
  if ! grep -q "${MODE}.*ACTIVE" "$CD/reroute.txt" 2>/dev/null; then
    echo "R252_REROUTE_GATE_FAIL_$MODE (no ACTIVE line -- see reroute.txt)"
    echo "REROUTE_FAIL" > "$CD/outcome.txt"; FAILED="$FAILED ${MODE}_reroute"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    return 0
  fi

  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R252_PROBE_CP_FAIL_$MODE"; FAILED="$FAILED ${MODE}_probe"; \
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true; return 0; }
  docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" \
    /root/serve_full.log 2>/dev/null | head -3 > "$CD/markers.txt" || true
  POOL=$(kv_pool); echo "POOL=$POOL" | tee -a "$CD/markers.txt"

  # warmup + the coherence face (fixed mt64 temp-0 completion, banked:
  # W4A8 changes numerics so texts may differ from the control's --
  # the gate is coherence class, assessed in the VERDICT)
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8-27b\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  curl -s -m 120 -o "$CD/coherence.json" http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","prompt":"Write a concise summary of the attention mechanism in modern language models.","max_tokens":64,"temperature":0}' \
    && python3 -c "import json;print(json.load(open('$CD/coherence.json'))['choices'][0]['text'])" > "$CD/coherence.text" 2>/dev/null || true

  group  8192 1
  group  8192 8
  group 32768 1
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "=== [r252:$MODE] done $(date +%H:%M:%S) ==="
}

echo "=== [r252:w4a8ab] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R252_WARN_PORT8000_STILL_HELD"
sleep 5

run_cell w4a16
run_cell w4a8

cd "$P"
find . -type f \( -name "*.txt" -o -name "*.json" -o -name "*.log" -o -name "*.text" \) \
  -exec sha256sum {} \; > SHA256SUMS 2>/dev/null || true
if [ -z "$FAILED" ]; then
  echo "W4A8_AB_PASS $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi
echo "W4A8_AB_FAILED:$FAILED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 1
