#!/bin/bash
# r171_fastcycle.sh -- P6.47 improvement-loop FAST cycle (owner directive
# 2026-11-13: the full ~4.5h gate starves the improvement loop; iterate
# cheap, run the full battery only at convergence).
#
# One cycle = single model gemma4-31b (ARC_SPEC=0 -- no MTP, isolates the
# plain decode path) x the three MX KV dtypes (mxfp8 / mxfp4 / mxk8v4),
# SIMPLE corruption check only. Budget <= ~30 min per cycle: 3 boots
# (~5-7 min each) + battery + decode battery + a light cold/churn/rehit
# smoke probe + erraudit per leg. NO longprobe 4x3-face program, NO
# 3-cycle churn, NO ctrl leg, NO gemma_compare -- those live in the
# r170-pattern full battery, which runs when a change line converges.
# Gemma quality gates are SHAPE-only (reploop-classifier-baseline law:
# think-spam REP-LOOP is native-KV baseline; determinism is baseline-
# churny on 4-bit V -- r170 m4g evidence), so the smoke probe gates:
# non-empty answers, no traceback, not all-finish=length, server alive
# after churn. Promote to the full battery before any cert/release.
# W14 quiesce before, trap restore after. Artifacts ->
# ${P647_BANK:-/root/build/p647/fastcycle}/<leg>/.
set -uo pipefail
R=/root/build
P=${P647_BANK:-$R/p647/fastcycle}
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25z}
CNAME=arc-fc
CYCLE_FAIL=0
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647fc.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "CYCLE_FAIL_STALE_CONTAINER $CNAME"; return 1
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

smoke_probe() {  # $1 = leg dir $2 = served model — SIMPLE corruption check:
  # 6 fixed chat prompts, temp-0, max_tokens=256 (keeps gemma think-spam
  # OFF the length cap -- the full-face REP-LOOP program is the r170
  # battery's job). Round A cold; 60s churn; round B rehit. Gates:
  # 6/6 PROMPT_OK non-empty in BOTH rounds, no traceback, not every
  # answer finish=length (an all-cap face = decode-path breakage, not
  # baseline spam), server alive after churn. Hashes are BANKED, never
  # gated (baseline churn law).
  local CD=$1 mdl=$2 r b lens
  for r in A B; do
    : > "$CD/smoke_$r.out"
    for q in "What is the capital of France? Answer with just the city name." \
             "Write a Python function that returns the n-th Fibonacci number using iteration." \
             "Name three chemical elements that are gases at room temperature." \
             "Summarize the plot of Hamlet in two sentences." \
             "Convert 42 from decimal to hexadecimal." \
             "List the first four planets of the solar system in order."; do
      curl -s -m 120 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mdl\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":256,\"temperature\":0,\"seed\":171}" \
        | python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]; m=c.get("message",{}); t=m.get("content") or m.get("reasoning_content") or m.get("reasoning") or ""; print("PROMPT_OK", len(t), c.get("finish_reason"), repr(t[:120]))' >> "$CD/smoke_$r.out" 2>&1
    done
    [ "$r" = A ] || break
    python3 - > "$CD/churn.out" 2>&1 << 'PYEOF' || true
import json, time, urllib.request
t_end = time.time() + 60
i = 0
while time.time() < t_end:
    i += 1
    d = json.dumps({"model": "gemma4-31b", "prompt": f"churn {i} " + "x" * 400,
                    "max_tokens": 256, "temperature": 0}).encode()
    urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions", d,
        {"Content-Type": "application/json"}), timeout=600).read()
print("churn rounds=", i)
PYEOF
  done
  for r in A B; do
    b=$(grep -c "^PROMPT_OK" "$CD/smoke_$r.out" || true)
    [ "$b" = "6" ] || { echo "SMOKE: FAIL (round $r: ${b:-0}/6 answered)"; CYCLE_FAIL=1; }
    grep -q "Traceback" "$CD/smoke_$r.out" && { echo "SMOKE: FAIL (traceback round $r)"; CYCLE_FAIL=1; }
    lens=$(grep -cE "^PROMPT_OK 0 " "$CD/smoke_$r.out" || true)
    [ "${lens:-0}" = "0" ] || { echo "SMOKE: FAIL (round $r: empty answers)"; CYCLE_FAIL=1; }
    local cap; cap=$(grep -c "length" "$CD/smoke_$r.out" || true)
    [ "${cap:-0}" -lt 6 ] || { echo "SMOKE: FAIL (round $r: every answer hit the cap)"; CYCLE_FAIL=1; }
  done
  echo "SMOKE: $([ "$CYCLE_FAIL" = 0 ] && echo PASS || echo FAIL) in $CD"
}

run_leg() {
  # $1=legname $2..=env assignments
  local LEGN=$1; shift
  local CD=$P/$LEGN
  mkdir -p "$CD"
  echo "=== [$LEGN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$LEGN"; then
    echo "CYCLE_FAIL_BOOT_$LEGN"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -30 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    CYCLE_FAIL=1; return 1
  fi
  echo "HEALTH_OK_$LEGN $(date +%H:%M:%S)"
  {
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "lmcache_lines=$(gcount 'LMCacheConnector')"
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
    docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" /root/serve_full.log 2>/dev/null | head -2
  } > "$CD/markers.txt"
  cat "$CD/markers.txt"
  grep -Eq "mx_backend_lines=[1-9]" "$CD/markers.txt" \
    || { echo "MX_MARKER: FAIL $LEGN"; CYCLE_FAIL=1; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma4-31b","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  smoke_probe "$CD" gemma4-31b
  python3 $R/r113_battery.py decode gemma4-31b > "$CD/battery.txt" 2>&1 || echo "BATTERY_NONZERO_EXIT $LEGN"
  tail -6 "$CD/battery.txt"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || echo "ERROR=?" > "$CD/erraudit.txt"
  cat "$CD/erraudit.txt"
  grep -q "ERROR=0" "$CD/erraudit.txt" || { echo "ERRAUDIT: FAIL ($LEGN)"; CYCLE_FAIL=1; }
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1
  verify_removed || { CYCLE_FAIL=1; return 1; }
  sleep 3
  echo "=== [$LEGN] done $(date +%H:%M:%S) ==="
}

GEMMA="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=mxfp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

echo "=== [r171:fastcycle] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
docker rm -f $CNAME >/dev/null 2>&1 || true
sleep 5

T0=$(date +%s)
run_leg m8g $GEMMA ARC_KVDTYPE=mxfp8   || true
run_leg m4g $GEMMA ARC_KVDTYPE=mxfp4   || true
run_leg mkg $GEMMA ARC_KVDTYPE=mxk8v4  || true
T1=$(date +%s)

cd "$P"
find . -type f \( -name "*.out" -o -name "*.txt" -o -name "serve.log" \) -exec sha256sum {} \; > SHA256SUMS 2>/dev/null || true
echo "P647_FASTCYCLE_DONE fail=$CYCLE_FAIL wall=$(( (T1 - T0) / 60 ))min (budget 30) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ "$CYCLE_FAIL" = 0 ]
