#!/bin/bash
# r154_cert24_gate.sh -- cert24 (P6.47 MXFP4/MXFP8 KV lane) §10 gate.
# Image vllm-arc:v0.30.0-b70-cert24 = cert23 + patches/mx_kv/ (7 files) +
# patches/{mx_dtype_map,mx_kv_mode}.py + platform.py routing. Legs:
#   L0 ctrl   qwen fp8_e4m3 spec3 (NO ARC_KVDTYPE) -- default-off
#      neutrality: boots the cert23 posture byte-identically, battery +
#      markers only (no long suite; the posture is cert23-certified).
#   L1 m8q    qwen ARC_KVDTYPE=mxfp8 ARC_SPEC=3 (champion chain, LMC=1)
#   L2 m4q    qwen ARC_KVDTYPE=mxfp4 ARC_SPEC=3
#   L3 m8g    gemma4-31b ARC_KVDTYPE=mxfp8 ARC_SPEC=0 (cert22 gemma line)
#   L4 m4g    gemma4-31b ARC_KVDTYPE=mxfp4 ARC_SPEC=0
# L1-L4 each run the FULL corruption program (owner directive 2026-11-11:
# end-to-end garbage/long-return validation on EVERY MX dtype, both
# models): 3-prompt temp-0 text battery (p1 MUST contain Paris), suite-13
# arc_garbage_probe (malformed inputs + state-integrity), suite-14
# longprobe full + 3 churn rehits (garble flags + cross-cycle hash
# determinism), erraudit ERROR=0/TRACEBACK=0, metrics, serve log bank.
# trap restores W14. Artifacts -> /root/build/p647/cert24/<leg>/.
set -uo pipefail
R=/root/build
P=$R/p647/cert24
IMG=vllm-arc:v0.30.0-b70-cert24
CNAME=arc-c24
GATE_FAIL=0
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647c24.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "GATE_FAIL_STALE_CONTAINER $CNAME"; return 1
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

garble_check() {  # $1 = longprobe out file — r98 law: grep -F (r84-era BRE
  # escapes never matched; do not reintroduce)
  local lines empty
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" || true)
  empty=$(( $(grep -cF "garble={'cold': [], 'rehit': [], 'rehit2': []}" "$1" || true)
          + $(grep -cF "garble={'r1': [], 'r2': []}" "$1" || true) ))
  echo "GARBLE: class_lines=$lines empty_garble=$empty (expect equal, >=4) in $(basename "$1")"
  { [ "$lines" -ge 4 ] && [ "$lines" = "$empty" ]; } || { echo "ZERO-GARBLE: FAIL ($1)"; GATE_FAIL=1; }
}

longreturn_summary() {  # $1 = tag $2 = leg dir — r99 law: probe writes its
  # jsons to CWD; summary MUST run from the leg dir
  ( cd "$2" && python3 - "$1" > longreturn_summary.txt 2>&1 << 'PYEOF' || true
import json, sys
tag = sys.argv[1]
def load(fn):
    with open(fn) as f:
        return json.load(f)
full = load(f"longprobe_{tag}_full.json")
rehits = [load(f"longprobe_{tag}r{c}_rehit.json") for c in (1, 2, 3)]
fail = []
t1_miss = []
for cls in ["essay", "code", "multi", "simple"]:
    v = full.get(cls, {})
    hashes = set(v.get("hashes", []))
    garble_all = list(v.get("garble", {}).values()) + [v.get("t1_garble", [])]
    for r in rehits:
        rv = r.get(cls, {})
        hashes |= set(rv.get("hashes", []))
        garble_all += list(rv.get("garble", {}).values())
    flat = [g for g in garble_all if g]
    if len(hashes) != 1:
        fail.append(f"{cls}: cross-cycle hash drift {sorted(hashes)}")
    if flat:
        fail.append(f"{cls}: garble flags {flat}")
    if any(fin != "stop" for fin in v.get("finish", [])):
        fail.append(f"{cls}: truncation finish={v.get('finish')}")
    if not v.get("t1_seed_identical"):
        t1_miss.append(cls)
    print(f"[{cls}] hashes={sorted(hashes)} finish={v.get('finish')} "
          f"chars={v.get('chars')} t1_seed_identical={v.get('t1_seed_identical')} "
          f"garble_empty={not flat}")
print("LONGRETURN:", "PASS" if not fail else "FAIL " + "; ".join(fail))
print("T1_SEED:", "PASS" if not t1_miss else "MISS " + ",".join(t1_miss))
PYEOF
  )
  grep -q "LONGRETURN: PASS" "$2/longreturn_summary.txt" || GATE_FAIL=1
  grep -q "T1_SEED: PASS" "$2/longreturn_summary.txt" || GATE_FAIL=1
  cat "$2/longreturn_summary.txt"
}

churn() {  # $1 = seed base $2 = served model — r105 churn with the model
  # parameterized (gemma legs need their own name)
  local base=$1 mdl=$2
  python3 > "$P/churn_$base.out" 2>&1 << PYEOF || true
import json, threading, time, urllib.request
def one(i, out):
    body = {"model": "$mdl", "stream": False, "temperature": 0.0,
            "max_tokens": 256, "seed": $base + i,
            "messages": [{"role": "user", "content":
                f"Essay {i}: explain cache coherence in distributed systems, 200 words."}]}
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=600))
    out.append((time.perf_counter() - t0, r["usage"]["completion_tokens"]))
res = []
ts = [threading.Thread(target=one, args=(i, res)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("CHURN 4x256 done:", [f"{d:.1f}s/{n}tok" for d, n in res])
PYEOF
  tail -1 "$P/churn_$base.out"
}

text_battery() {  # $1 = leg dir $2 = served model $3 = endpoint
  # (completions|chat) — the r149 3-prompt temp-0 battery (trace25-28
  # validated set); p1 MUST contain Paris. Gemma is chat-only: the raw
  # /v1/completions endpoint has document-continuation semantics there
  # (m8g run: prompt-echo + further questions, dtype-INDEPENDENT — an
  # endpoint artifact, not KV corruption), so gemma legs use the chat
  # endpoint with the message content identical.
  local CD=$1 mdl=$2 EP=${3:-completions}
  : > "$CD/probe.out"
  for q in "What is the capital of France? Answer with just the city name." \
           "Write a Python function that returns the n-th Fibonacci number using iteration." \
           "Name three chemical elements that are gases at room temperature."; do
    if [ "$EP" = chat ]; then
      curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mdl\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]; t=c.get("message",{}).get("content") or c.get("message",{}).get("reasoning_content") or ""; print("PROMPT_OK", len(t), repr(t[:160]))' >> "$CD/probe.out" 2>&1
    else
      curl -s -m 180 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mdl\",\"prompt\":\"$q\",\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c 'import json,sys; r=json.load(sys.stdin); print("PROMPT_OK", len(r["choices"][0]["text"]), repr(r["choices"][0]["text"][:160]))' >> "$CD/probe.out" 2>&1
    fi
  done
  cat "$CD/probe.out"
  grep -q "PROMPT_OK" "$CD/probe.out" || { echo "TEXT_BATTERY: FAIL (no PROMPT_OK) $CD"; GATE_FAIL=1; return; }
  [ "$(grep -c PROMPT_OK "$CD/probe.out" || true)" = "3" ] || { echo "TEXT_BATTERY: FAIL (missing answers) $CD"; GATE_FAIL=1; }
  grep -q "Paris" "$CD/probe.out" || { echo "P1_PARIS: FAIL $CD"; GATE_FAIL=1; }
  # shape errors surface as python tracebacks inside probe.out (2>&1)
  grep -q "Traceback" "$CD/probe.out" && { echo "TEXT_BATTERY: FAIL (traceback) $CD"; GATE_FAIL=1; }
  sha256sum "$CD/probe.out" > "$CD/probe.sha256"
}

erraudit() {  # $1 = leg dir
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$1/erraudit.txt" 2>/dev/null || echo "ERROR=?" > "$1/erraudit.txt"
  cat "$1/erraudit.txt"
  grep -q "ERROR=0" "$1/erraudit.txt" || { echo "ERRAUDIT: FAIL ($1)"; GATE_FAIL=1; }
}

run_leg() {
  # $1=legname $2=served $3=fullsuite(1|0) $4..=env assignments
  local LEGN=$1 SERVED=$2 FULL=$3; shift 3
  local CD=$P/$LEGN
  # gemma is chat-only at the probe layer (completions = document-
  # continuation semantics, dtype-independent — m8g run1 evidence)
  case $SERVED in gemma*) local EP=chat ;; *) local EP=completions ;; esac
  mkdir -p "$CD"
  echo "=== [$LEGN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$LEGN"; then
    echo "GATE_FAIL_BOOT_$LEGN"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -40 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    GATE_FAIL=1; return 1
  fi
  echo "HEALTH_OK_$LEGN $(date +%H:%M:%S)"
  sleep 5
  # MX lane markers + posture evidence
  {
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "lmcache_lines=$(gcount 'LMCacheConnector')"
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
    docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" /root/serve_full.log 2>/dev/null | head -4
  } > "$CD/markers.txt"
  cat "$CD/markers.txt"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  # text battery (all legs — the corruption gate)
  text_battery "$CD" "$SERVED" "$EP"
  # decode battery (all legs)
  python3 $R/r113_battery.py decode "$SERVED" > "$CD/battery.txt" 2>&1 || echo "BATTERY_NONZERO_EXIT $LEGN"
  tail -6 "$CD/battery.txt"
  if [ "$FULL" = 1 ]; then
    # suite-13: malformed-input hardening (host python3, urllib only)
    ARC_PROBE_MODEL="$SERVED" python3 $R/p647/arc_garbage_probe.py > "$CD/garbage.out" 2>&1
    tail -5 "$CD/garbage.out"
    grep -q "GARBAGE_SUITE_PASS" "$CD/garbage.out" || { echo "SUITE13: FAIL $LEGN"; GATE_FAIL=1; }
    # suite-14: long-return full + 3 churn rehits + hash determinism
    ( cd "$CD" && ARC_PROBE_MODEL="$SERVED" python3 $R/p647/p647_longprobe.py c24$LEGN full > longprobe_full.out 2>&1 )
    grep -E "^\[|LONGPROBE" "$CD/longprobe_full.out"
    garble_check "$CD/longprobe_full.out"
    for cyc in 1 2 3; do
      churn $(( 900 * cyc )) "$SERVED"
      ( cd "$CD" && ARC_PROBE_MODEL="$SERVED" python3 $R/p647/p647_longprobe.py c24${LEGN}r$cyc rehit > longprobe_rehit$cyc.out 2>&1 )
      grep -E "^\[|LONGPROBE" "$CD/longprobe_rehit$cyc.out"
      garble_check "$CD/longprobe_rehit$cyc.out"
    done
    longreturn_summary c24$LEGN "$CD"
  fi
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || echo "serve.log MISSING $LEGN"
  docker exec $CNAME grep -c "xcheck3" /root/serve_full.log > "$CD/xcheck3_count.txt" 2>/dev/null || true
  erraudit "$CD"
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1
  verify_removed || { GATE_FAIL=1; return 1; }
  sleep 3
  echo "=== [$LEGN] done $(date +%H:%M:%S) ==="
}

# derived longprobe with served-model env override (p618 hardcodes the
# qwen name; gemma legs need their own — one-line edit, provenance kept)
mkdir -p $R/p647
python3 - << 'PYEOF'
import re
src = open("/root/build/p618/p618_longprobe.py", encoding="utf-8").read()
if "ARC_PROBE_MODEL" not in src:
    assert src.count('MODEL = "qwen3.8-27b-fp8"') == 1, "p618 model line drifted"
    src = src.replace('MODEL = "qwen3.8-27b-fp8"',
                      'MODEL = os.environ.get("ARC_PROBE_MODEL", "qwen3.8-27b-fp8")')
    src = src.replace("import hashlib\n", "import hashlib\nimport os\n", 1)
open("/root/build/p647/p647_longprobe.py", "w", encoding="utf-8", newline="\n").write(src)
print("p647_longprobe.py derived OK")
PYEOF
[ -f $R/p647/arc_garbage_probe.py ] || { echo "MISSING $R/p647/arc_garbage_probe.py (ship it first)"; exit 1; }

docker stop radiance-test >/dev/null 2>&1
sleep 5

QWEN_FP8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1"
GEMMA_M8="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=mxfp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

# Leg registry: run1 = all five (2026-11-12 07:41 UTC); run2 = the gemma
# pair re-run after the two gate-probe fixes (chat-endpoint battery +
# ARC_PROBE_MODEL on longprobe — run1's gemma longprobe cells 404'd on the
# default qwen model name and the battery hit gemma's completions-continuation
# semantics; both artifacts, engine exonerated by m8g==m4g text equality).
# Prior leg dirs are preserved as <leg>.run1 before a re-run.
leg_ctrl() {
  run_leg ctrl qwen3.8-27b-fp8 0 $QWEN_FP8 || true
  MX_LINES=$(grep -c "mx_backend_lines=0" "$P/ctrl/markers.txt" || true)
  [ "$MX_LINES" = "1" ] || { echo "DEFAULT-OFF: FAIL (MX backend engaged without ARC_KVDTYPE)"; GATE_FAIL=1; }
}
leg_m8q() {
  run_leg m8q qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxfp8 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m8q/markers.txt" || { echo "MX_MARKER: FAIL m8q (expect >=1 worker)"; GATE_FAIL=1; }
}
leg_m4q() {
  run_leg m4q qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxfp4 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m4q/markers.txt" || { echo "MX_MARKER: FAIL m4q (expect >=1 worker)"; GATE_FAIL=1; }
}
leg_m8g() {
  [ -d "$P/m8g" ] && mv "$P/m8g" "$P/m8g.run1"
  run_leg m8g gemma4-31b 1 $GEMMA_M8 ARC_KVDTYPE=mxfp8 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m8g/markers.txt" || { echo "MX_MARKER: FAIL m8g (expect >=1 worker)"; GATE_FAIL=1; }
}
leg_m4g() {
  [ -d "$P/m4g" ] && mv "$P/m4g" "$P/m4g.run1"
  run_leg m4g gemma4-31b 1 $GEMMA_M8 ARC_KVDTYPE=mxfp4 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m4g/markers.txt" || { echo "MX_MARKER: FAIL m4g (expect >=1 worker)"; GATE_FAIL=1; }
}

if [ $# -gt 0 ]; then
  for LEG in "$@"; do
    case $LEG in
      ctrl|m8q|m4q|m8g|m4g) leg_$LEG ;;
      *) echo "UNKNOWN_LEG $LEG (ctrl|m8q|m4q|m8g|m4g)"; GATE_FAIL=1 ;;
    esac
  done
else
  leg_ctrl; leg_m8q; leg_m4q; leg_m8g; leg_m4g
fi

cd "$P"
find . -type f \( -name "*.out" -o -name "*.txt" -o -name "*.json" -o -name "*.sha256" -o -name "serve.log" -o -name "metrics.txt" \) -exec sha256sum {} \; > SHA256SUMS
echo "LEG_ARTIFACTS $P"
if [ "$GATE_FAIL" = 0 ]; then
  echo "P647_CERT24_GATE_COMPLETE -- PASS (analysis/bank next)"
else
  echo "P647_CERT24_GATE_COMPLETE -- GATE_FAIL=$GATE_FAIL (fix before cert)"
fi
exit $GATE_FAIL
