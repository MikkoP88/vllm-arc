#!/bin/bash
# r161_cert25y_gate.sh (r159 + cert25y fixes) -- cert25y (P6.47 MXFP4/MXFP8
# KV lane, v3 decode + s_eff floor + native-q + batched verify) §10 gate.
# Image vllm-arc:v0.30.0-b70-cert25y = cert25x content + the A9 bake +
# _split_verify_decode (one mx_decode per MTP verify step).
# vs r159:
#   - churn rehit outs bank per-leg ($CD/churn_$base.out) — r159 wrote
#     them to the bank root, so each leg clobbered the previous leg's
#     churn evidence (gate-clobber law).
#   - g0 runs the v4 longprobe as UNGATED evidence — the native-KV
#     baseline discriminator for gemma thinking-stream REP-LOOP (r159's
#     g0 was battery-only; the m8g CORRUPTION_SIGNAL disposition needs
#     the native lane's long-return shape; thinking-mode reasoning
#     degeneration is documented gemma baseline behavior at temp-0).
#   - gate start asserts the cert25y image content markers (s_eff floor,
#     engine cast-free decode path) — a stale cert25x image would
#     silently re-run yesterday's kernel.
# Legs (unchanged semantics):
#   L0 ctrl   qwen fp8_e4m3 spec3 (NO ARC_KVDTYPE) -- default-off
#      neutrality: boots the cert23 posture byte-identically, battery +
#      markers only (no long suite; the posture is cert23-certified).
#   L1 m8q    qwen ARC_KVDTYPE=mxfp8 ARC_SPEC=3 (champion chain, LMC=1)
#   L2 m4q    qwen ARC_KVDTYPE=mxfp4 ARC_SPEC=3
#   L3 m8g    gemma4-31b ARC_KVDTYPE=mxfp8 ARC_SPEC=0 (cert22 gemma line)
#   L4 m4g    gemma4-31b ARC_KVDTYPE=mxfp4 ARC_SPEC=0
#   L5 mkq    qwen ARC_KVDTYPE=mxk8v4 (e4m3 K + e2m1 V)
#   L6 mkg    gemma ARC_KVDTYPE=mxk8v4
#   L7 g0     gemma native fp8_e4m3 KV control (+ ungated longprobe v4)
# L1-L6 each run the FULL corruption program (owner directive 2026-11-11:
# end-to-end garbage/long-return validation on EVERY MX dtype, both
# models): 3-prompt temp-0 text battery, suite-13 arc_garbage_probe
# (malformed inputs + state-integrity), suite-14 longprobe v4 full + 3
# churn rehits (garble flags + top-window counts + raw heads +
# cross-cycle hash determinism), erraudit ERROR=0/TRACEBACK=0, metrics,
# serve log bank. QWEN legs gate ABSOLUTE (p1 MUST contain Paris;
# zero-garble). GEMMA legs gate SHAPE + COMPARATIVELY vs the same-image
# g0 native control (r164 disposition 2026-11-12: think-spam degeneration
# — universal cross-class attractor, P1 newline-spam — is BASELINE on
# native fp8 KV; gemma_compare enforces no-new-flag-modes, no window
# amplification, rehit hash determinism). trap restores W14.
# Artifacts -> ${P647_BANK:-/root/build/p647/cert25y}/<leg>/.
set -uo pipefail
R=/root/build
P=${P647_BANK:-$R/p647/cert25y}
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25y}
CNAME=arc-c24
GATE_FAIL=0
mkdir -p "$P"

# ---- image-content markers: refuse a stale cert25x image ----
IMG_OK=$(docker run --rm -i --entrypoint /opt/venv/bin/python "$IMG" - <<'PYEOF' 2>/dev/null
import re
from vllm_gaudi.xpu.patches.mx_kv import engine
from vllm_gaudi.xpu.patches.mx_kv import triton_mx_decode as t
tsrc = open(t.__file__).read()
esrc = open(engine.__file__).read()
ok = "tl.maximum(1, n_tiles)" in tsrc          # v3 s_eff 1-tile floor
ok = ok and not re.search(r"query\.float\(\)", esrc)  # native-q decode path
ok = ok and "_split_verify_decode" in esrc     # batched MTP-verify decode
print("IMG_CONTENT_OK" if ok else "IMG_CONTENT_STALE")
PYEOF
) || IMG_OK=""
# vllm's import-time plugin logger writes INFO lines to STDOUT (a stdout
# handler is installed at import), so the capture is multi-line — match
# the verdict by grep, never by string equality (maiden-run lesson,
# 2026-11-12: equality refused a live cert25y image).
echo "$IMG_OK" | grep -q "IMG_CONTENT_OK" \
  || { echo "R161_STALE_IMAGE (missing s_eff floor / engine casts still present) IMG=$IMG"; exit 1; }
echo "R161_IMAGE_CONTENT_OK $IMG"

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

garble_check() {  # $1 = longprobe out file $2 = gate(1|0, default 1) —
  # r98 law: grep -F (r84-era BRE escapes never matched; do not
  # reintroduce). Gemma legs pass 0: bank-only (the gemma baseline
  # trips REP-LOOP on native KV — r164 g0 evidence 2026-11-12; the
  # gemma verdict is comparative, see gemma_compare).
  local lines empty GATE=${2:-1}
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" || true)
  empty=$(( $(grep -cF "garble={'cold': [], 'rehit': [], 'rehit2': []}" "$1" || true)
          + $(grep -cF "garble={'r1': [], 'r2': []}" "$1" || true) ))
  echo "GARBLE: class_lines=$lines empty_garble=$empty (gated=$GATE) in $(basename "$1")"
  if [ "$GATE" = 1 ]; then
    { [ "$lines" -ge 4 ] && [ "$lines" = "$empty" ]; } || { echo "ZERO-GARBLE: FAIL ($1)"; GATE_FAIL=1; }
  fi
}

longreturn_summary() {  # $1 = tag $2 = leg dir $3 = gate(1|0, default 1)
  # — r99 law: probe writes its jsons to CWD; summary MUST run from the
  # leg dir. Gemma legs pass 0: bank-only (the native g0 baseline trips
  # REP-LOOP + finish=length — r164 evidence 2026-11-12; the gemma
  # verdict is comparative, see gemma_compare).
  local GATE=${3:-1}
  ( cd "$2" && python3 - "$1" > longreturn_summary.txt 2>&1 << 'PYEOF' || true
import json, os, sys
tag = sys.argv[1]
def load(fn):
    with open(fn) as f:
        return json.load(f)
full = load(f"longprobe_{tag}_full.json")
rehits = [load(f"longprobe_{tag}r{c}_rehit.json") for c in (1, 2, 3)
          if os.path.exists(f"longprobe_{tag}r{c}_rehit.json")]
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
  if [ "$GATE" = 1 ]; then
    grep -q "LONGRETURN: PASS" "$2/longreturn_summary.txt" || GATE_FAIL=1
    grep -q "T1_SEED: PASS" "$2/longreturn_summary.txt" || GATE_FAIL=1
  fi
  cat "$2/longreturn_summary.txt"
}

gemma_compare() {  # comparative corruption verdict for the gemma legs vs
  # the same-image native g0 leg (reploop-classifier-baseline law; r164
  # disposition 2026-11-12: gemma4-31b think-spam degeneration — universal
  # cross-class attractor + P1 newline-spam — is BASELINE on native fp8
  # KV, so the MX gemma legs gate on NO-NEW-MODES + no window
  # amplification + rehit determinism, never vs CLEAN).
  python3 - "$P" > "$P/gemma_compare.txt" 2>&1 << 'PYEOF' || true
import json, sys
from pathlib import Path
bank = Path(sys.argv[1])
CLASSES = ("essay", "code", "multi", "simple")
def load_full(leg):
    d = bank / leg
    cands = sorted(d.glob("longprobe_*_full.json")) if d.is_dir() else []
    if not cands:
        return None
    with open(cands[0]) as f:
        return json.load(f)
def flags_of(doc):
    out = {}
    for cls in CLASSES:
        v = doc.get(cls, {}) if doc else {}
        fl = set()
        for g in list(v.get("garble", {}).values()) + [v.get("t1_garble", [])]:
            fl |= set(g or [])
        out[cls] = fl
    return out
def topwin_n(doc):
    out = {}
    for cls in CLASSES:
        v = doc.get(cls, {}) if doc else {}
        n = 0
        ws = v.get("win_stats") or {}
        for ph in (ws.values() if isinstance(ws, dict) else []):
            for w in (ph.get("top_windows") or [])[:1]:
                n = max(n, int(w.get("n", 0)))
        out[cls] = n
    return out
g0 = load_full("g0")
if g0 is None:
    print("GEMMA_COMPARE: FAIL (no g0 baseline longprobe banked)")
    sys.exit(0)
g0f, g0n = flags_of(g0), topwin_n(g0)
fails = []
for leg in ("m8g", "m4g", "mkg"):
    doc = load_full(leg)
    if doc is None:
        print(f"[{leg}] no longprobe banked (skipped)")
        continue
    lf, ln = flags_of(doc), topwin_n(doc)
    for cls in CLASSES:
        new = lf[cls] - g0f[cls]
        if new:
            fails.append(f"{leg}/{cls}: NEW flag modes {sorted(new)} (g0 {sorted(g0f[cls])})")
        if ln[cls] > max(2 * max(g0n[cls], 1), 8):
            fails.append(f"{leg}/{cls}: window amplification leg={ln[cls]} vs g0={g0n[cls]}")
        hashes = set(doc.get(cls, {}).get("hashes", []))
        if len(hashes) > 1:
            fails.append(f"{leg}/{cls}: rehit hash churn {sorted(hashes)}")
    print(f"[{leg}] flags={ {c: sorted(lf[c]) for c in CLASSES} } "
          f"topwin={ln} (g0 topwin={g0n})")
print("GEMMA_COMPARE:", "PASS (no new modes / no amplification / deterministic)"
      if not fails else "FAIL " + "; ".join(fails))
PYEOF
  cat "$P/gemma_compare.txt"
  grep -q "GEMMA_COMPARE: PASS" "$P/gemma_compare.txt" \
    || { echo "GEMMA_COMPARE: FAIL (MX lane worse than native baseline)"; GATE_FAIL=1; }
}

churn() {  # $1 = seed base $2 = served model — r105 churn with the model
  # parameterized (gemma legs need their own name). Out banks per-leg
  # (${CD}/churn_$base.out, CD dynamic-scoped from run_leg) — r159 wrote
  # to the bank root and each leg clobbered the previous leg's evidence.
  local base=$1 mdl=$2
  python3 > "${CD:-$P}/churn_$base.out" 2>&1 << PYEOF || true
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
  tail -1 "${CD:-$P}/churn_$base.out"
}

text_battery() {  # $1 = leg dir $2 = served model $3 = endpoint
  # (completions|chat) — the r149 3-prompt temp-0 battery (trace25-28
  # validated set); p1 MUST contain Paris. Gemma is chat-only: the raw
  # /v1/completions endpoint has document-continuation semantics there
  # (m8g run: prompt-echo + further questions, dtype-INDEPENDENT — an
  # endpoint artifact, not KV corruption), so gemma legs use the chat
  # endpoint with the message content identical. $4 = mode (abs|gemma,
  # default abs): gemma's baseline degenerates into think-spam even on
  # NATIVE fp8 KV (r164 g0 disposition 2026-11-12 — the native control
  # fails P1_PARIS), so gemma mode gates response SHAPE only (3x
  # PROMPT_OK with non-empty text, no traceback); content quality is
  # the comparative gemma_compare verdict.
  local CD=$1 mdl=$2 EP=${3:-completions} MODE=${4:-abs}
  : > "$CD/probe.out"
  for q in "What is the capital of France? Answer with just the city name." \
           "Write a Python function that returns the n-th Fibonacci number using iteration." \
           "Name three chemical elements that are gases at room temperature."; do
    if [ "$EP" = chat ]; then
      curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mdl\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c 'import json,sys; r=json.load(sys.stdin); c=r["choices"][0]; m=c.get("message",{}); t=m.get("content") or m.get("reasoning_content") or m.get("reasoning") or ""; print("PROMPT_OK", len(t), repr(t[:160]))' >> "$CD/probe.out" 2>&1
    else
      curl -s -m 180 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
        -d "{\"model\":\"$mdl\",\"prompt\":\"$q\",\"max_tokens\":64,\"temperature\":0}" \
        | python3 -c 'import json,sys; r=json.load(sys.stdin); print("PROMPT_OK", len(r["choices"][0]["text"]), repr(r["choices"][0]["text"][:160]))' >> "$CD/probe.out" 2>&1
    fi
  done
  cat "$CD/probe.out"
  grep -q "PROMPT_OK" "$CD/probe.out" || { echo "TEXT_BATTERY: FAIL (no PROMPT_OK) $CD"; GATE_FAIL=1; return; }
  [ "$(grep -c PROMPT_OK "$CD/probe.out" || true)" = "3" ] || { echo "TEXT_BATTERY: FAIL (missing answers) $CD"; GATE_FAIL=1; }
  if [ "$MODE" = abs ]; then
    grep -q "Paris" "$CD/probe.out" || { echo "P1_PARIS: FAIL $CD"; GATE_FAIL=1; }
  else
    # gemma shape gate: non-empty text on every answer (baseline
    # think-spam still emits reasoning; EMPTY = a lane defect)
    local nonempty
    nonempty=$(grep -cE "^PROMPT_OK [1-9]" "$CD/probe.out" || true)
    [ "$nonempty" = "3" ] || { echo "TEXT_BATTERY: FAIL (empty answers: ${nonempty:-0}/3) $CD"; GATE_FAIL=1; }
  fi
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
  # continuation semantics, dtype-independent — m8g run1 evidence) and
  # gates its quality program comparatively (r164 g0 disposition)
  case $SERVED in gemma*) local EP=chat GMODE=gemma GQ=0 ;; *) local EP=completions GMODE=abs GQ=1 ;; esac
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
  # text battery (all legs — the corruption gate; gemma shape-mode)
  text_battery "$CD" "$SERVED" "$EP" "$GMODE"
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
    garble_check "$CD/longprobe_full.out" "$GQ"
    for cyc in 1 2 3; do
      churn $(( 900 * cyc )) "$SERVED"
      ( cd "$CD" && ARC_PROBE_MODEL="$SERVED" python3 $R/p647/p647_longprobe.py c24${LEGN}r$cyc rehit > longprobe_rehit$cyc.out 2>&1 )
      grep -E "^\[|LONGPROBE" "$CD/longprobe_rehit$cyc.out"
      garble_check "$CD/longprobe_rehit$cyc.out" "$GQ"
    done
    longreturn_summary c24$LEGN "$CD" "$GQ"
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
# + the reasoning-field capture: gemma-4 is a thinking model that IGNORES
# enable_thinking:false (kwarg accepted, template unaffected) and vLLM
# 0.30 emits the stream as message.reasoning (NOT reasoning_content —
# run2 m8g live evidence: content null, reasoning carries "The capital
# of France is Paris."). Without this read every gemma longprobe hash
# is EMPTY -> spurious EMPTY/longreturn fails.
mkdir -p $R/p647
python3 - << 'PYEOF'
import re
src = open("/root/build/p618/p618_longprobe.py", encoding="utf-8").read()
if "ARC_PROBE_MODEL" not in src:
    assert src.count('MODEL = "qwen3.8-27b-fp8"') == 1, "p618 model line drifted"
    src = src.replace('MODEL = "qwen3.8-27b-fp8"',
                      'MODEL = os.environ.get("ARC_PROBE_MODEL", "qwen3.8-27b-fp8")')
    src = src.replace("import hashlib\n", "import hashlib\nimport os\n", 1)
if "msg.get(\"reasoning\")" not in src:
    assert src.count('rsn = msg.get("reasoning_content") or ""') == 1, \
        "p618 reasoning line drifted"
    src = src.replace('rsn = msg.get("reasoning_content") or ""',
                      'rsn = msg.get("reasoning_content") or msg.get("reasoning") or ""')
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
# semantics; both artifacts, engine exonerated by m8g==m4g text equality);
# run3 = run2 + the reasoning-field capture (gemma thinking ignores
# enable_thinking:false and vLLM 0.30 emits message.reasoning; without
# the read, battery + longprobe texts are EMPTY on gemma — run2 m8g live
# curl evidence). Prior leg dirs are preserved as <leg>.runN before a
# re-run.
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
leg_mkq() {  # mxk8v4 (e4m3 K + e2m1 V) — the m4q quality-fix lane (3b92b9f8)
  run_leg mkq qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxk8v4 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/mkq/markers.txt" || { echo "MX_MARKER: FAIL mkq (expect >=1 worker)"; GATE_FAIL=1; }
}
leg_mkg() {
  [ -d "$P/mkg" ] && mv "$P/mkg" "$P/mkg.run1"
  run_leg mkg gemma4-31b 1 $GEMMA_M8 ARC_KVDTYPE=mxk8v4 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/mkg/markers.txt" || { echo "MX_MARKER: FAIL mkg (expect >=1 worker)"; GATE_FAIL=1; }
}
leg_g0() {  # gemma NATIVE-KV control: battery texts on the default
  # fp8_e4m3 KV — the discriminator for gemma thinking-stream degeneration
  # ("our our" REP-LOOP in reasoning, deterministic across boots/run2+run3).
  # If g0 shows the same p2/p3 shape, it is gemma baseline behavior at
  # temp-0 thinking, NOT MX-KV corruption; m8g/m4g/mkg then compare
  # dtype-vs-dtype only. mx_backend_lines MUST be 0 here.
  run_leg g0 gemma4-31b 0 $GEMMA_M8 || true
  grep -q "mx_backend_lines=0" "$P/g0/markers.txt" || { echo "G0_NATIVE: FAIL (MX backend engaged on the control leg)"; GATE_FAIL=1; }
  # r161: g0 longprobe v4 as UNGATED evidence — r159's g0 was battery-only,
  # so the m8g CORRUPTION_SIGNAL had no native-lane long-return shape to
  # compare against. v4 banks top-window counts + raw heads, which is
  # exactly the disposition evidence (benign quoting ~4-8x vs pathological
  # loops 100x+). Deliberately NOT garble_check'd: gemma thinking-mode
  # baseline may flag; the gate compares MX-vs-native, not vs CLEAN.
  ( cd "$P/g0" && ARC_PROBE_MODEL=gemma4-31b \
      python3 $R/p647/p647_longprobe.py c25yg0 full > longprobe_full.out 2>&1 ) || true
  grep -E "^\[|LONGPROBE" "$P/g0/longprobe_full.out" || true
  echo "G0_LONGPROBE_EVIDENCE_BANKED (ungated baseline discriminator)"
}

if [ $# -gt 0 ]; then
  for LEG in "$@"; do
    case $LEG in
      ctrl|m8q|m4q|m8g|m4g|mkq|mkg|g0) leg_$LEG ;;
      compare) gemma_compare ;;
      *) echo "UNKNOWN_LEG $LEG (ctrl|m8q|m4q|m8g|m4g|mkq|mkg|g0|compare)"; GATE_FAIL=1 ;;
    esac
  done
else
  # full battery: qwen legs gate ABSOLUTE (clean baseline), gemma legs
  # bank-only + the comparative gemma_compare verdict after the g0
  # native control lands (r164 disposition law)
  leg_ctrl; leg_m8q; leg_m4q; leg_m8g; leg_m4g; leg_mkq; leg_mkg; leg_g0
  gemma_compare
fi

cd "$P"
find . -type f \( -name "*.out" -o -name "*.txt" -o -name "*.json" -o -name "*.sha256" -o -name "serve.log" -o -name "metrics.txt" \) -exec sha256sum {} \; > SHA256SUMS
echo "LEG_ARTIFACTS $P"
if [ "$GATE_FAIL" = 0 ]; then
  echo "P647_CERT25_GATE_COMPLETE -- PASS (analysis/bank next)"
else
  echo "P647_CERT25_GATE_COMPLETE -- GATE_FAIL=$GATE_FAIL (fix before cert)"
fi
exit $GATE_FAIL
