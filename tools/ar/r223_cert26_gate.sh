#!/bin/bash
# r223_cert26_gate.sh (r197 + cert26 deltas) -- cert26 (P6.47 Phase-8
# convergence: arc7 MX-native decode line) §10 gate. Image
# vllm-arc:v0.30.0-b70-cert26 = cert25zc content + the arc7mx wheel
# (vllm_xpu_kernels 0.1.14.1+arc7mx, md5 62f234c8 byte-pinned, BAKED)
# + the fork mx_kv overlay native_verify bridge (ARC_MXKV_NATIVE gate,
# default-off; _NATIVE_MINLEN per-lane thresholds; r212f capacity-
# priced walk; capture telemetry). Certified MX posture on the QWEN
# legs: ARC_KVDTYPE + ARC_MXKV_NATIVE=1 on m8q/m4q, NSPLIT unset (C++
# auto at the validated optimum -- r219b A/A within 0.08 pct of the
# pinned path, chain 341 bw -> 130 reduce -> 128 policy), P64NATIVE
# unset (_16 sub-tile frame -- the r220 e2e arbiter closed the 16->64
# tile lever; offline wins inverted e2e +13.4 pct @32K, not rescued at
# any split depth). mkq runs the cert25zc TRITON lanes: the battery's
# first mkq cell (native) FAILED its absolute LONGRETURN gate (REP-LOOP
# all classes, deterministic) while the uniform-dtype native lanes are
# clean -- the mixed-fmt (e4m3 K + e2m1 V) lane is refused pending the
# r224 engine-stride diagnosis (the failed cell is banked as refusal
# evidence). Gemma MX legs keep the cert25zc triton lanes (native-off):
# the r216-r221 native evidence chain is qwen-only; gemma gates
# corruption comparatively as before. Evidence: r219b (auto 48.5/51.2/
# 56.8; tps 65.4 @32K; chunks 53-54; 0 errors), r220b ctl (posture
# -0.64 pct vs banked ns128), r220c value gate v2 (d_16_64 <= 0.000977,
# det 0, nan 0), r220d split bracket + r221a/r221b mechanism
# eliminations. vs r197 (cert25zc):
#   - image-content markers extended to the cert26 set (all cert25zc
#     markers PLUS the native bridge, the per-lane MINLEN defaults,
#     the wheel import surface) + manifest fork_sha refuse (passed as
#     $1 -- the content commit).
#   - qwen MX legs run ARC_MXKV_NATIVE=1 and assert the native
#     telemetry + splits=auto posture in the serve log.
# Leg semantics otherwise unchanged from r197 (qwen absolute, gemma
# comparative vs same-image g0 native control; r170b churn amendment
# carried). trap restores W14.
# Artifacts -> ${P647_BANK:-/root/build/p647/cert26}/<leg>/.
set -uo pipefail
R=/root/build
P=${P647_BANK:-$R/p647/cert26}
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert26}
CNAME=arc-c26
FORK_SHA=${1:?usage: r223_cert26_gate.sh <FORK_SHA> [legs...]}
shift || true
GATE_FAIL=0
mkdir -p "$P"

# ---- image-content markers: refuse a stale cert26 image ----
IMG_OK=$(docker run --rm -i --entrypoint /opt/venv/bin/python "$IMG" - <<'PYEOF' 2>/dev/null
import re
from vllm_gaudi.xpu.patches.mx_kv import engine
from vllm_gaudi.xpu.patches.mx_kv import native_decode as n
from vllm_gaudi.xpu.patches.mx_kv import triton_mx_decode as t
from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode
tsrc = open(t.__file__).read()
esrc = open(engine.__file__).read()
nsrc = open(n.__file__).read()
ok = "tl.maximum(1, n_tiles)" in tsrc          # v3 s_eff 1-tile floor
ok = ok and not re.search(r"query\.float\(\)", esrc)  # native-q decode path
ok = ok and "_split_verify_decode" in esrc     # batched MTP-verify decode
ok = ok and "_mx_decode_split_rows" in tsrc    # v4 verify-row kernel
ok = ok and "_mx_decode_split_dot" in tsrc     # v5 XMX tl.dot kernel
ok = ok and "_V5_DEFAULT" in esrc              # v5 default flip
ok = ok and engine._V5_DEFAULT == ("mxfp8", "mxk8v4")
ok = ok and "ARC_MXKV_VERIFY_DECODER" in esrc  # verify-decoder override env
ok = ok and "mx_decode_rows_dot" in esrc       # v5 dispatch
ok = ok and "_f16_scale" in tsrc and "_e2m1_byte" in tsrc  # K2/K3 helpers
ok = ok and 'ARC_MXKV_K2", "1") != "0"' in tsrc           # K2 default-on
ok = ok and 'ARC_MXKV_K3", "1") != "0"' in tsrc           # K3 default-on
ok = ok and 'ARC_MXKV_V5_TFLOOR", "2") or 2' in tsrc      # r195 tf2 launcher default
ok = ok and "_V5_G_MIN" in esrc and engine._V5_G_MIN == 8  # G-adaptive knee (n8 face)
ok = ok and "_mx_decode_split_dot2" in tsrc               # v7 research kernel present
ok = ok and 'ARC_MXKV_V7", "0") == "1"' in tsrc           # v7 dormant default-off
# cert26 native line
ok = ok and "from .native_decode import mx_decode_native" in esrc  # bridge import
ok = ok and 'os.environ.get("ARC_MXKV_NATIVE", "") == "1"' in esrc  # native default-off
ok = ok and engine._NATIVE_MINLEN == {"mxfp8": (16384, 16384),
                                      "mxfp4": (12288, 8192),
                                      "mxk8v4": (8192, 8192)}
ok = ok and "ARC_MXKV_NSPLIT" in nsrc         # split-depth instrument retained
ok = ok and "MXKV native call fmt=" in nsrc    # capture telemetry
ok = ok and callable(n.mx_decode_native) and callable(mx_paged_decode)  # wheel live
print("IMG_CONTENT_OK" if ok else "IMG_CONTENT_STALE")
PYEOF
) || IMG_OK=""
# vllm's import-time plugin logger writes INFO lines to STDOUT (a stdout
# handler is installed at import), so the capture is multi-line — match
# the verdict by grep, never by string equality (r161 maiden-run lesson).
echo "$IMG_OK" | grep -q "IMG_CONTENT_OK" \
  || { echo "R223_STALE_IMAGE (missing cert26 native line) IMG=$IMG"; exit 1; }
echo "R223_IMAGE_CONTENT_OK $IMG"
# manifest provenance refuse: fork_sha must be the cert26 content commit
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert26_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "$MF_SHA" | grep -q "cert26_fork_sha=$FORK_SHA" \
  || { echo "R223_FORK_SHA_MISMATCH ($MF_SHA vs $FORK_SHA)"; exit 1; }
echo "R223_FORK_SHA_OK $MF_SHA"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647c26.log 2>&1 &
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
  # amplification, never vs CLEAN).
  #
  # r170b amendment (2026-11-13, evidence-based): the first run's
  # absolute per-leg rehit-determinism criterion contradicted the
  # comparative law this function's own header cites. The r170 FULL
  # discriminator + the cert25y bank close it: (1) g0's own LONGRETURN
  # FAILs on the same REP-LOOP flags natively (2125-char truncation
  # spam, all four classes) — the native baseline IS the attractor;
  # (2) the churn pairs are byte-identical in cert25y's banked compare
  # (m8g/essay 12c0e71e5be4->fc2dddcc7ce6; m4g/simple 3-hash ladder;
  # m8g==mkg cold agreement cross-dtype) — the churn predates cert25zb
  # content, survives the LMC posture change, and is itself
  # DETERMINISTIC; (3) the churn destination class is REP-LOOP = g0's
  # own class set, so the existing NO-NEW-MODES check already gates
  # destination parity. Determinism therefore gates COMPARATIVELY:
  # churn into baseline classes is banked as KNOWN-CLASS churn for
  # review, never an automatic FAIL. The qwen target lanes stay
  # ABSOLUTE (4/4 PASS, byte-exact r168 — unaffected by this
  # amendment).
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
print(f"[g0] baseline flags={ {c: sorted(g0f[c]) for c in CLASSES} } "
      f"topwin={g0n} (native LONGRETURN fails these same classes)")
fails = []
churn_notes = []
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
        hashes = sorted(set(doc.get(cls, {}).get("hashes", [])))
        if len(hashes) > 1:
            # r170b: churn whose destination classes sit inside g0's own
            # class set is baseline-class (documented above) — bank for
            # review; a churn destination OUTSIDE g0's classes trips the
            # NEW-modes check above.
            churn_notes.append(f"{leg}/{cls}: KNOWN-CLASS churn {hashes}")
    print(f"[{leg}] flags={ {c: sorted(lf[c]) for c in CLASSES} } "
          f"topwin={ln} (g0 topwin={g0n})")
for note in churn_notes:
    print(f"[churn] {note}")
print("GEMMA_COMPARE:", "PASS (no new modes / no amplification; "
      "known-class churn banked for review)"
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
    echo "native_lines=$(gcount 'MXKV native call')"
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
    ( cd "$CD" && ARC_PROBE_MODEL="$SERVED" python3 $R/p647/p647_longprobe.py c26$LEGN full > longprobe_full.out 2>&1 )
    grep -E "^\[|LONGPROBE" "$CD/longprobe_full.out"
    garble_check "$CD/longprobe_full.out" "$GQ"
    for cyc in 1 2 3; do
      churn $(( 900 * cyc )) "$SERVED"
      ( cd "$CD" && ARC_PROBE_MODEL="$SERVED" python3 $R/p647/p647_longprobe.py c26${LEGN}r$cyc rehit > longprobe_rehit$cyc.out 2>&1 )
      grep -E "^\[|LONGPROBE" "$CD/longprobe_rehit$cyc.out"
      garble_check "$CD/longprobe_rehit$cyc.out" "$GQ"
    done
    longreturn_summary c26$LEGN "$CD" "$GQ"
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

# ARC_LMC is NOT pinned here: the serve script's default applies per
# lane. Phase 6 posture amendment (commit cde08345, 2026-11-18,
# r176 A/B evidence): auto-off extends to ALL MX lanes including mxfp8
# (LMC-off e2e-neutral-to-positive — ITL unchanged, TTFT -26/-259 ms,
# store worker + blob capacity freed; decode-parity runs MX
# LMC-free). LMC stays ON for the q8 ctrl lane only (the incumbent's
# own store tax, not this fork's regression). Leg postures assert it.
QWEN_FP8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1"
GEMMA_M8="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_QUANT=mxfp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"
# cert26 native posture: the qwen MX legs pass ARC_MXKV_NATIVE=1 (the
# r216-r221 evidence chain is qwen-only; NSPLIT + P64NATIVE stay unset
# = C++ auto at the validated optimum on the _16 frame). Gemma MX legs
# keep the cert25zc triton lanes — no native e2e evidence exists for
# gemma on this wheel, and the gemma program gates corruption
# comparatively, not decode superiority.

lmc_autooff_check() {  # $1 = leg dir — the e2m1-V lanes MUST come up with
  # LMCache off: the serve script's auto-off NOTE in boot.txt (a stale
  # rig serve script silently re-enables the double-quant driver) and
  # zero LMCacheConnector lines in the serve log.
  local CD=$1
  grep -q "NOTE: ARC_LMC auto-off" "$CD/boot.txt" \
    || { echo "LMC_POSTURE: FAIL ($CD) -- serve script stale or posture not applied"; GATE_FAIL=1; }
  grep -q "lmcache_lines=0" "$CD/markers.txt" \
    || { echo "LMC_POSTURE: FAIL ($CD) -- LMCache engaged on an e2m1-V lane"; GATE_FAIL=1; }
}

native_posture_check() {  # $1 = leg dir — the qwen native legs MUST show
  # the capture telemetry (>=1 'MXKV native call' per worker) with
  # splits=auto (NSPLIT retired; the C++ chain owns the pick). serve.log
  # is the docker-cp'd copy banked before container removal.
  local CD=$1
  grep -Eq "native_lines=[1-9]" "$CD/markers.txt" \
    || { echo "NATIVE_POSTURE: FAIL ($CD) -- no native capture telemetry"; GATE_FAIL=1; }
  grep -q "splits=auto" "$CD/serve.log" 2>/dev/null \
    || { echo "NATIVE_POSTURE: FAIL ($CD) -- no splits=auto evidence"; GATE_FAIL=1; }
}

leg_ctrl() {
  run_leg ctrl qwen3.8-27b-fp8 0 $QWEN_FP8 || true
  MX_LINES=$(grep -c "mx_backend_lines=0" "$P/ctrl/markers.txt" || true)
  [ "$MX_LINES" = "1" ] || { echo "DEFAULT-OFF: FAIL (MX backend engaged without ARC_KVDTYPE)"; GATE_FAIL=1; }
  # ctrl/m8q keep the LMC-on certified posture (e4m3-V; NOT auto-off)
  grep -Eq "lmcache_lines=[1-9]" "$P/ctrl/markers.txt" \
    || { echo "LMC_POSTURE: FAIL (ctrl lost the certified LMC-on posture)"; GATE_FAIL=1; }
}
leg_m8q() {
  run_leg m8q qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m8q/markers.txt" || { echo "MX_MARKER: FAIL m8q (expect >=1 worker)"; GATE_FAIL=1; }
  # Phase 6 posture (cde08345): ALL MX lanes LMC-off by serve-script
  # auto-off — m8q included (r176 A/B: neutral-to-positive; r184's
  # LMC-on expectation predates the amendment)
  lmc_autooff_check "$P/m8q"
  native_posture_check "$P/m8q"
}
leg_m4q() {
  run_leg m4q qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m4q/markers.txt" || { echo "MX_MARKER: FAIL m4q (expect >=1 worker)"; GATE_FAIL=1; }
  lmc_autooff_check "$P/m4q"
  native_posture_check "$P/m4q"
}
leg_m8g() {
  run_leg m8g gemma4-31b 1 $GEMMA_M8 ARC_KVDTYPE=mxfp8 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m8g/markers.txt" || { echo "MX_MARKER: FAIL m8g (expect >=1 worker)"; GATE_FAIL=1; }
}
leg_m4g() {
  run_leg m4g gemma4-31b 1 $GEMMA_M8 ARC_KVDTYPE=mxfp4 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/m4g/markers.txt" || { echo "MX_MARKER: FAIL m4g (expect >=1 worker)"; GATE_FAIL=1; }
}
leg_mkq() {
  # cert26 posture amendment (2026-11-22, this battery's own evidence):
  # the FIRST mkq cell ran ARC_MXKV_NATIVE=1 and FAILED its absolute
  # LONGRETURN gate (REP-LOOP garble all 4 classes + rehits; T1_SEED
  # PASS = deterministic numeric drift, not a race) while m8q/m4q
  # native are absolute-clean and cert25zc's triton mkq was clean --
  # the failure attributes to the native MIXED-fmt lane (e4m3 K +
  # e2m1 V): uniform-dtype lanes pass, the mixed lane is exactly where
  # per-tensor K/V layout divergence bites (kv-store separate-strides
  # law, P6.47), and the r220c offline value gate built a CONTIGUOUS
  # synthetic pool (same K/V layout) so it structurally could not see
  # an engine-stride defect. Native mxk8v4 REFUSED at this gate; mkq
  # serves the cert25zc triton lanes. The failed native cell is banked
  # as the refusal evidence; r224 = the engine-stride-faithful value
  # gate + native-vs-triton mkq A/B (reopen path).
  run_leg mkq qwen3.8-27b-fp8 1 $QWEN_FP8 ARC_KVDTYPE=mxk8v4 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/mkq/markers.txt" || { echo "MX_MARKER: FAIL mkq (expect >=1 worker)"; GATE_FAIL=1; }
  lmc_autooff_check "$P/mkq"
}
leg_mkg() {
  run_leg mkg gemma4-31b 1 $GEMMA_M8 ARC_KVDTYPE=mxk8v4 || true
  grep -Eq "mx_backend_lines=[1-9]" "$P/mkg/markers.txt" || { echo "MX_MARKER: FAIL mkg (expect >=1 worker)"; GATE_FAIL=1; }
}
leg_g0() {  # gemma NATIVE-KV control. FULL=1 (r170): the full suite
  # including 3 churn rehits, UNGATED — the churn-classification
  # discriminator for r161's gemma rehit hash-churn fails. If g0's own
  # cross-cycle hashes churn (KV pool 52,822 tokens vs max_len 40,960 =
  # 1.29x concurrency; churn evicts prefixes and flips cold/cached
  # text), the MX gemma churn fails are eviction artifacts, not
  # corruption. mx_backend_lines MUST be 0 here.
  run_leg g0 gemma4-31b 1 $GEMMA_M8 || true
  grep -q "mx_backend_lines=0" "$P/g0/markers.txt" || { echo "G0_NATIVE: FAIL (MX backend engaged on the control leg)"; GATE_FAIL=1; }
  echo "G0_FULL_SUITE_BANKED (ungated native baseline + churn classification)"
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
  # full battery: qwen legs gate ABSOLUTE (clean baseline; m4q/mkq now
  # expected CLEAN under the r168 auto-off posture), gemma legs bank-only
  # + the comparative gemma_compare verdict after the g0 native control
  leg_ctrl; leg_m8q; leg_m4q; leg_m8g; leg_m4g; leg_mkq; leg_mkg; leg_g0
  gemma_compare
fi

cd "$P"
find . -type f \( -name "*.out" -o -name "*.txt" -o -name "*.json" -o -name "*.sha256" -o -name "serve.log" -o -name "metrics.txt" \) -exec sha256sum {} \; > SHA256SUMS
echo "LEG_ARTIFACTS $P"
if [ "$GATE_FAIL" = 0 ]; then
  echo "P647_CERT26_GATE_COMPLETE -- PASS (analysis/bank next)"
else
  echo "P647_CERT26_GATE_COMPLETE -- GATE_FAIL=$GATE_FAIL (fix before cert)"
fi
exit $GATE_FAIL
