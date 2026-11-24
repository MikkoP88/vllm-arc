#!/bin/bash
# r239b_gemma_rhmxfp4_fix.sh -- P6.47 the two r239 boot-fail cells re-run
# with the diagnosed fixes:
#   - gemma L1 record: ARC_MAXLEN 131072 -> 118784. r239's 131072 boot
#     refused on cert28 (5.83 GiB KV needed > 5.64 available; estimated
#     max 120704 -- the L1 lane's KV headroom shrank since cert19's
#     131K boots: engine growth). 118784 preserves the face120k class
#     exactly (110028-ptok face + 16 gen = 110044 << 118784) with
#     boot margin; every other face needs far less.
#   - rhmxfp4: no posture change -- r239's failure was the container
#     name conflict (the failed gemma boot left arc-r239 up; the serve
#     docker run refused; the poll read the stale traceback). Every
#     boot-fail path here rm's the container before returning.
# qurk + aur cells already banked by r239 (banked under p647/r239/);
# this leg banks to p647/r239b/ and the close-out merges both.
# Same gates as r239 (erraudit 0/0, G1 3/3, probe98 clean, spec-gate
# acceptance >= 0.85, one-sided x1.20 cross-era band; face32k_2 ungated).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_MQA=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r239b
L=$R/p647/r239log
CNAME=arc-r239b

fail() { echo "R239B_ABORT: $*"; }

# ---- refuses (r237/r239 pattern) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert28_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
read -r SO_A SO_M <<< "$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; vp=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/')[0]; print(hashlib.md5(open(vp+'libattn_kernels_xe_2.so','rb').read()).hexdigest(), hashlib.md5(open(glob.glob(vp+'libmqa_logits*.so')[0],'rb').read()).hexdigest())\"" 2>/dev/null)" || true
[ "${SO_A:-}" = "$SO_ATTN" ] || { fail "image libattn md5 ${SO_A:-?} != build28 pin"; exit 1; }
[ "${SO_M:-}" = "$SO_MQA" ] || { fail "image libmqa md5 ${SO_M:-?} != pin"; exit 1; }
for M in /models/gemma4-31b /models/qwen3.8-27b-mxfp4xfp8; do
  [ -d "$M" ] || { fail "model dir missing: $M"; exit 1; }
done
[ -f "$R/p65/p65_timeline.py" ] || { fail "p65_timeline missing"; exit 1; }
[ -f "$R/arc_spec_gate.py" ] || { fail "arc_spec_gate missing"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R239B refuses passed (fork_sha=$FORK_SHA libattn=$SO_A libmqa=$SO_M)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "R239B: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  echo "R239B: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r239b.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R239B: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R239B: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[boot] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb
      tb=$(docker exec $CNAME grep -c "Traceback (most recent call last)" /root/serve_full.log 2>/dev/null)
      [ "${tb:-0}" -gt 0 ] && { echo "[boot] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[boot] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

erraudit() {  # $1 = cell dir
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$1/erraudit.txt" 2>/dev/null || true
}

# ---------------- gemma L1 record cell (maxlen fixed) ----------------
cell_gemma() {
  local CD=$P/gemma
  mkdir -p "$CD"
  echo "=== [r239b:gemma] boot L1 record posture (graph=1 spec=0 lmc=0 maxlen=118784) $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma-4-31b \
    ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_FP8=0 \
    ARC_MAXLEN=118784 ARC_RPC_TIMEOUT=1200 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R239B_BOOT_FAIL_gemma"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f $CNAME >/dev/null 2>&1 || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"; return 0
  fi
  echo "HEALTH_OK_r239b_gemma $(date +%H:%M:%S)"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"gemma-4-31b","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3

  python3 - > "$CD/gemma_faces.txt" 2>&1 <<'PYEOF' || true
import json, time, threading, urllib.request
BASE = "http://127.0.0.1:8000"
MODEL = "gemma-4-31b"
INSTR = "Continue the passage naturally. "
FILLER = "the "

def completions(prompt, max_tokens, timeout=900):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=timeout))
    return time.perf_counter() - t0, r["usage"]["prompt_tokens"], r["choices"][0]["finish_reason"]

# probe98 (98-token generation probe)
w, pt, fin = completions("Hello", 98)
print(f"probe98 wall={w:.1f}s ptok={pt} finish={fin} CLEAN={fin=='length'}")

# G1 3-probe t0 quality (r104 shape, chat)
def chat(q, mt):
    body = {"model": MODEL, "max_tokens": mt, "temperature": 0,
            "messages": [{"role": "user", "content": q}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=300))
    m = r["choices"][0]["message"]
    return r["choices"][0]["finish_reason"], r["usage"]["completion_tokens"], \
        ((m.get("reasoning") or "") + "|" + (m.get("content") or ""))[:200]
for name, q, mt in [
    ("factual", "Say the number seven in words, then stop.", 24),
    ("reason", "If a train leaves at 3pm traveling 60 km/h and another at 4pm traveling 90 km/h the same direction, when does the second catch the first? Answer with just the time.", 96),
    ("code", "Write a Python one-liner that reverses a string s. Output only the code.", 64)]:
    fin, ct, txt = chat(q, mt)
    print(f"[G1:{name}] finish={fin} tok={ct} {txt!r}")

# filler calibration (slope/intercept of ptok vs filler count)
_, p0, _ = completions(INSTR + FILLER * 200, 1)
_, p1, _ = completions(INSTR + FILLER * 400, 1)
slope = (p1 - p0) / 200.0
intercept = p0 - slope * 200
def build(target):
    n = int(round((target - intercept) / slope))
    return INSTR + FILLER * n

# g2 faces (16 max_tokens, TTFT-dominated walls) + re-hit
p32k = None
for name, target in [("face8k", 8028), ("face32k", 32028), ("face120k", 110028)]:
    prompt = build(target)
    if name == "face32k":
        p32k = prompt
    w, pt, fin = completions(prompt, 16)
    print(f"{name} wall={w:.1f}s finish={fin} ptok={pt} ctok=16")
w, pt, fin = completions(p32k, 16)
print(f"face32k_2 wall={w:.1f}s finish={fin} ptok={pt} ctok=16 (prefix re-hit)")
print("G2_DONE")

# decode battery (temp 0): solo conc1 x3 (256 ctok), agg8 conc8 x3 (2048 ctok)
SEEDS = [f"Story seed {i}: a lantern, a harbor, and a long winter evening. Continue. "
         for i in range(8)]
def one(i, mt, out):
    w, _, _ = completions(SEEDS[i], mt)
    out[i] = mt / w
for rep in (1, 2, 3):
    out = [0.0] * 1
    one(0, 256, out)
    print(f"solo conc1 rep{rep} tps={out[0]:.1f}")
for rep in (1, 2, 3):
    out = [0.0] * 8
    th = [threading.Thread(target=one, args=(i, 2048, out)) for i in range(8)]
    barrier = time.perf_counter()
    [t.start() for t in th]; [t.join() for t in th]
    print(f"agg8 conc8 rep{rep} tps_mean={sum(out)/8:.1f} span={time.perf_counter()-barrier:.1f}s")
print("DECODE_BATTERY_DONE")
PYEOF
  cat "$CD/gemma_faces.txt"

  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  erraudit "$CD"
  echo "FACES_OK" > "$CD/outcome.txt"
  docker rm -f $CNAME >/dev/null 2>&1 || true
  echo "=== [r239b:gemma] done $(date +%H:%M:%S) ==="
}

# ---------------- rhmxfp4 cell (r104 flow, container fixed) ----------------
cell_rhmxfp4() {
  local TAG=rhmxfp4 MODEL=/models/qwen3.8-27b-mxfp4xfp8 SERVED=qwen3.8-27b-mxfp4xfp8 CD=$P/rhmxfp4
  mkdir -p "$CD"
  echo "=== [r239b:$TAG] boot ($SERVED, graph=1 spec=3 lmc=1) $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=$MODEL ARC_SERVED=$SERVED ARC_QUANT= \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R239B_BOOT_FAIL_$TAG"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f $CNAME >/dev/null 2>&1 || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"; return 0
  fi
  echo "HEALTH_OK_r239b_$TAG $(date +%H:%M:%S)"
  curl -s -m 180 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3

  python3 - "$SERVED" > "$CD/g1_probes.txt" 2>&1 <<'PYEOF' || true
import json, sys, urllib.request
served = sys.argv[1]
probes = [
    ("factual",  "Say the number seven in words, then stop.", 24),
    ("reason",   "If a train leaves at 3pm traveling 60 km/h and another at 4pm traveling 90 km/h the same direction, when does the second catch the first? Answer with just the time.", 96),
    ("code",     "Write a Python one-liner that reverses a string s. Output only the code.", 64),
]
for name, q, mt in probes:
    body = {"model": served, "max_tokens": mt, "temperature": 0,
            "messages": [{"role": "user", "content": q}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=300))
    m = r["choices"][0]["message"]
    txt = (m.get("reasoning") or "") + "|" + (m.get("content") or "")
    u = r["usage"]
    print(f"[{name}] finish={r['choices'][0]['finish_reason']} tok={u['completion_tokens']}")
    print(f"  {txt[:400]!r}")
PYEOF
  cat "$CD/g1_probes.txt"

  echo "--- [$TAG] spec gate $(date +%H:%M:%S) ---"
  python3 $R/arc_spec_gate.py --model "$SERVED" --url http://127.0.0.1:8000 \
    > "$CD/spec_gate.txt" 2>&1 || true
  tail -6 "$CD/spec_gate.txt"

  echo "--- [$TAG] p65 timeline G2 faces $(date +%H:%M:%S) ---"
  local TL=$R/p65/p65_timeline_r239b_$TAG.py
  cp $R/p65/p65_timeline.py $TL
  sed -i "s|^MODEL = .*|MODEL = \"$SERVED\"|" $TL
  grep -q "MODEL = \"$SERVED\"" $TL || { echo "FATAL: harness sed failed"; return 0; }
  python3 $TL "$CD/timeline.json" > "$CD/timeline_stdout.txt" 2>&1 || echo "timeline exit=$?"
  grep -A6 "long-stream faces" "$CD/timeline_stdout.txt" || tail -20 "$CD/timeline_stdout.txt"
  grep -A3 "baseline:" "$CD/timeline_stdout.txt" | head -4 || true

  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  erraudit "$CD"
  echo "FACES_OK" > "$CD/outcome.txt"
  docker rm -f $CNAME >/dev/null 2>&1 || true
  echo "=== [r239b:$TAG] done $(date +%H:%M:%S) ==="
}

# ---- 1. arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5
mkdir -p "$P" "$L"

# ---- 2. cells ----
cell_gemma
cell_rhmxfp4

# ---- 3. report + gate (recomputed from THIS bank) ----
echo "=== [r239b] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import os, re, sys
P = sys.argv[1]
ANCH = {
    "gemma":   {"face8k": 4.9, "face32k": 17.2, "face120k": 110.9, "solo": 28.45, "agg8": 195.0},
    "rhmxfp4": {"U32k": 29.01, "C32k": 1.68, "U8k": 7.37, "U64k": 62.46},
}
BAND = 1.20
TARGET_PTOK = {"face8k": 8028, "face32k": 32028, "face120k": 110028}
ok = True

def gate_erraudit(cell):
    global ok
    f = os.path.join(P, cell, "erraudit.txt")
    if not os.path.exists(f):
        print(f"  {cell}: erraudit MISSING"); ok = False; return
    txt = open(f).read()
    e = re.search(r"ERROR=(\d+)", txt); t = re.search(r"TRACEBACK=(\d+)", txt)
    if not e or int(e.group(1)) or int(t.group(1)):
        print(f"  {cell}: erraudit {txt.strip()!r}"); ok = False
    else:
        print(f"  {cell}: erraudit 0/0")

gf = os.path.join(P, "gemma", "gemma_faces.txt")
print("gemma L1 record cell (cert19-era anchors, maxlen=118784):")
if os.path.exists(gf):
    txt = open(gf).read()
    for line in txt.splitlines():
        m = re.match(r"(face\w+) wall=([\d.]+)s finish=(\w+) ptok=(\d+) ", line)
        if m:
            name, wall, fin, ptok = m.group(1), float(m.group(2)), m.group(3), int(m.group(4))
            a = ANCH["gemma"].get(name)
            if a and name in TARGET_PTOK:
                norm = wall * TARGET_PTOK[name] / ptok
                r = norm / a
                print(f"  {name:10s} wall={wall:6.1f}s ptok={ptok:6d} (target {TARGET_PTOK[name]}) norm={norm:6.1f}s anchor={a:6.1f} ratio={r:.3f}"
                      + (" REGRESSION" if r > BAND else ""))
                if r > BAND: ok = False
            elif name:
                print(f"  {name:10s} wall={wall:6.1f}s ptok={ptok:6d} (banked, cache face ungated)")
    for line in txt.splitlines():
        m = re.match(r"solo conc1 rep\d tps=([\d.]+)", line)
        if m:
            tps = float(m.group(1)); r = ANCH["gemma"]["solo"] / tps
            print(f"  solo tps={tps:.1f} anchor={ANCH['gemma']['solo']} ratio={r:.3f}" + (" REGRESSION" if r > BAND else ""))
            if r > BAND: ok = False
        m = re.match(r"agg8 conc8 rep\d tps_mean=([\d.]+)", line)
        if m:
            tps = float(m.group(1)); r = ANCH["gemma"]["agg8"] / tps
            print(f"  agg8 tps={tps:.1f} anchor={ANCH['gemma']['agg8']} ratio={r:.3f}" + (" REGRESSION" if r > BAND else ""))
            if r > BAND: ok = False
    if "CLEAN=True" not in txt:
        print("  probe98 not CLEAN"); ok = False
    for k in ("[G1:factual]", "[G1:reason]", "[G1:code]"):
        if k not in txt:
            print(f"  {k} MISSING"); ok = False
    gate_erraudit("gemma")
else:
    print("  gemma faces MISSING"); ok = False

tf = os.path.join(P, "rhmxfp4", "timeline_stdout.txt")
print("rhmxfp4 checkpoint-lane cell (cert11c-era anchors):")
if os.path.exists(tf):
    txt = open(tf).read()
    for m in re.finditer(r"(U\d+k|C\d+k): TTFT=\s*([\d.]+)s", txt):
        face, val = m.group(1), float(m.group(2))
        a = ANCH["rhmxfp4"].get(face)
        if a:
            r = val / a
            print(f"  {face:5s} TTFT={val:7.2f}s anchor={a:6.2f} ratio={r:.3f}" + (" REGRESSION" if r > BAND else ""))
            if r > BAND: ok = False
    b = re.search(r"baseline: n=\d+ p50=\s*([\d.]+)ms", txt)
    if b:
        print(f"  bystander p50={b.group(1)}ms (reference)")
else:
    print("  timeline MISSING"); ok = False
sg = os.path.join(P, "rhmxfp4", "spec_gate.txt")
if os.path.exists(sg):
    stxt = open(sg).read()
    acc = re.search(r"acceptance[_-]ratio[=:\s]+([\d.]+)", stxt, re.I)
    if acc:
        a = float(acc.group(1))
        print(f"  spec-gate acceptance={a:.4f}" + (" FAIL(<0.85)" if a < 0.85 else " PASS"))
        if a < 0.85: ok = False
    else:
        print("  spec-gate acceptance line not found (inspect spec_gate.txt)")
g1 = os.path.join(P, "rhmxfp4", "g1_probes.txt")
if os.path.exists(g1) and all(k in open(g1).read() for k in ("[factual]", "[reason]", "[code]")):
    print("  G1 3/3 present")
else:
    print("  G1 MISSING"); ok = False
gate_erraudit("rhmxfp4")

print(f"R239B_GATE: {'PASS' if ok else 'FAIL'} (gemma+rhmxfp4 re-discriminated; cross-era band x{BAND})")
PY
echo "R239B_DONE $(date +%H:%M:%S)"
exit 0
