#!/bin/bash
# r234_cert28_itl.sh -- P6.47 cert28 perf re-discrimination + the
# shallow-ctx record faces. The r231d clamp changes shallow-ctx
# decode, so the certified record on cert28 needs BOTH the long-ctx
# anchors re-measured AND the first banked shallow rows (the dynamic
# serving win every fresh request rides).
#
# Cells (one boot each, the certified posture: ARC_GRAPH=1 spec k3
# fp8 weights; wheel BAKED in cert28 -- no mounts, no env overrides):
#   ctrl : ARC_KVDTYPE unset -- the fp8 ITL anchors (3 long faces)
#   m4q  : ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1 -- 5 faces (the
#          headline record incl. the r231de-validated shallow win)
#   m8q  : ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 -- 5 faces
#   mkq  : ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1 -- 5 faces
# Faces: p139_conc.py n1 MT=192 at c2048/c4096/c8192/c16384/c32768.
# Gates: chunks <= 80 (r216b law), errs 0, MX cells native telemetry
# with splits logged; ctrl within +5 pct of the r230 anchors
# (40.4/41.1/42.6); m4q within +5 pct of the r230+r231de anchors
# (43.4/45.2/48.5/52.1/58.1); m8q/mkq long faces within +5 pct of
# their r230 first-rows (50.9/55.3/64.4 resp. 50.0/53.9/61.9) --
# shallow m8q/mkq rows are FIRST measurements, banked ungated.
# Bank -> /root/build/p647/r234/
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_MQA=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r234
CNAME=arc-r234
PROBE=$R/p139_conc.py

fail() { echo "R234_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert28_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
read -r SO_A SO_M <<< "$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; vp=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/')[0]; print(hashlib.md5(open(vp+'libattn_kernels_xe_2.so','rb').read()).hexdigest(), hashlib.md5(open(glob.glob(vp+'libmqa_logits*.so')[0],'rb').read()).hexdigest())\"" 2>/dev/null)" || true
[ "${SO_A:-}" = "$SO_ATTN" ] || { fail "image libattn md5 ${SO_A:-?} != build28 pin"; exit 1; }
[ "${SO_M:-}" = "$SO_MQA" ] || { fail "image libmqa md5 ${SO_M:-?} != pin"; exit 1; }
[ -f "$PROBE" ] || { fail "probe harness missing ($PROBE)"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R234 refuses passed (fork_sha=$FORK_SHA libattn=$SO_A libmqa=$SO_M)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "R234: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  echo "R234: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r234.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R234: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R234: WARN W14 not 200 after restore poll (last=$c)"
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

group() {  # ctx conc  (r230 verbatim)
  local CTX=$1 NC=$2
  local TAG="c${1}_n${2}"
  local GD="$CD/$TAG"
  local MT; [ "$NC" = 1 ] && MT=192 || MT=128
  mkdir -p "$GD"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_before.txt" || true
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" qwen3.8-27b-fp8 0.0
  local cap=$(( 300 + CTX / 8192 * 90 + 240 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec $CNAME test -f /root/p139_conc_$TAG.json && break
    docker inspect $CNAME >/dev/null 2>&1 || break
    sleep 4; i=$((i+1))
  done
  if ! docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s)"; echo "TIMEOUT" > "$GD/outcome.txt"; return 0
  fi
  docker cp $CNAME:/root/p139_conc_$TAG.json "$GD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG"
}

run_cell() {  # tag kvdtype faces...
  local CELL=$1 KVD="$2"; shift 2
  local FACES="$*"
  CD=$P/$CELL
  mkdir -p "$CD"
  local KVDENV=()
  [ -n "$KVD" ] && KVDENV=(ARC_KVDTYPE="$KVD" ARC_MXKV_NATIVE=1)
  echo "=== [r234:$CELL] boot kvd=${KVD:-fp8_e4m3(default)} faces=$FACES $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 \
    ${KVDENV+"${KVDENV[@]}"} \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R234_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r234_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R234_PROBE_CP_FAIL_$CELL"; return 0; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  local CTX
  for CTX in $FACES; do
    group "$CTX" 1
  done
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME grep "MXKV native call" /root/serve_full.log 2>/dev/null \
    > "$CD/telemetry.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1 || true
  echo "=== [r234:$CELL] done $(date +%H:%M:%S) ==="
}

# ---- 1. arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5
mkdir -p "$P"

# ---- 2. cells (long faces for all; shallow faces on the MX lanes) ----
run_cell ctrl "" 8192 16384 32768
run_cell m4q mxfp4 2048 4096 8192 16384 32768
run_cell m8q mxfp8 2048 4096 8192 16384 32768
run_cell mkq mxk8v4 2048 4096 8192 16384 32768

# ---- 3. report + gate (recomputed from banked JSONs) ----
echo "=== [r234] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, re, sys
P = sys.argv[1]
# anchors: r230 cert27 long-ctx rows + the r231de build28 shallow rows
ANCH = {
    "ctrl": {8192: 40.4, 16384: 41.1, 32768: 42.6},
    "m4q": {2048: 43.4, 4096: 45.2, 8192: 48.5, 16384: 52.1, 32768: 58.1},
    "m8q": {8192: 50.9, 16384: 55.3, 32768: 64.4},
    "mkq": {8192: 50.0, 16384: 53.9, 32768: 61.9},
}
BAND = 1.05  # one-sided regression band (oracle-band law)

def cell(name, ctx):
    f = os.path.join(P, name, "c%d_n1" % ctx, "p139_conc_c%d_n1.json" % ctx)
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    rs = [r for r in j["requests"] if r["error"] is None]
    if not rs:
        return None
    itl = sum(r["itl_ms_p50"] for r in rs) / len(rs)
    chunks = sum(r["n_stream_chunks"] for r in rs) / len(rs)
    tps = j.get("per_user_tps")
    tps = tps[0] if isinstance(tps, list) and tps else None
    errs = len([r for r in j["requests"] if r["error"] is not None])
    return itl, chunks, tps, errs

ok = True
print("r234 cert28 perf re-discrimination -- qwen3.8-27b-fp8 n1 MT=192")
print(f" {'cell':>5} | {'ctx':>6} | {'itl_ms':>7} | {'anchor':>7} | {'ratio':>6} | {'tps':>6} | {'chunks':>7} | qual | errs")
rows = {}
for name in ("ctrl", "m4q", "m8q", "mkq"):
    for ctx in (2048, 4096, 8192, 16384, 32768):
        if ctx not in ANCH[name] and name == "ctrl":
            continue  # ctrl has no shallow anchors (fp8 lane untouched by the clamp)
        r = cell(name, ctx)
        if r is None:
            if ctx in ANCH[name]:
                print(f" {name:>5} | {ctx:>6} | MISSING (gated face)")
                ok = False
            else:
                print(f" {name:>5} | {ctx:>6} | MISSING (ungated first-row)")
            continue
        itl, chunks, tps, errs = r
        rows.setdefault(name, {})[ctx] = itl
        anchor = ANCH[name].get(ctx)
        ratio = f"{itl / anchor:.3f}" if anchor else "-"
        anchor_s = f"{anchor:.1f}" if anchor else "-"
        qual = "OK" if chunks <= 80 else "INVALID_FOR_SPEED"
        if chunks > 80 or errs:
            ok = False
        print(f" {name:>5} | {ctx:>6} | {itl:7.1f} | {anchor_s:>7} | {ratio:>6} | {tps if tps else 0:6.1f} | {chunks:7.1f} | {qual} | {errs}")

# gates: every anchored face within +5 pct (or better)
for name, rec in ANCH.items():
    for ctx, anchor in rec.items():
        got = rows.get(name, {}).get(ctx)
        if got is None or got > anchor * BAND:
            print(f"GATE_REGRESSION {name} c{ctx}: itl {got if got else -1:.1f} > anchor {anchor:.1f} x {BAND}")
            ok = False
# gates: MX cells actually ran native with splits telemetry
for name in ("m4q", "m8q", "mkq"):
    tf = os.path.join(P, name, "telemetry.txt")
    nl = sum(1 for _ in open(tf)) if os.path.exists(tf) else 0
    sp = None
    if os.path.exists(tf):
        vals = set()
        for line in open(tf):
            m = re.search(r"splits=(\d+|auto)", line)
            if m:
                vals.add(m.group(1))
        sp = sorted(vals) if vals else None
    if nl < 1 or not sp:
        print(f"GATE_NO_NATIVE {name}: telemetry lines={nl} splits={sp}")
        ok = False
    else:
        print(f"native {name}: lines={nl} splits={sp}")

if ok:
    print("R234_GATE: PASS (cert28 record re-discriminated; shallow rows banked)")
else:
    print("R234_GATE: FAIL")
PY
echo "R234_DONE $(date +%H:%M:%S)"
# ---- 4. W14 restore via the EXIT trap ----
exit 0
