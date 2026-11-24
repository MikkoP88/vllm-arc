#!/bin/bash
# r235_ctrl_shallow.sh -- P6.47 the fp8-ctrl shallow record rows. The
# r234 record banked first shallow rows for the three MX lanes but the
# ctrl cell ran long faces only, so every MX-vs-fp8 shallow comparison
# is cross-context ("MX 2K within 8 pct of fp8 8K"). This leg fills the
# gap: ONE ctrl boot, the two first shallow faces (c2048/c4096, n1
# MT=192) plus the c8192 face as the same-boot sanity anchor (gate:
# within +5 pct of the r230/r234 anchor 40.4). Measurement only -- no
# engine change, no cert cut (r230 precedent). Derived from the
# certified r234 harness (boot/group/report bytes verbatim where
# possible).
# Bank -> /root/build/p647/r235/
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_MQA=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r235
CNAME=arc-r235
PROBE=$R/p139_conc.py

fail() { echo "R235_ABORT: $*"; }

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
echo "R235 refuses passed (fork_sha=$FORK_SHA libattn=$SO_A libmqa=$SO_M)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "R235: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  echo "R235: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r235.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R235: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R235: WARN W14 not 200 after restore poll (last=$c)"
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

group() {  # ctx conc  (r234 verbatim)
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

run_cell() {  # tag kvdtype faces...  (r234 verbatim, ctrl path)
  local CELL=$1 KVD="$2"; shift 2
  local FACES="$*"
  CD=$P/$CELL
  mkdir -p "$CD"
  local KVDENV=()
  [ -n "$KVD" ] && KVDENV=(ARC_KVDTYPE="$KVD" ARC_MXKV_NATIVE=1)
  echo "=== [r235:$CELL] boot kvd=${KVD:-fp8_e4m3(default)} faces=$FACES $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 \
    ${KVDENV+"${KVDENV[@]}"} \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R235_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r235_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R235_PROBE_CP_FAIL_$CELL"; return 0; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  local CTX
  for CTX in $FACES; do
    group "$CTX" 1
  done
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1 || true
  echo "=== [r235:$CELL] done $(date +%H:%M:%S) ==="
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

# ---- 2. the ctrl cell: 2 shallow first-rows + the c8192 sanity anchor ----
run_cell ctrl "" 2048 4096 8192

# ---- 3. report + gate (recomputed from banked JSONs) ----
echo "=== [r235] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, sys
P = sys.argv[1]
# anchor: the r230/r234 ctrl c8192 row (same-boot sanity gate);
# c2048/c4096 are FIRST ctrl shallow rows, banked ungated
ANCH = {8192: 40.4}
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
print("r235 ctrl shallow rows -- qwen3.8-27b-fp8 n1 MT=192 (fp8 ctrl, cert28)")
print(f" {'cell':>5} | {'ctx':>6} | {'itl_ms':>7} | {'anchor':>7} | {'ratio':>6} | {'tps':>6} | {'chunks':>7} | qual | errs")
rows = {}
for ctx in (2048, 4096, 8192):
    r = cell("ctrl", ctx)
    if r is None:
        print(f" {'ctrl':>5} | {ctx:>6} | MISSING")
        if ctx in ANCH:
            ok = False
        continue
    itl, chunks, tps, errs = r
    rows[ctx] = itl
    anchor = ANCH.get(ctx)
    ratio = f"{itl / anchor:.3f}" if anchor else "-"
    anchor_s = f"{anchor:.1f}" if anchor else "-"
    qual = "OK" if chunks <= 80 else "INVALID_FOR_SPEED"
    if chunks > 80 or errs:
        ok = False
    print(f" {'ctrl':>5} | {ctx:>6} | {itl:7.1f} | {anchor_s:>7} | {ratio:>6} | {tps if tps else 0:6.1f} | {chunks:7.1f} | {qual} | {errs}")

for ctx, anchor in ANCH.items():
    got = rows.get(ctx)
    if got is None or got > anchor * BAND:
        print(f"GATE_REGRESSION ctrl c{ctx}: itl {got if got else -1:.1f} > anchor {anchor:.1f} x {BAND}")
        ok = False

if ok:
    print("R235_GATE: PASS (ctrl shallow rows banked; c8192 sanity within band)")
else:
    print("R235_GATE: FAIL")
PY
echo "R235_DONE $(date +%H:%M:%S)"
# ---- 4. W14 restore via the EXIT trap ----
exit 0
