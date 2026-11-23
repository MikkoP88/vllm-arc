#!/bin/bash
# r231d_e2e.sh -- P6.47 r231d e2e face leg: the seqused-tightened
# split clamp on the SERVING engine (m4q = the record lane).
#
# Cells (one boot each, r230 posture exactly: ARC_GRAPH=1 spec k3
# fp8 weights, ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1):
#   m4q25 : cert27 as-is (build25 baked) -- same-day baseline; its
#           c8192/c32768 faces also re-discriminate the r230 record
#           (banked 49.4/58.1 ms) = cross-boot dice bound
#   m4q28a: cert27 + build28 wheel (ARC_MXKV_WHEEL pre-install,
#           in-container libattn md5 verified 61d2f5a3...)
#   m4q28b: second build28 boot -- the A/A spread bound (the
#           single-sample cross-wheel law)
# Faces: p139_conc.py c2048/c4096/c8192/c32768 n1 MT=192. The
# shallow faces are the WIN faces (walked ctx 1792-4032 -> clamp
# 31/63 vs capacity 128); c8192 is the clamp no-op boundary (blocks
# 512 -> S=128) and c32768 the record context -- both PARITY faces.
# Gates: PARITY b28 c8192/c32768 within +5 pct of m4q25 same-day
# (one-sided band law); WIN b28 mean(c2048) < m4q25 c2048 x 0.97;
# chunks <= 80 (r216b), errs 0, native telemetry present.
# Bank -> /root/build/p647/r231d_e2e/
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert27
FORK_SHA=d39ab5a26bf737d006d47a3f3125a404dca6e928
SOMD5_25=2d73ff7a3c2aa2caab86179288ca3711
SOMD5_28=61d2f5a365f05084b3dd72c4ae1da3e0
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=e6d8d337e43b907504537061414128f8
P=$R/p647/r231d_e2e
CNAME=arc-r231de
PROBE=$R/p139_conc.py
CTXS="2048 4096 8192 32768"

fail() { echo "R231DE_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert27_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
SO_MD5=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels*.so'); print(hashlib.md5(open(ps[0],'rb').read()).hexdigest())\"" 2>/dev/null) || SO_MD5=""
[ "$SO_MD5" = "$SOMD5_25" ] || { fail "image .so md5 $SO_MD5 != build25 pin"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" 2>/dev/null | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { fail "wheel md5 $GOTMD5 != build28 pin $WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$PROBE" ] || { fail "probe harness missing ($PROBE)"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  fail "arc leg already in flight"; exit 2
fi
md5sum "$WHEELDIR/$WHEELNAME" > /dev/null
echo "R231DE refuses passed (fork_sha=$FORK_SHA so25=$SO_MD5 wheel=$GOTMD5)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  # own container dies FIRST so the in-flight guard does not match it
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "R231DE: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  echo "R231DE: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r231de.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R231DE: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R231DE: WARN W14 not 200 after restore poll (last=$c)"
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

run_cell() {  # tag wheel(yes|no)
  local CELL=$1 USEWHEEL=$2
  CD=$P/$CELL
  mkdir -p "$CD"
  local WHEELENV=()
  if [ "$USEWHEEL" = "yes" ]; then
    WHEELENV=(ARC_MXKV_WHEEL_DIR="$WHEELDIR" ARC_MXKV_WHEEL="/wheels/$WHEELNAME")
  fi
  echo "=== [r231de:$CELL] wheel=$USEWHEEL $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 \
    ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1 \
    ${WHEELENV+"${WHEELENV[@]}"} \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R231DE_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r231de_$CELL $(date +%H:%M:%S)"
  # the cell's wheel posture, verified from the live container
  docker exec "$CNAME" /opt/venv/bin/python -c \
    "import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libattn_kernels*.so'); print('CELL_SO_MD5', hashlib.md5(open(ps[0],'rb').read()).hexdigest())" \
    > "$CD/cell_so_md5.txt" 2>/dev/null || true
  local CELL_SO
  CELL_SO=$(awk '{print $2}' "$CD/cell_so_md5.txt")
  local WANT_SO=$SOMD5_25
  [ "$USEWHEEL" = "yes" ] && WANT_SO=$SOMD5_28
  if [ "$CELL_SO" != "$WANT_SO" ]; then
    echo "R231DE_SO_MISMATCH_$CELL got=$CELL_SO want=$WANT_SO"
    echo "SO_MISMATCH" > "$CD/outcome.txt"
    docker rm -f $CNAME >/dev/null 2>&1 || true
    return 0
  fi
  echo "R231DE_SO_OK_$CELL $CELL_SO"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R231DE_PROBE_CP_FAIL_$CELL"; return 0; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  local CTX
  for CTX in $CTXS; do
    group "$CTX" 1
  done
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME grep "MXKV native call" /root/serve_full.log 2>/dev/null \
    > "$CD/telemetry.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1 || true
  echo "=== [r231de:$CELL] done $(date +%H:%M:%S) ==="
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
md5sum "$WHEELDIR/$WHEELNAME" > "$P/wheel_md5.txt"

# ---- 2. cells ----
run_cell m4q25 no
run_cell m4q28a yes
run_cell m4q28b yes

# ---- 3. report + gate (recomputed from banked JSONs) ----
echo "=== [r231de] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, sys
P = sys.argv[1]
# banked r230 cert27 record (same image+env, 2026-11-23): context only
REC = {8192: 49.4, 32768: 58.1}
BAND = 1.05   # one-sided parity band (oracle-band law)
WIN = 0.97    # shallow face must beat baseline by >3 pct

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
print("r231d e2e face leg -- qwen3.8-27b-fp8 mxfp4 native n1 MT=192")
print(f" {'cell':>7} | {'ctx':>6} | {'itl_ms':>7} | {'vs25':>6} | {'tps':>6} | {'chunks':>7} | qual | errs")
rows = {}
for name in ("m4q25", "m4q28a", "m4q28b"):
    for ctx in (2048, 4096, 8192, 32768):
        r = cell(name, ctx)
        if r is None:
            print(f" {name:>7} | {ctx:>6} | MISSING")
            ok = False
            continue
        itl, chunks, tps, errs = r
        rows.setdefault(name, {})[ctx] = itl
        qual = "OK" if chunks <= 80 else "INVALID_FOR_SPEED"
        if chunks > 80 or errs:
            ok = False
        print(f" {name:>7} | {ctx:>6} | {itl:7.1f} | {'-':>6} | {tps if tps else 0:6.1f} | {chunks:7.1f} | {qual} | {errs}")

base = rows.get("m4q25", {})
a = rows.get("m4q28a", {})
b = rows.get("m4q28b", {})
print("-- ratios vs m4q25 (same day, same image env) --")
for ctx in (2048, 4096, 8192, 32768):
    if base.get(ctx) and a.get(ctx):
        aa = f"{(abs(a[ctx]-b[ctx])/((a[ctx]+b[ctx])/2)*100):.1f}%" if b.get(ctx) else "-"
        print(f" c{ctx}: b28a {a[ctx]/base[ctx]:.3f}x  b28b "
              f"{(b[ctx]/base[ctx]) if b.get(ctx) else 0:.3f}x  A/A spread {aa}"
              f"  (banked r230: {REC.get(ctx, '-')} ms)")

# PARITY gates: b28 boots within +5 pct of same-day baseline at the
# no-op/record contexts
for ctx in (8192, 32768):
    for name in ("m4q28a", "m4q28b"):
        got = rows.get(name, {}).get(ctx)
        ref = base.get(ctx)
        if got is None or ref is None or got > ref * BAND:
            print(f"GATE_PARITY_FAIL {name} c{ctx}: {got} vs base {ref} x{BAND}")
            ok = False
        else:
            print(f"GATE_PARITY_OK {name} c{ctx}: {got:.1f} <= {ref:.1f} x{BAND}")
# WIN gates: the MEAN of the two b28 boots beats baseline by >3 pct at
# both shallow faces
for ctx in (2048, 4096):
    got = [rows.get(n, {}).get(ctx) for n in ("m4q28a", "m4q28b")]
    ref = base.get(ctx)
    if None in got or ref is None:
        print(f"GATE_WIN_MISSING c{ctx}")
        ok = False
        continue
    m = sum(got) / 2
    if m < ref * WIN:
        print(f"GATE_WIN_OK c{ctx}: b28 mean {m:.1f} < base {ref:.1f} x{WIN} ({(m/ref-1)*100:.1f} pct)")
    else:
        print(f"GATE_WIN_FAIL c{ctx}: b28 mean {m:.1f} >= base {ref:.1f} x{WIN} ({(m/ref-1)*100:.1f} pct)")
        ok = False
# native telemetry present on every cell
for name in ("m4q25", "m4q28a", "m4q28b"):
    tf = os.path.join(P, name, "telemetry.txt")
    nl = sum(1 for _ in open(tf)) if os.path.exists(tf) else 0
    if nl < 1:
        print(f"GATE_NO_NATIVE {name}: telemetry lines={nl}")
        ok = False
    else:
        print(f"native {name}: lines={nl}")

print(f"R231DE_GATE: {'PASS' if ok else 'FAIL'}")
PY
echo "R231DE_DONE $(date +%H:%M:%S)"
# ---- 4. W14 restore via the EXIT trap ----
exit 0
