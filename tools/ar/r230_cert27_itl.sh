#!/bin/bash
# r230_cert27_itl.sh -- P6.47 r230: the cert27 perf re-discrimination,
# long-ctx ITL faces. The record numbers are cert26-content
# measurements (r219b auto-at-128: mxfp4 48.5/51.2/56.8 ms @8/16/32K;
# fp8 anchors 40.4/41.1/42.5), and the r225 wheel fixes touch the
# split-KV path's precision (fix3 = fp32 intermediate buffers), so the
# faces must be re-run on the shipping cert27 image before the record
# can claim cert27 (improvement-convergence law: the battery ran on
# cert27; the agg8 half of the re-discrimination is already banked in
# results/2026-11-23-p647-cert27-battery/*/battery.txt vs the r223
# bank: ctrl/m8q parity, m4q +2.9 pct, native mkq +7.9 pct vs its
# cert26 Triton lane).
#
# Cells (one boot each, the r219b record posture exactly: ARC_GRAPH=1
# spec k3 fp8 weights; wheel+overlay BAKED in cert27 -- no mounts):
#   ctrl : ARC_KVDTYPE unset (fp8_e4m3) -- the fp8 ITL anchors
#   m4q  : ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1 -- the headline record
#   m8q  : ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1
#   mkq  : ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1 -- first native ITL
#          row for this lane on the fixed wheel
# Faces: p139_conc.py c8192/c16384/c32768 n1 MT=192 (the r219b
# group()). Gates: chunks <= 80 (r216b law), errs 0, MX cells must
# show native telemetry with splits logged; m4q/ctrl within +5 pct of
# the banked anchors or better -> R230_GATE: PASS.
# Bank -> /root/build/p647/r230/{ctrl,m4q,m8q,mkq}
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert27
FORK_SHA=d39ab5a26bf737d006d47a3f3125a404dca6e928
SOMD5=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r230
CNAME=arc-r230
PROBE=$R/p139_conc.py

fail() { echo "R230_ABORT: $*"; }

# ---- refuses ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert27_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
SO_MD5=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libmqa_logits*.so'); print(hashlib.md5(open(ps[0],'rb').read()).hexdigest())\"" 2>/dev/null) || SO_MD5=""
[ "$SO_MD5" = "$SOMD5" ] || { fail "image libmqa_logits .so md5 $SO_MD5 != build25 pin"; exit 1; }
[ -f "$PROBE" ] || { fail "probe harness missing ($PROBE)"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  fail "arc leg already in flight"; exit 2
fi
echo "R230 refuses passed (fork_sha=$FORK_SHA so_md5=$SO_MD5)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "R230: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R230: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r230.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R230: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R230: WARN W14 not 200 after restore poll (last=$c)"
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

group() {  # ctx conc  (r219b verbatim)
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

run_cell() {  # tag kvdtype (empty = ctrl) extra-env...
  local CELL=$1 KVD="$2"; shift 2
  CD=$P/$CELL
  mkdir -p "$CD"
  local KVDENV=()
  [ -n "$KVD" ] && KVDENV=(ARC_KVDTYPE="$KVD" ARC_MXKV_NATIVE=1)
  echo "=== [r230:$CELL] boot kvd=${KVD:-fp8_e4m3(default)} extra=$* $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 \
    ${KVDENV+"${KVDENV[@]}"} "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R230_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r230_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R230_PROBE_CP_FAIL_$CELL"; return 0; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  group 8192 1
  group 16384 1
  group 32768 1
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME grep "MXKV native call" /root/serve_full.log 2>/dev/null \
    > "$CD/telemetry.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1 || true
  echo "=== [r230:$CELL] done $(date +%H:%M:%S) ==="
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

# ---- 2. cells ----
run_cell ctrl ""
run_cell m4q mxfp4
run_cell m8q mxfp8
run_cell mkq mxk8v4

# ---- 3. report + gate (recomputed from banked JSONs) ----
echo "=== [r230] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, re, sys
P = sys.argv[1]
# the banked cert26-content record this leg re-discriminates
REC_M4Q = {8192: 48.5, 16384: 51.2, 32768: 56.8}   # r219b auto/ns128
REC_FP8 = {8192: 40.4, 16384: 41.1, 32768: 42.5}   # fp8 anchors
BAND = 1.05  # +5 pct regression band (oracle-band one-sided law)

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

def native_lines(name):
    tf = os.path.join(P, name, "telemetry.txt")
    return sum(1 for _ in open(tf)) if os.path.exists(tf) else 0

def splits_logged(name):
    tf = os.path.join(P, name, "telemetry.txt")
    if not os.path.exists(tf):
        return None
    vals = set()
    for line in open(tf):
        m = re.search(r"splits=(\d+|auto)", line)
        if m:
            vals.add(m.group(1))
    return sorted(vals) if vals else None

ok = True
print("r230 cert27 long-ctx ITL re-discrimination -- qwen3.8-27b-fp8 n1")
print(f" {'cell':>5} | {'ctx':>6} | {'itl_ms':>7} | {'anchor':>7} | {'ratio':>6} | {'tps':>6} | {'chunks':>7} | qual | errs")
rows = {}
for name in ("ctrl", "m4q", "m8q", "mkq"):
    for ctx in (8192, 16384, 32768):
        r = cell(name, ctx)
        if r is None:
            print(f" {name:>5} | {ctx:>6} | MISSING")
            ok = False
            continue
        itl, chunks, tps, errs = r
        rows.setdefault(name, {})[ctx] = (itl, chunks, tps, errs)
        anchor = REC_FP8[ctx] if name == "ctrl" else (REC_M4Q[ctx] if name == "m4q" else None)
        ratio = f"{itl / anchor:.3f}" if anchor else "-"
        anchor_s = f"{anchor:.1f}" if anchor else "-"
        qual = "OK" if chunks <= 80 else "INVALID_FOR_SPEED"
        if chunks > 80 or errs:
            ok = False
        print(f" {name:>5} | {ctx:>6} | {itl:7.1f} | {anchor_s:>7} | {ratio:>6} | {tps if tps else 0:6.1f} | {chunks:7.1f} | {qual} | {errs}")

# gates: m4q + ctrl within +5 pct of the banked anchors (or better)
for name, rec in (("ctrl", REC_FP8), ("m4q", REC_M4Q)):
    for ctx, anchor in rec.items():
        got = rows.get(name, {}).get(ctx)
        if got is None or got[0] > anchor * BAND:
            print(f"GATE_REGRESSION {name} c{ctx}: itl {got[0] if got else -1:.1f} > anchor {anchor:.1f} x {BAND}")
            ok = False
# gates: MX cells actually ran native with splits telemetry
for name in ("m4q", "m8q", "mkq"):
    nl = native_lines(name)
    sp = splits_logged(name)
    if nl < 1 or not sp:
        print(f"GATE_NO_NATIVE {name}: telemetry lines={nl} splits={sp}")
        ok = False
    else:
        print(f"native {name}: lines={nl} splits={sp}")

if ok:
    print("R230_GATE: PASS (cert27 ITL re-discrimination clean; record updated)")
else:
    print("R230_GATE: FAIL")
PY
echo "R230_DONE $(date +%H:%M:%S)"
# ---- 4. W14 restore via the EXIT trap ----
exit 0
