#!/bin/bash
# r237_n8ss_cert28.sh -- P6.47 the n8 steady-state long-ctx faces on
# cert28. The n1 record (r234/r235) and the agg8 record (r236 mining)
# are re-discriminated on the clamp wheel; the last un-re-measured
# face class is n8 steady-state long-ctx (last measured cert25zb-era
# S1 ladder + B1: m4q 1703 ss @32K post-v5-flip, m8q 1743, mkq 1824,
# ctrl 1671 -- pre-cert27-fixes, pre-clamp). This leg closes it.
#
# Cells (one boot each, the certified posture: ARC_GRAPH=1 spec k3
# fp8 weights; wheel BAKED in cert28): ctrl/m4q/m8q/mkq, n8 MT=128
# at c16384/c32768. Metric: per-slot steady-state ITL =
# (total_s - ttft_s) * 1000 / (n_stream_chunks - 1), MEDIAN across
# clean slots (ramp-interleaved ITL law: aggregate p50s mix prefill
# stalls -- never gated on). Gates: 32K faces one-sided +5 pct vs the
# cert25zb-era anchors; 16K faces are first cert28 rows, banked
# ungated. chunks<=80, errs 0. Bank -> /root/build/p647/r237/
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a
SO_ATTN=61d2f5a365f05084b3dd72c4ae1da3e0
SO_MQA=190e261edc79b3b497d8d8e801e65757
P=$R/p647/r237
CNAME=arc-r237
PROBE=$R/p139_conc.py

fail() { echo "R237_ABORT: $*"; }

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
echo "R237 refuses passed (fork_sha=$FORK_SHA libattn=$SO_A libmqa=$SO_M)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "R237: W14_RESTORE_SKIPPED (arc leg in flight)"; return
  fi
  echo "R237: restoring W14 ($(date +%H:%M:%S))"
  docker start radiance-test >/dev/null 2>&1 || true
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r237.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R237: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R237: WARN W14 not 200 after restore poll (last=$c)"
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

group() {  # ctx conc  (r185 ladder shape: MT=128 at n8, KV-cap guard, +240 cap)
  local CTX=$1 NC=$2
  local TAG="c${1}_n${2}"
  local GD="$CD/$TAG"
  local MT; [ "$NC" = 1 ] && MT=192 || MT=128
  mkdir -p "$GD"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_before.txt" || true
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" qwen3.8-27b-fp8 0.0
  local cap=$(( 300 + CTX / 8192 * 90 + 120 + (NC == 8 ? 240 : 0) ))
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

run_cell() {  # tag kvdtype faces...  (r234 verbatim)
  local CELL=$1 KVD="$2"; shift 2
  local FACES="$*"
  CD=$P/$CELL
  mkdir -p "$CD"
  local KVDENV=()
  [ -n "$KVD" ] && KVDENV=(ARC_KVDTYPE="$KVD" ARC_MXKV_NATIVE=1)
  echo "=== [r237:$CELL] boot kvd=${KVD:-fp8_e4m3(default)} faces=$FACES $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 \
    ${KVDENV+"${KVDENV[@]}"} \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R237_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r237_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R237_PROBE_CP_FAIL_$CELL"; return 0; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  local CTX
  for CTX in $FACES; do
    group "$CTX" 8
  done
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME grep "MXKV native call" /root/serve_full.log 2>/dev/null \
    > "$CD/telemetry.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1 || true
  echo "=== [r237:$CELL] done $(date +%H:%M:%S) ==="
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

# ---- 2. cells: n8 steady-state long faces ----
run_cell ctrl "" 16384 32768
run_cell m4q mxfp4 16384 32768
run_cell m8q mxfp8 16384 32768
run_cell mkq mxk8v4 16384 32768

# ---- 3. report + gate (recomputed from banked JSONs) ----
echo "=== [r237] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, sys
P = sys.argv[1]
# anchors: cert25zb-era n8 ss medians @32K (S1 ladder r185 + B1 v5-flip
# r188 for m4q; m8/mkq per-format defaults already v5-class at G8).
# 16K faces are FIRST cert28 n8-ss rows, banked ungated.
ANCH = {
    "ctrl": {32768: 1671.0},
    "m4q":  {32768: 1703.0},
    "m8q":  {32768: 1743.0},
    "mkq":  {32768: 1824.0},
}
BAND = 1.05  # one-sided regression band (oracle-band law)

def cell(name, ctx):
    f = os.path.join(P, name, "c%d_n8" % ctx, "p139_conc_c%d_n8.json" % ctx)
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    rs = [r for r in j["requests"] if r["error"] is None]
    if not rs:
        return None
    ss = sorted((r["total_s"] - r["ttft_s"]) * 1000.0 / (r["n_stream_chunks"] - 1)
                for r in rs)
    med = ss[len(ss) // 2]
    agg = sum(r["itl_ms_p50"] for r in rs) / len(rs)  # reference only
    chunks = sum(r["n_stream_chunks"] for r in rs) / len(rs)
    errs = len([r for r in j["requests"] if r["error"] is not None])
    return med, agg, chunks, errs

ok = True
print("r237 n8 steady-state long faces -- qwen3.8-27b-fp8 n8 MT=128 (cert28)")
print(" ss = per-slot (total-ttft)*1000/(chunks-1), median across slots (ramp-interleaved law)")
print(f" {'cell':>5} | {'ctx':>6} | {'ss_ms':>7} | {'anchor':>7} | {'ratio':>6} | {'agg_p50':>8} | {'chunks':>7} | qual | errs")
rows = {}
for name in ("ctrl", "m4q", "m8q", "mkq"):
    for ctx in (16384, 32768):
        r = cell(name, ctx)
        if r is None:
            print(f" {name:>5} | {ctx:>6} | MISSING")
            if ctx in ANCH[name]:
                ok = False
            continue
        med, agg, chunks, errs = r
        rows.setdefault(name, {})[ctx] = med
        anchor = ANCH[name].get(ctx)
        ratio = f"{med / anchor:.3f}" if anchor else "-"
        anchor_s = f"{anchor:.0f}" if anchor else "-"
        qual = "OK" if chunks <= 80 else "INVALID_FOR_SPEED"
        if chunks > 80 or errs:
            ok = False
        print(f" {name:>5} | {ctx:>6} | {med:7.1f} | {anchor_s:>7} | {ratio:>6} | {agg:8.1f} | {chunks:7.1f} | {qual} | {errs}")

for name, rec in ANCH.items():
    for ctx, anchor in rec.items():
        got = rows.get(name, {}).get(ctx)
        if got is None or got > anchor * BAND:
            print(f"GATE_REGRESSION {name} c{ctx} n8ss: {got if got else -1:.1f} > anchor {anchor:.0f} x {BAND}")
            ok = False
# lane-vs-ctrl ss ratios at 32K (the superiority face)
if all(32768 in rows.get(n, {}) for n in ("ctrl", "m4q", "m8q", "mkq")):
    c = rows["ctrl"][32768]
    for n in ("m4q", "m8q", "mkq"):
        print(f"ss_ratio {n}/ctrl @32K = {rows[n][32768] / c:.3f}")

if ok:
    print("R237_GATE: PASS (n8 ss faces banked; 32K anchors within band)")
else:
    print("R237_GATE: FAIL")
PY
echo "R237_DONE $(date +%H:%M:%S)"
# ---- 4. W14 restore via the EXIT trap ----
exit 0
