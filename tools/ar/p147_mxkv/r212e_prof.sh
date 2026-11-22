#!/bin/bash
# r212e_prof.sh -- arc7 Phase-8 r212e: IN-SITU kernel telemetry. The
# r212d-diag ladder proved the kernel at the ENGINE'S EXACT offline
# call shape runs 1271.9 us (1.00x base, canary clean) while the e2e
# native step pays ~53 ms/layer -- the 40x lives in HOW the engine
# invokes the op inside the full-model graph, not in the op. This leg
# brackets ONE c16384 n1 generation with the torch profiler on the
# native posture and answers: is the mx kernel slow in situ (device
# time per call), or is the time between kernels (host/graph)?
#
# Method = the proven r67b recipe (results/2026-10-31-p67-r67a-
# compile-ab/p67_gate2_prof.sh): profiler routed by serve flags
#   --profiler-config.profiler=torch
#   --profiler-config.torch_profiler_dir=/root/prof
#   --profiler-config.torch_profiler_with_stack=false
# + a one-line bind-mounted fix of core
#   vllm/v1/worker/gpu_worker.py activities ["CPU","CUDA"] ->
#   ["CPU","XPU"] (0.30 hardcodes CUDA; the XPU key exists in
#   profiler/wrapper.py TorchProfilerActivityMap). /start_profile
#   takes no body when the dir comes from flags.
# Parse ON THE RIG (traces are per-rank .pt.trace.json.gz, kineto
# cat=kernel events): per-kernel-name totals+counts+max, the SYCL-
# compat KernelFunctor bucket (FA2 hides there per r67b; the mx
# kernel may too), graph-launch cpu_op visibility (XPU-graph replay
# kernels may be invisible -- then the launch-op wall is the floor),
# and the MXKV telemetry lines for the profiled window.
# Bank -> /root/build/p647/r212e/
set -uo pipefail
R=/root/build
P=$R/p647/r212e
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r212e
PROBE=$R/p139_conc.py
GW=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R212E_NO_PROBE"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R212E_NO_OVERLAY"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212E_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212E_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212E_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212E_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212e.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "STALE_CONTAINER $CNAME"; return 1
}

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

run_face() {  # CTX MT NC TAG
  local CTX=$1 MT=$2 NC=$3 TAG=$4
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" qwen3.8-27b-fp8 0.0
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec $CNAME test -f /root/p139_conc_$TAG.json && break
    docker inspect $CNAME >/dev/null 2>&1 || break
    sleep 4; i=$((i+1))
  done
  if docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    docker cp $CNAME:/root/p139_conc_$TAG.json "$P/$TAG.json" >/dev/null 2>&1
    echo "FACE_${TAG}_DONE"
  else
    echo "FACE_${TAG}_TIMEOUT"
  fi
}

mkdir -p "$P"

echo "=== [r212e] gpu_worker activities CUDA->XPU (r67b recipe) ==="
docker run --rm --entrypoint cat "$IMG" $GW > "$P/gpu_worker_orig.py"
N=$(grep -c 'activities=\["CPU", "CUDA"\],' "$P/gpu_worker_orig.py")
[ "$N" = "1" ] || { echo "R212E_ANCHOR_COUNT activities=$N (expected 1)"; exit 1; }
sed -i 's|activities=\["CPU", "CUDA"\],|activities=["CPU", "XPU"],|' "$P/gpu_worker_orig.py"
mv "$P/gpu_worker_orig.py" "$P/gpu_worker_xpu.py"
grep -q 'activities=\["CPU", "XPU"\],' "$P/gpu_worker_xpu.py" || { echo "R212E_SED_FAIL"; exit 1; }
echo "gpu_worker_xpu.py ready (1-line activities patch)"

SRV_PR=$R/serve_arc_b70_r212e.sh
python3 - "$R/serve_arc_b70.sh" "$SRV_PR" "$P/gpu_worker_xpu.py" "$GW" <<'PYEOF'
import sys
src, dst, patched, gwc = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = open(src).read()
old = "  $COMPILATION_FLAG \\\n"
new = ("  $COMPILATION_FLAG "
       "--profiler-config.profiler=torch "
       "--profiler-config.torch_profiler_dir=/root/prof "
       "--profiler-config.torch_profiler_with_stack=false \\\n")
assert s.count(old) == 1, f"COMPILATION_FLAG usage count={s.count(old)}"
s = s.replace(old, new)
old2 = '    --entrypoint /bin/bash "$ARC_IMAGE" -c \'sleep infinity\'\n'
new2 = f'    -v {patched}:{gwc}:ro \\\n' + old2
assert s.count(old2) == 1, f"entrypoint line count={s.count(old2)}"
s = s.replace(old2, new2)
open(dst, "w").write(s)
print("serve r212e variant written (profiler flags + activities mount)")
PYEOF
grep -q "profiler-config.profiler=torch" "$SRV_PR" || { echo "R212E_FLAG_EDIT_FAIL"; exit 1; }
grep -q "gpu_worker_xpu.py" "$SRV_PR" || { echo "R212E_MOUNT_EDIT_FAIL"; exit 1; }

echo "=== [r212e] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

echo "=== [r212e] boot native posture + profiler $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
  ARC_MXKV_NATIVE=1 \
  ARC_MXKV_OVERLAY=$OVERLAY \
  ARC_MXKV_WHEEL_DIR=$WHEELDIR \
  ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
  bash "$SRV_PR" > "$P/boot.txt" 2>&1 &
if ! boot_wait; then
  echo "R212E_BOOT_FAIL"
  docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  exit 1
fi
echo "HEALTH_OK_r212e $(date +%H:%M:%S)"
docker exec $CNAME grep -c 'activities=\["CPU", "XPU"\],' $GW \
  || { echo "R212E_MOUNT_NOT_EFFECTIVE"; exit 1; }
docker cp "$PROBE" $CNAME:/root/p139_conc.py || { echo "R212E_PROBE_CP_FAIL"; exit 1; }
docker exec $CNAME mkdir -p /root/prof

echo "=== [r212e] warm face c16384 n1 (unprofiled; graphs+cache warm) $(date +%H:%M:%S) ==="
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
sleep 3
run_face 16384 192 1 c16384_n1w

echo "=== [r212e] profile bracket $(date +%H:%M:%S) ==="
curl -s -m 30 -X POST http://127.0.0.1:8000/start_profile | head -c 200
echo " <- start_profile"
sleep 2
run_face 16384 192 1 c16384_n1p
curl -s -m 60 -X POST http://127.0.0.1:8000/stop_profile | head -c 200
echo " <- stop_profile"

echo "=== [r212e] wait trace flush $(date +%H:%M:%S) ==="
FLUSHED=0
for i in $(seq 1 36); do
  sleep 10
  C=$(docker exec $CNAME sh -c "ls /root/prof/*.gz 2>/dev/null | wc -l")
  echo "poll$i gz=$C"
  if [ "$C" -ge 2 ]; then sleep 5; FLUSHED=1; break; fi
done
docker exec $CNAME sh -c 'ls -la /root/prof/' | tee "$P/prof_ls.txt"
[ "$FLUSHED" = "1" ] || echo "R212E_TRACE_FLUSH_INCOMPLETE (continuing with what flushed)"

echo "=== [r212e] trace summary (on-rig parse) $(date +%H:%M:%S) ==="
docker exec $CNAME /opt/venv/bin/python - <<'PY' 2>&1 | tee "$P/summary.txt"
import collections
import glob
import gzip
import json

files = sorted(glob.glob("/root/prof/*.gz"))
print("trace files:", files)
MX_PAT = ("mx", "mxkv", "paged", "decode")
for f in files:
    try:
        with gzip.open(f, "rt") as fh:
            d = json.load(fh)
    except Exception as e:
        print(f, "PARSE_FAIL", repr(e)[:120]); continue
    ev = d.get("traceEvents", [])
    if not ev:
        print(f, "NO_EVENTS"); continue
    kern = collections.defaultdict(lambda: [0, 0.0, 0])
    cops = collections.defaultdict(lambda: [0, 0.0])
    tmin = None; tmax = None
    for e in ev:
        if e.get("ph") != "X":
            continue
        dur = float(e.get("dur", 0))
        ts = e.get("ts")
        if isinstance(ts, (int, float)):
            tmin = ts if tmin is None else min(tmin, ts)
            tmax = (ts + dur) if tmax is None else max(tmax, ts + dur)
        cat = e.get("cat", "")
        if cat == "kernel":
            a = kern[e.get("name", "?")]
            a[0] += 1; a[1] += dur
            if dur > a[2]: a[2] = dur
        elif cat == "cpu_op":
            a = cops[e.get("name", "?")]
            a[0] += 1; a[1] += dur
    tot = sum(v[1] for v in kern.values()) or 1.0
    span = ((tmax - tmin) / 1e6) if isinstance(tmin, (int, float)) else -1
    print(f"\n== {f.rsplit('/',1)[1]} == kernel events "
          f"{sum(v[0] for v in kern.values())}, device total "
          f"{tot/1e3:.1f} ms, span {span:.2f} s")
    kf = [(k, v) for k, v in kern.items()]
    kf.sort(key=lambda kv: -kv[1][1])
    for name, (n, us, mx) in kf[:25]:
        low = name.lower()
        tag = "MXDECODE" if all(p in low for p in ("mx", "decode")) else (
              "MX-ish" if any(p in low for p in MX_PAT) else "")
        print(f"  {us/1e3:9.2f} ms n={n:6d} max={mx:9.1f}us {tag:<8} {name[:100]}")
    kfam = sum(v[1] for k, v in kern.items()
               if "KernelFunctor" in k)
    print(f"  [SYCL-compat bucket] compat::experimental::detail::KernelFunctor "
          f"total {kfam/1e3:.2f} ms")
    cf = sorted(cops.items(), key=lambda kv: -kv[1][1])[:8]
    print("  [cpu_op top]")
    for name, (n, us) in cf:
        print(f"  {us/1e3:9.2f} ms n={n:6d} {name[:100]}")
PY

docker exec $CNAME grep -E "MXKV native call|MXKV native dispatch" /root/serve_full.log 2>/dev/null \
  > "$P/telemetry.txt" || true
docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
mkdir -p "$P/traces"
for gz in $(docker exec $CNAME sh -c 'ls /root/prof/*.gz 2>/dev/null'); do
  bn=$(basename "$gz")
  docker cp "$CNAME:$gz" "$P/traces/$bn" >/dev/null 2>&1 || echo "TRACE_CP_FAIL $bn"
done
docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
echo "R212E_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
