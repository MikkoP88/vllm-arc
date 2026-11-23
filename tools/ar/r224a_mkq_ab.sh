#!/bin/bash
# r224a_mkq_ab.sh — P6.47 r224 leg 0: in-vivo native-vs-Triton A/B on the
# refused mxk8v4 lane (the cert26 battery's mkq native REP-LOOP refusal).
#
# WHAT THIS ANSWERS: does the native kernel diverge from the certified
# Triton v2 decoder ON THE ENGINE'S OWN TENSORS — same q_rows, same pool
# view (with its REAL shape/strides), same bt/sl rows? The r220c offline
# value gate built a contiguous synthetic pool and could not see an
# engine-stride/offset defect (kv-store separate-strides law); this leg
# captures the divergence at the exact call site and banks the payload
# (pool window + inputs + both outputs) for offline bisection.
#
# HOW (diagnostic boot, one arc window):
#   1. W14 down (docker stop radiance-test — arc window law).
#   2. Boot cert26 with the REFUSAL posture (ARC_KVDTYPE=mxk8v4
#      ARC_MXKV_NATIVE=1, LMC auto-off) — the certified serve script.
#   3. Capture the running serve's exact argv+environ from /proc BEFORE
#      killing it, kill it, docker cp the instrumented engine.py (the
#      fork tree's mx_kv/engine.py with the ARC_MXKV_AB block, md5
#      verified at ship) over the baked overlay, rm the module pycache.
#   4. Re-serve from the captured bytes with exactly two env edits:
#      VLLM_XPU_ENABLE_XPU_GRAPH 1->0 (the AB's D2H payload saves are
#      illegal under capture — pageable-H2D law; the block self-guards,
#      eager lets it fire) + ARC_MXKV_AB=1 (budget 3 payloads).
#      DEVIATION NOTE: graph-off is a diagnosis-only posture; no perf
#      claim is made from this leg (value defect, T1_SEED-deterministic
#      per the refusal evidence — eager-reproducible by construction).
#   5. One small generation (fires the first AB payloads at small ctx),
#      one long essay-class prompt (~4K tokens; the refusal's failing
#      class), pull payloads + logs, run the in-container analyzer
#      (r224b_ab_analyze.py) while the container still has its venv.
#   6. Teardown arc container, restore W14 (w14_boot.sh fullres), poll
#      health-200 — the script does not return until W14 is up.
#
# Refuses: image tag present + manifest cert26_fork_sha == $1; shipped
# engine.py + analyzer present with expected md5s.
#
# Usage: r224a_mkq_ab.sh <CERT26_FORK_SHA>
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert26
CNAME=arc-r224
BANK=$R/p647/r224a
ENGSRC=$R/r224a_engine.py
ANALYZER=$R/r224b_ab_analyze.py
FORK_SHA="${1:?usage: r224a_mkq_ab.sh <CERT26_FORK_SHA>}"
OVERLAY=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv

mkdir -p "$BANK"
cd "$BANK" || exit 2

fail() { echo "R224A_ABORT: $*"; }

# ---- refuses (staged-ctx law) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert26_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
[ -f "$ENGSRC" ] || { fail "instrumented engine.py not shipped ($ENGSRC)"; exit 1; }
[ -f "$ANALYZER" ] || { fail "analyzer not shipped ($ANALYZER)"; exit 1; }
ENGMD5=$(md5sum "$ENGSRC" | cut -d' ' -f1)
ANAMD5=$(md5sum "$ANALYZER" | cut -d' ' -f1)
grep -q "ARC_MXKV_AB" "$ENGSRC" || { fail "engine.py lacks the AB block (wrong file?)"; exit 1; }
echo "R224A refuses passed (fork_sha=$FORK_SHA eng_md5=$ENGMD5 ana_md5=$ANAMD5)"

RESTORE_DONE=0
finish() {
  # guarded fail-safe: even a mid-flight death must not leave the rig on
  # an arc image (overnight law) — restore W14 exactly once.
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R224A: restoring W14 ($(date +%H:%M:%S))"
  nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore.log 2>&1 &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R224A: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R224A: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

# ---- 1. arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
sleep 5

# ---- 2. refusal-posture boot ----
echo "=== boot $IMG mkq native $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME ARC_KVDTYPE=mxk8v4 ARC_MXKV_NATIVE=1 \
  bash $R/serve_arc_b70.sh > "$BANK/boot.txt" 2>&1 &
SEEN=0
for i in $(seq 1 100); do
  sleep 15
  c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
  [ "$c" = "200" ] && { SEEN=1; break; }
  if ! docker inspect "$CNAME" >/dev/null 2>&1; then
    # container gone pre-health: server failed, bank + bail
    break
  fi
done
if [ "$SEEN" != 1 ]; then
  echo "GATE_FAIL_BOOT (last=$c)"
  docker cp $CNAME:/root/serve_full.log "$BANK/serve_bootfail.log" >/dev/null 2>&1 || true
  exit 1
fi
echo "HEALTH_OK first serve $(date +%H:%M:%S)"
docker exec $CNAME grep -c "MXKV native dispatch" /root/serve_full.log > "$BANK/native_dispatch_count.txt" 2>/dev/null || true
docker exec $CNAME grep -E "MXKV native (dispatch|call)" /root/serve_full.log | head -6 > "$BANK/native_markers.txt" 2>/dev/null || true

# ---- 3. capture argv+environ, kill, inject instrument ----
# NOTE: -i is REQUIRED — a non-interactive docker exec never passes the
# heredoc stdin and python - executes nothing (first-attempt failure).
docker exec -i $CNAME /opt/venv/bin/python - <<'PYEOF' > "$BANK/proc_capture.txt" 2>&1
import os
import shlex
cands = []
for p in sorted(int(x) for x in os.listdir("/proc") if x.isdigit()):
    try:
        with open(f"/proc/{p}/cmdline", "rb") as f:
            argv = f.read().split(b"\0")[:-1]
    except OSError:
        continue
    joined = b" ".join(argv)
    if b"vllm" in joined or b"VLLM::" in joined:
        cands.append((p, joined))
if not cands:
    print("NO_SERVE_PROC cands=[]")
    raise SystemExit(1)
master = None
for p, joined in cands:
    if b"serve" in joined and b"--kv-cache-dtype" in joined:
        master = p
        break
if master is None:
    # filter on the FULL joined bytes; truncate only in the diagnostic
    # print (a truncated filter missed --kv-cache-dtype on attempt 2)
    print(f"NO_MASTER cands={[(p, j[:160]) for p, j in cands]!r}")
    raise SystemExit(1)
with open(f"/proc/{master}/cmdline", "rb") as f:
    argv = f.read().split(b"\0")[:-1]
env = {}
with open(f"/proc/{master}/environ", "rb") as f:
    for kv in f.read().split(b"\0")[:-1]:
        k, _, v = kv.partition(b"=")
        env[k] = v
env[b"VLLM_XPU_ENABLE_XPU_GRAPH"] = b"0"
env[b"ARC_MXKV_AB"] = b"1"
# r224a-2: attempt-1's whole budget (64) was eaten by the eager dummy
# executions (rows=152/4 rounds, sl<=101) before any real decode ran.
# MINSL=1000 skips dummy + small gen; 60 calls = ~60 layer-calls of the
# first real essay steps (51 layers/step) — all banked with bt-indexed
# pool windows.
env[b"ARC_MXKV_AB_N"] = b"60"
env[b"ARC_MXKV_AB_MINSL"] = b"1000"
with open("/tmp/r224_serve.sh", "wb") as f:
    for k, v in sorted(env.items()):
        f.write(b"export %s=%s\n" % (k, shlex.quote(v.decode()).encode()))
    f.write(b"exec %s >> /root/serve_full.log 2>&1\n"
            % b" ".join(shlex.quote(a.decode()).encode() for a in argv))
print(f"CAPTURED pid={master} argc={len(argv)} env={len(env)} "
      f"cands={len(cands)} -> /tmp/r224_serve.sh")
PYEOF
grep -q "CAPTURED pid=" "$BANK/proc_capture.txt" || { echo "GATE_FAIL_PROCCAPTURE"; cat "$BANK/proc_capture.txt"; exit 1; }
cat "$BANK/proc_capture.txt"

# kill by the CAPTURED pid (kill-by-PID law), then a -9 sweep for stragglers
MPID=$(grep -o "pid=[0-9]*" "$BANK/proc_capture.txt" | head -1 | cut -d= -f2)
docker exec $CNAME kill "$MPID" 2>/dev/null || true
sleep 8
docker exec $CNAME bash -c "pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f 'VLLM::' 2>/dev/null" || true
sleep 4
docker cp "$ENGSRC" "$CNAME:$OVERLAY/engine.py" || { echo "GATE_FAIL_CP_ENGINE"; exit 1; }
docker exec $CNAME bash -c "rm -rf $OVERLAY/__pycache__" || true
docker exec $CNAME bash -c "md5sum $OVERLAY/engine.py" > "$BANK/injected_md5.txt"
grep -q "$ENGMD5" "$BANK/injected_md5.txt" || { echo "GATE_FAIL_MD5_MISMATCH"; cat "$BANK/injected_md5.txt"; exit 1; }
echo "INSTRUMENT_INJECTED md5=$ENGMD5"

# ---- 4. re-serve (eager + AB) ----
docker exec -d $CNAME bash /tmp/r224_serve.sh
SEEN=0
for i in $(seq 1 100); do
  sleep 15
  c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
  [ "$c" = "200" ] && { SEEN=1; break; }
done
[ "$SEEN" = 1 ] || { echo "GATE_FAIL_REBOOT (last=$c)"; exit 1; }
echo "HEALTH_OK re-serve (eager, AB armed) $(date +%H:%M:%S)"
sleep 5
docker exec $CNAME grep -c "ARC_MXKV_AB\|MXKV AB" /root/serve_full.log > "$BANK/ab_armed_count.txt" 2>/dev/null || true

# ---- 5. traffic: small gen, then essay-class long prompt ----
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup r224a","max_tokens":16,"temperature":0}' \
  > "$BANK/gen_small.json" || true
sleep 5
python3 - <<'PYEOF' > "$BANK/essay_prompt.json"
import json
base = ("The archive room held rows of ledgers, each spine numbered by hand, and "
        "every evening the archivist walked the aisles recording which volumes had "
        "been consulted that day, a ritual older than the building itself. ")
prompt = {"model": "qwen3.8-27b-fp8", "prompt": base * 130, "max_tokens": 96,
          "temperature": 0}
print(json.dumps(prompt))
PYEOF
curl -s -m 300 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d @"$BANK/essay_prompt.json" > "$BANK/gen_essay.json" || true
sleep 10

# ---- 6. harvest + in-container analysis ----
docker exec $CNAME sh -c 'ls -la /tmp/mxkv_ab.*.pt 2>/dev/null' > "$BANK/payload_inventory.txt" || true
cat "$BANK/payload_inventory.txt"
docker exec $CNAME grep -E "MXKV AB" /root/serve_full.log > "$BANK/ab_lines.txt" 2>/dev/null || true
echo "=== AB verdict lines ==="; cat "$BANK/ab_lines.txt" || true
docker exec $CNAME sh -c 'wc -l /root/serve_full.log' > "$BANK/serve_log_wc.txt" || true
docker cp $CNAME:/root/serve_full.log "$BANK/serve.log" >/dev/null 2>&1 || true
# the analyzer must run INSIDE the container (its venv has torch; the
# rig host does not) — ship it in first (attempt-1 passed the RIG path
# to docker exec, where it does not exist)
docker cp "$ANALYZER" "$CNAME:/tmp/r224b_ab_analyze.py" >/dev/null 2>&1 || true
docker exec $CNAME /opt/venv/bin/python /tmp/r224b_ab_analyze.py /tmp > "$BANK/r224b_report.txt" 2>&1 || true
echo "=== r224b analysis ==="; cat "$BANK/r224b_report.txt" || true
for f in $(docker exec $CNAME sh -c 'ls /tmp/mxkv_ab.*.pt 2>/dev/null'); do
  docker cp "$CNAME:$f" "$BANK/" >/dev/null 2>&1 || true
done

echo "R224A_LEG_DONE $(date +%H:%M:%S) payloads=$(ls "$BANK"/mxkv_ab.*.pt 2>/dev/null | wc -l)"
# ---- 7. teardown + W14 restore via the EXIT trap ----
exit 0
