#!/bin/bash
# r226a_mkq_ab25.sh — P6.47 r226 leg a: e2e re-entry of the FIXED
# mxk8v4 native lane (build25 wheel, r225t-gated md5
# aeca45daa740e9973a1c51bfc5ea2bc4) on the engine's own tensors.
#
# WHAT THIS ANSWERS: r225 closed the refusal offline (payload replay,
# q-cast-matched gate 12/12 PASS). This leg re-runs the r224a in-vivo
# A/B harness with the fixed wheel installed in the cert26 container
# and scores the FRESHLY BANKED payloads at the corrected criterion —
# each lane vs the fp64 walk of its own input (r225r law: the banked q
# is fp16, the oracle eats fp16, the native eats the bf16 cast; the
# raw lane gap ~5e-2 on sensitive payloads is INPUT, not kernel).
#
# HOW (diagnostic boot, one arc window; r224a flow verbatim plus the
# wheel step):
#   1. W14 down (docker stop radiance-test — arc window law).
#   2. Boot cert26 with the REFUSAL posture (ARC_KVDTYPE=mxk8v4
#      ARC_MXKV_NATIVE=1, LMC auto-off) — the certified serve script.
#   3. Capture the running serve's exact argv+environ from /proc,
#      kill it, INSTALL THE BUILD25 WHEEL (pip force-reinstall, md5
#      refuse == the r225t-gated md5), docker cp the instrumented
#      engine.py over the baked overlay, rm the module pycache.
#   4. Re-serve from the captured bytes with the AB env edits
#      (graph-off for the D2H payload saves + ARC_MXKV_AB=1,
#      N=60, MINSL=1000 — the essay-class refusal window).
#   5. One small generation, one long essay-class prompt; harvest
#      payloads + logs; in-container analyzer (lane gap report) AND
#      the r225t q-cast-matched scorer on the first 8 COVERED banked
#      payloads (fp64 walks, ~40 s each) — the leg's verdict is the
#      scorer's.
#   6. Teardown arc container, restore W14 (w14_boot.sh fullres),
#      poll health-200 — the script does not return until W14 is up.
#
# PAYLOAD CLASSES (first-run law): the instrument banks at most 96
# blocks per call; the essay window produces an 8199-token class (129
# blocks needed -> banked window TRUNCATED, not walkable offline) and
# a 5463-token class (86 blocks -> COVERED). r224's bank held the same
# mix; its replayed payloads were the covered class. Payloads are
# docker-cp'd to the bank BEFORE the scoring gate so an abort cannot
# lose them (the first run lost all payloads that way).
#
# Refuses: image tag present + manifest cert26_fork_sha == $1; shipped
# engine.py + analyzer + scorer present; wheel md5 == build25 exactly.
#
# Usage: r226a_mkq_ab25.sh <CERT26_FORK_SHA>
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert26
CNAME=arc-r226
BANK=$R/p647/r226a
ENGSRC=$R/r224a_engine.py
ANALYZER=$R/r224b_ab_analyze.py
SCORER=$R/r225_tools/r225t_replay.py
ISOLIB=$R/r225_tools/r225j_iso.py
WHEELS=$R/wheels-arc-vxk
WANT_MD5=aeca45daa740e9973a1c51bfc5ea2bc4
FORK_SHA="${1:?usage: r226a_mkq_ab25.sh <CERT26_FORK_SHA>}"
OVERLAY=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv

mkdir -p "$BANK"
cd "$BANK" || exit 2

fail() { echo "R226A_ABORT: $*"; }

# ---- refuses (staged-ctx law) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert26_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
[ -f "$ENGSRC" ] || { fail "instrumented engine.py not shipped ($ENGSRC)"; exit 1; }
[ -f "$ANALYZER" ] || { fail "analyzer not shipped ($ANALYZER)"; exit 1; }
[ -f "$SCORER" ] || { fail "scorer not shipped ($SCORER)"; exit 1; }
[ -f "$ISOLIB" ] || { fail "iso lib not shipped ($ISOLIB)"; exit 1; }
ENGMD5=$(md5sum "$ENGSRC" | cut -d' ' -f1)
ANAMD5=$(md5sum "$ANALYZER" | cut -d' ' -f1)
grep -q "ARC_MXKV_AB" "$ENGSRC" || { fail "engine.py lacks the AB block (wrong file?)"; exit 1; }
WHEEL=$(ls -t "$WHEELS"/vllm_xpu_kernels-0.1.14.1+arc7mx*.whl 2>/dev/null | grep -v r210keeper | head -1)
[ -n "$WHEEL" ] || { fail "no arc7mx wheel in $WHEELS"; exit 1; }
WHEEL_MD5=$(md5sum "$WHEEL" | cut -d' ' -f1)
[ "$WHEEL_MD5" = "$WANT_MD5" ] \
  || { fail "wheel md5 $WHEEL_MD5 != r225t-gated build25 $WANT_MD5"; exit 1; }
echo "R226A refuses passed (fork_sha=$FORK_SHA wheel_md5=$WHEEL_MD5 eng_md5=$ENGMD5 ana_md5=$ANAMD5)"

RESTORE_DONE=0
finish() {
  # guarded fail-safe: even a mid-flight death must not leave the rig on
  # an arc image (overnight law) — restore W14 exactly once.
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R226A: restoring W14 ($(date +%H:%M:%S))"
  nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R226A: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R226A: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

# ---- 1. arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
sleep 5

# ---- 2. refusal-posture boot ----
echo "=== boot $IMG mkq native + build25 wheel $(date +%H:%M:%S)) ==="
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

# ---- 3. capture argv+environ, kill, install wheel, inject instrument ----
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

# ---- 3b. THE r226 DELTA: install the gated build25 wheel ----
WHEEL_BASE=$(basename "$WHEEL")
docker cp "$WHEEL" "$CNAME:/tmp/$WHEEL_BASE" || { echo "GATE_FAIL_CP_WHEEL"; exit 1; }
docker exec $CNAME /opt/venv/bin/pip install --force-reinstall --no-deps \
  "/tmp/$WHEEL_BASE" > "$BANK/wheel_install.log" 2>&1 \
  || { fail "pip install rc=$?"; exit 1; }
tail -2 "$BANK/wheel_install.log"
docker exec $CNAME /opt/venv/bin/python -c \
  "import vllm_xpu_kernels, hashlib, pathlib; p=list(pathlib.Path(vllm_xpu_kernels.__file__).parent.rglob('*.so'))[0]; print('SO_MD5', hashlib.md5(p.read_bytes()).hexdigest(), p.name)" \
  > "$BANK/so_md5.txt" 2>&1 || true
cat "$BANK/so_md5.txt" || true

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
echo "HEALTH_OK re-serve (eager, AB armed, build25) $(date +%H:%M:%S)"
sleep 5
docker exec $CNAME grep -c "ARC_MXKV_AB\|MXKV AB" /root/serve_full.log > "$BANK/ab_armed_count.txt" 2>/dev/null || true

# ---- 5. traffic: small gen, then essay-class long prompt ----
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup r226a","max_tokens":16,"temperature":0}' \
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

# ---- 6. harvest + in-container analysis + q-cast-matched scoring ----
docker exec $CNAME sh -c 'ls -la /tmp/mxkv_ab.*.pt 2>/dev/null' > "$BANK/payload_inventory.txt" || true
cat "$BANK/payload_inventory.txt"
docker exec $CNAME grep -E "MXKV AB" /root/serve_full.log > "$BANK/ab_lines.txt" 2>/dev/null || true
echo "=== AB verdict lines ==="; cat "$BANK/ab_lines.txt" || true
docker exec $CNAME sh -c 'wc -l /root/serve_full.log' > "$BANK/serve_log_wc.txt" || true
docker cp $CNAME:/root/serve_full.log "$BANK/serve.log" >/dev/null 2>&1 || true
docker cp "$ANALYZER" "$CNAME:/tmp/r224b_ab_analyze.py" >/dev/null 2>&1 || true
docker exec $CNAME /opt/venv/bin/python /tmp/r224b_ab_analyze.py /tmp > "$BANK/r224b_report.txt" 2>&1 || true
echo "=== r224b lane-gap analysis (raw lanes; q-cast sensitivity expected) ==="
cat "$BANK/r224b_report.txt" || true

# HARVEST FIRST (first-run law): bank every payload BEFORE any scoring
# gate so an abort cannot lose them (the first run lost all payloads to
# the scorer crash + trap teardown).
for f in $(docker exec $CNAME sh -c 'ls /tmp/mxkv_ab.*.pt 2>/dev/null'); do
  docker cp "$CNAME:$f" "$BANK/" >/dev/null 2>&1 || true
done
echo "harvested payloads: $(ls "$BANK"/mxkv_ab.*.pt 2>/dev/null | wc -l)"

# payload-class scan: max_sl <= nbt*64 means the 96-block banked window
# covers the whole sequence (walkable offline); else TRUNCATED -> skip
# by class. NOTE: docker exec needs -i for the heredoc (r224b law).
docker exec -i $CNAME /opt/venv/bin/python - <<'PYEOF' > "$BANK/payload_classes.txt" 2>&1
import glob
import torch
for f in sorted(glob.glob("/tmp/mxkv_ab.*.pt")):
    p = torch.load(f, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    msl = int(p["sl"][:r].max())
    nbt = int(p["bt_head"].numel())
    print(f"CLASS {f} max_sl={msl} nbt={nbt} "
          f"{'COVERED' if msl <= nbt * 64 else 'TRUNCATED'}")
PYEOF
grep -q "^CLASS" "$BANK/payload_classes.txt" || { fail "payload class scan empty"; exit 1; }
cat "$BANK/payload_classes.txt"
NCOV=$(grep -c " COVERED$" "$BANK/payload_classes.txt" || true)
NTRU=$(grep -c " TRUNCATED$" "$BANK/payload_classes.txt" || true)
echo "payload classes: covered=$NCOV truncated=$NTRU"

# THE VERDICT: r225t q-cast-matched scorer on the first 8 COVERED
# payloads (each lane vs the fp64 walk of its OWN input). Strict: a
# scorer crash aborts the leg — a skipped cell must never pass silently.
docker cp "$SCORER" "$CNAME:/tmp/r225t_replay.py" >/dev/null 2>&1 || true
docker cp "$ISOLIB" "$CNAME:/tmp/r225j_iso.py" >/dev/null 2>&1 || true
SLOG=$BANK/r226c_score.log
: > "$SLOG"
ninv=0
for f in $(grep " COVERED$" "$BANK/payload_classes.txt" | awk '{print $2}' | head -8); do
  ninv=$((ninv+1))
  echo "=== R226C score $f ($(date +%H:%M:%S)) ===" | tee -a "$SLOG"
  docker exec $CNAME env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
    /tmp/r225t_replay.py base "$f" >> "$SLOG" 2>&1 \
    || { fail "scorer rc=$? on $f"; exit 1; }
done
cat "$SLOG"
nscored=$(grep -c "^R225T_CELL" "$SLOG" || true)
npassed=$(grep -c "^R225T_CELL .* PASS$" "$SLOG" || true)
echo "R226C_INVOKED=$ninv R226C_SCORED=$nscored R226C_PASSED=$npassed"
[ "$ninv" -eq 8 ] || { fail "fewer than 8 COVERED payloads scored ($ninv; covered=$NCOV)"; exit 1; }
[ "$nscored" -eq "$ninv" ] || { fail "scorer cells $nscored != invocations $ninv (crashed cell)"; exit 1; }
[ "$npassed" = "$nscored" ] || { fail "q-cast-matched scoring FAILED ($npassed/$nscored)"; exit 1; }

echo "R226A_LEG_DONE $(date +%H:%M:%S) payloads=$(ls "$BANK"/mxkv_ab.*.pt 2>/dev/null | wc -l) score=$npassed/$nscored"
# ---- 7. teardown + W14 restore via the EXIT trap ----
exit 0
