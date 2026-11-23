#!/bin/bash
# r229s_fmt_sweep.sh -- P6.47 r229 band-calibration sweep: score ALL
# banked COVERED payloads of a format (no gate, pure measurement) to
# derive the fmt's true native-vs-fp64-walk floor distribution. Born
# from the r229a mxfp8 first gate: cell 1030.24 measured dmax 2.704e-02
# vs the mxk8v4-derived BAND_DMAX 2.5e-2 while every corruption
# signature stayed clean (triton bit-exact, native stable nan=0, mism
# 1/3072, ulp 0.51, cross-lane 3.125e-02 <= 5e-2) and sibling cells
# ranged 7.6e-3..1.4e-2 -- a wider-fmt floor hypothesis that needs a
# full distribution before any band moves (no unevidenced
# customization). The band decision cites this sweep's max/p90.
#
# Refuses: cert27 image + manifest fork_sha; the bank holds payloads of
# the requested fmt; scorer + iso shipped. Boots ONE plain cert27
# container (no AB, no traffic -- the payloads are already banked),
# scores every COVERED payload, aggregates the distribution.
#
# Usage: r229s_fmt_sweep.sh <CERT27_FORK_SHA> <mxfp8|mxfp4> <BANK_DIR>
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert27
FORK_SHA="${1:?usage: r229s_fmt_sweep.sh <CERT27_FORK_SHA> <fmt> <bank_dir>}"
FMT="${2:?usage: r229s_fmt_sweep.sh <CERT27_FORK_SHA> <fmt> <bank_dir>}"
BANK="${3:?usage: r229s_fmt_sweep.sh <CERT27_FORK_SHA> <fmt> <bank_dir>}"
CNAME=arc-r229s
SCORER=$R/r229_tools/r229_replay_fmt.py
ISOLIB=$R/r225_tools/r225j_iso.py

fail() { echo "R229S_ABORT: $*"; }

docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert27_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
[ -d "$BANK" ] || { fail "bank dir $BANK missing"; exit 1; }
ls "$BANK"/mxkv_ab.*.pt >/dev/null 2>&1 || { fail "no banked payloads in $BANK"; exit 1; }
for f in "$SCORER" "$ISOLIB"; do
  [ -f "$f" ] || { fail "tool not shipped ($f)"; exit 1; }
done
SO_MD5=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libmqa_logits*.so'); print(hashlib.md5(open(ps[0],'rb').read()).hexdigest())\"" 2>/dev/null) || SO_MD5=""
[ "$SO_MD5" = "190e261edc79b3b497d8d8e801e65757" ] \
  || { fail "image libmqa_logits .so md5 $SO_MD5 != build25 pin"; exit 1; }
echo "R229S refuses passed (fork_sha=$FORK_SHA fmt=$FMT bank=$BANK so_md5=$SO_MD5)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R229S: restoring W14 ($(date +%H:%M:%S))"
  nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R229S: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R229S: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

# ---- 1. arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
sleep 5

# ---- 2. plain boot (same posture as the gate leg, minus AB: payloads
# are already banked; ARC_KVDTYPE keeps the decode env identical) ----
echo "=== boot $IMG fmt=$FMT sweep $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=0 ARC_SPEC=3 ARC_FP8=1 \
  ARC_KVDTYPE=$FMT ARC_MXKV_NATIVE=1 \
  bash $R/serve_arc_b70.sh > "$BANK/sweep_boot.txt" 2>&1 &
SEEN=0
for i in $(seq 1 100); do
  sleep 15
  c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
  [ "$c" = "200" ] && { SEEN=1; break; }
  if ! docker inspect "$CNAME" >/dev/null 2>&1; then
    break
  fi
done
if [ "$SEEN" != 1 ]; then
  echo "R229S_ABORT_BOOT (last=$c)"
  docker cp $CNAME:/root/serve_full.log "$BANK/sweep_bootfail.log" >/dev/null 2>&1 || true
  exit 1
fi
echo "HEALTH_OK fmt=$FMT $(date +%H:%M:%S)"

# ---- 3. class scan on the BANKED payloads, inside the container ----
docker cp "$SCORER" "$CNAME:/tmp/r229_replay_fmt.py" >/dev/null 2>&1 || true
docker cp "$ISOLIB" "$CNAME:/tmp/r225j_iso.py" >/dev/null 2>&1 || true
docker exec $CNAME mkdir -p /tmp/r229bank
for f in "$BANK"/mxkv_ab.*.pt; do
  docker cp "$f" "$CNAME:/tmp/r229bank/" >/dev/null 2>&1 || true
done
docker exec -i $CNAME /opt/venv/bin/python - <<'PYEOF' > "$BANK/sweep_classes.txt" 2>&1
import glob
import torch
for f in sorted(glob.glob("/tmp/r229bank/mxkv_ab.*.pt")):
    p = torch.load(f, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    msl = int(p["sl"][:r].max())
    nbt = int(p["bt_head"].numel())
    print(f"CLASS {f} fmt={p['fmt']} max_sl={msl} nbt={nbt} "
          f"{'COVERED' if msl <= nbt * 64 else 'TRUNCATED'}")
PYEOF
grep -q "^CLASS" "$BANK/sweep_classes.txt" || { fail "class scan empty"; exit 1; }
grep -q "fmt=$FMT " "$BANK/sweep_classes.txt" \
  || { fail "banked payloads are not fmt=$FMT"; exit 1; }
NCOV=$(grep -c " COVERED$" "$BANK/sweep_classes.txt" || true)
echo "sweep population: covered=$NCOV"

# ---- 4. score EVERY covered payload; one metric line per cell ----
SLOG=$BANK/r229s_sweep.log
: > "$SLOG"
n=0
for f in $(grep " COVERED$" "$BANK/sweep_classes.txt" | awk '{print $2}'); do
  n=$((n+1))
  docker exec $CNAME env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
    /tmp/r229_replay_fmt.py "$f" >> "$SLOG" 2>&1 || true
done
echo "scored $n cells ($(grep -c '^R229F_CELL' "$SLOG" || true) verdict lines)"

# ---- 5. aggregate the distribution (the calibration evidence) ----
python3 - "$SLOG" > "$BANK/r229s_dist.txt" <<'PYEOF'
import re
import sys

dn, cx, cells = [], [], 0
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if line.startswith("R229F_CELL"):
        cells += 1
    elif line.startswith("   NATIVE_VS_WALK "):
        m = re.search(r"dmax=([0-9.e+-]+)", line)
        if m:
            dn.append(float(m.group(1)))
    elif line.startswith("   NATIVE_VS_QMATCHED_ORACLE "):
        m = re.search(r"dmax=([0-9.e+-]+)", line)
        if m:
            cx.append(float(m.group(1)))


def dist(xs, name):
    xs = sorted(xs)
    if not xs:
        print(f"  {name}: NO DATA")
        return
    k = len(xs)
    print(f"  {name}  n={k} min={xs[0]:.3e} p50={xs[k // 2]:.3e} "
          f"p90={xs[int(k * 0.9)]:.3e} max={xs[-1]:.3e}")


print(f"SWEEP_DIST cells={cells}")
dist(dn, "native_vs_walk")
dist(cx, "cross_qmatched")
PYEOF
cat "$BANK/r229s_dist.txt"
echo "R229S_SWEEP_DONE $(date +%H:%M:%S) fmt=$FMT cells=$n bank=$BANK"
# ---- 6. teardown + W14 restore via the EXIT trap ----
exit 0
