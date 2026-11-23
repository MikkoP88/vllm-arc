#!/bin/bash
# r225a_replay.sh — P6.47 r225 leg 0: factor-matrix GPU replay of the
# banked r224a payloads against the mxk8v4 native refusal.
#
# WHAT THIS ANSWERS: r224 root-caused the refusal to the wheel kernel
# (NATIVE_WRONG, 156-245x vs float64; magnitude-dependent, grows with
# layer depth). Static reading of the wheel source (r225) verified every
# format-specific site and found each dequant leg e2e-certified in its
# SYMMETRIC format — mxk8v4 is the only MIXED instantiation (e4m3 K +
# e2m1 V). The remaining suspects are runtime-selectable, so this leg
# bisects them by replay, one factor per cell, all inside ONE arc
# container on the banked payloads (no engine boot):
#
#   base    shipping config — must reproduce the banked native output
#           bit-exactly (r224c law) before any factor means anything
#   dbf0    ARC_MXKV_DIRECTBF16=0 — float-assembly decode both legs
#           (isolates the direct-bf16 exponent fold)
#   nsplit1 num_splits_kv=1 — single split, no ReduceSplitK combine
#           (isolates the split/combine machinery)
#   p64     ARC_MXKV_P64NATIVE=1 — the r220a multi-SG _64 policy frame
#           (isolates the certified _16 sub-tile frame)
#
# Then r225b_arbitrate.py scores every factor's output against the
# float64 ground truth; the repaired factor is the root cause.
#
# Rig laws honored: W14 down for the window and restored by a guarded
# EXIT trap with a health-200 poll; warm triton cache mounted (the
# fresh-autotune stall law); engine bt-width 4114 inside the tools;
# kill-by-container (docker rm -f), never pkill patterns.
#
# Refuses: image tag present + manifest cert26_fork_sha == $1; the
# three arbitration payloads present in the r224a bank; shipped tools
# present with expected md5s; warm triton cache directory present.
#
# Usage: r225a_replay.sh <CERT26_FORK_SHA>
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert26
CNAME=arc-r225
BANK=$R/p647/r224a
OUT=$R/p647/r225a
TOOLS=$R/r225_tools
FORK_SHA="${1:?usage: r225a_replay.sh <CERT26_FORK_SHA>}"
PAYLOADS="mxkv_ab.2447.18.pt mxkv_ab.2447.30.pt mxkv_ab.2547.18.pt"
OVERLAY=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv

mkdir -p "$OUT" "$TOOLS"
cd "$OUT" || exit 2

fail() { echo "R225A_ABORT: $*"; }

# ---- refuses (staged-ctx law) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert26_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
for f in $PAYLOADS; do
  [ -f "$BANK/$f" ] || { fail "banked payload missing ($BANK/$f)"; exit 1; }
done
[ -f "$TOOLS/r225a_replay.py" ] || { fail "r225a_replay.py not shipped"; exit 1; }
[ -f "$TOOLS/r225b_arbitrate.py" ] || { fail "r225b_arbitrate.py not shipped"; exit 1; }
REPLAY_MD5=$(md5sum "$TOOLS/r225a_replay.py" | cut -d' ' -f1)
ARB_MD5=$(md5sum "$TOOLS/r225b_arbitrate.py" | cut -d' ' -f1)
[ -d "$R/triton_cache" ] || { fail "warm triton cache missing ($R/triton_cache)"; exit 1; }
echo "R225A refuses passed (fork_sha=$FORK_SHA replay_md5=$REPLAY_MD5 arb_md5=$ARB_MD5)"

RESTORE_DONE=0
finish() {
  # guarded fail-safe: even a mid-flight death must not leave the rig on
  # an arc image (overnight law) — restore W14 exactly once.
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R225A: restoring W14 ($(date +%H:%M:%S))"
  nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R225A: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R225A: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

# ---- arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run -d --name "$CNAME" \
  --device /dev/dri --ipc=host \
  -v "$R/triton_cache:/root/.triton_cache" \
  -v "$BANK:/bank:ro" \
  -v "$OUT:/out" \
  --entrypoint /bin/bash "$IMG" -c "sleep infinity" \
  || { fail "container boot failed"; exit 1; }
docker cp "$TOOLS/r225a_replay.py" "$CNAME:/tmp/r225a_replay.py"
docker cp "$TOOLS/r225b_arbitrate.py" "$CNAME:/tmp/r225b_arbitrate.py"

PARGS=""
for f in $PAYLOADS; do PARGS="$PARGS /bank/$f"; done

cell() {  # cell <name> [ENV=VAL...] — one replay factor
  local name=$1; shift
  local envargs=()
  for e in "$@"; do envargs+=(-e "$e"); done
  echo "=== R225A cell $name ($(date +%H:%M:%S)) ==="
  docker exec "${envargs[@]}" "$CNAME" \
    env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
    /tmp/r225a_replay.py "$name" "/out/r225a_${name}.pt" $PARGS \
    > "$OUT/cell_${name}.log" 2>&1
  local rc=$?
  tail -20 "$OUT/cell_${name}.log"
  [ $rc -eq 0 ] || { fail "cell $name rc=$rc (see $OUT/cell_${name}.log)"; exit 1; }
  grep -q "R225A_DONE" "$OUT/cell_${name}.log" \
    || { fail "cell $name produced no DONE line"; exit 1; }
}

cell base
cell dbf0 ARC_MXKV_DIRECTBF16=0
cell nsplit1
cell p64 ARC_MXKV_P64NATIVE=1

echo "=== R225A arbitration (float64, CPU) ($(date +%H:%M:%S)) ==="
STEMS=""
for f in $PAYLOADS; do STEMS="$STEMS ${f%.pt}"; done
docker exec "$CNAME" env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
  /tmp/r225b_arbitrate.py /bank /out $STEMS \
  > "$OUT/arbitration.log" 2>&1
rc=$?
cat "$OUT/arbitration.log"
[ $rc -eq 0 ] || { fail "arbitration rc=$rc"; exit 1; }
grep -q "R225B_VERDICT" "$OUT/arbitration.log" \
  || { fail "arbitration produced no verdict"; exit 1; }

# teardown + W14 restore via the EXIT trap
docker rm -f "$CNAME" >/dev/null 2>&1 || true
echo "R225A: cells + arbitration complete; restoring W14"
