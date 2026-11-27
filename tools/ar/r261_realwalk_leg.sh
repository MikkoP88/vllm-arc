#!/bin/bash
# r261_realwalk_leg.sh -- the REAL-WALK table replay leg (P6.47 r261,
# host-side driver; run ON THE RIG as /root/build/r261_realwalk_leg.sh).
# Boots a NO-SERVE harness container on the cert29 image and runs
# tools/ar/r261_realwalk_replay.py inside (docker exec -i, the r258b
# pattern).
#
# THE CORRECTION: r258b's make_table(tail=True) placed the 18 windowed
# ids at columns 622..639; the kernel's window walk reads columns
# 36..53 -- every r258b cell read BLOCK 0 eighteen times and the real
# crash ids never entered the walk. This leg replays the walk the
# engine actually performs: the dump rows verbatim (36 zero-pads at
# 0..35, the 18 real ids at 36..53), crossed with four crash tables
# (m8g / r257 / the r260 V3 + V4 dumps) + the consecutive-fresh
# control, x four pool fills x 3 seeds = 60 cells.
#   any HANG/DEVLOST -> the offline repro is in hand (the kernel +
#     real walk is the surface; iterate fills/ids offline);
#   all 60 clean     -> the remaining delta is the engine's same-step
#     store/interleave context (r262: the ARC_MXKV_NATIVE=0 lane
#     ablation on the V4 e2e repro).
# Crash outcomes are DATA: exit is always 0; the client polls
# P647_R261_LEG_DONE. trap restores W14 (stop container + restore
# serve; the session-end law).
set -uo pipefail
R=/root/build
P=$R/p647/r261
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert29}
CNAME=arc-r261
mkdir -p "$P" "$P/devcores"

# ---- pre-refuses ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R261_BUSY (arc leg in flight)"; exit 2
fi
[ -f $R/p647/r261_realwalk_replay.py ] || { echo "R261_NO_HARNESS (ship p647/r261_realwalk_replay.py first)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R261_NO_IMAGE $IMG"; exit 1; }
echo "R261_HARNESS_MD5 $(md5sum $R/p647/r261_realwalk_replay.py)"
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "R261_IMAGE $IMG $MF_SHA"

w14_restore() {
  rm -f "$P/watch_on"
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  if ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    docker start radiance-test >/dev/null 2>&1
    nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r261.log 2>&1 &
    echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  else
    echo "W14_ALREADY_HEALTHY (restore skipped)"
  fi
}
trap w14_restore EXIT

devcore_preclear() {
  local d
  for d in /sys/class/drm/card*/device/devcoredump; do
    [ -d "$d" ] || continue
    echo 1 > "$d/data" 2>/dev/null || true
    echo "DEVCORE_PRECLEARED $(basename "$(dirname "$(dirname "$d")")")"
  done
  true
}

devcore_watch() {
  local d card nm sz
  while [ -f "$P/watch_on" ]; do
    for d in /sys/class/drm/card*/device/devcoredump; do
      [ -f "$d/data" ] || continue
      card=$(basename "$(dirname "$(dirname "$d")")")
      nm="$P/devcores/${card}_$(date +%H%M%S)"
      if timeout 30 cat "$d/data" > "${nm}.devcore" 2>/dev/null \
         && [ -s "${nm}.devcore" ]; then
        sz=$(stat -c%s "${nm}.devcore" 2>/dev/null || echo 0)
        echo "DEVCORE_CAPTURED $card $(date +%H:%M:%S) bytes=$sz"
        echo 1 > "$d/data" 2>/dev/null || true
        dmesg -T > "${nm}_dmesg.txt" 2>/dev/null || true
      fi
    done
    sleep 2
  done
}

# ---- quiesce W14 (its serve holds 0.955 GMU), arm the watcher ----
docker stop radiance-test >/dev/null 2>&1
sleep 5
devcore_preclear
touch "$P/watch_on"
devcore_watch > "$P/devcores/watch.log" 2>&1 &

# ---- boot the harness container ----
echo "=== [boot] $CNAME on $IMG (no serve; harness only) $(date +%H:%M:%S) ==="
docker run -d --name $CNAME \
  --privileged --network host --ipc host \
  --shm-size 4g \
  --entrypoint /bin/bash \
  -v /root/build/neo_cache:/root/.neo_cache \
  -v /root/build/triton_cache:/root/.triton_cache \
  "$IMG" -c "sleep 10800" >/dev/null || { echo "R261_CONTAINER_FAIL"; exit 1; }
sleep 8
docker exec $CNAME /opt/venv/bin/python -c "import torch; \
  print('R261_TORCH_OK', torch.__version__, torch.xpu.is_available(), \
        torch.xpu.get_device_name(0))" || { echo "R261_XPU_FAIL"; exit 1; }

# ---- run the matrix (60 cells x <=45 s watchdog + fills; exec 5400 s) --
echo "=== [matrix] $(date +%H:%M:%S) ==="
( timeout 5400 docker exec -i $CNAME /opt/venv/bin/python - /root/r261_out.json \
    < $R/p647/r261_realwalk_replay.py > "$P/matrix.out" 2>&1 ) || true
i=0
while [ $i -lt 90 ]; do
  grep -q "R261_DONE" "$P/matrix.out" 2>/dev/null && break
  sleep 10; i=$((i + 1))
done
grep -q "R261_DONE" "$P/matrix.out" 2>/dev/null \
  || echo "R261_MATRIX_CEILING (no DONE line)"
docker cp $CNAME:/root/r261_out.json "$P/results.json" >/dev/null 2>&1 || true
grep -E "^R261_(START|TORCH|CELL|DONE|DEVICE_WEDGED)" "$P/matrix.out" | tail -70
echo "-- hang/devlost cells:"
grep -E "HANG|DEVLOST" "$P/matrix.out" | grep -v "^R261_DONE" || echo "  (none)"

# ---- capture + teardown ----
dmesg -T > "$P/dmesg_final.txt" 2>/dev/null || true
docker rm -f $CNAME >/dev/null 2>&1
sleep 3

rm -f "$P/watch_on"
sleep 4   # let the watcher loop observe the off switch
cp $R/p647/r261_realwalk_replay.py "$P/" >/dev/null 2>&1 || true
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.out" -o -name "*.json" \
  -o -name "*.devcore" -o -name "*.py" \) -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "R261_BANK $P ($(find . -type f | wc -l) files)"
echo "P647_R261_LEG_DONE -- bank ready for pull"
exit 0
