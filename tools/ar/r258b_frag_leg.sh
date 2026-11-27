#!/bin/bash
# r258b_frag_leg.sh -- the r258b synthetic fragmented-table replay leg
# (P6.47 r258, host-side driver; run ON THE RIG as
# /root/build/r258b_frag_leg.sh). Boots a NO-SERVE harness container on
# the cert29 image (the crash image: cert28 + build34 page-128 wheel +
# the 16effb49 P-C2 delta), runs tools/ar/r258b_frag_replay.py inside
# (docker exec -i, the r257 pattern), and banks the matrix verdicts.
#
# The harness replays the crash step's native calls DIRECTLY (no
# engine): the five windowed page-64 masked twins (the new P-C2
# surface) + the full-attn page-128 triton twin, crossed with the exact
# banked crash tables (m8g / r257) vs a consecutive-fresh control, and
# four pool fills (zero / recycled / random / nanband) x 3 seeds. A
# repro of the DEVICE_LOST class prints DEVLOST/HANG on the cell; the
# devcore watcher banks any GPU reset's dumps the moment it lands
# (2 s cadence, the r257 apparatus).
#
# Crash outcomes are DATA: exit is always 0; the client polls
# P647_R258B_LEG_DONE. trap restores W14 (stop container + restore
# serve; the session-end law).
set -uo pipefail
R=/root/build
P=$R/p647/r258b
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert29}
CNAME=arc-r258b
mkdir -p "$P" "$P/devcores"

# ---- pre-refuses ----
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R258B_BUSY (arc leg in flight)"; exit 2
fi
[ -f $R/p647/r258b_frag_replay.py ] || { echo "R258B_NO_HARNESS (ship p647/r258b_frag_replay.py first)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R258B_NO_IMAGE $IMG"; exit 1; }
echo "R258B_HARNESS_MD5 $(md5sum $R/p647/r258b_frag_replay.py)"
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -o "cert29_fork_sha=.*" /etc/vllm-arc/IMAGE_MANIFEST' 2>/dev/null) || MF_SHA=""
echo "R258B_IMAGE $IMG $MF_SHA"

w14_restore() {
  rm -f "$P/watch_on"
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  if ! curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    docker start radiance-test >/dev/null 2>&1
    nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r258b.log 2>&1 &
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
  "$IMG" -c "sleep 7200" >/dev/null || { echo "R258B_CONTAINER_FAIL"; exit 1; }
sleep 8
docker exec $CNAME /opt/venv/bin/python -c "import torch; \
  print('R258B_TORCH_OK', torch.__version__, torch.xpu.is_available(), \
        torch.xpu.get_device_name(0))" || { echo "R258B_XPU_FAIL"; exit 1; }

# ---- run the matrix (36 cells x <=45 s worst case; exec-capped 2400 s) ----
echo "=== [matrix] $(date +%H:%M:%S) ==="
( timeout 2400 docker exec -i $CNAME /opt/venv/bin/python - /root/r258b_out.json \
    < $R/p647/r258b_frag_replay.py > "$P/matrix.out" 2>&1 ) || true
i=0
while [ $i -lt 200 ]; do
  grep -q "R258B_DONE" "$P/matrix.out" 2>/dev/null && break
  sleep 10; i=$((i + 1))
done
grep -q "R258B_DONE" "$P/matrix.out" 2>/dev/null \
  || echo "R258B_MATRIX_CEILING (no DONE line)"
docker cp $CNAME:/root/r258b_out.json "$P/results.json" >/dev/null 2>&1 || true
grep -E "^R258B_(START|TRITON|CELL|DONE|DEVICE_WEDGED)" "$P/matrix.out" | tail -50
echo "-- hang/devlost cells:"
grep -E "HANG|DEVLOST" "$P/matrix.out" | grep -v "^R258B_DONE" || echo "  (none)"

# ---- capture + teardown ----
dmesg -T > "$P/dmesg_final.txt" 2>/dev/null || true
docker rm -f $CNAME >/dev/null 2>&1
sleep 3

rm -f "$P/watch_on"
sleep 4   # let the watcher loop observe the off switch
cp $R/p647/r258b_frag_replay.py "$P/" >/dev/null 2>&1 || true
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.out" -o -name "*.json" \
  -o -name "*.devcore" -o -name "*.py" \) -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "R258B_BANK $P ($(find . -type f | wc -l) files)"
echo "P647_R258B_LEG_DONE -- bank ready for pull"
exit 0
