#!/bin/bash
# r262_w14_restore.sh -- W14 restore with the stale-baton cleanup (P6.47 r262).
#
# ROOT CAUSE of the "W14 boot hang" (first banked after r261, then twice
# after this leg): the W14 image's expert-offload JIT
# (vllm_gaudi/models/qwen4_exp/xpu/expert_offload.py::_host_view_ext)
# builds host_view_ext through torch cpp_extension.load into a PER-PID
# build dir (/tmp/iexp_host_view_<pid>) under a torch FileBaton. A W14
# boot killed inside the ~27 s build window (a leg entry's docker stop
# while the previous restore boot is still warming up) leaves a 0-byte
# `lock` in that dir. Every later boot whose workers land on the SAME
# pids -- a fresh container PID namespace is deterministic (Worker_TP0
# 561, Worker_TP1 562) -- finds the baton held and waits forever in
# file_baton.py:wait: "JIT kernel warmup finished" is the last serve
# line, both GPUs idle, port 8000 refused. docker restart alone cannot
# clear it (same pids, same locks); it "worked" once only because
# diagnostic docker execs had bumped the pid counter.
#
# Remedy: with NO serve running, delete the per-pid dirs that hold no
# built .so (lock-only), then boot. This script does restart -> cleanup
# -> GPU-free wait -> serve -> health + one generation.
# Exit: 0 healthy + generating, 1 not, 3 refused (arc leg in flight).
set -uo pipefail
R=/root/build
LOGDIR=$R/logs
STAMP=$(date -u +%Y%m%d_%H%M%S)
EV=$LOGDIR/w14_restore_$STAMP
mkdir -p "$EV"
SERVE_LOG=$EV/w14_serve_only.out

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "W14_RESTORE_REFUSED (arc leg in flight -- its trap owns the restore)"
  exit 3
fi

mem_used() {  # $1 = device index -> used MiB (integer), empty on error
  xpu-smi dump --device "$1" --metrics MEMORY --number 1 2>/dev/null \
    | tail -1 | awk -F',' '{printf "%d", $5}'
}

health() { curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; }

if health; then
  echo "W14_ALREADY_HEALTHY $(date -u +%Y-%m-%dT%H:%M:%SZ) (nothing to do)"
  exit 0
fi

# evidence of the state we are fixing (serve log + per-pid dirs + stacks)
docker cp radiance-test:/root/serve_posture.log "$EV/serve_posture_before.log" >/dev/null 2>&1 || true
docker exec radiance-test bash -c 'for d in /tmp/iexp_host_view_*; do [ -d "$d" ] || continue; echo "== $d"; ls -la --time-style=full-iso "$d"; done' \
  > "$EV/host_view_dirs_before.txt" 2>&1 || true
for p in $(pgrep -f 'VLLM::'); do echo "=== py-spy dump hostpid=$p"; timeout 40 py-spy dump --pid "$p" 2>&1 | head -45; done \
  > "$EV/pyspy_before.txt" 2>&1 || true

# 1. quiesce: a restart kills every serve process and resets the pid namespace
docker restart radiance-test >/dev/null 2>&1 || docker start radiance-test >/dev/null 2>&1
sleep 3

# 2. the cleanup (no serve is running in a just-restarted container)
docker exec radiance-test bash -c '
  n=0
  for d in /tmp/iexp_host_view_*; do
    [ -d "$d" ] || continue
    if [ ! -f "$d/iexp_host_view_ext.so" ]; then rm -rf "$d"; n=$((n + 1)); echo "removed $d"; fi
  done
  echo "W14_STALE_BATON_DIRS_REMOVED=$n"' 2>&1 | tee "$EV/cleanup.txt"

# 3. wait for both cards to release their memory (up to 90 s)
i=0
while [ $i -lt 18 ]; do
  U0=$(mem_used 0); U1=$(mem_used 1)
  [ -n "$U0" ] && [ -n "$U1" ] && [ "$U0" -lt 2000 ] && [ "$U1" -lt 2000 ] && break
  i=$((i + 1)); sleep 5
done
echo "W14_GPU_MEM_BEFORE_BOOT dev0=${U0:-?}MiB dev1=${U1:-?}MiB"

# 4. serve
nohup bash $R/w14_serve_only.sh > "$SERVE_LOG" 2>&1 &
echo "W14_SERVE_LAUNCHED $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$SERVE_LOG"

# 5. health (a healthy boot takes ~5 min; allow 14)
i=0
while [ $i -lt 84 ]; do
  health && break
  i=$((i + 1)); sleep 10
done
if ! health; then
  echo "W14_RESTORE_FAIL $(date -u +%Y-%m-%dT%H:%M:%SZ) (no health after $((i * 10)) s)"
  docker exec radiance-test bash -c 'ls -la --time-style=full-iso -d /tmp/iexp_host_view_* 2>/dev/null; tail -4 /root/serve_posture.log | cut -c1-200' 2>&1 | tee "$EV/after_fail.txt"
  for p in $(pgrep -f 'VLLM::'); do echo "=== py-spy dump hostpid=$p"; timeout 40 py-spy dump --pid "$p" 2>&1 | head -45; done \
    > "$EV/pyspy_after_fail.txt" 2>&1 || true
  exit 1
fi
echo "W14_HEALTH_OK $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 6. one real generation (health alone is an existence-only signal)
sleep 5
GEN=$(docker exec radiance-test curl -s -m 180 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flashnext","prompt":"The capital of Finland is","max_tokens":8,"temperature":0}' 2>/dev/null)
echo "$GEN" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    c = d["choices"][0]
    print("W14_GEN_OK text=%r finish=%s" % (c["text"][:60], c.get("finish_reason")))
except Exception as e:  # noqa: BLE001
    print("W14_GEN_FAIL %r" % (e,)); sys.exit(1)
' | tee "$EV/gen.txt"
RC=${PIPESTATUS[0]}
docker exec radiance-test bash -c 'for d in /tmp/iexp_host_view_*; do [ -d "$d" ] || continue; echo "== $d"; ls -la --time-style=full-iso "$d"; done' \
  > "$EV/host_view_dirs_after.txt" 2>&1 || true
docker exec radiance-test bash -c 'ps -eo pid,comm | grep VLLM::' > "$EV/pids_after.txt" 2>&1 || true
echo "W14_FINAL rc=$RC evidence=$EV"
exit "$RC"
