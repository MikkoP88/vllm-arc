#!/bin/bash
# r71a_ccl_diag.sh — p71 lane-1 confirm: which transport/IPC path does
# the image's oneCCL 2022.0 actually select on the certified TP2 boot?
# Owner directive 2026-10-31: "oneCCL never selected the P2P path (most
# common, pure config) — check: boot log with CCL_LOG_LEVEL=2".
#
# LEVEL LAW (attempt-1 evidence, 2026-10-31): this oneCCL build takes
# ENUM NAMES, not numbers — `CCL_LOG_LEVEL=2` throws at env parse
# (env_parser.hpp:100 enum_by_str: "unexpected value: 2, expected
# values: error, warn, info, debug, trace") and kills both workers at
# the first all_reduce. Evidence:
# results/.../r71a_attempt1_invalid_numeric_level/serve_diag.log.
# The diagnostic level is therefore `debug` (trace = per-op flood).
#
# ONE change vs the certified line (one-change law): a single
# `export CCL_LOG_LEVEL=debug` inserted before the frozen champion CCL
# block. No measurement faces run here — boot, health, one request,
# capture the CCL diagnostic lines, teardown, W14 restored (trap).
# IDLE WINDOW ONLY (never mid-leg / mid-soak).
#
# Expected reads (oneCCL 2022.0 ofi ATL): per-rank transport selection
# lines, ze IPC handle exchange mode (champion env sets
# CCL_ZE_IPC_EXCHANGE=pidfd + CCL_TOPO_P2P_ACCESS=1 + ATL=ofi with ze
# and shm providers compiled into the bundled libfabric — env surface
# extracted from the image 2026-10-31, see
# results/2026-10-31-p71-r71-p2p-analysis/).
set -u
B=/root/build
D=$B/r71; mkdir -p "$D"
IMG=vllm-arc:v0.30.0-b70-cert4
SRC=$B/serve_arc_b70_p70.sh   # byte-canonical serve line
                              # (md5 2c20cec58cd6d940cf4d8561ab181d28)
ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $*"; }

w14_restore() {
  docker rm -f arc-r71a >/dev/null 2>&1 || true
  docker start radiance-test >/dev/null 2>&1 || true
  bash "$B/w14_serve_only.sh" >> "$D/w14_restore.log" 2>&1 || true
  log "W14_RESTORED"
}
trap 'rc=$?; log "r71a exit rc=$rc"; w14_restore' EXIT

cp "$SRC" "$D/r71a_serve.sh"
perl -pi -e 's/^(export CCL_ENABLE_SYCL_KERNELS=1)$/export CCL_LOG_LEVEL=debug\n$1/' "$D/r71a_serve.sh"
inserts=$(grep -c '^export CCL_LOG_LEVEL=debug$' "$D/r71a_serve.sh" || true)
if [ "$inserts" != "1" ]; then
  log "perl anchor MISS (inserts=$inserts) — aborting, certified line untouched"
  exit 2
fi
log "serve line prepared: exactly one CCL_LOG_LEVEL=debug insert (verified)"

docker stop radiance-test >/dev/null 2>&1 || true
ARC_IMAGE=$IMG ARC_NAME=arc-r71a ARC_SPEC=3 ARC_GRAPH=1 \
  bash "$D/r71a_serve.sh" > "$D/boot.out" 2>&1

deadline=$(( $(date +%s) + 1200 ))
while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 \
    http://127.0.0.1:8000/health 2>/dev/null || true)
  [ "$code" = "200" ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "HEALTH TIMEOUT — capturing boot log anyway"
    docker exec arc-r71a cat /root/serve_full.log > "$D/serve_diag.log" 2>/dev/null || true
    exit 1
  fi
  sleep 10
done
log "health 200 — firing one request, then capturing CCL lines"

curl -s -m 60 http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"Count from 1 to 50.","max_tokens":64,"temperature":0}' \
  > "$D/first_resp.json"

docker exec arc-r71a cat /root/serve_full.log > "$D/serve_diag.log" 2>/dev/null || true
grep -aE 'CCL_|transport|ofi|ipc|IPC|ze_|selected' "$D/serve_diag.log" \
  > "$D/ccl_diag_lines.txt" || true
log "captured: $D/ccl_diag_lines.txt ($(grep -ac . "$D/ccl_diag_lines.txt" || true) lines)"
log "full diagnostic serve log: $D/serve_diag.log"
