#!/bin/bash
# r187_resumewait.sh -- P6.47 S1 chain auto-resume. The foreign
# arc-test serve (external actor, created 23:05:26Z, serving since
# 23:23 at gmu 0.95 on both XPUs + host port 8000) starved r185b's
# mkq boot (xpu:1 free 1.1/31.89 GiB) and blocks r186 + the W14
# restore. Foreign workloads are never killed here; this watcher
# polls until arc-test exits, then fires the remaining chain
# unattended: preserve the boot-fail mkq evidence -> r185b (mkq
# fresh) -> r186 (ctrl cell) -> W14 verify. Cap 12 h -> timeout
# marker for the session cron to surface.
set -uo pipefail
R=/root/build
P=$R/p647/acceptA23
LOG=$R/r187_resumewait.log
CAP_S=43200          # 12 h
POLL_S=120
MARK=$R/p647/CHAIN_DONE.txt

log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$LOG"; }

log "=== r187 resumewait start: waiting for arc-test exit (cap ${CAP_S}s) ==="
T=0
while [ $T -lt $CAP_S ]; do
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'arc-test'; then
    log "arc-test NOT RUNNING (exited or removed) -- window open"
    break
  fi
  sleep $POLL_S; T=$((T + POLL_S))
done
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'arc-test'; then
  log "WATCH_TIMEOUT: arc-test still running after ${CAP_S}s -- surfacing to session"
  echo "WATCH_TIMEOUT $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARK"
  exit 3
fi

# guard: our own chain must not double-fire
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  log "ABORT: arc-r container already running (manual chain?) -- standing down"
  exit 4
fi

# port must actually be free (linger safety)
Q=0
while [ $Q -lt 60 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 5; Q=$((Q + 5))
done
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  log "ABORT: port 8000 still held after arc-test exit (unknown holder) -- surfacing"
  echo "PORT_HELD $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARK"
  exit 5
fi
log "port 8000 free -- launching chain"

# preserve the 23:24 boot-fail mkq artifacts WITHOUT touching the
# original mkq_interfered evidence (r185b's own preserve would
# overwrite it: rm -rf mkq_interfered on rerun)
if [ -d "$P/mkq" ] && [ ! -f "$P/mkq/outcome.txt" ] \
   || grep -q BOOT_FAIL "$P/mkq/outcome.txt" 2>/dev/null; then
  rm -rf "$P/mkq_bootfail_gpuconflict"
  mv "$P/mkq" "$P/mkq_bootfail_gpuconflict" 2>/dev/null \
    && log "preserved boot-fail mkq -> mkq_bootfail_gpuconflict" \
    || log "WARN: mkq preserve skipped (dir absent?)"
fi

log "firing r185b (mkq rerun)"
bash $R/r185b_mkq_rerun.sh >> $R/r185b.log 2>&1
log "r185b rc=$? -- bank trio report in $P/acceptknee.txt"

log "firing r186 (ctrl cell)"
bash $R/r186_ctrlcell.sh >> $R/r186.log 2>&1
log "r186 rc=$? -- ctrl report in $R/p647/ctrlA24/ctrlcurve.txt"

# final W14 state (r186's trap restores it); verify + report
if curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
  W14=UP
elif docker exec radiance-test pgrep -f EngineCore >/dev/null 2>&1; then
  W14=BOOTING
else
  W14=DOWN
fi
log "final W14 state: $W14"
{
  echo "CHAIN_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "w14=$W14"
  echo "trio_report=$P/acceptknee.txt"
  echo "ctrl_report=$R/p647/ctrlA24/ctrlcurve.txt"
} > "$MARK"
log "=== r187 resumewait complete ==="
exit 0
