#!/bin/bash
# r190_quietwait.sh -- P6.47 S1 chain re-arm (r187 v2). r187 timed
# out at 12:12:51Z with arc-test still Up; by 18:39Z the actor had
# removed arc-test entirely but stayed live on the rig (litellm-proxy
# restarted ~18:18Z, radiance-test swept a third time, exit 137). So
# the fire predicate widens from "arc-test gone" to FULL quiet: no
# arc-test, no litellm-proxy, no ^arc-r containers, stable for two
# consecutive polls -- the chain only grabs both XPUs once the
# actor's whole session has wound down. Port guard likewise widens:
# port 8000 held by radiance-test's own serve is the known W14
# holder that r185b quiesces (docker stop at its line 161); any
# other holder aborts PORT_HELD. Chain body is r187 verbatim:
# preserve boot-fail mkq -> r185b (mkq) -> r186 (ctrl) -> W14
# verify -> CHAIN_DONE marker. Foreign workloads are never killed.
#
# v6 (2026-11-18): (a) CHAIN_DONE stand-down guard, checked every
# poll -- a watcher re-firing after a completed chain would rerun
# r185b, whose internal preserve destroys the mkq_interfered
# evidence; (b) R190_FORCE_FIRE=1 manual-fire path: skips the quiet
# wait after direct arc-test/GPU/port verification, keeps every
# guard, and exempts only the CPU-idle restart-policy litellm-proxy
# from the fire predicate (a reboot cannot clear it, so full-quiet
# alone can never fire once maintenance ends).
set -uo pipefail
R=/root/build
P=$R/p647/acceptA23
LOG=$R/r190_quietwait.log
CAP_S=43200          # 12 h
POLL_S=120
QUIET_N=2            # consecutive quiet polls required
MARK=$R/p647/CHAIN_DONE.txt

log() { echo "[$(date -u +%H:%M:%SZ)] $*" >> "$LOG"; }

is_quiet() {
  local names
  names=$(docker ps --format '{{.Names}}' 2>/dev/null)
  echo "$names" | grep -qx 'arc-test' && return 1
  echo "$names" | grep -qx 'litellm-proxy' && return 1
  echo "$names" | grep -q '^arc-r' && return 1
  return 0
}

log "=== r190 quietwait start: waiting for full quiet (cap ${CAP_S}s, ${QUIET_N} stable polls) ==="

if [ "${R190_FORCE_FIRE:-0}" = "1" ]; then
  if [ -f "$MARK" ]; then
    log "CHAIN_DONE present -- standing down (no force fire)"
    exit 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'arc-test'; then
    log "ABORT: arc-test present -- refusing force fire"
    exit 6
  fi
  log "R190_FORCE_FIRE=1 -- firing chain directly (manual, post-verification)"
else
  T=0
  Q=0
  while [ $T -lt $CAP_S ]; do
    if [ -f "$MARK" ]; then
      log "CHAIN_DONE present -- standing down"
      exit 0
    fi
    if is_quiet; then
      Q=$((Q + 1))
      [ $Q -ge $QUIET_N ] && break
    else
      Q=0
    fi
    sleep $POLL_S; T=$((T + POLL_S))
  done
  if [ $Q -lt $QUIET_N ]; then
    log "WATCH_TIMEOUT: rig not quiet after ${CAP_S}s -- surfacing to session"
    echo "WATCH_TIMEOUT $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARK"
    exit 3
  fi
  log "rig quiet (${QUIET_N} stable polls) -- window open"
fi

# guard: our own chain must not double-fire
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  log "ABORT: arc-r container already running (manual chain?) -- standing down"
  exit 4
fi

# port must be free, or held solely by W14's own serve (r185b stops
# radiance-test at its start; that holder is expected and safe)
Q2=0
while [ $Q2 -lt 60 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test' && break
  sleep 5; Q2=$((Q2 + 5))
done
if ss -tln 2>/dev/null | grep -q ':8000 ' \
   && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
  log "ABORT: port 8000 held by unknown holder -- surfacing"
  echo "PORT_HELD $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$MARK"
  exit 5
fi
log "port 8000 clear (or W14-held) -- launching chain"

# preserve the boot-fail mkq artifacts WITHOUT touching the original
# mkq_interfered evidence (r185b's own preserve would overwrite it:
# rm -rf mkq_interfered on rerun)
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
log "=== r190 quietwait complete ==="
exit 0
