#!/bin/bash
# r268c_pull.sh -- run LOCALLY (git-bash) after the r268c trace leg
# completes on the rig (R268C_DONE rc= and the exit trap's W14_RESTORE
# rc= in /root/build/p647/r268c_trace/logs/launch.log). Pulls the trace
# bank (out/{disc.json,disc.out,serve.log,trace_report.json,
# analysis_*.txt} + logs/{launch.log,boot.out,w14_restore_r268c.log})
# into results/2026-11-28-p647-r268c-trace/. serve.log is 29.8 MB of
# 145,680 MXKVTRACE lines: pulled RAW for the transfer oracle,
# redacted byte-exact (host literals), then gzipped locally -- the bank
# holds serve.log.gz; the bank's trace_disc.txt (MXKVTRACE-only subset
# of serve.log) is pulled with the bank then REMOVED (derivable via
# grep MXKVTRACE serve.log; noted in VERDICT.md). Rig-side SHA256SUMS are generated at pull time
# and double as the transfer oracle: every entry must match the pulled
# bytes or belong to a file that carries a host literal (redacted
# next). Writes the post-redaction data-file SHA256SUMS over the
# banked (gzipped) form. Idempotent (VERDICT.md guard). Pull-only,
# read-only on the rig.
# Rig address from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r268c_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-28-p647-r268c-trace
SRC=/root/build/p647/r268c_trace
LOG=$SRC/logs/launch.log

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the trace leg AND its exit-trap W14 restore are complete"
STATE=$(ssh "$RIG" "grep -E '^(R268C_DONE|W14_RESTORE rc=)' $LOG 2>/dev/null | cut -c1-200 || true")
echo "$STATE"
echo "$STATE" | grep -q "^R268C_DONE" \
  || { echo "R268C_STILL_RUNNING (no DONE line yet) -- wait"; exit 2; }
echo "$STATE" | grep -q "^W14_RESTORE rc=" \
  || { echo "R268C_RESTORE_PENDING (leg done, W14 restore still running) -- wait"; exit 3; }

echo "== rig-side sums over the PULLED file set (raw bytes, transfer oracle)"
ssh "$RIG" "cd $SRC && find . -type f ! -name SHA256SUMS.rig ! -name trace_disc.txt -exec sha256sum {} \; | sort -k2 > SHA256SUMS.rig" \
  || { echo "R268C_RIG_SUM_FAIL"; exit 1; }

echo "== pulling the r268c trace bank (serve.log RAW for the oracle)"
mkdir -p "$ROOT/$B"
scp -r -q "$RIG:$SRC/." "$ROOT/$B/"
cd "$ROOT/$B"
mv SHA256SUMS.rig ./SHA256SUMS.rig.tmp 2>/dev/null || true
rm -f out/trace_disc.txt

echo "== transfer oracle: rig-side sums vs the pulled files"
if [ -f SHA256SUMS.rig.tmp ]; then
  MATCH=0; ADDR=0; BAD=0; ABSENT=0
  while read -r H P; do
    P=${P#./}; P=${P#\*}
    [ -f "$P" ] || { echo "   RIG_SUM_ENTRY_MISSING_FILE $P"; ABSENT=$((ABSENT+1)); continue; }
    if [ "$(sha256sum "$P" | cut -d" " -f1)" = "$H" ]; then
      MATCH=$((MATCH+1))
    elif grep -qIE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' "$P"; then
      ADDR=$((ADDR+1))
    else
      echo "   RIG_SUM_MISMATCH $P"; BAD=$((BAD+1))
    fi
  done < SHA256SUMS.rig.tmp
  echo "rig_sum_entries=$(wc -l < SHA256SUMS.rig.tmp | tr -d ' ') match=$MATCH address_bearing=$ADDR mismatch=$BAD missing=$ABSENT"
  [ "$BAD" = 0 ] && [ "$ABSENT" = 0 ] || { echo "R268C_PULL_INTEGRITY_FAIL (pull again)"; exit 1; }
  mv SHA256SUMS.rig.tmp SHA256SUMS.rig
else
  echo "   no rig-side sums (ssh sum step failed earlier) -- integrity UNPROVEN"
  exit 1
fi

echo "== byte-exact redaction (every text file holding a host literal): $B"
grep -rIlZ -e '10\.20\.3\.65' -e '10\.100\.8\.6' . 2>/dev/null |
  xargs -0 -r perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10\.100\.8.REDACTED/g' || true

echo "== gzip the 29.8 MB serve.log (banked as serve.log.gz)"
gzip -9 -n out/serve.log
mv out/serve.log.gz out/serve.log.gz

find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" ! -name "facts.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the disc verdicts"
grep -E "^(R268C_DISC|R268C_TRACE|R268C_VERDICT|R268C_DONE)" logs/launch.log || true
grep -E "^\[p268c\] VERDICT" out/disc.out || true
echo "-- trace counts:"
grep -E "^trace lines" logs/launch.log || true
echo "-- W14 restore:"
grep -E "^W14_" logs/w14_restore_r268c.log 2>/dev/null | tail -6 || true
echo "== address audit over the bank (must be clean):"
grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' . 2>/dev/null | grep -v REDACTED || echo "   clean"
