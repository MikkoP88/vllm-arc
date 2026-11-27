#!/bin/bash
# r258b_pull.sh -- run LOCALLY (git-bash) after the r258b frag-replay leg
# completes on the rig (P647_R258B_LEG_DONE in
# /root/build/logs/r258b_launch.log). Pulls:
#   /root/build/p647/r258b      -> run2/  (the real matrix bank)
#   /root/build/p647/r258b_run1 -> run1/  (the vacuous run-1: the fmt-slot
#                                harness bug; banked as evidence, no claim)
#   /root/build/logs/r258b_launch.log -> launch.log
# into results/2026-11-27-p647-r258b-frag-replay/. Redacts rig + client
# host literals byte-exact over the TEXT artifacts only (perl, NOT sed --
# MSYS sed injects CRLF); the binary devcoredumps are SHA'd but never
# text-rewritten. Writes the data-file SHA256SUMS (VERDICT.md is
# convention-excluded, written after pull). Idempotent (VERDICT.md
# guard). Pull-only: the VERDICT + docs/journal updates are local
# follow-ups. Rig address from the caller's environment (law 8: never
# in-tree): ARC_RIG=<user>@<rig-host> tools/ar/r258b_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-r258b-frag-replay

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the leg is complete (P647_R258B_LEG_DONE)"
DONE=$(ssh "$RIG" 'grep -c "P647_R258B_LEG_DONE" /root/build/logs/r258b_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R258B_STILL_RUNNING (no DONE line yet) -- wait"; exit 2; }

echo "== pulling the r258b banks (run2 + the vacuous run1)"
mkdir -p "$ROOT/$B/run2" "$ROOT/$B/run1"
scp -r "$RIG:/root/build/p647/r258b/." "$ROOT/$B/run2/" >/dev/null
scp -r "$RIG:/root/build/p647/r258b_run1/." "$ROOT/$B/run1/" >/dev/null
scp "$RIG:/root/build/logs/r258b_launch.log" "$ROOT/$B/launch.log" >/dev/null

cd "$ROOT/$B"
echo "== byte-exact redaction (text artifacts only): $B"
find run1 run2 launch.log -type f \( -name "*.log" -o -name "*.txt" -o -name "*.out" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: run-2 matrix summary + run-1 vacuity"
grep -E "^R258B_(START|DONE|DEVICE_WEDGED)" run2/matrix.out || true
echo "-- run2 cell verdict census:"
grep -oE "\[(f0|w[0-9])=(OK|HANG|DEVLOST|ERR:[A-Za-z]*)" run2/matrix.out | sort | uniq -c
echo "-- run1 (expected: all cells ERR on w0, the fmt-slot bug):"
grep -c "ERR:ValueError" run1/matrix.out || true
echo "-- hang/devlost lines across both runs:"
grep -hE "HANG|DEVLOST" run1/matrix.out run2/matrix.out | grep -v "^R258B_DONE" || echo "  (none)"
echo "-- devcores banked: $(ls run2/devcores/*.devcore 2>/dev/null | wc -l)"
