#!/bin/bash
# r261_pull.sh -- run LOCALLY (git-bash) after the r261 real-walk replay
# leg completes on the rig (P647_R261_LEG_DONE in
# /root/build/logs/r261_launch.log). Pulls:
#   /root/build/p647/r261 -> r261/  (matrix.out, results.json,
#                                   dmesg_final.txt, devcores/, the
#                                   harness copy, the rig-side SHA256SUMS)
#   /root/build/logs/r261_launch.log -> launch.log
#   /root/build/logs/w14_hang_20261127_0601 -> w14_hang/  (the W14
#                                   boot-hang evidence captured after
#                                   this leg's first restore; the
#                                   remedy is the r262 leg's restore
#                                   recipe)
# into results/2026-11-27-p647-r261-realwalk-replay/. Redacts rig +
# client host literals byte-exact over the TEXT artifacts only (perl,
# NOT sed -- MSYS sed injects CRLF); binary devcoredumps are SHA'd but
# never text-rewritten. Writes the data-file SHA256SUMS (VERDICT.md is
# convention-excluded, written after pull). Idempotent (VERDICT.md
# guard). Pull-only: the VERDICT + docs/journal updates are local
# follow-ups. Rig address from the caller's environment (law 8: never
# in-tree): ARC_RIG=<user>@<rig-host> tools/ar/r261_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-r261-realwalk-replay

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the leg is complete (P647_R261_LEG_DONE)"
DONE=$(ssh "$RIG" 'grep -c "P647_R261_LEG_DONE" /root/build/logs/r261_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R261_STILL_RUNNING (no DONE line yet) -- wait"; exit 2; }

echo "== pulling the r261 bank"
mkdir -p "$ROOT/$B"
scp -r "$RIG:/root/build/p647/r261/." "$ROOT/$B/r261/" >/dev/null
scp "$RIG:/root/build/logs/r261_launch.log" "$ROOT/$B/launch.log" >/dev/null
if ssh "$RIG" 'test -d /root/build/logs/w14_hang_20261127_0601'; then
  scp -r "$RIG:/root/build/logs/w14_hang_20261127_0601/." "$ROOT/$B/w14_hang/" >/dev/null
fi

cd "$ROOT/$B"
echo "== byte-exact redaction (text artifacts only; devcores stay binary): $B"
find r261 w14_hang launch.log -type f \( -name "*.log" -o -name "*.txt" -o -name "*.out" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the matrix"
grep -E "^R261_(START|DONE|DEVICE_WEDGED|HANG|CELL.*(HANG|DEVLOST))" r261/matrix.out launch.log || true
echo "-- cell tally by outcome:"
grep -oE "^R261_CELL .* (CLEAN|OK|HANG|DEVLOST|MISMATCH)[A-Z_]*" r261/matrix.out 2>/dev/null | awk '{print $NF}' | sort | uniq -c || true
echo "-- cells logged: $(grep -c '^R261_CELL' r261/matrix.out 2>/dev/null || true)"
echo "-- devcores banked: $(ls r261/devcores/*.devcore 2>/dev/null | wc -l)"
