#!/bin/bash
# r259_pull.sh -- run LOCALLY (git-bash) after the r259 eager-suite leg
# completes on the rig (P647_R259_LEG_DONE in
# /root/build/logs/r259_launch.log). Pulls:
#   /root/build/p647/r259 -> r259/  (boot + markers + suite + crash
#                                   evidence + devcores)
#   /root/build/logs/r259_launch.log -> launch.log
# into results/2026-11-27-p647-r259-eager-suite/. Redacts rig + client
# host literals byte-exact over the TEXT artifacts only (perl, NOT sed
# -- MSYS sed injects CRLF); the binary devcoredumps are SHA'd but
# never text-rewritten. Writes the data-file SHA256SUMS (VERDICT.md is
# convention-excluded, written after pull). Idempotent (VERDICT.md
# guard). Pull-only: the VERDICT + docs/journal updates are local
# follow-ups. Rig address from the caller's environment (law 8: never
# in-tree): ARC_RIG=<user>@<rig-host> tools/ar/r259_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-r259-eager-suite

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the leg is complete (P647_R259_LEG_DONE)"
DONE=$(ssh "$RIG" 'grep -c "P647_R259_LEG_DONE" /root/build/logs/r259_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R259_STILL_RUNNING (no DONE line yet) -- wait"; exit 2; }

echo "== pulling the r259 bank"
mkdir -p "$ROOT/$B"
scp -r "$RIG:/root/build/p647/r259/." "$ROOT/$B/r259/" >/dev/null
scp "$RIG:/root/build/logs/r259_launch.log" "$ROOT/$B/launch.log" >/dev/null

cd "$ROOT/$B"
echo "== byte-exact redaction (text artifacts only; devcores stay binary): $B"
find r259 launch.log -type f \( -name "*.log" -o -name "*.txt" -o -name "*.out" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: verdict + posture + crash evidence"
grep -E "^R259_(EAGER|BOOT_FAIL|POSTURE|HEALTH|IMAGE|LONGPROBE)|^P647_R259" launch.log || true
cat r259/erraudit.txt 2>/dev/null || true
echo "-- markers (the graph-mode delta evidence):"
cat r259/markers.txt 2>/dev/null || true
echo "-- crash context head:"
head -6 r259/devicelost_context.txt 2>/dev/null || echo "  (no devicelost context)"
echo "-- devcores banked: $(ls r259/devcores/*.devcore 2>/dev/null | wc -l)"
grep -h "Engine reset\|UR_RESULT_ERROR\|GuC" r259/dmesg_final.txt 2>/dev/null | tail -6 || true
