#!/bin/bash
# r260_pull.sh -- run LOCALLY (git-bash) after the r260 face-bisect leg
# completes on the rig (P647_R260_LEG_DONE in
# /root/build/logs/r260_launch.log). Pulls:
#   /root/build/p647/r260 -> r260/  (v1..v4 phase dirs: boot + markers +
#                                   probe.out + longprobe_seq_*.json/.txt
#                                   + crash evidence + devcores)
#   /root/build/logs/r260_launch.log -> launch.log
# into results/2026-11-27-p647-r260-face-bisect/. Redacts rig + client
# host literals byte-exact over the TEXT artifacts only (perl, NOT sed
# -- MSYS sed injects CRLF); the binary devcoredumps are SHA'd but
# never text-rewritten. Writes the data-file SHA256SUMS (VERDICT.md is
# convention-excluded, written after pull). Idempotent (VERDICT.md
# guard). Pull-only: the VERDICT + docs/journal updates are local
# follow-ups. Rig address from the caller's environment (law 8: never
# in-tree): ARC_RIG=<user>@<rig-host> tools/ar/r260_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-r260-face-bisect

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the leg is complete (P647_R260_LEG_DONE)"
DONE=$(ssh "$RIG" 'grep -c "P647_R260_LEG_DONE" /root/build/logs/r260_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R260_STILL_RUNNING (no DONE line yet) -- wait"; exit 2; }

echo "== pulling the r260 bank"
mkdir -p "$ROOT/$B"
scp -r "$RIG:/root/build/p647/r260/." "$ROOT/$B/r260/" >/dev/null
scp "$RIG:/root/build/logs/r260_launch.log" "$ROOT/$B/launch.log" >/dev/null

cd "$ROOT/$B"
echo "== byte-exact redaction (text artifacts only; devcores stay binary): $B"
find r260 launch.log -type f \( -name "*.log" -o -name "*.txt" -o -name "*.out" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: verdicts + per-phase summaries"
grep -E "^R260_(V[0-9]|SUMMARY|BANK|POSTURE|BOOT_FAIL)|^P647_R260|^DEVCORE_CAPTURED" launch.log || true
echo "-- per-phase error audits:"
for d in r260/v1_essay_multi r260/v2_code_multi r260/v3_multi_multi r260/v4_simple_multi; do
  echo "-- $d:"; cat "$d/erraudit.txt" 2>/dev/null || echo "  (no erraudit)"
done
echo "-- crash ordinal legible in the probe outs (R260_GEN / ERROR lines):"
grep -h "R260_GEN face=" r260/v*/probe.out 2>/dev/null | tail -30 || true
echo "-- devcores banked: $(ls r260/devcores/*.devcore 2>/dev/null | wc -l)"
grep -h "Engine reset\|UR_RESULT_ERROR\|GuC" r260/v*/dmesg_final.txt 2>/dev/null | tail -6 || true
