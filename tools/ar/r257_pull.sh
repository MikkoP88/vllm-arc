#!/bin/bash
# r257_pull.sh -- run LOCALLY (git-bash) after the r257 faithful replay
# completes on the rig (P647_R257_REPLAY_COMPLETE in
# /root/build/logs/r257_replay_launch.log). Pulls:
#   /root/build/p647/r257 -> r257/  (replay + faithful banks + devcores)
#   /root/build/logs/r257_replay_launch.log -> launch.log
# into results/2026-11-27-p647-r257-faithful-replay/. Redacts rig +
# client host literals byte-exact over the TEXT artifacts only (perl,
# NOT sed -- MSYS sed injects CRLF); the binary devcoredumps are SHA'd
# but never text-rewritten. Writes the data-file SHA256SUMS (VERDICT.md
# is convention-excluded, written after pull). Idempotent (VERDICT.md
# guard). Pull-only: the VERDICT + docs/journal updates are local
# follow-ups. Rig address from the caller's environment (law 8: never
# in-tree): ARC_RIG=<user>@<rig-host> tools/ar/r257_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-r257-faithful-replay

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the leg is complete (P647_R257_REPLAY_COMPLETE)"
DONE=$(ssh "$RIG" 'grep -c "P647_R257_REPLAY_COMPLETE" /root/build/logs/r257_replay_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R257_STILL_RUNNING (no COMPLETE line yet) -- wait"; exit 2; }

echo "== pulling the r257 bank (both phases + the devcoredumps)"
mkdir -p "$ROOT/$B"
scp -r "$RIG:/root/build/p647/r257/." "$ROOT/$B/r257/" >/dev/null
scp "$RIG:/root/build/logs/r257_replay_launch.log" "$ROOT/$B/launch.log" >/dev/null

cd "$ROOT/$B"
echo "== byte-exact redaction (text artifacts only; devcores stay binary): $B"
find r257 -type f \( -name "*.log" -o -name "*.txt" -o -name "*.out" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100\.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: phase verdicts + probe lines + devcores"
grep -E "^R257_(REPLAY|FAITHFUL|FULLSTATE|SUMMARY|POSTURE_WARN|IMAGE)" launch.log || true
for ph in replay faithful; do
  echo "-- [$ph] erraudit: $(cat r257/$ph/erraudit.txt 2>/dev/null | tr '\n' ' ')"
  grep -E "^RUNG |^CALIB |^FULL_MULTI" r257/$ph/probe.txt 2>/dev/null \
    || grep -E "^\[(essay|code|multi|simple)\]|LONGPROBE" r257/$ph/longprobe_full.out 2>/dev/null \
    || echo "[$ph] no probe lines"
done
echo "-- devcores banked: $(ls r257/devcores/*.devcore 2>/dev/null | wc -l)"
ls -la r257/devcores/ 2>/dev/null | grep -E "devcore|dmesg" || true
grep -h "Engine reset" r257/*/dmesg_final.txt 2>/dev/null | tail -10 || true
