#!/bin/bash
# r256_pull.sh -- run LOCALLY (git-bash) after the r256 DEVICE_LOST
# repro completes on the rig (P647_R256_REPRO_COMPLETE in
# /root/build/logs/r256_repro_launch.log). Pulls:
#   /root/build/p647/r256                  -> r256/  (the 3-phase bank)
#   /root/build/p647/r256_run1_probe_defects -> run1_probe_defects/
#   /root/build/logs/r256_repro_launch.log -> launch.log
# into results/2026-11-27-p647-r256-devicelost-repro/. Redacts rig +
# client host literals byte-exact over the TEXT artifacts only (perl,
# NOT sed -- MSYS sed injects CRLF); the binary devcoredumps are
# SHA'd but never text-rewritten. Writes the data-file SHA256SUMS
# (VERDICT.md is convention-excluded, written after pull). Idempotent
# (VERDICT.md guard). Pull-only: the VERDICT + docs/journal updates
# are local follow-ups. Rig address from the caller's environment
# (law 8: never in-tree): ARC_RIG=<user>@<rig-host> tools/ar/r256_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-r256-devicelost-repro

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the leg is complete (P647_R256_REPRO_COMPLETE)"
DONE=$(ssh "$RIG" 'grep -c "P647_R256_REPRO_COMPLETE" /root/build/logs/r256_repro_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R256_STILL_RUNNING (no COMPLETE line yet) -- wait"; exit 2; }

echo "== pulling the r256 bank (run2 + the run1 probe-defect record)"
mkdir -p "$ROOT/$B"
scp -r "$RIG:/root/build/p647/r256/." "$ROOT/$B/r256/" >/dev/null
scp -r "$RIG:/root/build/p647/r256_run1_probe_defects/." "$ROOT/$B/run1_probe_defects/" >/dev/null
scp "$RIG:/root/build/logs/r256_repro_launch.log" "$ROOT/$B/launch.log" >/dev/null

cd "$ROOT/$B"
echo "== byte-exact redaction (text artifacts only; devcores stay binary): $B"
find r256 run1_probe_defects -type f \( -name "*.log" -o -name "*.txt" -o -name "*.out" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: phase verdicts + probe lines + devcores"
grep -E "^R256_(REPRO|LADDER|CONTROL|SUMMARY|PROBE_CEILING|POSTURE_WARN)" launch.log || true
for ph in repro ladder control; do
  echo "-- [$ph] erraudit: $(cat r256/$ph/erraudit.txt 2>/dev/null | tr '\n' ' ')"
  grep -E "^RUNG |^LADDER_|^CALIB|^FULL_" r256/$ph/probe.txt 2>/dev/null || echo "[$ph] no probe lines"
done
echo "-- devcores banked: $(ls r256/devcores/*.devcore 2>/dev/null | wc -l)"
grep -h "Engine reset" r256/*/dmesg_final.txt 2>/dev/null | head -8 || true
