#!/bin/bash
# r255_pull.sh -- run LOCALLY (git-bash) after the r255 cert29 gate
# completes on the rig (P647_CERT29_GATE_COMPLETE in /root/build/logs/
# r255_gate_launch.log). Pulls the gate bank:
#   /root/build/p647/cert29 -> results/2026-11-26-p647-cert29-gate/
# plus the gate launch log (logs/r255_gate_launch.log -> launch.log).
# Redacts rig host addresses byte-exact (perl, NOT sed -- MSYS sed
# injects CRLF), writes the data-file SHA256SUMS (VERDICT.md is
# convention-excluded, written after pull). Idempotent: a bank with
# VERDICT.md present is skipped whole (resume law). Pull-only: the
# VERDICT + docs/currency/tag/release updates are local follow-ups.
# Rig address comes from the caller's environment (law 8: never
# in-tree): ARC_RIG=<user>@<rig-host> tools/ar/r255_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-26-p647-cert29-gate

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the gate is complete (P647_CERT29_GATE_COMPLETE)"
DONE=$(ssh "$RIG" 'grep -c "P647_CERT29_GATE_COMPLETE" /root/build/logs/r255_gate_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R255_GATE_STILL_RUNNING (no COMPLETE line yet) -- wait"; exit 2; }

echo "== pulling the r255 gate bank"
mkdir -p "$ROOT/$B"
scp -r "$RIG:/root/build/p647/cert29/." "$ROOT/$B/" >/dev/null
scp "$RIG:/root/build/logs/r255_gate_launch.log" "$ROOT/$B/launch.log" >/dev/null
# the r255a gref-only rerun log (harness fix after the first-pass gref
# evidence loss); tolerated absent if the rerun has not run yet
scp "$RIG:/root/build/logs/r255_gref_rerun_launch.log" "$ROOT/$B/gref_rerun_launch.log" >/dev/null 2>&1 || \
  echo "note: no gref rerun log on the rig yet"

cd "$ROOT/$B"
echo "== byte-exact redaction (rig + client host literals): $B"
find . -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: per-leg marker decomposition + gate verdict"
grep -E "^P647_CERT29_GATE_COMPLETE" launch.log || true
for leg in ctrl m8q m4q mkq lmc gref m8g m4g mkg g0; do
  if [ -f "$leg/markers.txt" ]; then
    printf '[%s] %s\n' "$leg" "$(grep -E "^(mx_backend_lines|native_lines|pc2_windowed_impl|lmcache_lines|interop_live|tracebacks)=" "$leg/markers.txt" | tr '\n' ' ')"
  else
    echo "[$leg] NO markers.txt banked"
  fi
done
echo "-- lmc interop faces:"
grep -E "^G[1-5]_|^LMC_INTEROP" launch.log || true
echo "-- gref refusal face:"
grep -E "^GREF_" launch.log || true
echo "-- gemma comparative verdict:"
grep -E "^GEMMA_COMPARE:" gemma_compare.txt 2>/dev/null || true
echo "-- boot failures (must be none):"
grep -c "GATE_FAIL_BOOT" launch.log || true
