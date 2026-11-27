#!/bin/bash
# r262_pull.sh -- run LOCALLY (git-bash) after the r262 expand-view A/B
# leg completes on the rig (P647_R262_LEG_DONE in
# /root/build/logs/r262_launch.log). Pulls:
#   /root/build/p647/r262 -> r262/  (the three phase dirs a_probe_fixon /
#                                   c_suite_fixon / b_ctrl_fixoff with
#                                   serve.log, btprobe.txt, probe.out,
#                                   markers.txt, overlay_identity.txt;
#                                   overlay_{probe,on,off}/, the script
#                                   copies, devcores/, the rig-side
#                                   SHA256SUMS)
#   /root/build/logs/r262_launch.log -> launch.log
#   the latest /root/build/logs/w14_restore_* -> w14_hang/  (the W14
#                                   stale-baton hang evidence: the hung
#                                   serve log, the per-pid JIT build dirs
#                                   before/after, the py-spy stacks, the
#                                   cleanup and the post-restore
#                                   generation) + the restore driver log
# into results/2026-11-27-p647-r262-expandview-ab/. Redacts rig + client
# host literals byte-exact over the TEXT artifacts only (perl, NOT sed
# -- MSYS sed injects CRLF); binary devcoredumps are SHA'd but never
# text-rewritten. Writes the data-file SHA256SUMS (VERDICT.md is
# convention-excluded, written after pull). Idempotent (VERDICT.md
# guard). Pull-only: the VERDICT + docs/journal updates are local
# follow-ups. Rig address from the caller's environment (law 8: never
# in-tree): ARC_RIG=<user>@<rig-host> tools/ar/r262_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-r262-expandview-ab

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the leg is complete (P647_R262_LEG_DONE)"
DONE=$(ssh "$RIG" 'grep -c "P647_R262_LEG_DONE" /root/build/logs/r262_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R262_STILL_RUNNING (no DONE line yet) -- wait"; exit 2; }

echo "== pulling the r262 bank"
mkdir -p "$ROOT/$B"
scp -r "$RIG:/root/build/p647/r262/." "$ROOT/$B/r262/" >/dev/null
scp "$RIG:/root/build/logs/r262_launch.log" "$ROOT/$B/launch.log" >/dev/null
WDIR=$(ssh "$RIG" 'ls -d /root/build/logs/w14_restore_* 2>/dev/null | tail -1 || true')
if [ -n "$WDIR" ]; then
  mkdir -p "$ROOT/$B/w14_hang"
  scp -r "$RIG:$WDIR/." "$ROOT/$B/w14_hang/" >/dev/null
  scp "$RIG:/root/build/logs/r262_w14_restore.log" "$ROOT/$B/w14_hang/restore.log" >/dev/null 2>&1 || true
fi

cd "$ROOT/$B"
echo "== byte-exact redaction (text artifacts only; devcores stay binary): $B"
find r262 w14_hang launch.log -type f \( -name "*.log" -o -name "*.txt" -o -name "*.out" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the A/B"
grep -E "^R262_(IMAGE|OVERLAY|PHASE|SUMMARY|BANK)|^W14_|^P647_R262" launch.log || true
for PH in a_probe_fixon c_suite_fixon b_ctrl_fixoff; do
  echo "-- $PH: gens=$(grep -c '^R260_GEN face=' "r262/$PH/probe.out" 2>/dev/null || true) $(cat "r262/$PH/erraudit.txt" 2>/dev/null | tr '\n' ' ') probe_calls=$(grep -c 'MXKVBTPROBE call' "r262/$PH/btprobe.txt" 2>/dev/null || true) probe_rows=$(grep -c 'MXKVBTPROBE row=' "r262/$PH/btprobe.txt" 2>/dev/null || true) beyond_rows=$(grep -c 'MXKVBTPROBE row=.* BEYOND ' "r262/$PH/btprobe.txt" 2>/dev/null || true) materialized=$(grep -c 'MXKV native bt materialized' "r262/$PH/btprobe.txt" 2>/dev/null || true) read_failed=$(grep -c 'READ_FAILED' "r262/$PH/btprobe.txt" 2>/dev/null || true)"
  grep -E "^(bridge_marker_lines|bridge_mode_lines|graph_capture_lines|eager_posture_lines|native_lines|lmcache_lines|tracebacks)=" "r262/$PH/markers.txt" 2>/dev/null | tr '\n' ' ' || true
  echo
done
echo "-- devcores banked: $(ls r262/devcores/*.devcore 2>/dev/null | wc -l)"
echo "-- probe.out terminal lines:"
for PH in a_probe_fixon c_suite_fixon b_ctrl_fixoff; do
  echo "   [$PH] $(grep -E '^R260_DONE' "r262/$PH/probe.out" 2>/dev/null || echo 'no R260_DONE line')"
done
echo "-- W14 restore:"
grep -E "removed |W14_" w14_hang/restore.log 2>/dev/null || true
echo "== address audit over the bank (must be empty):"
grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' . 2>/dev/null | grep -v '^./SHA256SUMS$' || echo "   clean"
