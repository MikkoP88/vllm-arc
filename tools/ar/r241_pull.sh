#!/bin/bash
# r241_pull.sh -- run LOCALLY (git-bash) after the r241 posture-
# correction leg completes on the rig (R241_COMPLETE in
# /root/build/p647/r241log/launch.log). Pulls the bank:
#   /root/build/p647/r241  -> results/2026-11-25-p647-r241-mxkv-lmcfree/
# plus the leg launch log (r241log/launch.log -> launch.log).
# Redacts rig host addresses byte-exact (perl, NOT sed -- MSYS sed
# injects CRLF), writes the bank SHA256SUMS (data files only --
# VERDICT.md convention-excluded, written after pull). Idempotent:
# a bank with VERDICT.md present is skipped whole (resume law).
# Pull-only: the VERDICT + docs updates are local follow-ups.
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
# rig address comes from the caller's environment (law 8: never in-tree)
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B241=2026-11-25-p647-r241-mxkv-lmcfree

if [ -f "$ROOT/$B241/VERDICT.md" ]; then
  echo "== $B241 already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== pulling the r241 bank"
mkdir -p "$ROOT/$B241"
scp -r "$RIG:/root/build/p647/r241/." "$ROOT/$B241/" >/dev/null
scp "$RIG:/root/build/p647/r241log/launch.log" "$ROOT/$B241/launch.log" >/dev/null

cd "$ROOT/$B241"
echo "== byte-exact redaction (rig + client host literals): $B241"
find . -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100\.8.REDACTED/g'
# VERDICT.md is bank convention-excluded from the sums (r239/r240
# pattern: SHA256SUMS covers data files only; VERDICT is written
# after pull and cites the sums).
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B241"
