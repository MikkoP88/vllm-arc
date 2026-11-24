#!/bin/bash
# r239_pull.sh -- run LOCALLY (git-bash) after r239 + r239b complete on the
# rig. Pulls /root/build/p647/r239 (the 4-cell record-lane matrix: gemma L1
# + rhmxfp4 boot-failed, qurk + aur clean) and /root/build/p647/r239b (the
# two fix cells: gemma at ARC_MAXLEN=118784, rhmxfp4 with the
# container-name-cascade fix) into ONE merged bank:
#   results/2026-11-24-p647-r239-record-lanes/{r239,r239b}/
# Redacts rig host addresses byte-exact (perl, NOT sed -- MSYS sed injects
# CRLF), writes SHA256SUMS. Pull-only: the VERDICT + docs edits are local.
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
# rig address comes from the caller's environment (law 8: never in-tree)
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
BANK=$(dirname "$0")/../../results/2026-11-24-p647-r239-record-lanes
mkdir -p "$BANK/r239" "$BANK/r239b"

echo "== pulling r239 + r239b trees"
scp -r "$RIG:/root/build/p647/r239/." "$BANK/r239/" >/dev/null
scp -r "$RIG:/root/build/p647/r239b/." "$BANK/r239b/" >/dev/null

cd "$BANK"
echo "== byte-exact redaction (rig + client host literals)"
find . -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'

echo "== hashes"
find . -type f ! -name "SHA256SUMS*" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $BANK"
