#!/bin/bash
# r140_pull.sh -- run LOCALLY (git-bash) after the rig-side r140 gemma
# 128K annex gate completes. Pulls /root/build/r140 into the annex bank
# dir, redacts rig host addresses byte-exact (perl, NOT sed), writes
# SHA256SUMS + SHA256SUMS.redacted. Pull-only: the reports are generated
# by tools/ar/p139/p139_report.py from the bank.
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
# rig address comes from the caller's environment (law 8: never in-tree)
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
SRC=/root/build/r140
BANK=$(dirname "$0")/../../results/2026-11-11-r140-gemma128k
mkdir -p "$BANK"

echo "== pulling r140 tree"
scp -r "$RIG:$SRC/." "$BANK/" >/dev/null

cd "$BANK"
echo "== byte-exact redaction (rig + client host literals)"
find . -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'

echo "== hashes"
find . -type f \( -name "p139_conc_*.json" -o -name "conc.log" -o -name "outcome.txt" \
  -o -name "posture.txt" -o -name "serve.log" -o -name "boot.txt" -o -name "SHA256SUMS" \) \
  -exec sha256sum {} \; | sort -k2 > SHA256SUMS.redacted
echo "banked: $(find . -type f | wc -l) files -> $BANK"
