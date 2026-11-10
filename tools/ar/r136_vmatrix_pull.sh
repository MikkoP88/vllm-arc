#!/bin/bash
# r136_vmatrix_pull.sh -- run LOCALLY (git-bash) after the rig-side matrix
# completes. Pulls /root/build/p635/vmatrix into the bank dir, redacts rig
# host addresses byte-exact (perl, NOT sed -- MSYS sed injects CRLF),
# writes SHA256SUMS + SHA256SUMS.redacted. Pull-only: analysis is the local
# analyzer (tools/ar/p135/p135_sweep_analyze.py).
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
# rig address comes from the caller's environment (law 8: never in-tree)
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
SRC=/root/build/p635/vmatrix
BANK=$(dirname "$0")/../../results/2026-11-10-p635-telemetry/vmatrix
mkdir -p "$BANK"

echo "== pulling vmatrix tree"
scp -r "$RIG:$SRC/." "$BANK/" >/dev/null

cd "$BANK"
echo "== byte-exact redaction (rig + client host literals)"
find . -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'

echo "== hashes"
find . -type f \( -name "p135_face_*.json" -o -name "battery.txt" -o -name "db_markers.txt" \
  -o -name "posture.txt" -o -name "metrics.txt" -o -name "serve.log" -o -name "boot.txt" \) \
  -exec sha256sum {} \; | sort -k2 > SHA256SUMS.redacted
echo "banked: $(find . -type f | wc -l) files -> $BANK"
