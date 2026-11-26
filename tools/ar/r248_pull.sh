#!/bin/bash
# r248_pull.sh -- P-H1 bank pull: the r248 triage artifacts from the
# rig to results/2026-11-26-p647-ph1-r248-lmc-mx-triage/ with the law-8
# byte-exact redaction pass (banked serve logs carry rig/client
# addresses; perl -- MSYS sed injects CRLF). Local files sweep-clean
# BEFORE git add (leak-sweep-before-add law). Usage:
#   ARC_RIG=root@<rig> bash tools/ar/r248_pull.sh
set -uo pipefail
RIG=${ARC_RIG:?usage: ARC_RIG=root@<rig> bash tools/ar/r248_pull.sh}
BANK=results/2026-11-26-p647-ph1-r248-lmc-mx-triage
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0

mkdir -p "$BANK"
for f in boot.txt posture.txt outcome.txt diag_face1.txt diag_full.txt \
         warmup.json warmup.code p139_conc_r248f1.json p139_conc_r248f2.json \
         SHA256SUMS launch.log launch2.log; do
  scp -o StrictHostKeyChecking=no "$RIG:/root/build/p248/$f" \
    "$BANK/$f" 2>/dev/null || echo "PULL_MISS $f"
done
scp -o StrictHostKeyChecking=no "$RIG:/root/build/p248/serve.log" \
  "$BANK/serve.log" || { echo "PULL_MISS serve.log"; exit 1; }

# law-8: byte-exact host-address redaction (escaped literals only --
# never the raw bytes in this file). Client subnet pattern added
# 2026-11-26 (REDACTION-2026-11-26-r248-r249: the first pull missed it
# and shipped 21 client-LAN access lines).
perl -pi -e 's/\b10\.20\.3\.65\b/RIG_ADDR/g; s/\b10\.20\.3\.\d+\b/CLIENT_ADDR/g; s/\b10\.100\.8\.\d+\b/10.100.8.REDACTED/g' \
  "$BANK"/*.txt "$BANK"/*.log "$BANK"/*.json "$BANK"/serve.log 2>/dev/null || true

( cd "$BANK" && sha256sum $(ls | grep -v '^SHA256SUMS$') > SHA256SUMS )
echo "R248_PULL_DONE -> $BANK"
