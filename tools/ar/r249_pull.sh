#!/bin/bash
# r249_pull.sh -- P-H2 bank pull: the r249 interop-validation artifacts
# from the rig to results/2026-11-26-p647-ph2-r249-lmc-mx-interop/ with
# the law-8 byte-exact redaction pass (banked serve logs carry
# rig/client addresses; perl -- MSYS sed injects CRLF). Local files
# sweep-clean BEFORE git add (leak-sweep-before-add law). Usage:
#   ARC_RIG=root@<rig> bash tools/ar/r249_pull.sh
set -uo pipefail
RIG=${ARC_RIG:?usage: ARC_RIG=root@<rig> bash tools/ar/r249_pull.sh}
BANK=results/2026-11-26-p647-ph2-r249-lmc-mx-interop
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0

mkdir -p "$BANK"
for f in boot.txt posture.txt outcome.txt failed.txt kvshape_line.txt \
         g1.txt g2.txt g3.txt g4.txt g5.txt warmup.json warmup.code \
         aa1.json aa2.json aa1.text aa2.text aa_times.txt aaprompt.json \
         p139_conc_r249a.json p139_conc_r249d_n1_run1.json \
         p139_conc_r249d_n1_run2.json p139_conc_r249d_n2_run3.json \
         p139_conc_r249d_n4_run4.json; do
  scp -o StrictHostKeyChecking=no "$RIG:/root/build/p249/$f" \
    "$BANK/$f" 2>/dev/null || echo "PULL_MISS $f"
done
scp -o StrictHostKeyChecking=no "$RIG:/root/build/p249/serve.log" \
  "$BANK/serve.log" || { echo "PULL_MISS serve.log"; exit 1; }
scp -o StrictHostKeyChecking=no "$RIG:/root/build/p249/SHA256SUMS" \
  "$BANK/SHA256SUMS.rig" 2>/dev/null || true
scp -o StrictHostKeyChecking=no "$RIG:/root/build/p249b_launch.log" \
  "$BANK/launch.log" 2>/dev/null || scp -o StrictHostKeyChecking=no \
  "$RIG:/root/build/p249_launch.log" "$BANK/launch.log" 2>/dev/null \
  || echo "PULL_MISS launch.log"

# law-8: byte-exact host-address redaction (escaped literals only --
# never the raw bytes in this file). Client subnet pattern added
# 2026-11-26 (REDACTION-2026-11-26-r248-r249: the first pull missed it
# and shipped 21 client-LAN access lines).
perl -pi -e 's/\b10\.20\.3\.65\b/RIG_ADDR/g; s/\b10\.20\.3\.\d+\b/CLIENT_ADDR/g; s/\b10\.100\.8\.\d+\b/10.100.8.REDACTED/g' \
  "$BANK"/*.txt "$BANK"/*.log "$BANK"/*.json "$BANK"/*.text "$BANK"/serve.log \
  "$BANK"/SHA256SUMS.rig 2>/dev/null || true

( cd "$BANK" && sha256sum $(ls | grep -v -E '^SHA256SUMS(\.rig)?$') > SHA256SUMS )
echo "R249_PULL_DONE -> $BANK"
