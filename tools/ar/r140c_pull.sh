#!/bin/bash
# r140c_pull.sh -- run LOCALLY (git-bash) after the rig-side r140c
# probe + face re-run completes (PROBE_GATE_PASS path). Pulls
# /root/build/r140c into a staging dir, redacts rig host addresses
# byte-exact (perl, NOT sed), merges the three corrected 131K faces
# over the banked HTTP-400 originals (the originals stay preserved
# rig-side under r140/gm4x/http400_orig/ -- local pre-overwrite
# sha256s print below for the redaction note), lands the re-run aux
# files under _r140c suffixes plus the two boundary-probe jsons, and
# regenerates SHA256SUMS.redacted over the full merged bank.
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
# rig address comes from the caller's environment (law 8: never in-tree)
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
SRC=/root/build/r140c
BANK=$(cd "$(dirname "$0")/../.." && pwd)/results/2026-11-11-r140-gemma128k
STAGE=$(mktemp -d)

echo "== pulling r140c tree"
scp -r "$RIG:$SRC/." "$STAGE/" >/dev/null

echo "== byte-exact redaction (rig + client host literals)"
find "$STAGE" -type f -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'

echo "== sha256 of the banked HTTP-400 originals (provenance, pre-overwrite)"
cd "$BANK/gm4x"
sha256sum p139_conc_gm4x_c131072_n1.json p139_conc_gm4x_c131072_n2.json \
  p139_conc_gm4x_c131072_n1_t1.0.json | tee "${TMP:-/tmp}/r140c_orig_400.sha256"

echo "== md5 spot-check faces vs rig (pre-redaction bytes)"
ssh "$RIG" md5sum "$SRC/gm4x/p139_conc_gm4x_c131072_n1.json" \
  "$SRC/gm4x/p139_conc_gm4x_c131072_n2.json" \
  "$SRC/gm4x/p139_conc_gm4x_c131072_n1_t1.0.json"
md5sum "$STAGE/gm4x/p139_conc_gm4x_c131072_n1.json" \
  "$STAGE/gm4x/p139_conc_gm4x_c131072_n2.json" \
  "$STAGE/gm4x/p139_conc_gm4x_c131072_n1_t1.0.json"

echo "== merge: faces + probes overwrite-in; aux files land with _r140c suffix"
for f in p139_conc_gm4x_c131072_n1.json p139_conc_gm4x_c131072_n2.json \
         p139_conc_gm4x_c131072_n1_t1.0.json \
         p140_probe_gm4x_probe_b131000.json p140_probe_gm4x_probe_b130300.json; do
  cp "$STAGE/gm4x/$f" "$BANK/gm4x/$f"
done
for f in serve.log conc.log boot.txt posture.txt outcome.txt probe.log; do
  cp "$STAGE/gm4x/$f" "$BANK/gm4x/${f%.*}_r140c.${f##*.}"
done
cp "$STAGE/SHA256SUMS" "$BANK/SHA256SUMS.r140c"

echo "== hashes (full merged bank, redacted bytes)"
cd "$BANK"
find . -type f ! -name 'SHA256SUMS*' -exec sha256sum {} \; | sort -k2 > SHA256SUMS.redacted
echo "banked: $(find . -type f ! -name 'SHA256SUMS*' | wc -l) files -> $BANK"
