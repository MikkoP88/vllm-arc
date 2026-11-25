#!/bin/bash
# r240_pull.sh -- run LOCALLY (git-bash) after the r240 campaign legs
# complete on the rig (r240a solo faces + r240b conc sweep + r240c
# gemma 128K annex). Pulls the three banks:
#   /root/build/p647/r240   -> results/2026-11-24-p647-r240-reports/
#   /root/build/p647/r240b  -> results/2026-11-24-p647-r240b-conc/
#   /root/build/p647/r240c  -> results/2026-11-24-p647-r240c-gemma128k/
# plus the leg launch logs into the matching bank (r240a: its original
# launch.log only if not yet banked locally; r240b: r240log/launch.log
# banked as launch_b.log -- the relaunch reused the redirect name;
# r240c: launch_c.log). Redacts rig host addresses byte-exact (perl,
# NOT sed -- MSYS sed injects CRLF), writes one SHA256SUMS per bank
# (data files only -- VERDICT.md convention-excluded). Idempotent for
# r240a: a bank with VERDICT.md present is skipped whole.
# Pull-only: the VERDICT + docs/reports regeneration are local.
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
# rig address comes from the caller's environment (law 8: never in-tree)
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B240=2026-11-24-p647-r240-reports
B240B=2026-11-24-p647-r240b-conc
B240C=2026-11-24-p647-r240c-gemma128k

echo "== pulling the three r240 banks"
mkdir -p "$ROOT/$B240B" "$ROOT/$B240C"
# r240a is already pulled + banked locally (VERDICT written at its
# close-out) AND its rig-side launch.log was overwritten by the r240b
# relaunch redirect -- re-pulling would clobber the banked original.
if [ -f "$ROOT/$B240/VERDICT.md" ]; then
  echo "== $B240 already banked (VERDICT present) -- skipping r240a"
else
  mkdir -p "$ROOT/$B240"
  scp -r "$RIG:/root/build/p647/r240/." "$ROOT/$B240/" >/dev/null
fi
scp -r "$RIG:/root/build/p647/r240b/." "$ROOT/$B240B/" >/dev/null
scp -r "$RIG:/root/build/p647/r240c/." "$ROOT/$B240C/" >/dev/null

echo "== pulling launch logs"
# r240b's leg log lives at r240log/launch.log (the 19:52 relaunch reused
# the original redirect name); banked locally as launch_b.log. The
# pre-restart narrative was overwritten on the rig -- the per-cell
# conc.log files (append-mode) carry the full face history and the plan
# records the restart. r240c writes launch_c.log at its launch.
if [ ! -f "$ROOT/$B240/VERDICT.md" ]; then
  scp "$RIG:/root/build/p647/r240log/launch.log" "$ROOT/$B240/launch.log" >/dev/null
fi
scp "$RIG:/root/build/p647/r240log/launch.log"  "$ROOT/$B240B/launch_b.log" >/dev/null
scp "$RIG:/root/build/p647/r240log/launch_c.log" "$ROOT/$B240C/launch_c.log" >/dev/null

BANKS=("$B240B" "$B240C")
if [ ! -f "$ROOT/$B240/VERDICT.md" ]; then
  BANKS=("$B240" "${BANKS[@]}")
fi
for bank in "${BANKS[@]}"; do
  cd "$ROOT/$bank"
  echo "== byte-exact redaction (rig + client host literals): $bank"
  find . -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" \) -print0 |
    xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100\.8.REDACTED/g'
  # VERDICT.md is bank convention-excluded from the sums (r239/r140
  # pattern: SHA256SUMS covers data files only; VERDICT is written
  # after pull and cites the sums).
  find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
  echo "banked: $(find . -type f | wc -l) files -> $ROOT/$bank"
done
