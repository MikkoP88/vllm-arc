#!/bin/bash
# r245_pull.sh -- run LOCALLY (git-bash) after the r245 e2e grid
# completes on the rig (R245_DONE in /root/build/logs/
# r245_e2e_launch.log). Pulls the e2e bank:
#   /root/build/p647/r245 -> results/2026-11-26-p647-r245-page128/e2e/
# plus the leg launch log (logs/r245_e2e_launch.log -> e2e/launch.log).
# The parent bank (offline r245a oracle + the build5 failrun) is
# already committed at PR #149 and stays untouched -- results are
# immutable; the e2e grid banks as a self-contained sub-bank with its
# own VERDICT.md + SHA256SUMS. Redacts rig host addresses byte-exact
# (perl, NOT sed -- MSYS sed injects CRLF), writes the e2e SHA256SUMS
# (data files only -- VERDICT.md convention-excluded, written after
# pull, on the r239/r240/r241 pattern). Idempotent: a bank with
# e2e/VERDICT.md present is skipped whole (resume law). Pull-only:
# the VERDICT + docs updates are local follow-ups.
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
# rig address comes from the caller's environment (law 8: never in-tree)
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B245=2026-11-26-p647-r245-page128

if [ -f "$ROOT/$B245/e2e/VERDICT.md" ]; then
  echo "== $B245/e2e already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the grid is complete (R245_DONE in the launch log)"
DONE=$(ssh "$RIG" 'grep -c "R245_DONE" /root/build/logs/r245_e2e_launch.log 2>/dev/null || true')
[ "$DONE" -ge 1 ] || { echo "R245_E2E_STILL_RUNNING (no R245_DONE yet) -- wait"; exit 2; }

echo "== pulling the r245 e2e bank"
mkdir -p "$ROOT/$B245/e2e"
scp -r "$RIG:/root/build/p647/r245/." "$ROOT/$B245/e2e/" >/dev/null
scp "$RIG:/root/build/logs/r245_e2e_launch.log" "$ROOT/$B245/e2e/launch.log" >/dev/null

cd "$ROOT/$B245/e2e"
echo "== byte-exact redaction (rig + client host literals): $B245/e2e"
find . -type f \( -name "*.log" -o -name "*.txt" -o -name "*.json" \) -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10\.100\.8.REDACTED/g'
# SHA256SUMS covers the e2e data files only (VERDICT.md is written
# after pull and cites the sums, like launch.log which is redacted
# too -- launch.log IS a data file and IS covered).
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B245/e2e"
echo "== semantic audit: n_ok per face (existence-only-green-gate law)"
python3 - <<'PYEOF'
import glob, json, sys
dead = []
total = 0
for f in sorted(glob.glob("*/p139_conc_*.json")):
    total += 1
    try:
        n_ok = json.load(open(f))["n_ok"]
    except Exception:
        n_ok = -1
    if n_ok <= 0:
        dead.append((f, n_ok))
print(f"faces={total} dead={len(dead)}")
for f, n in dead:
    print(f"DEAD_FACE {f} n_ok={n}")
PYEOF
