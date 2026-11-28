#!/bin/bash
# r266_pull.sh -- run LOCALLY (git-bash) after the r266 gemma analysis matrix
# completes on the rig (R266_END and the exit trap's W14_RESTORE rc= line in
# /root/build/p647/r266_gemma/logs/launch.log). Pulls the leg banks
# (<leg>/{leg_env.txt,boot.txt,matrix.out,matrix_<leg>.json,g1_probes.txt,
# engine_config.txt,markers.txt,erraudit.txt,serve.log,models.txt,kernel_faults.txt})
# plus logs/{launch.log,w14_restore_r266.log} into
# results/2026-11-28-p647-r266-gemma-analysis/.
# Rig-side SHA256SUMS are generated at pull time (the runner does not sum) and
# double as the transfer oracle: every entry must match the pulled bytes or
# belong to a file that carries a host literal (redacted next). Redaction is
# byte-exact perl over every text file holding a host literal (MSYS sed
# injects CRLF; binaries never rewritten). Writes the post-redaction data-file
# SHA256SUMS (VERDICT.md + facts.md are convention-excluded, written after
# pull). Idempotent (VERDICT.md guard). Pull-only, read-only on the rig.
# Rig address from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r266_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results
HERE=$(cd "$(dirname "$0")" && pwd)

B=2026-11-28-p647-r266-gemma-analysis
SRC=/root/build/p647/r266_gemma
LOG=$SRC/logs/launch.log

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the matrix AND its exit-trap W14 restore are complete"
STATE=$(ssh "$RIG" "grep -E '^(R266_END|W14_RESTORE rc=)' $LOG 2>/dev/null | cut -c1-200 || true")
echo "$STATE"
echo "$STATE" | grep -q "^R266_END" \
  || { echo "R266_MATRIX_STILL_RUNNING (no END line yet) -- wait"; exit 2; }
echo "$STATE" | grep -q "^W14_RESTORE rc=" \
  || { echo "R266_RESTORE_PENDING (matrix done, W14 restore still running) -- wait"; exit 3; }

echo "== rig-side sums (raw bytes, pre-redaction, transfer oracle)"
ssh "$RIG" "cd $SRC && find . -type f ! -name SHA256SUMS.rig -exec sha256sum {} \; | sort -k2 > SHA256SUMS.rig" \
  || { echo "R266_RIG_SUM_FAIL"; exit 1; }

echo "== pulling the r266 bank"
mkdir -p "$ROOT/$B"
scp -r -q "$RIG:$SRC/." "$ROOT/$B/"
cd "$ROOT/$B"
mv SHA256SUMS.rig ./SHA256SUMS.rig.tmp 2>/dev/null || true

echo "== transfer oracle: rig-side sums vs the pulled files"
if [ -f SHA256SUMS.rig.tmp ]; then
  MATCH=0; ADDR=0; BAD=0; ABSENT=0
  while read -r H P; do
    P=${P#./}; P=${P#\*}
    [ -f "$P" ] || { echo "   RIG_SUM_ENTRY_MISSING_FILE $P"; ABSENT=$((ABSENT+1)); continue; }
    if [ "$(sha256sum "$P" | cut -d" " -f1)" = "$H" ]; then
      MATCH=$((MATCH+1))
    elif grep -qIE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' "$P"; then
      ADDR=$((ADDR+1))
    else
      echo "   RIG_SUM_MISMATCH $P"; BAD=$((BAD+1))
    fi
  done < SHA256SUMS.rig.tmp
  echo "rig_sum_entries=$(wc -l < SHA256SUMS.rig.tmp | tr -d ' ') match=$MATCH address_bearing=$ADDR mismatch=$BAD missing=$ABSENT"
  [ "$BAD" = 0 ] && [ "$ABSENT" = 0 ] || { echo "R266_PULL_INTEGRITY_FAIL (pull again)"; exit 1; }
  mv SHA256SUMS.rig.tmp SHA256SUMS.rig
else
  echo "   no rig-side sums (ssh sum step failed earlier) -- integrity UNPROVEN"
  exit 1
fi

echo "== byte-exact redaction (every text file holding a host literal): $B"
grep -rIlZ -e '10\.20\.3\.65' -e '10\.100\.8\.6' . 2>/dev/null |
  xargs -0 -r perl -pi -e 's/10\.20\.3\.65/10.20\.3.REDACTED/g; s/10\.100\.8\.6/10\.100\.8.REDACTED/g' || true
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" ! -name "facts.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the matrix legs"
grep -E "^(R266_LEG|KERNEL_HEALTH)" logs/launch.log | cut -c1-200 || true
echo "-- per-leg probe rc:"
grep -E "probe done .*rc=" logs/launch.log | cut -c1-200 || true
echo "-- W14 restore:"
grep -E "^W14_" logs/w14_restore_r266.log 2>/dev/null | tail -6 || true
echo "== address audit over the bank (must be clean):"
grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' . 2>/dev/null | grep -v REDACTED || echo "   clean"

echo "== fact table (p266_report.py over the leg dirs)"
python "$HERE/p266/p266_report.py" $(find . -maxdepth 1 -mindepth 1 -type d ! -name logs | sort) --out facts.md
head -40 facts.md
