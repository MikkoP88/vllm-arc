#!/bin/bash
# r266b_pull.sh -- run LOCALLY (git-bash) after the r266 BATCH-2 legs
# complete on the rig (R266_END and the exit trap's W14_RESTORE rc= in
# /root/build/p647/r266_gemma/logs/launch.log). Batch-1 is already
# banked at results/2026-11-28-p647-r266-gemma-analysis/ (PR #173) and
# that pull's VERDICT guard refuses re-pulls, so batch-2 banks as a
# SIBLING dir: results/2026-11-28-p647-r266b-gemma-batch2/ with its own
# VERDICT (cross-referencing batch-1's). Pulls ONLY the batch-2 legs:
#   e2_m8g_eager_29 e1_g3_eager_29 b5_g3_19 b4_g3_20
#   c1_bf16_29 c2_fa2off_29 b3_g4_29
# plus logs/{launch.log,w14_restore_r266.log} (the runner's trap
# rewrites w14_restore_r266.log per batch -- batch-1's copy is already
# banked). Rig-side SHA256SUMS over exactly the pulled set, generated at
# pull time, double as the transfer oracle: every entry must match the
# pulled bytes or belong to a file that carries a host literal (redacted
# next, byte-exact perl -- MSYS sed injects CRLF). Writes the
# post-redaction data-file SHA256SUMS (VERDICT.md + facts.md are
# convention-excluded, written after pull). Idempotent (VERDICT.md
# guard). Pull-only, read-only on the rig.
# Rig address from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r266b_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results
HERE=$(cd "$(dirname "$0")" && pwd)

B=2026-11-28-p647-r266b-gemma-batch2
SRC=/root/build/p647/r266_gemma
LOG=$SRC/logs/launch.log
LEGS="e2_m8g_eager_29 e1_g3_eager_29 b5_g3_19 b4_g3_20 c1_bf16_29 c2_fa2off_29 b3_g4_29"

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the batch-2 matrix AND its exit-trap W14 restore are complete"
STATE=$(ssh "$RIG" "grep -E '^(R266_END|W14_RESTORE rc=)' $LOG 2>/dev/null | tail -4 | cut -c1-200 || true")
echo "$STATE"
echo "$STATE" | grep -q "^R266_END" \
  || { echo "R266_BATCH2_STILL_RUNNING (no END line yet) -- wait"; exit 2; }
echo "$STATE" | grep -q "^W14_RESTORE rc=" \
  || { echo "R266_RESTORE_PENDING (matrix done, W14 restore still running) -- wait"; exit 3; }

echo "== all batch-2 legs banked on the rig (each has serve.log + matrix json)"
MISSING=$(ssh "$RIG" "cd $SRC; for L in $LEGS; do [ -f \$L/serve.log ] && [ -f \$L/matrix_\$L.json ] || echo \$L; done" 2>/dev/null || true)
[ -z "$MISSING" ] || { echo "R266B_LEGS_INCOMPLETE: $MISSING"; exit 4; }

echo "== rig-side sums over the batch-2 set (raw bytes, pre-redaction, transfer oracle)"
ssh "$RIG" "cd $SRC && find $LEGS logs/launch.log logs/w14_restore_r266.log -type f -exec sha256sum {} \; | sort -k2 > SHA256SUMS.r266b" \
  || { echo "R266B_RIG_SUM_FAIL"; exit 1; }

echo "== pulling the batch-2 bank"
mkdir -p "$ROOT/$B"
for L in $LEGS; do
  scp -r -q "$RIG:$SRC/$L" "$ROOT/$B/" || { echo "R266B_PULL_FAIL $L"; exit 1; }
done
mkdir -p "$ROOT/$B/logs"
scp -q "$RIG:$SRC/logs/launch.log" "$RIG:$SRC/logs/w14_restore_r266.log" "$ROOT/$B/logs/"
scp -q "$RIG:$SRC/SHA256SUMS.r266b" "$ROOT/$B/SHA256SUMS.rig.tmp"

echo "== transfer oracle: rig-side sums vs the pulled files"
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
[ "$BAD" = 0 ] && [ "$ABSENT" = 0 ] || { echo "R266B_PULL_INTEGRITY_FAIL (pull again)"; exit 1; }
mv SHA256SUMS.rig.tmp SHA256SUMS.rig

echo "== byte-exact redaction (every text file holding a host literal): $B"
grep -rIlZ -e '10\.20\.3\.65' -e '10\.100\.8\.6' . 2>/dev/null |
  xargs -0 -r perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10\.100\.8.REDACTED/g' || true
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" ! -name "facts.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the batch-2 legs"
grep -E "boot 0|HEALTH_OK_|probe done" logs/launch.log | grep -A99 "R266_BEGIN.*e2" | cut -c1-160 || true
echo "-- W14 restore:"
grep -E "^W14_" logs/w14_restore_r266.log 2>/dev/null | tail -6 || true
echo "== address audit over the bank (must be clean):"
grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' . 2>/dev/null | grep -v REDACTED || echo "   clean"

echo "== fact table (p266_report.py over the batch-2 leg dirs)"
python "$HERE/p266/p266_report.py" $(for L in $LEGS; do echo "$ROOT/$B/$L"; done) --out "$ROOT/$B/facts.md"
head -30 "$ROOT/$B/facts.md"
