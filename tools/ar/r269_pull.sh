#!/bin/bash
# r269_pull.sh -- run LOCALLY (git-bash) after the r269 fix-gate leg
# completes on the rig (R269_DONE rc= and the exit trap's W14_RESTORE
# rc= in /root/build/p647/r269_fixgate/logs/launch.log). Pulls the
# fix-gate bank (f3_fixm8g_29/{boot.txt,engine_config.txt,erraudit.txt,
# g1_probes.txt,kernel_faults.txt,leg_env.txt,markers.txt,
# matrix_f3_fixm8g_29.json,matrix.out,models.txt,serve.log} +
# logs/{launch.log,w14_restore_r269.log}) into
# results/2026-11-28-p647-r269-fixgate/. Rig-side SHA256SUMS are
# generated at pull time and double as the transfer oracle: every
# entry must match the pulled bytes or belong to a file that carries a
# host literal (redacted next). Writes the post-redaction data-file
# SHA256SUMS over the banked form. Idempotent (VERDICT.md guard).
# Pull-only, read-only on the rig.
# Rig address from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r269_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-28-p647-r269-fixgate
SRC=/root/build/p647/r269_fixgate
LOG=$SRC/logs/launch.log

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the fix-gate leg AND its exit-trap W14 restore are complete"
STATE=$(ssh "$RIG" "grep -E '^(R269_DONE|W14_RESTORE rc=)' $LOG 2>/dev/null | cut -c1-200 || true")
echo "$STATE"
echo "$STATE" | grep -q "^R269_DONE" \
  || { echo "R269_STILL_RUNNING (no DONE line yet) -- wait"; exit 2; }
echo "$STATE" | grep -q "^W14_RESTORE rc=" \
  || { echo "R269_RESTORE_PENDING (leg done, W14 restore still running) -- wait"; exit 3; }

echo "== rig-side sums over the PULLED file set (raw bytes, transfer oracle)"
ssh "$RIG" "cd $SRC && find . -type f ! -name SHA256SUMS.rig -exec sha256sum {} \; | sort -k2 > SHA256SUMS.rig" \
  || { echo "R269_RIG_SUM_FAIL"; exit 1; }

echo "== pulling the r269 fix-gate bank"
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
  [ "$BAD" = 0 ] && [ "$ABSENT" = 0 ] || { echo "R269_PULL_INTEGRITY_FAIL (pull again)"; exit 1; }
  mv SHA256SUMS.rig.tmp SHA256SUMS.rig
else
  echo "   no rig-side sums (ssh sum step failed earlier) -- integrity UNPROVEN"
  exit 1
fi

echo "== byte-exact redaction (every text file holding a host literal): $B"
grep -rIlZ -e '10\.20\.3\.65' -e '10\.100\.8\.6' . 2>/dev/null |
  xargs -0 -r perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10\.100\.8.REDACTED/g' || true

find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the fix-gate verdicts"
grep -E "^(R269_|W14_|KERNEL_HEALTH|G1_|HEALTH)" logs/launch.log || true
grep -E "^(chat|raw|gate264)" f3_fixm8g_29/matrix_f3_fixm8g_29.json 2>/dev/null || true
echo "-- red-count recompute (must be 0 RED):"
python - <<'PYEOF'
import json

m = json.load(open("f3_fixm8g_29/matrix_f3_fixm8g_29.json", encoding="utf-8", errors="replace"))
MARKERS = (" our our", "our our our", " the the ", " and and ")
total_red = 0
total = 0
for cell in ("chat", "raw", "nll", "gate"):
    rows = [r for r in m.get(cell, []) if isinstance(r, dict)]
    if not rows:
        continue
    reds = [
        r
        for r in rows
        if str(r.get("http")) != "200"
        or int(r.get("lp_nan") or 0) > 0
        or any(mk in str(r.get("text", "")) for mk in MARKERS)
    ]
    total_red += len(reds)
    total += len(rows)
    print("%s: n=%d RED=%d CLEAN=%d" % (cell, len(rows), len(reds), len(rows) - len(reds)))
print("TOTAL: %d/%d RED (r268b same-posture baseline: 323/324 RED)" % (total_red, total))
PYEOF
echo "== address audit over the bank (must be clean):"
grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' . 2>/dev/null | grep -v REDACTED || echo "   clean"
