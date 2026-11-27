#!/bin/bash
# r264_pull.sh -- run LOCALLY (git-bash) after the r264 cert29 gate
# completes on the rig (P647_CERT29_GATE_COMPLETE AND the exit trap's
# W14_RESTORE rc= line in /root/build/logs/r264_gate.log). Pulls:
#   /root/build/p647/cert29_r263 -> .   the eleven leg dirs (<leg>/
#                                       boot.txt markers.txt battery.txt
#                                       garbage.out longprobe_*.out
#                                       serve.log ...), gemma_compare.txt,
#                                       _battery/ (the per-leg verdict
#                                       ledger + the whole-battery kernel
#                                       health), and the gate's own
#                                       rig-side SHA256SUMS, kept here as
#                                       SHA256SUMS.rig
#   /root/build/logs/r264_gate.log -> gate.log
#   /root/build/logs/w14_restore_r264.log (+ _entry.log, when W14 needed
#                                       a restore at gate entry)
#                                       -> w14_restore*.log
# into results/2026-11-27-p647-cert29-gate-r264/.
# The rig-side sums were taken ON THE RIG over the raw artifacts, before
# any redaction, so they double as a transfer oracle: every entry must
# either match the pulled bytes or belong to a file that carries a host
# literal (those are about to be redacted). A mismatch on an untouched
# file, or a missing file, is a corrupt or partial pull -> exit 1.
# Redacts rig + client host literals byte-exact over every TEXT file
# holding them (perl, NOT sed -- MSYS sed injects CRLF; binaries are
# never rewritten). Writes the post-redaction data-file SHA256SUMS
# (VERDICT.md is convention-excluded, written after pull). Idempotent
# (VERDICT.md guard). Pull-only and read-only on the rig.
# Rig address from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r264_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-cert29-gate-r264
SRC=/root/build/p647/cert29_r263
LOG=/root/build/logs/r264_gate.log
LEGS="gref m8g m4g mkg g0 ctrl m8q m4q mkq lmc"

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the gate AND its exit-trap W14 restore are complete"
STATE=$(ssh "$RIG" "grep -E '^(P647_CERT29_GATE_COMPLETE|W14_RESTORE rc=)' $LOG 2>/dev/null | cut -c1-200 || true")
echo "$STATE"
echo "$STATE" | grep -q "^P647_CERT29_GATE_COMPLETE" \
  || { echo "R264_GATE_STILL_RUNNING (no COMPLETE line yet) -- wait"; exit 2; }
echo "$STATE" | grep -q "^W14_RESTORE rc=" \
  || { echo "R264_RESTORE_PENDING (gate done, its W14 restore trap is still running) -- wait"; exit 3; }

echo "== pulling the r264 gate bank"
mkdir -p "$ROOT/$B"
scp -r -q "$RIG:$SRC/." "$ROOT/$B/"
scp -q "$RIG:$LOG" "$ROOT/$B/gate.log"
scp -q "$RIG:/root/build/logs/w14_restore_r264.log" "$ROOT/$B/w14_restore.log"
scp -q "$RIG:/root/build/logs/w14_restore_r264_entry.log" "$ROOT/$B/w14_restore_entry.log" 2>/dev/null \
  || echo "note: no entry restore log (W14 was healthy at gate entry)"

cd "$ROOT/$B"
[ -f SHA256SUMS ] && mv SHA256SUMS SHA256SUMS.rig

echo "== transfer oracle: rig-side sums (raw bytes, pre-redaction) vs the pulled files"
if [ -f SHA256SUMS.rig ]; then
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
  done < SHA256SUMS.rig
  echo "rig_sum_entries=$(wc -l < SHA256SUMS.rig | tr -d ' ') match=$MATCH address_bearing=$ADDR mismatch=$BAD missing=$ABSENT"
  [ "$BAD" = 0 ] && [ "$ABSENT" = 0 ] || { echo "R264_PULL_INTEGRITY_FAIL (pull again)"; exit 1; }
else
  echo "   no rig-side SHA256SUMS in the bank (the gate did not reach its summing step)"
fi

echo "== byte-exact redaction (every text file holding a host literal): $B"
grep -rIlZ -e '10\.20\.3\.65' -e '10\.100\.8\.6' . 2>/dev/null |
  xargs -0 -r perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g' || true
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the gate"
grep -E "^(LEG_VERDICT|KERNEL_HEALTH|STRIDE_FIX_ENGAGED|GEMMA_ABS_CLASSES|R264_CERT_PRECONDITION|R264_GATE_END|P647_CERT29_GATE_COMPLETE|W14_RESTORE)" gate.log | cut -c1-230 || true
echo "-- failed checks (must be none):"
grep -nE ": FAIL|_FAIL" gate.log | cut -c1-200 || echo "   none"
echo "-- gemma comparative verdict:"
grep -E "^GEMMA_COMPARE:" gemma_compare.txt 2>/dev/null || echo "   GEMMA_COMPARE: not-run"
echo "-- lmc interop faces:"
grep -E "^G[1-5]_|^LMC_INTEROP|^LMC_G" gate.log || true
echo "-- ride completion + probe errors per longprobe file:"
grep -E "^GARBLE:" gate.log || true
echo "-- marker decomposition per leg (boot snapshot + post-suite recount):"
for leg in $LEGS; do
  if [ -f "$leg/markers.txt" ]; then
    printf '[%s] %s\n' "$leg" "$(grep -E "^(mx_backend_lines|native_lines|pc2_windowed_impl|lmcache_lines|interop_live|tracebacks|materialized_lines_post|device_lost_lines_post|tracebacks_post|native_lines_post|serve_log_bytes|kernel_fault_lines)=" "$leg/markers.txt" | tr '\n' ' ')"
  else
    echo "[$leg] NO markers.txt banked"
  fi
done
echo "-- per-leg verdict ledger (_battery/leg_verdicts.txt):"
cat _battery/leg_verdicts.txt 2>/dev/null || echo "   missing"
echo "-- W14 restore:"
grep -E "^W14_" w14_restore.log 2>/dev/null | tail -6 || true
echo "== address audit over the bank (must be empty):"
grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' . 2>/dev/null || echo "   clean"
