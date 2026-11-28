#!/bin/bash
# r271_pull.sh -- run LOCALLY (git-bash) after the r271 cert30 gate
# completes on the rig (P647_CERT30_GATE_COMPLETE AND the wrapper's
# R271_DONE rc= AND the exit trap's W14_RESTORE rc= in
# /root/build/p647/cert30_r271/logs/launch.log). Pulls the whole gate
# bank /root/build/p647/cert30_r271 (the eleven leg dirs + _battery/ +
# gemma_compare.txt + the gate's own SHA256SUMS, kept as SHA256SUMS.leg,
# + logs/launch.log + logs/w14_restore_r271*.log) into
# results/2026-11-28-p647-cert30-gate-r271/.
# Rig-side SHA256SUMS.rig are REGENERATED at pull time over the raw
# artifacts and double as the transfer oracle: every entry must match
# the pulled bytes or belong to a file that carries a host literal
# (redacted next). Every serve.log > 1 MiB is then gzipped to
# serve.log.gz (the r268c banking precedent -- healthy full-suite boots
# run 10-40 MB each; the bank stays lean and under the GitHub artifact
# law) AFTER the oracle, so the raw-byte check is never faked; the
# post-gzip set gets the banked SHA256SUMS. Writes VERDICT.md notes
# there (convention-excluded from the data sums). Idempotent
# (VERDICT.md guard). Pull-only, read-only on the rig.
# Rig address from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r271_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-28-p647-cert30-gate-r271
SRC=/root/build/p647/cert30_r271
LOG=$SRC/logs/launch.log
LEGS="gref m8g m4g mkg g0 ctrl m8q m4q mkq lmc"

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the gate AND its exit-trap W14 restore are complete"
STATE=$(ssh "$RIG" "grep -E '^(R271_DONE|W14_RESTORE rc=)' $LOG 2>/dev/null | cut -c1-200 || true")
echo "$STATE"
echo "$STATE" | grep -q "^R271_DONE" \
  || { echo "R271_GATE_STILL_RUNNING (no DONE line yet) -- wait"; exit 2; }
echo "$STATE" | grep -q "^W14_RESTORE rc=" \
  || { echo "R271_RESTORE_PENDING (gate done, W14 restore still running) -- wait"; exit 3; }

echo "== rig-side sums over the PULLED file set (raw bytes, transfer oracle)"
ssh "$RIG" "cd $SRC && find . -type f ! -name SHA256SUMS.rig ! -name SHA256SUMS -exec sha256sum {} \; | sort -k2 > SHA256SUMS.rig" \
  || { echo "R271_RIG_SUM_FAIL"; exit 1; }

echo "== pulling the r271 cert30 gate bank"
mkdir -p "$ROOT/$B"
scp -r -q "$RIG:$SRC/." "$ROOT/$B/"
cd "$ROOT/$B"
mv SHA256SUMS.rig ./SHA256SUMS.rig.tmp 2>/dev/null || true
[ -f SHA256SUMS ] && mv SHA256SUMS SHA256SUMS.leg || true

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
  [ "$BAD" = 0 ] && [ "$ABSENT" = 0 ] || { echo "R271_PULL_INTEGRITY_FAIL (pull again)"; exit 1; }
  mv SHA256SUMS.rig.tmp SHA256SUMS.rig
else
  echo "   no rig-side sums (ssh sum step failed) -- integrity UNPROVEN"
  exit 1
fi

echo "== byte-exact redaction (every text file holding a host literal): $B"
grep -rIlZ -e '10\.20\.3\.65' -e '10\.100\.8\.6' . 2>/dev/null |
  xargs -0 -r perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10\.100\.8.REDACTED/g' || true

echo "== gzip the fat serve.logs (r268c precedent; oracle already ran on raw bytes)"
GZ=0
find . -type f -name serve.log -size +1M | while read -r f; do
  gzip -9 "$f" && echo "   gzipped $f -> $f.gz ($(wc -c < "$f.gz") bytes)"
done
GZ=$(find . -type f -name serve.log.gz | wc -l | tr -d ' ')
echo "serve.log.gz count=$GZ"

find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the r271 cert30 gate"
grep -E "^(R271_GATE_BEGIN|IMG_ENGINE_MD5|IMG_TRITON_MD5|R271_FORK_SHA_OK|R271_BASE_MANIFEST_OK|LEG_VERDICT|KERNEL_HEALTH|SKIN_G3|STRIDE_FIX_ENGAGED|GEMMA_ABS_CLASSES|R264_CERT_PRECONDITION|R271_GATE_END|P647_CERT30_GATE_COMPLETE|R271_DONE|W14_RESTORE)" logs/launch.log | cut -c1-230 || true
echo "-- failed checks (must be none):"
grep -nE ": FAIL|_FAIL" logs/launch.log | cut -c1-200 || echo "   none"
echo "-- logprob gates per leg probe.out (r268c corruption class must be absent:"
echo "   every leg 3x 'LP_NAN 0' and zero 'LP_NAN [1-9]' / 'LP_N 0 '):"
for leg in $LEGS; do
  [ -f "$leg/probe.out" ] || continue
  printf '[%s] LP_NAN0=%s LP_NANplus=%s LP_N0=%s\n' "$leg" \
    "$(grep -c 'LP_NAN 0' "$leg/probe.out" || true)" \
    "$(grep -cE 'LP_NAN [1-9]' "$leg/probe.out" || true)" \
    "$(grep -cE 'LP_N 0 ' "$leg/probe.out" || true)"
done
grep -E "LP_NAN: FAIL|LP_N: FAIL" logs/launch.log || echo "   no LP_NAN gate failures"
echo "-- gemma comparative verdict:"
grep -E "^GEMMA_COMPARE:" gemma_compare.txt 2>/dev/null || echo "   GEMMA_COMPARE: not-run"
echo "-- lmc interop faces:"
grep -E "^G[1-5]_|^LMC_INTEROP|^LMC_G" logs/launch.log || true
echo "-- ride completion + probe errors per longprobe file:"
grep -E "^GARBLE:" logs/launch.log || true
echo "-- marker decomposition per leg (skin + posture + post-suite recount):"
for leg in $LEGS; do
  if [ -f "$leg/markers.txt" ]; then
    printf '[%s] %s\n' "$leg" "$(grep -E "^(mx_backend_lines|native_lines|pc2_windowed_impl|lmcache_lines|interop_live|tracebacks|skin_template_lines|materialized_lines_post|device_lost_lines_post|tracebacks_post|native_lines_post|skin_template_lines_post|serve_log_bytes|kernel_fault_lines)=" "$leg/markers.txt" | tr '\n' ' ')"
  else
    echo "[$leg] NO markers.txt banked"
  fi
done
echo "-- per-leg verdict ledger (_battery/leg_verdicts.txt):"
cat _battery/leg_verdicts.txt 2>/dev/null || echo "   missing"
echo "-- W14 restore:"
grep -hE "^W14_(STALE_BATON|SERVE_LAUNCHED|HEALTH_OK|GEN_OK|FINAL|ALREADY)" logs/w14_restore_r271.log 2>/dev/null | tail -6 || true
echo "== address audit over the bank (must be empty):"
grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' . 2>/dev/null || echo "   clean"
