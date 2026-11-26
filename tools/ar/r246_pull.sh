#!/bin/bash
# r246_pull.sh -- P6.47 P-D close-out pull: bank the r246 (offline
# probe, 11 faces) + r246b (two-cell e2e) validation evidence into
# results/2026-11-26-p647-pd-lmc-mx-boot-gate/.
#
# Gates on the rig state, never assumes: r246.out must carry
# VERDICT: PASS and the run-3 launch log must carry R246B_DONE
# failed=[] (both legs green). Pulls the probe output, the final
# run-3 cells (pd_refuse GATE_FIRED + pd_clean CONC_OK), the run-1
# runner-defect snapshot and the run-2 leak-cell artifacts (the
# gate's posture kv_dtype=None lines are the cache_dtype-rename
# evidence), and all three r246b launch logs.
#
# Redaction is the byte-exact perl pass (MSYS sed injects CRLF):
# the rig subnet address and the client address -> the REDACTED
# forms (same mapping as the r245 precedent), applied to every
# pulled text artifact BEFORE SHA256SUMS (sums are computed over the
# redacted bytes; the remote-side pre-redaction sums are preserved
# as remote_SHA256SUMS.* provenance).
#
# Usage: ARC_RIG=<user>@<rig> bash tools/ar/r246_pull.sh
# (law 8: the rig address never lives in tracked files)
set -uo pipefail
RIG="${ARC_RIG:?ARC_RIG=<user>@<rig> must be set}"
R=/root/build
SRC="$R/p246"
SRCB="$R/p246b"
DEST="results/2026-11-26-p647-pd-lmc-mx-boot-gate"

# ---- rig-side gates ------------------------------------------------
ok=$(ssh "$RIG" "grep -c 'VERDICT: PASS' $SRC/logs/r246.out 2>/dev/null" 2>/dev/null || true)
[ "${ok:-0}" -ge 1 ] || { echo "R246_PULL_REFUSED: probe not PASS in $SRC/logs/r246.out"; exit 1; }
ok=$(ssh "$RIG" "grep -c 'R246B_DONE failed=\[\]' $R/logs/r246b3_launch.log 2>/dev/null" 2>/dev/null || true)
[ "${ok:-0}" -ge 1 ] || { echo "R246_PULL_REFUSED: run-3 not R246B_DONE failed=[] (rerun r246b)"; exit 1; }

mkdir -p "$DEST/probe" "$DEST/e2e/pd_refuse" "$DEST/e2e/pd_clean" \
  "$DEST/e2e/run1/pd_refuse" "$DEST/e2e/run1/pd_clean"

# ---- probe half ----------------------------------------------------
scp -q "$RIG:$SRC/logs/r246.out" "$DEST/probe/" || exit 1
scp -q "$RIG:$SRC/out/r246_summary.txt" "$DEST/probe/" || exit 1
scp -q "$RIG:$SRC/out/SHA256SUMS" "$DEST/probe/remote_SHA256SUMS" || exit 1

# ---- e2e half: run-3 cells + run-1/2 history ------------------------
for c in pd_refuse pd_clean; do
  for f in boot.txt outcome.txt posture.txt serve.log; do
    scp -q "$RIG:$SRCB/$c/$f" "$DEST/e2e/$c/" || exit 1
  done
done
for f in warmup.code warmup.json p139_conc_pd_clean.json; do
  scp -q "$RIG:$SRCB/pd_clean/$f" "$DEST/e2e/pd_clean/" || exit 1
done
scp -q "$RIG:$SRCB/SHA256SUMS" "$DEST/e2e/remote_SHA256SUMS" || exit 1
for c in pd_refuse pd_clean; do
  for f in boot.txt outcome.txt posture.txt serve.log; do
    scp -q "$RIG:$SRCB/run1/$c/$f" "$DEST/e2e/run1/$c/" || exit 1
  done
done
for f in warmup.code warmup.json p139_conc_pd_clean.json; do
  scp -q "$RIG:$SRCB/run1/pd_clean/$f" "$DEST/e2e/run1/pd_clean/" || exit 1
done
scp -q "$RIG:$R/logs/r246b_launch.log" "$DEST/e2e/r246b_run1_launch.log" || exit 1
scp -q "$RIG:$R/logs/r246b2_launch.log" "$DEST/e2e/r246b_run2_launch.log" || exit 1
scp -q "$RIG:$R/logs/r246b3_launch.log" "$DEST/e2e/r246b_run3_launch.log" || exit 1
echo "R246_PULL: $(find "$DEST" -type f | wc -l) files pulled"

# ---- byte-exact redaction (perl; MSYS sed injects CRLF) -------------
find "$DEST" -type f \( -name '*.txt' -o -name '*.log' -o -name '*.code' \
  -o -name '*.json' -o -name 'SHA256SUMS*' \) | while read -r f; do
  perl -i -pe 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g' "$f"
done
echo "R246_PULL: redaction pass done"

# ---- SHA256SUMS over the redacted data bytes -------------------------
cd "$DEST" || exit 1
find . -type f ! -name 'SHA256SUMS' ! -name 'VERDICT.md' | sort \
  | xargs sha256sum > SHA256SUMS
echo "R246_PULL: $(wc -l < SHA256SUMS) sums banked"
echo "R246_PULL_DONE $DEST"
