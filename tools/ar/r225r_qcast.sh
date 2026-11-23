#!/bin/bash
# r225r_qcast.sh — P6.47 r225 leg r: q-cast sensitivity quantifier
# (CPU-only container; W14 stays up, no GPU device passed).
#
# Usage: r225r_qcast.sh
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert26
BANK=$R/p647/r224a
OUT=$R/p647/r225r
TOOLS=$R/r225_tools

mkdir -p "$OUT"

fail() { echo "R225R_ABORT: $*"; }

docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
[ -f "$TOOLS/r225r_qcast.py" ] || { fail "r225r_qcast.py not shipped"; exit 1; }
[ -f "$TOOLS/r225j_iso.py" ] || { fail "r225j_iso.py not shipped"; exit 1; }
for f in mxkv_ab.2447.18.pt mxkv_ab.2447.30.pt mxkv_ab.2547.18.pt; do
  [ -f "$BANK/$f" ] || { fail "banked payload missing ($BANK/$f)"; exit 1; }
done
echo "R225R refuses passed"

docker run --rm \
  -v "$BANK:/bank:ro" \
  -v "$TOOLS:/tools:ro" \
  -v "$OUT:/out" \
  --entrypoint /opt/venv/bin/python "$IMG" \
  /tools/r225r_qcast.py \
  /bank/mxkv_ab.2447.18.pt /bank/mxkv_ab.2447.30.pt /bank/mxkv_ab.2547.18.pt \
  > "$OUT/qcast.log" 2>&1
rc=$?
cat "$OUT/qcast.log"
[ $rc -eq 0 ] || { fail "qcast rc=$rc"; exit 1; }
grep -q "R225R_QCAST_DONE" "$OUT/qcast.log" \
  || { fail "qcast produced no DONE line"; exit 1; }
n=$(grep -c "^R225R mxkv" "$OUT/qcast.log" || true)
[ "$n" -eq 3 ] || { fail "expected 3 payload lines, got $n"; exit 1; }
echo "R225R: q-cast quantification complete"
