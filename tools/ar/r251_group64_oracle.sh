#!/bin/bash
# r251_group64_oracle.sh -- P-I cycle 2 (r251): run the group-64
# E8M0 offline band oracle (tools/ar/p147_mxkv/p251_group64_oracle.py)
# in the cert28 container on the rig and pull the bank. CPU-only
# (bit-exact element math; no XPU requested) so the W14 incumbent
# is NOT quiesced -- recorded before/after, never touched. No ports,
# no serve boot, no P-D overlay: the oracle measures the CERTIFIED
# wheel's own mx_quant reference (file-path load). Ship and run in
# separate ssh calls with md5 verify (ssh-ship-race law).
# Usage: ARC_RIG=root@<rig> bash tools/ar/r251_group64_oracle.sh
set -uo pipefail
RIG=${ARC_RIG:?usage: ARC_RIG=root@<rig> bash tools/ar/r251_group64_oracle.sh}
IMG=vllm-arc:v0.30.0-b70-cert28
PYL=tools/ar/p147_mxkv/p251_group64_oracle.py
BANK=results/2026-11-26-p647-pi-r251-group64-band
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
SSH="ssh -o StrictHostKeyChecking=no $RIG"
SCP="scp -o StrictHostKeyChecking=no"
RDIR=/root/build/p251_stage

md5() { md5sum "$1" 2>/dev/null | cut -d" " -f1; }

echo "=== [r251] W14 health snapshot (never quiesced) ==="
W14_BEFORE=$($SSH "docker inspect radiance-test --format '{{.State.Status}}' 2>/dev/null")
echo "radiance-test: $W14_BEFORE"

echo "=== [r251] arc-leg-in-flight refusal gate ==="
$SSH "docker ps --format '{{.Names}}' | grep -E 'arc-serve|arc-probe|p249_gate|r249' | grep -v radiance" && {
  echo "R251_REFUSE_ARC_LEG_IN_FLIGHT"; exit 1; }

echo "=== [r251] ship the oracle (md5 verify both sides) ==="
LMD5=$(md5 "$PYL"); echo "local  md5 $LMD5"
$SSH "mkdir -p $RDIR"
$SCP "$PYL" "$RIG:$RDIR/p251_group64_oracle.py"
RMD5=$($SSH "md5sum $RDIR/p251_group64_oracle.py | cut -d' ' -f1")
echo "remote md5 $RMD5"
[ "$LMD5" = "$RMD5" ] || { echo "R251_REFUSE_MD5_MISMATCH"; exit 1; }

echo "=== [r251] run the oracle in $IMG (CPU-only, --rm) ==="
$SSH "cd /root/build && rm -rf $RDIR/out && \
  docker run --rm \
    -v $RDIR:/work:rw \
    --entrypoint /bin/bash \
    $IMG -c '/opt/venv/bin/python /work/p251_group64_oracle.py --out /work/out' \
    > $RDIR/stdout.log 2>&1; echo RC=\$?"
$SSH "tail -20 $RDIR/stdout.log; grep -c P251_ORACLE_DONE $RDIR/stdout.log"

echo "=== [r251] pull the bank ==="
mkdir -p "$BANK"
$SCP "$RIG:$RDIR/stdout.log" "$BANK/p251_stdout.log"
$SCP "$RIG:$RDIR/out/p251_group64.json" "$BANK/p251_group64.json"
$SCP "$RIG:$RDIR/p251_group64_oracle.py" "$BANK/p251_group64_oracle.py.shipped"

echo "=== [r251] law-8 redaction + sums ==="
perl -pi -e 's/\b10\.20\.3\.65\b/RIG_ADDR/g; s/\b10\.20\.3\.\d+\b/CLIENT_ADDR/g; s/\b10\.100\.8\.\d+\b/10.100.8.REDACTED/g' \
  "$BANK"/*.log "$BANK"/*.json 2>/dev/null || true
( cd "$BANK" && sha256sum * > SHA256SUMS )

W14_AFTER=$($SSH "docker inspect radiance-test --format '{{.State.Status}}' 2>/dev/null")
echo "radiance-test after: $W14_AFTER"
[ "$W14_BEFORE" = "$W14_AFTER" ] || { echo "R251_W14_STATE_CHANGED"; exit 1; }
echo "R251_DONE -> $BANK"
