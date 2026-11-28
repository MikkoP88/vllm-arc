#!/bin/bash
# r272_ship.sh -- run LOCALLY (git-bash) to install the r272 skin-law
# serve script on the rig, AFTER the r271 cert30 battery completes.
# The midflight-script-edit law forbids touching the live
# /root/build/serve_arc_b70.sh while any gate can still invoke it (the
# r271 gate pins md5 c2db777506e776a86426852b6bb80de4 at pre-flight and
# calls `bash $R/serve_arc_b70.sh` per leg), so this tool REFUSES until
# the launch log carries R271_DONE AND a W14_RESTORE rc=0 line.
#
# What it does (all md5-verified, ship and verify in SEPARATE ssh calls
# per the ssh-ship-race law):
#   1. refuses if the local serve script / selftest are dirty
#      (ships exactly the committed r272 bytes, PR #190);
#   2. refuses until R271_DONE + W14_RESTORE rc=0;
#   3. ships the serve script + tools/ar/r272_skin_selftest.sh to
#      /root/build/r272_ship/tools{,/ar}/ (the selftest's relative
#      layout), verifies the remote md5 == the local md5;
#   4. runs the selftest RIG-SIDE over the shipped bytes -- the 8/8
#      posture proof executed on the install target, bankable as
#      `R272_RIG_SELFTEST_OK`;
#   5. backs up the live pinned bytes to serve_arc_b70.r271_c2db7775,
#      installs, verifies the live md5, prints R272_NEXT_GATE_PIN (the
#      md5 every FUTURE gate must pin -- the r271 gate's c2db7775 pin
#      is retired with this ship).
# The rig's W14 incumbent is untouched (the gate's exit trap already
# restored it).
# Rig address from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r272_ship.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
S=$REPO/tools/serve_arc_b70.sh
T=$REPO/tools/ar/r272_skin_selftest.sh
LOG=/root/build/p647/cert30_r271/logs/launch.log
R=/root/build                     # the live serve-script home the gates pin
LIVE=$R/serve_arc_b70.sh
BACKUP=$R/serve_arc_b70.r271_c2db7775

echo "== r272 serve-script ship"
[ -f "$S" ] && [ -f "$T" ] || { echo "R272_NO_LOCAL_BYTES ($S / $T)"; exit 1; }

echo "== refusing dirty local bytes (ships exactly the committed r272 fix)"
git -C "$REPO" diff --quiet -- tools/serve_arc_b70.sh tools/ar/r272_skin_selftest.sh \
  || { echo "R272_SHIP_REFUSED (serve script / selftest dirty -- commit first)"; exit 1; }
MD5=$(md5sum "$S" | cut -d" " -f1)
echo "R272_LOCAL_MD5 $MD5"

echo "== refusing until the r271 battery is DONE and W14 is restored"
STATE=$(ssh "$RIG" "grep -E '^(R271_DONE|W14_RESTORE rc=)' $LOG 2>/dev/null | cut -c1-120 || true" < /dev/null)
echo "$STATE" | grep -q "^R271_DONE" \
  || { echo "R272_SHIP_REFUSED (r271 battery still live)"; exit 2; }
echo "$STATE" | grep -q "^W14_RESTORE rc=0" \
  || { echo "R272_SHIP_REFUSED (W14 restore not confirmed rc=0)"; exit 3; }

echo "== shipping (layout for the selftest's relative path)"
ssh "$RIG" "mkdir -p $R/r272_ship/tools/ar" < /dev/null
scp -q "$S" "$RIG:$R/r272_ship/tools/serve_arc_b70.sh"
scp -q "$T" "$RIG:$R/r272_ship/tools/ar/r272_skin_selftest.sh"
GOT=$(ssh "$RIG" "md5sum $R/r272_ship/tools/serve_arc_b70.sh | cut -d' ' -f1" < /dev/null)
[ "$GOT" = "$MD5" ] \
  || { echo "R272_SHIP_MD5_MISMATCH rig=$GOT local=$MD5"; exit 1; }
echo "R272_SHIP_MD5_OK $GOT"

echo "== rig-side executable proof (the 8 postures over the shipped bytes)"
OUT=$(ssh "$RIG" "bash $R/r272_ship/tools/ar/r272_skin_selftest.sh" < /dev/null)
echo "$OUT"
echo "$OUT" | grep -q "R272_SKIN_SELFTEST pass=8 fail=0" \
  || { echo "R272_RIG_SELFTEST_FAIL"; exit 1; }
echo "R272_RIG_SELFTEST_OK"

echo "== installing over the live copy (pinned bytes backed up first)"
[ "$(ssh "$RIG" "md5sum $LIVE 2>/dev/null | cut -d' ' -f1" < /dev/null)" = "c2db777506e776a86426852b6bb80de4" ] \
  || echo "   NOTE: live serve script is NOT the r271-pinned c2db7775 (check before continuing)"
ssh "$RIG" "cp -f $LIVE $BACKUP && cp -f $R/r272_ship/tools/serve_arc_b70.sh $LIVE && \
  md5sum $LIVE $BACKUP $R/r272_ship/tools/serve_arc_b70.sh" < /dev/null
echo "R272_SHIPPED live_md5=$MD5"
echo "R272_BACKUP $BACKUP (the r271-pinned bytes)"
echo "R272_NEXT_GATE_PIN $MD5   # every FUTURE gate pins this md5, not c2db7775"
