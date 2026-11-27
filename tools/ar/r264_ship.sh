#!/bin/bash
# r264_ship.sh -- run LOCALLY (git-bash) BEFORE r264_cert29_gate.sh and
# (re-)running r263_build_cert29.sh on the rig. Ships the rig-side
# scripts with an md5 check on the far side and a bash -n there:
#   tools/ar/r263_build_cert29.sh -> /root/build/r263_build_cert29.sh
#   tools/ar/r262_w14_restore.sh  -> /root/build/r262_w14_restore.sh
#   tools/ar/r264_cert29_gate.sh  -> /root/build/r264_cert29_gate.sh
# Ship ONLY (ssh-ship-race law: launch in its own ssh call). Rig address
# from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r264_ship.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
SUMS=$(mktemp)
for f in r263_build_cert29.sh r262_w14_restore.sh r264_cert29_gate.sh; do
  bash -n "tools/ar/$f"
  scp -q "tools/ar/$f" "$RIG:/root/build/$f"
  printf '%s  %s\n' "$(md5sum "tools/ar/$f" | cut -d" " -f1)" "$f" >> "$SUMS"
done
REMOTE=$(ssh "$RIG" 'cd /root/build && md5sum r263_build_cert29.sh r262_w14_restore.sh r264_cert29_gate.sh')
diff <(sort "$SUMS") <(echo "$REMOTE" | sort) \
  && echo "R264_SHIP_OK md5s equal on both sides:" || { echo "R264_SHIP_MD5_MISMATCH"; exit 1; }
cat "$SUMS"
ssh "$RIG" 'cd /root/build && for f in r263_build_cert29.sh r262_w14_restore.sh r264_cert29_gate.sh; do bash -n "$f" && echo "R264_SYNTAX_OK $f"; done'
rm -f "$SUMS"
