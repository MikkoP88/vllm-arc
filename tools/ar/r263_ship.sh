#!/bin/bash
# r263_ship.sh -- run LOCALLY (git-bash) BEFORE r263_build_cert29.sh.
# Stages the six vllm_gaudi files of the cert29 content commit STRAIGHT
# FROM THE GIT BLOBS (git show <commit>:<path> -- never the working
# tree, so a stray edit cannot ride into the image), pins them by md5
# against the build script's table, ships them flat to
# /root/build/cert29stage_r263 on the rig, and ships the build script
# to /root/build/r263_build_cert29.sh. Every transfer is md5-verified
# on the far side. Ship ONLY: the build is launched in its own ssh call
# (ssh-ship-race law: a cat-chain and a backgrounded launch in one ssh
# call empties the remote file). Rig address from the caller's
# environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r263_ship.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
COMMIT=${1:-4dd3a1b9710f1c638f7f0b5ac1885525cf57c3bc}
RSTAGE=/root/build/cert29stage_r263
RSCRIPT=/root/build/r263_build_cert29.sh

cd "$REPO"
git cat-file -e "$COMMIT^{commit}" || { echo "R263_NO_COMMIT $COMMIT"; exit 1; }

# blob -> flat name -> md5 (the build script's table; r254 pins for the
# four unchanged files, the r262 blobs for the two changed ones)
TABLE="
vllm_gaudi/envs.py envs.py 31ca4ac7cfcd996aed6446d4b24c2bcd
vllm_gaudi/xpu/platform.py platform.py 0733f0b8c7dd2fb89422bafa56ec35f5
vllm_gaudi/xpu/patches/lmc_mx_boot_gate.py lmc_mx_boot_gate.py 04408d2e12cbdd8587cac4633624b7bc
vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py vllm_v1_adapter.py 7ca7cff4e1e04d4e0ea73573b1234664
vllm_gaudi/xpu/patches/mx_kv/engine.py engine.py d1fa3c17cde044211cbf62e28ddcc1b1
vllm_gaudi/xpu/patches/mx_kv/native_decode.py native_decode.py 81e427590e7ad3482e46bdc25f4a62b2
"
STAGE=$(mktemp -d)
SUMS="$STAGE.md5s"
: > "$SUMS"
echo "== staging the six blobs of $COMMIT"
while read -r BLOB FLAT WANT; do
  [ -n "$BLOB" ] || continue
  git show "$COMMIT:$BLOB" > "$STAGE/$FLAT"
  GOT=$(md5sum "$STAGE/$FLAT" | cut -d" " -f1)
  [ "$GOT" = "$WANT" ] || { echo "R263_LOCAL_MD5_MISMATCH $BLOB $GOT != $WANT"; exit 1; }
  echo "$WANT  $FLAT" >> "$SUMS"
done <<< "$TABLE"
cat "$SUMS"

echo "== shipping the stage dir -> $RIG:$RSTAGE"
ssh "$RIG" "rm -rf $RSTAGE && mkdir -p $RSTAGE"
tar -C "$STAGE" -cf - . | ssh "$RIG" "tar -C $RSTAGE -xf -"
REMOTE=$(ssh "$RIG" "cd $RSTAGE && md5sum envs.py platform.py lmc_mx_boot_gate.py vllm_v1_adapter.py engine.py native_decode.py")
diff <(sort "$SUMS") <(echo "$REMOTE" | sort) \
  && echo "R263_STAGE_SHIPPED md5s equal on both sides" \
  || { echo "R263_STAGE_REMOTE_MD5_MISMATCH"; exit 1; }

echo "== shipping the build script -> $RIG:$RSCRIPT"
scp -q tools/ar/r263_build_cert29.sh "$RIG:$RSCRIPT"
LOCAL_MD5=$(md5sum tools/ar/r263_build_cert29.sh | cut -d" " -f1)
REMOTE_MD5=$(ssh "$RIG" "md5sum $RSCRIPT | cut -d' ' -f1")
[ "$LOCAL_MD5" = "$REMOTE_MD5" ] || { echo "R263_SCRIPT_MD5_MISMATCH local=$LOCAL_MD5 remote=$REMOTE_MD5"; exit 1; }
echo "R263_SCRIPT_SHIPPED md5=$LOCAL_MD5"
ssh "$RIG" "bash -n $RSCRIPT && echo R263_SCRIPT_SYNTAX_OK"
rm -rf "$STAGE" "$SUMS"
echo "R263_SHIP_DONE (launch the build in its own ssh call: bash $RSCRIPT $COMMIT)"
