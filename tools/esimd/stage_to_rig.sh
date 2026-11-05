#!/bin/bash
# stage_to_rig.sh — push the p72 kernel tree into the rig's esimd container
# at /src/esimd-fa (donor /src/esimd stays untouched — read-only law).
# Usage (from this dir, askpass law applies):
#   bash stage_to_rig.sh
# Then build+probe INSIDE the container:
#   docker exec esimd-w11 bash -lc 'cd /src/esimd-fa && bash fa_dense_build.sh'
#   docker exec esimd-w11 /opt/venv/bin/python /src/esimd-fa/probe_fa_dense.py
# GPU probes require stopping the W14 incumbent first (idle-window law);
# the build itself is CPU-only and may run any time.
set -euo pipefail
cd "$(dirname "$0")"
SRC=../../vllm_gaudi/xpu/esimd_fa
# RIG = the rig address — lives ONLY in the untracked layer (CLAUDE.local.md
# "Rig" section); tracked files never carry it (§11 rule 9).
: "${RIG:?set RIG=<rig address> from the untracked layer}"
R="root@${RIG}"

export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
ssh "$R" 'docker exec esimd-w11 sh -c "rm -rf /src/esimd-fa /src/esimd_fa && mkdir -p /src/esimd-fa"'
tar -C "$SRC/.." -czf - esimd_fa \
  | ssh "$R" 'docker exec -i esimd-w11 sh -c "tar -xzf - -C /src && mv /src/esimd_fa/* /src/esimd-fa/ && rmdir /src/esimd_fa"'
scp probe_fa_dense.py fa_dense_build.sh "$R":/tmp/
ssh "$R" 'docker cp /tmp/probe_fa_dense.py esimd-w11:/src/esimd-fa/ && docker cp /tmp/fa_dense_build.sh esimd-w11:/src/esimd-fa/ && rm /tmp/probe_fa_dense.py /tmp/fa_dense_build.sh'
ssh "$R" 'docker exec esimd-w11 sh -c "ls -la /src/esimd-fa/ /src/esimd-fa/csrc/"'
echo STAGED_OK
