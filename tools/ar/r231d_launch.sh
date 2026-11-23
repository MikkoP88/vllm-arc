#!/bin/bash
# r231d_launch.sh -- P6.47 r231d chain: apply the seqused-tightened
# split clamp to the pristine builder tree, build the build28 wheel,
# run the offline gate battery. One gated chain, banked log; run
# detached (setsid) from the client. Each stage refuses unless the
# previous one exited 0.
set -uo pipefail
P=/root/build/p647/r231d
LOG=$P/launch.log

exec >> "$LOG" 2>&1
echo "R231D_LAUNCH_START $(date +%H:%M:%S)"

# ---- 1. apply the patch (host python3 against the live tree) --------
python3 "$P/r231d_apply.py" || { echo "R231D_LAUNCH_ABORT apply rc=$?"; exit 1; }

# ---- 2. build28 (incremental; ~43 min) -------------------------------
bash "$P/r231d_build.sh" || { echo "R231D_LAUNCH_ABORT build rc=$?"; exit 1; }

# ---- 3. offline gates (arc window + bare container + restore) --------
bash "$P/r231d_offline.sh" || { echo "R231D_LAUNCH_ABORT offline rc=$?"; exit 1; }

echo "R231D_LAUNCH_DONE $(date +%H:%M:%S)"
