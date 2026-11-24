#!/bin/bash
# r232_cert28_launch.sh -- P6.47 cert28 chain: build the cert28
# image (cert27 + the r231d build28 clamp wheel), run the section-10
# gate battery, run the perf re-discrimination + shallow record
# faces. One gated chain, banked log; run detached (setsid) from
# the client. Each stage refuses unless the previous one exited 0.
set -uo pipefail
R=/root/build
P=$R/p647/cert28chain
LOG=$P/launch.log
FORK_SHA=43f84cf977f9444900ff21387715ca5865db4c1a

mkdir -p "$P"
exec >> "$LOG" 2>&1
echo "R232C_LAUNCH_START $(date +%H:%M:%S)"

# ---- 1. build cert28 (docker build; ~3 min) -------------------------
bash "$P/r232_build_cert28.sh" "$FORK_SHA" \
  || { echo "R232C_LAUNCH_ABORT build rc=$?"; exit 1; }

# ---- 2. section-10 gate battery (~2 h; 8 legs) ----------------------
bash "$P/r233_cert28_gate.sh" "$FORK_SHA" \
  || { echo "R232C_LAUNCH_ABORT gate rc=$?"; exit 1; }

# ---- 3. perf re-discrimination + shallow record faces (~50 min) -----
bash "$P/r234_cert28_itl.sh" \
  || { echo "R232C_LAUNCH_ABORT itl rc=$?"; exit 1; }

echo "R232C_LAUNCH_DONE $(date +%H:%M:%S)"
