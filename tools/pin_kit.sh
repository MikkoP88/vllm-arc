#!/bin/bash
# pin_kit.sh — (re)generate the md5 pin manifest for the arc wheel set.
# D-chain governance port (plan §6): every external artifact pinned by md5;
# the manifest is committed (external_patches/wheels.pins.md5) and enforced
# at image-build time by tools/wheel_guard.sh.
#
# Usage (on the rig, against the staged build context wheels):
#   pin_kit.sh /root/build/arc_build/wheels > external_patches/wheels.pins.md5
set -euo pipefail

dir=${1:?usage: pin_kit.sh <wheels-dir>}
cd "$dir"
echo "# vllm-arc wheel pins — ground truth = the validated radiance v0.30.0 image"
echo "# venv (captured 2026-10-27, results/2026-10-27-p0-venv-groundtruth/)."
echo "# Regenerate: tools/pin_kit.sh <wheels-dir>"
md5sum -- * | sort -k2
