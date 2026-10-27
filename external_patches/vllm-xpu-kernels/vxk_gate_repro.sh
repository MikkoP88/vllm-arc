#!/bin/bash
# vxk_gate_repro.sh — §6 device-gate: donor repro_v88_int32.py (cap727.pt)
# against a vxk wheel, inside a fresh container from the arc image (device
# access + torch 2.13 runtime). Gate: exit 0 CLEAN (LO+HI+BND).
#
# usage: vxk_gate_repro.sh fixed|bare <wheel-file-or-pypi-spec> [noseed]
#   fixed  — install the given wheel file, run the full gate
#   bare   — (positive control, run LAST: may DEVICE_LOST) install spec,
#            expect exit 1 BUGGY — proves the repro detects on this stack
set -uo pipefail

MODE=${1:?usage: vxk_gate_repro.sh fixed|bare <wheel-or-spec> [noseed]}
SPEC=${2:?wheel file (fixed) or pip spec (bare)}
NOSEED=${3:-}

OUT=/root/build/vxk_gate_${MODE}$([ -n "$NOSEED" ] && echo _noseed)
rm -rf "$OUT"; mkdir -p "$OUT"

R0=$(dmesg | grep -c "Engine reset" || true)
echo "resets_before=$R0" | tee "$OUT/env.txt"

docker run --rm --device /dev/dri --ipc host \
  -v /root/build/wheels-arc-vxk:/wheels:ro \
  -v /root/build/lce1:/caps:ro \
  -v /root/build/repro_v88_int32.py:/repro.py:ro \
  -v "$OUT":/out \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-p12 -c "
    set -xo pipefail
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
    /opt/venv/bin/pip install --force-reinstall --no-deps $SPEC 2>&1 | tail -2
    /opt/venv/bin/pip show vllm-xpu-kernels | sed -n 1,2p
    cd /out
    set +e
    /opt/venv/bin/python /repro.py /caps/cap727.pt /out $NOSEED > repro.log 2>&1
    rc=\$?
    set -e
    echo REPRO_EXIT=\$rc | tee -a repro.log
    exit \$rc
  " 2>&1 | tee "$OUT/gate_run.log"
rc=${PIPESTATUS[0]}
R1=$(dmesg | grep -c "Engine reset" || true)
echo "resets_after=$R1 exit=$rc" | tee -a "$OUT/env.txt"
exit $rc
