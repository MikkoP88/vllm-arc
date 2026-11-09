#!/bin/bash
# run_cut10.sh -- P6.34 build9 (SLM wheel) post-processing: unpack the arc9 wheel
# into arc-jm7, run the dual-variant gate (stage A x2 variants + stage B
# jm5/jm14 x SGF sweep), bank artifacts to /root/build/r128a/, render
# the cut-10 report. Run ON THE RIG after watch8.sh reports the wheel.
set -uo pipefail
W=vllm_xpu_kernels-0.1.14.1+arc9-cp312-cp312-linux_x86_64.whl
R=/root/build/r128a

# 1. wheel out of the builder (byte-exact, sha256 banked)
docker exec vxk-arc3-builder cat /wheels/$W > $R/$W
sha256sum $R/$W | tee $R/arc9_wheel.sha256

# 2. into arc-jm7 as an unpacked PYTHONPATH dir (certified venv shadowed,
#    not touched)
docker cp $R/$W arc-jm7:/tmp/$W
docker exec arc-jm7 bash -c "cd /tmp && rm -rf arc9wheel && mkdir arc9wheel && cd arc9wheel && /opt/venv/bin/python -m zipfile -e /tmp/$W . && rm /tmp/$W"

# 3. import + op probe
docker exec -e PYTHONPATH=/tmp/arc9wheel arc-jm7 /opt/venv/bin/python -c "
import importlib.metadata as md
import torch
import vllm_xpu_kernels._xpu_C  # noqa
print('version', md.version('vllm-xpu-kernels'))
assert hasattr(torch.ops._xpu_C, 'jm_fp8_gemm_w8a16'), 'w16'
assert hasattr(torch.ops._xpu_C, 'jm_fp8_gemm_w8a16_mxfp8'), 'm816'
assert hasattr(torch.ops._xpu_C, 'jm_fp4_gemm_w4a16'), 'g16'
print('OPS_OK')" || exit 1

# 4. gate cut 10 (dual variant; log tee'd to the rig bank)
docker cp $R/r128b_jm_gate.py arc-jm7:/tmp/r128b_jm_gate.py
docker exec -e PYTHONPATH=/tmp/arc9wheel -e R128B_OUT=/tmp/cut10_rows.json \
  arc-jm7 /opt/venv/bin/python /tmp/r128b_jm_gate.py 2>&1 \
  | tee $R/r128b_cut10.log

# 5. artifacts home + report
docker cp arc-jm7:/tmp/cut10_rows.json $R/r128b_cut10_rows.json
python3 $R/r128p_cut9_report.py $R/r128b_cut10_rows.json \
  | tee $R/r128p_cut9_report.txt
echo "CUT9_RUNBOOK_DONE $(date)"
