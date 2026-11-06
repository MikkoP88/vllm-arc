#!/bin/bash
# r102_mxfp4_recon6.sh — P6.26 recon part 6: ocp_mx_linear assignment,
# supports_mx on XPU/our platform, supported_quantization enforcement,
# MxFp4LinearKernel dispatch logic.
set -uo pipefail
IMG=vllm-arc:v0.30.0-b70-cert10
docker run --rm -i -v /root/build/p626:/p626:ro --entrypoint /bin/bash $IMG -s <<'EOS'
SP=/opt/venv/lib/python3.12/site-packages
Q=$SP/vllm/model_executor/layers/quantization/quark
echo "=== A. quark_ocp_mx.py tail (129-210): ocp_mx_linear + apply ==="
sed -n '129,210p' $Q/schemes/quark_ocp_mx.py
echo
echo "=== B. supports_mx defs (interface + xpu + ours) ==="
grep -rn "def supports_mx" -A 8 $SP/vllm/platforms/interface.py | head -14
grep -rn "supports_mx" $SP/vllm/platforms/xpu.py | head -4
echo "--- our vllm_gaudi xpu platform ---"
find / -maxdepth 6 -path "*vllm_gaudi/xpu/platform.py" 2>/dev/null | head -2
G=$(find / -maxdepth 6 -path "*vllm_gaudi/xpu/platform.py" 2>/dev/null | head -1)
grep -n "supports_mx\|supported_quantization\|class \|capability" "$G" 2>/dev/null | head -12
echo
echo "=== C. supported_quantization enforcement anywhere ==="
grep -rn "supported_quantization" $SP/vllm --include=*.py -l | head -8
grep -rn "supported_quantization" $SP/vllm/config/quantization.py 2>/dev/null | head -6
echo
echo "=== D. MxFp4LinearKernel dispatch (kernels/linear/__init__.py) ==="
grep -n "supports_mx\|XPU\|XPUMxFp4\|priority\|def init_mxfp4_linear_kernel" -B2 -A6 $SP/vllm/model_executor/kernels/linear/__init__.py | sed -n '1,70p'
echo
echo "=== E. staging progress ==="
tail -2 /p626/stage.log 2>/dev/null || echo "no stage log mounted"
EOS
