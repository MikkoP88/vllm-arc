#!/bin/bash
# r102_mxfp4_recon4.sh — P6.26 recon part 4: QuarkConfig scheme selection,
# the two mxfp4 schemes' kernel paths, and the XPU platform gate site.
set -uo pipefail
IMG=vllm-arc:v0.30.0-b70-cert10
docker run --rm -i --entrypoint /bin/bash $IMG -s <<'EOS'
SP=/opt/venv/lib/python3.12/site-packages
Q=$SP/vllm/model_executor/layers/quantization/quark
echo "=== A. quark.py scheme selection (dtype->scheme table) ==="
grep -n "def get_scheme\|def _select\|is_mx\|is_nvfp4\|is_w4a16\|is_w8a8\|is_fp8\|is_int\|elif\|if " $Q/quark.py | sed -n '1,60p'
echo
echo "=== B. quark.py full get_scheme region ==="
sed -n "$(grep -n 'def get_scheme' $Q/quark.py | head -1 | cut -d: -f1),+80p" $Q/quark.py
echo
echo "=== C. quark_ocp_mx.py — head + kernel usage ==="
wc -l $Q/schemes/quark_ocp_mx.py
grep -n "import\|class \|def \|aiter\|triton\|init_mxfp4\|current_platform\|is_rocm\|is_cuda\|device\|ValueError\|raise" $Q/schemes/quark_ocp_mx.py | head -40
echo
echo "=== D. quark_w4a8_mxfp4_fp8.py — head + kernel usage ==="
wc -l $Q/schemes/quark_w4a8_mxfp4_fp8.py
grep -n "import\|class \|def \|aiter\|triton\|init_mxfp4\|current_platform\|is_rocm\|is_cuda\|device\|ValueError\|raise" $Q/schemes/quark_w4a8_mxfp4_fp8.py | head -40
echo
echo "=== E. where does platform reject unsupported quant? ==="
grep -rn "supported_quantization" $SP/vllm/config/model.py | head -6
grep -rn "not supported on\|get_supported_quantization" $SP/vllm/config/model.py $SP/vllm/platforms/interface.py 2>/dev/null | head -8
EOS
