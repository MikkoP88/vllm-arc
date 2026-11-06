#!/bin/bash
# r102_mxfp4_recon5.sh — P6.26 recon part 5: quark dispatch table, OCP-MX
# scheme body, XPU supports_mx/capability, supported_quantization enforcement.
set -uo pipefail
IMG=vllm-arc:v0.30.0-b70-cert10
docker run --rm -i --entrypoint /bin/bash $IMG -s <<'EOS'
SP=/opt/venv/lib/python3.12/site-packages
Q=$SP/vllm/model_executor/layers/quantization/quark
echo "=== A. _get_scheme_cls_from_config table ==="
sed -n "$(grep -n '_get_scheme_cls_from_config' $Q/quark.py | head -2 | tail -1 | cut -d: -f1),+70p" $Q/quark.py
echo
echo "=== B. quark.py from_config fp4 region (lines 128-175) ==="
sed -n '128,175p' $Q/quark.py
echo
echo "=== C. quark_ocp_mx.py full body (210 lines) ==="
sed -n '40,210p' $Q/schemes/quark_ocp_mx.py
echo
echo "=== D. XPU supports_mx + capability ==="
grep -n "supports_mx\|device_capability\|_get_device_capability" $SP/vllm/platforms/xpu.py | head -6
echo
echo "=== E. our platform override (vllm_gaudi/xpu) supports_mx? ==="
python3 -c "
import glob
for f in glob.glob('$SP/vllm_gaudi/xpu/*.py'):
    src = open(f).read()
    if 'supports_mx' in src or 'supported_quantization' in src:
        print(f)
" 2>/dev/null || ls $SP/vllm_gaudi/xpu/ 2>/dev/null || echo "vllm_gaudi not in image venv (pip show?)"
echo
echo "=== F. supported_quantization enforcement site ==="
grep -rn "supported_quantization" $SP/vllm/config/*.py $SP/vllm/platforms/interface.py 2>/dev/null | grep -v "def \|#\|xpu.py\|rocm.py\|cuda.py" | head -10
EOS
