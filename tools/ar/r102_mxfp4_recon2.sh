#!/bin/bash
# r102_mxfp4_recon2.sh — P6.26 recon part 2: quark traces, registry keys,
# the real XPU mxfp4 kernel file, unknown-quant error path.
set -uo pipefail
IMG=vllm-arc:v0.30.0-b70-cert10
docker run --rm -i --entrypoint /bin/bash $IMG -s <<'EOS'
SP=/opt/venv/lib/python3.12/site-packages
echo "=== A. quark anywhere in core vllm? ==="
grep -rln "quark" $SP/vllm --include=*.py -i | head -8 || echo "no hits"
echo
echo "=== B. quant registry keys + XPU supported list ==="
python3 - <<'PY'
from vllm.model_executor.layers.quantization import QUANTIZATION_REGISTRY
print("registry keys:", sorted(QUANTIZATION_REGISTRY.keys()))
from vllm.platforms.xpu import XPUPlatform
print("xpu supported:", sorted(XPUPlatform.supported_quantization))
PY
echo
echo "=== C. mxfp4 linear kernel files ==="
find $SP/vllm -path "*mxfp4*" -name "*.py" | grep -i "linear\|kernels" | head -10
echo
echo "=== D. XPUMxFp4LinearKernel can_implement ==="
K=$(grep -rln "class XPUMxFp4LinearKernel" $SP/vllm --include=*.py | head -1)
echo "file: $K"
grep -n "def can_implement" -A 20 "$K" 2>/dev/null | head -26
echo
echo "=== E. unknown quant_method resolution error ==="
F=$SP/vllm/model_executor/models/utils.py
grep -rn "def get_quantization_config\|quant_method = quant_config.get(\|not in the registry\|ValueError" $SP/vllm/config/quantization.py 2>/dev/null | head -6
grep -rn "get_quantization_config" $SP/vllm/model_executor/models/utils.py | head -3
EOS
