#!/bin/bash
# r102_mxfp4_recon3.sh — P6.26 recon part 3: the quark PACKAGE (config +
# schemes), registry membership, XPU supported list.
set -uo pipefail
IMG=vllm-arc:v0.30.0-b70-cert10
docker run --rm -i --entrypoint /bin/bash $IMG -s <<'EOS'
SP=/opt/venv/lib/python3.12/site-packages
Q=$SP/vllm/model_executor/layers/quantization
echo "=== A. quark package tree ==="
find $Q/quark -name "*.py" | sort
echo
echo "=== B. registry: is quark a key? ==="
grep -n "\"quark\"\|'quark'" $SP/vllm/model_executor/layers/quantization/__init__.py | head -5
grep -n "QUANTIZATION_METHODS" -A 3 $SP/vllm/model_executor/layers/quantization/__init__.py | head -12
echo
echo "=== C. XPU supported_quantization list (full) ==="
grep -n "supported_quantization" -A 20 $SP/vllm/platforms/xpu.py | head -30
echo
echo "=== D. quark config: quant_method name + is_layer_skipped + scheme pick ==="
QC=$(find $Q/quark -maxdepth 1 -name "*.py" ! -name "__init__*" | head -1)
echo "config file: $QC"
grep -n "name\|fp4\|float4\|mxfp4\|dtype\|def get_quant_method\|scheme\|rocm\|platform\|device" $QC | head -40
echo
echo "=== E. quark scheme selection body ==="
grep -n "def get_scheme\|def _select_scheme\|is_fp4\|is_nvfp4\|is_mx" -A 25 $QC | head -70
echo
echo "=== F. quark mxfp4 scheme present? ==="
ls $Q/quark/schemes/
echo
echo "=== G. CPU-only import test of quark config ==="
python3 - <<'PY'
from vllm.model_executor.layers.quantization.quark.quark import QuarkConfig 2>/dev/null or None
PY
python3 -c "
import importlib
m = importlib.import_module('vllm.model_executor.layers.quantization.quark.quark')
print('QuarkConfig file:', m.__file__)
import inspect
src = inspect.getsource(m.QuarkConfig)
import re
print('has get_quant_method:', 'def get_quant_method' in src)
print('fp4 mentions:', len(re.findall(r'[Ff][Pp]4|mxfp4|float4', src)))
" 2>&1 | tail -4
EOS
