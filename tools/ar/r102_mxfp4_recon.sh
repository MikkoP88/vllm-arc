#!/bin/bash
# r102_mxfp4_recon.sh — P6.26 recon (NO GPU): read the wheel's handlers for
# the two named checkpoints' quant formats + arch precedent.
#   - amd/Qwen3.8-27B-Quark-AWQ-MXFP4   -> quant_method "quark", fp4 real_quantized
#   - RedHatAI/Qwen3.8-27B-MXFP4xFP8    -> compressed-tensors mixed-precision
#                                         (mlp mxfp4 w + fp8 dyn act; attn fp8-block w)
set -uo pipefail
IMG=vllm-arc:v0.30.0-b70-cert10
docker run --rm -i -v /models:/models:ro --entrypoint /bin/bash $IMG -s <<'EOS'
SP=/opt/venv/lib/python3.12/site-packages
echo "=== 1. our certified fp8 target arch (precedent for ForConditionalGeneration?) ==="
python3 - <<'PY'
import json, glob
for p in ["/models/qwen3.8-27b-fp8/config.json", "/models/qwen3.8-27b/config.json"]:
    try:
        c = json.load(open(p))
        q = c.get("quantization_config", {})
        print(p)
        print("  architectures:", c.get("architectures"), "| model_type:", c.get("model_type"))
        print("  quant_method:", q.get("quant_method") or q.get("format"), "| has vision:", "vision_config" in c)
        if "config_groups" in q:
            for g, spec in q["config_groups"].items():
                w = spec.get("weights", {})
                a = spec.get("input_activations")
                print(f"   group {g}: fmt={spec.get('format')} w={w.get('num_bits')}bit/{w.get('strategy')}/{w.get('group_size')}"
                      + (f" act={a.get('num_bits')}bit/{a.get('strategy')}/{a.get('group_size')}/dyn={a.get('dynamic')}" if a else " act=None"))
    except FileNotFoundError:
        print(p, "MISSING")
PY
echo
echo "=== 2. quark.py — fp4/mxfp4 surface ==="
Q=$SP/vllm/model_executor/layers/quantization/quark.py
ls -la $Q 2>/dev/null || echo "NO quark.py"
grep -n "mxfp4\|float4\|e2m1\|e8m0\|fp4" $Q 2>/dev/null | head -20 || true
echo "--- quark weight-only / dtype gates ---"
grep -n "def \|int4\|int8\|weight_only\|dtype" $Q 2>/dev/null | head -40
echo
echo "=== 3. quark registered in XPU platform? ==="
grep -n "quark" $SP/vllm/platforms/xpu.py || echo "quark NOT in xpu supported_quantization"
echo
echo "=== 4. CT _is_mxfp4 + scheme dispatch (mixed-precision per-group) ==="
CT=$SP/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
grep -n "def _is_mxfp4" -A 14 $CT
echo "--- dispatch site ---"
grep -n "_is_mxfp4\|W4A4Mxfp4" $CT | head -12
echo
echo "=== 5. W4A4Mxfp4 scheme: what act key reaches init_mxfp4_linear_kernel ==="
S=$SP/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py
sed -n '1,60p' $S 2>/dev/null | grep -n "class \|def \|activation\|kMxfp4\|dynamic" || echo "scheme file missing"
grep -n "activation_quant_key\|kMxfp4\|init_mxfp4_linear_kernel" $S 2>/dev/null | head -15
echo
echo "=== 6. XPU mxfp4 linear kernel can_implement (the gate) ==="
K=$SP/vllm/model_executor/layers/kernels/linear/mxfp4/xpu.py
grep -n "class \|def can_implement\|kMxfp4Dynamic\|activation_quant_key\|only supports\|ValueError" $K | head -20
echo
echo "=== 7. Qwen3_5 VL arch in registry ==="
grep -n "Qwen3_5\|qwen3_5" $SP/vllm/model_executor/models/registry.py | head -8
echo
echo "=== 8. CT fp8-block attn group route on XPU (W8A8 float block 128x128) ==="
grep -n "fp8_per_block\|block" $SP/vllm/platforms/xpu.py | head -5
grep -rn "class CompressedTensorsW8A8" $SP/vllm/model_executor/layers/quantization/compressed_tensors/schemes/*.py | head -6
EOS
