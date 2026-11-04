#!/bin/bash
# p619d probe — who references/patches the TQ decode inside the cert6 wheel?
SP=/opt/venv/lib/python3.12/site-packages
echo "=== references outside the two decode modules ==="
grep -rn "triton_turboquant_decode_attention" "$SP/vllm/" "$SP/vllm_gaudi/" \
  --include="*.py" |
  grep -v "ops/turboquant_soa/triton_turboquant_decode" |
  grep -v "ops/triton_turboquant_decode.py" |
  grep -v "^Binary"
echo "=== backend call sites + surrounding dispatch (855-885, 1195-1215) ==="
sed -n '855,885p;1195,1215p' "$SP/vllm/v1/attention/backends/turboquant_attn.py"
echo "=== any xpu/xpu-specific override files touching turboquant ==="
grep -rln "turboquant" "$SP/vllm_gaudi/" --include="*.py" 2>/dev/null | head
