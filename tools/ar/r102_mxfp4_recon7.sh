#!/bin/bash
# r102_mxfp4_recon7.sh — P6.26 recon part 7 (final): init_mxfp4_linear_kernel
# dispatch + supports_mx role + model.py enforcement + CT scheme constraints.
set -uo pipefail
IMG=vllm-arc:v0.30.0-b70-cert10
docker run --rm -i -v /models:/models:ro -v /root/build/p626:/p626:ro --entrypoint /bin/bash $IMG -s <<'EOS'
SP=/opt/venv/lib/python3.12/site-packages
L=$SP/vllm/model_executor/kernels/linear/__init__.py
echo "=== A. supports_mx usage inside kernel dispatch ==="
grep -n "supports_mx" $L
echo
echo "=== B. mxfp4 kernel table + init fn ==="
sed -n "$(grep -n '_POSSIBLE_MXFP4_KERNELS' $L | head -1 | cut -d: -f1),+30p" $L
sed -n "$(grep -n 'def init_mxfp4_linear_kernel' $L | head -1 | cut -d: -f1),+40p" $L
echo
echo "=== C. model.py enforcement semantics ==="
grep -n "supported_quantization" -B4 -A14 $SP/vllm/config/model.py | head -40
echo
echo "=== D. emulation kernel class + where appended ==="
grep -n "emulation\|Emulation" $L | head -8
echo
echo "=== E. staged model sizes (real quant payloads) ==="
du -sh /models/qwen3.8-27b-mxfp4xfp8 /models/qwen3.8-27b-quark-awq-mxfp4 2>/dev/null
ls /models/qwen3.8-27b-mxfp4xfp8 2>/dev/null | head -12
echo
echo "=== F. staged checkpoints complete? ==="
tail -1 /p626/stage.log 2>/dev/null | tr '\r' '\n' | tail -2
grep -c "safetensors" <(ls /models/qwen3.8-27b-mxfp4xfp8 2>/dev/null) || true
EOS
