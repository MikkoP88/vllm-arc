#!/bin/bash
# r102_mxfp4_recon8.sh — P6.26 recon part 8: the platform-gate raise site
# + every supports_mx call site (fix-scope decision).
set -uo pipefail
IMG=vllm-arc:v0.30.0-b70-cert10
docker run --rm -i --entrypoint /bin/bash $IMG -s <<'EOS'
SP=/opt/venv/lib/python3.12/site-packages
echo "=== A. the raise site ==="
grep -rn "is currently not supported in" $SP/vllm --include=*.py | head -5
echo
echo "=== B. its function context ==="
F=$(grep -rln "quantization is currently not supported" $SP/vllm --include=*.py | head -1)
grep -n "quantization is currently not supported" -B 22 "$F" | head -34
echo
echo "=== C. every supports_mx call site in the wheel ==="
grep -rn "supports_mx()" $SP/vllm --include=*.py | grep -v "def supports_mx" | head -20
echo
echo "=== D. supports_mx overrides (platforms that return True) ==="
grep -rn "def supports_mx" -A 2 $SP/vllm/platforms/*.py | head -20
EOS
