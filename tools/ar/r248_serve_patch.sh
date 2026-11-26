#!/bin/bash
# r248_serve_patch.sh rev 2 -- P-H1 serve-script extension (two
# idempotent steps; run with no arc leg in flight -- mid-flight law):
#   1. the two env-optional lmcint diag mounts (ARC_PD_LMCDIAG_DET /
#      ARC_PD_LMCDIAG_AD over the lmcint kit files; the P-D ARC_PD_*
#      mount pattern).
#   2. the P-D gate knob pass-through into the engine's docker-exec
#      env block: ${VLLM_ARC_LMC_MX_GATE:+export ...} -- UNSET keeps
#      the certified gate-ON posture byte-identical; exporting 0 on
#      the serve process reaches the engine (run 1 lesson: the runner
#      exported VLLM_ARC_LMC_MX_GATE=0 but the exec block never
#      passed it -- the gate refused the boot at connector
#      construction, 3 min in, with zero diag lines).
set -euo pipefail
S=/root/build/serve_arc_b70.sh
cp "$S" "$S.bak_r248b"

# --- step 1: diag mounts (idempotent) ---
if ! grep -q ARC_PD_LMCDIAG_DET "$S"; then
  grep -q 'ARC_PD_GATE' "$S" || { echo "R248_NO_PD_GATE_ANCHOR"; exit 1; }
  awk '{
    print
    if ($0 ~ /ARC_PD_GATE/) {
      print "    ${ARC_PD_LMCDIAG_DET:+-v \"$ARC_PD_LMCDIAG_DET\":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmcint/lmc_kvfmt_detector.py:ro} \\"
      print "    ${ARC_PD_LMCDIAG_AD:+-v \"$ARC_PD_LMCDIAG_AD\":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py:ro} \\"
    }
  }' "$S" > "$S.r248new" && bash -n "$S.r248new" && mv "$S.r248new" "$S"
  echo "step1: diag mount lines added"
else
  echo "step1: already patched"
fi

# --- step 2: gate knob pass-through (idempotent) ---
if ! grep -q 'VLLM_ARC_LMC_MX_GATE' "$S"; then
  grep -q 'ARC_SKINNY_M' "$S" || { echo "R248_NO_SKINNY_ANCHOR"; exit 1; }
  awk '{
    print
    if ($0 ~ /export ARC_SKINNY_M=/) {
      print "${VLLM_ARC_LMC_MX_GATE:+export VLLM_ARC_LMC_MX_GATE=${VLLM_ARC_LMC_MX_GATE};}"
    }
  }' "$S" > "$S.r248new" && bash -n "$S.r248new" && mv "$S.r248new" "$S"
  echo "step2: gate knob pass-through added"
else
  echo "step2: already patched"
fi

bash -n "$S"
echo "diag mounts: $(grep -c 'ARC_PD_LMCDIAG' "$S")  knob lines: $(grep -c 'VLLM_ARC_LMC_MX_GATE' "$S")"
md5sum "$S"
echo "R248_SERVE_PATCHED"
