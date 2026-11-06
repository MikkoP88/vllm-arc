#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r101b_overlap_bench.sh — P6.25 GPU window 2: the missing r101 legs.
#
# Window 1 banked stock async/census/ar/coten (r101_stock_*.json) but mock
# died on the q-slice shape bug (o_proj consumes qkv[:, :3072] — fixed in
# ar_overlap_bench.py, PEAK_TFS re-referenced to the census's own ~150 TF/s
# rank-local dense ceiling). This window runs ONLY what's missing:
#   L1 stock : mock                                                -> r101_stock_mock.json
#   L2 fp8   : async + ar + coten + mock (CCL_SYCL_ALLREDUCE_FP8=1) -> r101_fp8_*.json
# The fp8 leg is the live question: its AR kernel mix (amax+quant+wire/2+
# dequant) may contend with GEMM far less than the stock ring did
# (stock coten: gemm_inflation 1.974) — if fp8 coten ~1.0x and fp8 mock
# >= 1.08x, the COMPOSED posture (fp8 wire + overlap) opens r102.
# PRE-REGISTERED GATE (plan P6.25): best-k mock >= 1.08x serial -> r102;
# else the overlap door closes measured.
# W14: stopped for the window, restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p625
CCL_BLOCK="export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1 CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0"

w14_start() {
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p625.log 2>&1 < /dev/null &
}
trap w14_start EXIT

mkdir -p $CT/bench
cp $CT/ar_overlap_bench.py $CT/bench/ || { echo "FATAL: bench py missing at $CT"; exit 1; }

docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

run_leg() {  # $1 label, $2 out, $3 extra env, $4 mode(s)
  docker run --rm --privileged --ipc host --shm-size 32g \
    -v $CT/bench:/bench:ro -v $CT:/work -w /work \
    -e ZE_AFFINITY_MASK=0,1 \
    --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert10 -c "
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
    $CCL_BLOCK
    export $3
    for mode in $4; do
      echo \"=== [\$(date +%H:%M:%S)] mode=\$mode label=$1 ===\"
      /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 2 \
        /bench/ar_overlap_bench.py --mode \$mode --label $1 --out /work/$(basename $2 .json)_\$mode.json \
      || { echo \"MODE \$mode FAILED\"; exit 5; }
    done
  " > "${2%.json}.log" 2>&1 || { echo "LEG $1 FAILED — tail:"; tail -30 "${2%.json}.log"; exit 6; }
  echo "leg $1 done: $(ls $CT/$(basename $2 .json)_*.json 2>/dev/null | xargs -n1 basename | tr '\n' ' ')"
}

echo "=== r101b window start $(date +%H:%M:%S) ==="
run_leg stock $CT/r101_stock.json "ARC_NOP=1" "mock"
run_leg fp8   $CT/r101_fp8.json   "CCL_SYCL_ALLREDUCE_FP8=1" "async ar coten mock"
echo "=== r101b window end $(date +%H:%M:%S) ==="
echo "--- summary ---"
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob('/root/build/p625/r101_*.json')):
    d = json.load(open(f))
    keys = ['enqueue_ms_med','verdict','ar_ms','gemm_inflation','ar_inflation',
            'serial_ms','overlap_k2_speedup','overlap_k3_speedup','overlap_k4_speedup',
            'best_k','best_speedup','gate_1p08','overlap_bitwise','pct_peak','layer_gemm_ms']
    print(f.split('/')[-1], {k: d[k] for k in keys if k in d})
PY
