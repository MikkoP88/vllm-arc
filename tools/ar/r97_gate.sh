#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r97_gate.sh — P6.24 phase 1: fp8-wire GATE window on the rig.
#
# Preconditions: build_oneccl_arc.sh arc finished (stage-arc present);
# r96 fidelity PASS (phase-0 law: no fp8 evidence on a drifting base).
#
# One GPU window, legs (cert9 image, r95/r96 conditions byte-identical):
#   leg A fp8-off  : rebuilt libs, CCL_SYCL_ALLREDUCE_FP8=0 — control,
#                    must equal r96 baseline within noise
#   leg B fp8-on   : CCL_SYCL_ALLREDUCE_FP8=1 — pre-registered GATE
#                    >=1.3x at the dominant 76,677,120 B (<=6.0 ms);
#                    randn numerics green (cos >= 0.9999, rel-rms <= 2e-2)
#   leg C stats    : FP8=1 + STATS=1, sizes pinned to the dominant class,
#                    3 iters — telemetry evidence (declared perturbation)
# W14 restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p624
STAGE=$CT/stage-arc
SIZES="150994944,83886080,76677120,68157440,15779840,2621440"
CCL_BLOCK="export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1 CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0"

w14_start() {
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p624b.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

mkdir -p $CT/bench
cp $R/p624-kit/tools/ar_ceiling.py $CT/bench/ar_ceiling.py
[ -f "$STAGE/BUILD-MANIFEST.txt" ] || { echo "FATAL: $STAGE missing (run build_oneccl_arc.sh arc first)"; exit 1; }

run_leg() {  # $1 label, $2 out, $3 extra env, $4 sizes
  docker run --rm --privileged --ipc host --shm-size 32g \
    -v $CT/bench:/bench:ro -v $CT:/work -v $STAGE:/stage:ro -w /work \
    -e ZE_AFFINITY_MASK=0,1 \
    --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert9 -c "
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
    $CCL_BLOCK
    cp /stage/libccl.so.1.0 /opt/venv/lib/libccl.so.1.0
    cp /stage/libccl.so.1.0 /opt/venv/lib/libccl.so.1
    export $3
    /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 2 \
      /bench/ar_ceiling.py --role dist --guard randn --label $1 \
      --sizes-bytes $4 --out $2
  " 2>&1
}

echo "=== leg A: rebuilt libs, fp8 OFF (control) ==="
run_leg arc-fp8-off /work/r97_fp8off.json "CCL_SYCL_ALLREDUCE_FP8=0" "$SIZES" \
  | tee $CT/r97_legA.out | grep -E "PASS|FAIL|ARC-FP8-OFF|wrote"

echo "=== leg B: fp8 ON (gate) ==="
run_leg arc-fp8-on /work/r97_fp8on.json "CCL_SYCL_ALLREDUCE_FP8=1" "$SIZES" \
  | tee $CT/r97_legB.out | grep -E "PASS|FAIL|ARC-FP8-ON|wrote"

echo "=== leg C: fp8 ON + stats (telemetry, dominant class only) ==="
run_leg arc-fp8-stats /work/r97_stats.json "CCL_SYCL_ALLREDUCE_FP8=1 CCL_SYCL_ALLREDUCE_FP8_STATS=1" "76677120" \
  | tee $CT/r97_legC.out | grep -E "fp8 allreduce|ARC-FP8|PASS|FAIL" | head -8

echo "=== gate verdict ==="
docker run --rm -v $CT:/work --entrypoint /opt/venv/bin/python \
  vllm-arc:v0.30.0-b70-cert9 -c '
import json
off = {r["bytes"]: r["ms"] for r in json.load(open("/work/r97_fp8off.json"))}
on = {r["bytes"]: r["ms"] for r in json.load(open("/work/r97_fp8on.json"))}
dom = 76677120
ok_ctrl, ok_gate = True, True
for b in sorted(off):
    sp = off[b] / on[b]
    gate = (sp >= 1.3) if b == dom else None
    if b == dom and sp < 1.3: ok_gate = False
    ctrl = abs(on[b] / off[b] - 1) * 100
    note = ""
    if b == 2621440:
        note = " (sub-threshold: must be ~1.0x — stock path)"
        if abs(on[b] / off[b] - 1) > 0.10: ok_ctrl = False
    print(f"S={b:>9d}B off={off[b]:7.3f} on={on[b]:7.3f} speedup={sp:5.2f}x"
          + (f" GATE:{\"PASS\" if gate else \"FAIL\"}" if gate else "") + note)
print(f"anchor: stock 7.780 ms; fp8-on delta vs anchor: {(on[dom]/7.780-1)*100:+.1f}%")
print("R97_CONTROL:", "PASS" if ok_ctrl else "FAIL")
print("R97_GATE(>=1.3x @ 76,677,120B):", "PASS" if ok_gate else "FAIL")
'
echo "R97_GATE: DONE (W14 restore by trap)"
