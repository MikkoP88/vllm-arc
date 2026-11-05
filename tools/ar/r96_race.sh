#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r96_race.sh — P6.24 phase 0: stock-rebuild FIDELITY race on the rig.
#
# One GPU window, two legs, byte-identical conditions to the r95 anchor
# (cert9 image, same CCL block, same sizes, torchrun 2-proc dist role):
#   leg A baseline  : cert9 image untouched
#   leg B rebuilt   : /stage-stock libccl.so.1* overlaid onto /opt/venv/lib
# PASS bar (pre-registered): leg B within +-3% of leg A at the dominant
# 76,677,120 B class AND vs the r95 anchor 7.78 ms; randn numerics guard
# green on both legs. FAIL -> STOP (no fp8 build on a drifting base).
# W14 restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p624
STAGE=$CT/stage-stock
SIZES="150994944,83886080,76677120,68157440,15779840,2621440"
CCL_BLOCK="export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1 CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0"

w14_start() {
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p624.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

mkdir -p $CT/bench
cp $R/p624-kit/tools/ar_ceiling.py $CT/bench/ar_ceiling.py

run_leg() {  # $1 = label, $2 = out json, $3 = overlay shell prefix ("" for baseline)
  docker run --rm --privileged --ipc host --shm-size 32g \
    -v $CT/bench:/bench:ro -v $CT:/work -v $STAGE:/stage:ro -w /work \
    -e ZE_AFFINITY_MASK=0,1 \
    --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert9 -c "
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
    $CCL_BLOCK
    $3
    /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 2 \
      /bench/ar_ceiling.py --role dist --guard randn --label $1 \
      --sizes-bytes $SIZES --out $2
  " 2>&1
}

echo "=== leg A: baseline (cert9 untouched) ==="
run_leg stock /work/r96_baseline.json "" | tee $CT/r96_legA.out | grep -E "PASS|FAIL|STOCK|wrote"

echo "=== leg B: rebuilt-stock overlay ==="
OVERLAY='cp -v /stage/libccl.so.1.0 /opt/venv/lib/libccl.so.1.0; cp -v /stage/libccl.so.1.0 /opt/venv/lib/libccl.so.1; ls -la /opt/venv/lib/libccl.so.1*
# cert9 ships .so.1 and .so.1.0 as TWO regular files (not symlinks); the
# loader resolves the DT_NEEDED soname libccl.so.1 — both must be replaced'
run_leg rebuilt /work/r96_rebuilt.json "$OVERLAY" | tee $CT/r96_legB.out | grep -E "PASS|FAIL|REBUILT|wrote|libccl"

echo "=== fidelity verdict ==="
docker run --rm -v $CT:/work --entrypoint /opt/venv/bin/python \
  vllm-arc:v0.30.0-b70-cert9 -c '
import json
base = {r["bytes"]: r["ms"] for r in json.load(open("/work/r96_baseline.json"))}
reb = {r["bytes"]: r["ms"] for r in json.load(open("/work/r96_rebuilt.json"))}
dom = 76677120
ok = True
for b in sorted(base):
    d = (reb[b] - base[b]) / base[b] * 100
    if b == dom and abs(d) > 3.0:
        flag, ok = "DRIFT-GATE", False
    elif b != dom and abs(d) > 5.0:
        flag = "drift-warn"
    else:
        flag = "ok"
    print(f"S={b:>9d}B base={base[b]:7.3f} rebuilt={reb[b]:7.3f} delta={d:+5.1f}% {flag}")
print(f"anchor r95: 7.780 ms at {dom}B; rebuilt delta vs anchor: {(reb[dom]-7.780)/7.780*100:+.1f}%")
print("R96_FIDELITY:", "PASS" if ok else "FAIL")
'
echo "R96_RACE: DONE (W14 restore by trap)"
