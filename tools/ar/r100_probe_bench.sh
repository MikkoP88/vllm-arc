#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r100_probe_bench.sh — P6.24b bottleneck scan, GPU window 1: IPC wire-rate
# probe + v4-rate control on the probe build.
#
# Owner directive (2026-11-05): beat the cert9 baseline on EVERY p65 face
# with the highest possible margin; find bottlenecks end-to-end until no
# options remain. E-law: every door gets a measured close.
#
# Legs (cert9 container, stage-probe overlay = v4 + arc-fp8-probe.patch):
#   leg P: FP8=1 + FP8_PROBE=1, dominant size only — the one-shot probe
#          prints remote write4/8/16, remote read4/8/16, remote rw4, local
#          write4/read4 over the REAL peer mapping (r100 decision data:
#          v5 read-reduce is justified only if remote reads outrun the
#          9.6 GB/s write ceiling; wider stores only if the u32 law is
#          instruction-issue, not fabric).
#   leg C: FP8=1 (no probe), full size census — control: the probe build's
#          algorithm path must reproduce r97 v4 rates (probe patch is
#          additive; any drift convicts the build, not the algorithm).
# W14 restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p624
STAGE=$CT/stage-probe
SIZES="150994944,83886080,76677120,68157440,15779840,2621440"
CCL_BLOCK="export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1 CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0"

w14_start() {
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p624g.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

mkdir -p $CT/bench
cp $R/p624-kit/tools/ar_ceiling.py $CT/bench/ar_ceiling.py
[ -f "$STAGE/libccl.so.1.0" ] || { echo "FATAL: $STAGE missing (run build_oneccl_arc.sh probe first)"; exit 1; }
[ -f "$STAGE/BUILD-MANIFEST.txt" ] || { echo "FATAL: $STAGE manifest missing"; exit 1; }
grep -q "arc-fp8-probe.patch" "$STAGE/BUILD-MANIFEST.txt" \
  || { echo "FATAL: stage-probe manifest lacks probe row (stale stage?)"; exit 1; }
echo "--- stage-probe manifest:"; cat "$STAGE/BUILD-MANIFEST.txt"

run_leg() {  # $1 label, $2 out, $3 extra env, $4 sizes, $5 logfile
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
      /bench/ar_ceiling.py --role dist --guard randn --label $1 --tol-mean 0.035 \
      --sizes-bytes $4 --out $2
  " > "$5" 2>&1
}

echo "=== leg P: wire-rate probe (first-call one-shot) $(date +%H:%M:%S) ==="
run_leg arc-probe /work/r100_probe.json "CCL_SYCL_ALLREDUCE_FP8=1 CCL_SYCL_ALLREDUCE_FP8_PROBE=1" "76677120" $CT/r100_legP.out
grep -E "FP8AR_PROBE|PASS|FAIL|ARC-FP8" $CT/r100_legP.out | sort | uniq -c | sort -rn | head -30

echo "=== leg C: probe-build control (v4 rates must hold) $(date +%H:%M:%S) ==="
run_leg arc-probe-ctrl /work/r100_ctrl.json "CCL_SYCL_ALLREDUCE_FP8=1" "$SIZES" $CT/r100_legC.out
grep -E "PASS|FAIL|ARC-FP8-ON|wrote" $CT/r100_legC.out

echo "=== r100 verdict (rates + control vs r97 v4) $(date +%H:%M:%S) ==="
python3 - << 'PYEOF'
import json, re

# probe rates: collapse the two ranks' lines (they measure the same wire)
rates = {}
pat = re.compile(r"FP8AR_PROBE rank(\d) (\S+)\s+[\d.]+ MB width=\s*(\d+)B :\s+([\d.]+) GB/s")
for line in open("/root/build/p624/r100_legP.out", errors="replace"):
    m = pat.search(line)
    if m:
        rates.setdefault(m.group(2), []).append(float(m.group(4)))
for k in sorted(rates):
    v = rates[k]
    print(f"{k:16s}: {max(v):7.2f} GB/s (ranks {v})")

ctrl = {r["bytes"]: r["ms"] for r in json.load(open("/root/build/p624/r100_ctrl.json"))}
dom = 76677120
r97 = 4.731
if dom in ctrl:
    print(f"control @dominant: {ctrl[dom]:.3f} ms (r97 v4: {r97} ms, drift {ctrl[dom]/r97-1:+.1%})")
w = rates.get("remote_write4", [0])
r = rates.get("remote_read4", [0])
if w and r:
    verdict = "READ" if max(r) > max(w) * 1.10 else ("WRITE-BOUND" if max(w) >= max(r) * 0.95 else "PARITY")
    print(f"R100_WIRE_LAW: remote write4 {max(w):.2f} vs read4 {max(r):.2f} GB/s -> {verdict}")
w8 = rates.get("remote_write8", [0])
if w and w8 and max(w) > 0:
    print(f"width scaling: write4 {max(w):.2f} -> write8 {max(w8):.2f} GB/s ({max(w8)/max(w):.2f}x)")
print("R100_PROBE: DONE (v5 decision in VERDICT)")
PYEOF
echo "R100_PROBE_BENCH: DONE (W14 restore by trap)"
