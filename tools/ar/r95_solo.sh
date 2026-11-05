#!/bin/bash
# r95_solo.sh — P6.22 r95 ceiling bench as a STANDALONE short GPU window
# (the r94 attempt-4 serve capture succeeded; only the bench leg needs a
# re-run after two attempt-4 [8] defects: the size picker took the six
# SMALLEST >=1MiB classes, and /kit/platform.py shadowed stdlib `platform`
# via sys.path[0] when running /kit/ar_ceiling.py — this variant copies
# the bench scripts to /work/first and pins the DOMINANT census sizes:
# rows x 5120 x fp16 classes 14745(144MiB)/8192/7488/6656 + tail).
set -uo pipefail
R=/root/build
CT=$R/p622
KIT=$CT/kit
CCL_BLOCK="export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1 CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0"

w14_start() {
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p622c.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

# dominant classes (census r94a: 150994944/83886080/76677120/68157440
# + 15779840 + decode-class 2621440)
SIZES="150994944,83886080,76677120,68157440,15779840,2621440"
echo "r95 sizes: $SIZES"
mkdir -p $CT/bench
cp $KIT/ar_ceiling.py $CT/bench/ar_ceiling.py
docker run --rm --privileged --ipc host --shm-size 32g \
  -v $CT/bench:/bench:ro -v $CT:/work -w /work \
  -e ZE_AFFINITY_MASK=0,1 \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert9 -c "
  export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
  $CCL_BLOCK
  /opt/venv/bin/python /bench/ar_ceiling.py --role oneshot --sizes-bytes $SIZES --out /work/r95_oneshot.json &&
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 2 \
    /bench/ar_ceiling.py --role dist --sizes-bytes $SIZES --out /work/r95_dist.json
  " 2>&1 | tee $CT/r95_bench.out
echo "R95_SOLO: DONE (W14 restore by trap)"
