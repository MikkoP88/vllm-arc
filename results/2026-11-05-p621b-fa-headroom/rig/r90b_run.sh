#!/bin/bash
# p621b r89 — attention inventory + SDPA rate table on cert9 (throwaway,
# GPUs quiet; W14 stop/restore same laws as r87/r88).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert9
OUT=$R/p621b/r87
cd $OUT
w14_start() {
  docker rm -f arc-bench >/dev/null 2>&1
  (nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p621b.log 2>&1 &)
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1
sleep 20
docker run --rm --name arc-bench --privileged --network host --ipc host \
  --shm-size 16g -v /root/build/neo_cache:/root/.neo_cache \
  -v $OUT:/work -w /work -e ZE_AFFINITY_MASK=0,1 \
  --entrypoint /bin/bash $IMG -c \
  'cd /work && /opt/venv/bin/python r90b_backend_matrix.py -u --out r90b_backend_matrix.json' \
  > $OUT/r89_bench.out 2>&1
echo BENCH_RC=$?
cat $OUT/r89_bench.out
