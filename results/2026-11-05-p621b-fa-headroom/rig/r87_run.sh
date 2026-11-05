#!/bin/bash
# p621b r87 — FA headroom microbench on cert9 (throwaway container, GPUs
# quiet; NO serve, NO image change). Decides the P6.21 FA lane by the
# pre-registered S_lo rule in r87_fa_headroom.py's header.
#
# Window: stop W14 -> run bench on both cards -> trap restores W14
# (w14_boot.sh fullres; FRANCE probe verified after, rig law).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert9
OUT=$R/p621b/r87
mkdir -p "$OUT"
cd "$OUT"

w14_start() {
  docker rm -f arc-bench >/dev/null 2>&1
  (nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p621b.log 2>&1 &)
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1
sleep 20

# p72 stage must hold fa_dense_ops.so (v1, md5 6f26eabc...); missing .so
# just records a SKIP row — the stock/ceiling bench never depends on it.
ls -l $R/p72/stage/fa_dense_ops.so || echo "WARN: v1 .so absent -> V1 SKIP"

# r87a fix: /p72stage on PYTHONPATH shadows stdlib `platform` (the dir
# holds the p72 overlay platform.py) and kills torch import — copy ONLY
# the .so into a clean dir for the v1 import path.
mkdir -p "$OUT/v1so"
cp $R/p72/stage/fa_dense_ops.so "$OUT/v1so/"

docker run --rm --name arc-bench --privileged --network host --ipc host \
  --shm-size 16g \
  -v /root/build/neo_cache:/root/.neo_cache \
  -v "$OUT":/work -w /work \
  -e ZE_AFFINITY_MASK=0,1 \
  --entrypoint /bin/bash "$IMG" -c \
  'cd /work && PYTHONPATH=/work/v1so /opt/venv/bin/python r87_fa_headroom.py --out r87_fa_headroom.json' \
  > $OUT/r87_bench.out 2>&1
RC=$?
echo "BENCH_RC=$RC"
cat $OUT/r87_bench.out
