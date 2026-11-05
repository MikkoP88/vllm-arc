#!/bin/bash
# P6.21 v2.3 gate+ablation window — numerics ladder FIRST (any FAIL =>
# do not trust the rates), then the U32k level ladder + floor probes on
# the message-halved staging. W14 stop -> run -> EXIT-trap restore
# (verify FRANCE + W13H_BOOT_DONE before the next window).
set -u
OUT=/root/build/p621c/r92
trap '(nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore_p621c.log 2>&1 &)' EXIT
docker cp esimd-w11:/src/esimd-fa/build/lib.linux-x86_64-cpython-312/fa_dense_v2_ops.cpython-312-x86_64-linux-gnu.so $OUT/v2so/
md5sum $OUT/v2so/fa_dense_v2_ops.cpython-312-x86_64-linux-gnu.so
docker stop radiance-test >/dev/null 2>&1
sleep 20
docker run --rm --name arc-bench --privileged --network host --ipc host \
  --shm-size 16g -v /root/build/neo_cache:/root/.neo_cache -v $OUT:/work \
  -w /work -e ZE_AFFINITY_MASK=0,1 --entrypoint /bin/bash \
  vllm-arc:v0.30.0-b70-cert9 \
  -c 'cd /work && export PYTHONPATH=/work/v2so ARC_DENSE_FA_ESIMD_SO=/work/v2so/fa_dense_v2_ops.cpython-312-x86_64-linux-gnu.so && /opt/venv/bin/python numerics_v21.py && /opt/venv/bin/python bench_v21.py'
echo "=== window done ==="
