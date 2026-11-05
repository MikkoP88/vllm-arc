#!/bin/bash
# P6.21 r92 round-6b window — dpas layout probe (N=8 raw / SLM / N=16
# control). W14 stop -> run -> EXIT-trap restore (verify FRANCE +
# W13H_BOOT_DONE in the restore log before the next window).
set -u
OUT=/root/build/p621c/r92
trap '(nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore_p621c.log 2>&1 &)' EXIT
docker stop radiance-test >/dev/null 2>&1
sleep 20
docker run --rm --name arc-bench --privileged --network host --ipc host \
  --shm-size 16g -v /root/build/neo_cache:/root/.neo_cache -v $OUT:/work \
  -w /work -e ZE_AFFINITY_MASK=0,1 --entrypoint /bin/bash \
  vllm-arc:v0.30.0-b70-cert9 \
  -c 'PYTHONPATH=/work/v2so ARC_DENSE_FA_ESIMD_SO=/work/v2so/fa_dense_v2_ops.cpython-312-x86_64-linux-gnu.so /opt/venv/bin/python /work/probe_dpas_layout.py'
echo "=== window done ==="
