#!/bin/bash
# p135g_run.sh -- run p135g_ccl_bench (torchrun 2 ranks, ccl) on idle device
# (W14 stopped), always restoring W14 afterwards (trap).
set -u
R=/root/build
docker stop radiance-test >/dev/null 2>&1
restore() {
  docker rm -f p135g >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap restore EXIT
sleep 3
docker rm -f p135g >/dev/null 2>&1
docker run -d --name p135g --privileged --entrypoint sleep \
  vllm-arc:v0.30.0-b70-cert22 900 >/dev/null
docker cp $R/p635/p135g_ccl_bench.py p135g:/root/
docker exec -w /root p135g /opt/venv/bin/torchrun --standalone \
  --nproc_per_node=2 /root/p135g_ccl_bench.py
E=$?
for r in 0 1; do docker cp p135g:/root/p135g_bench_r$r.json $R/p635/ 2>/dev/null; done
exit $E
