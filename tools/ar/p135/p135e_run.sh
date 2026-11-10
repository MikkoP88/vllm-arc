#!/bin/bash
# p135e_run.sh -- run p135e_isolated_bench on an IDLE device (W14 stopped),
# always restoring W14 afterwards (trap). One ssh invocation, self-contained.
set -u
R=/root/build
docker stop radiance-test >/dev/null 2>&1
restore() {
  docker rm -f p135e >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap restore EXIT
sleep 3
docker rm -f p135e >/dev/null 2>&1
docker run -d --name p135e --privileged --entrypoint sleep \
  vllm-arc:v0.30.0-b70-cert22 900 >/dev/null
docker cp $R/p635/p135e_isolated_bench.py p135e:/root/
docker exec p135e /opt/venv/bin/python /root/p135e_isolated_bench.py
E=$?
docker cp p135e:/root/p135e_bench.json $R/p635/ 2>/dev/null
exit $E
