#!/bin/bash
# r28 teardown — arc leg law: restore the rig to W14 incumbent serving.
# (curl guard fixed — the p32 teardown died on curl's exit 7 mid-poll.)
docker stop arc-test >/dev/null && echo ARC-STOPPED
docker start radiance-test >/dev/null && echo RADIANCE-STARTED
sleep 3
bash /root/build/w14_serve_only.sh && echo W14_SERVE_LAUNCHED
code=000
for i in $(seq 1 60); do
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 3 http://127.0.0.1:8000/health 2>/dev/null || echo 000)
  [ "$code" = 200 ] && { echo "W14_HEALTH_200 $(date +%H:%M:%S)"; exit 0; }
  sleep 10
done
echo "W14_HEALTH_FAIL code=$code"
docker exec radiance-test tail -20 /root/serve_posture.log
exit 5
