#!/bin/bash
# p66_gate1b.sh — device-truth probe after gate1: brief W14 pause, GEMM
# ceiling + device props + model config in a cert4 container, then W14
# back. ~4 min.
set -uo pipefail
R=/root/build
CT=$R/p66
IMG=vllm-arc:v0.30.0-b70-cert4

w14_start() {
  docker rm -f p66dev >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p66b.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 3

docker rm -f p66dev >/dev/null 2>&1
docker run -d --name p66dev --privileged --ipc host \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  --entrypoint /bin/bash "$IMG" -c 'sleep 300' >/dev/null
docker cp $CT/p66_devprobe.py p66dev:/tmp/p66_devprobe.py
docker exec -e ZE_AFFINITY_MASK=0 p66dev /opt/venv/bin/python /tmp/p66_devprobe.py \
  > "$CT/p66_devprobe.out" 2>&1
cat "$CT/p66_devprobe.out"
echo "P66_GATE1B: DONE"
