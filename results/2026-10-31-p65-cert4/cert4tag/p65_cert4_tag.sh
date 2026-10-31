#!/bin/bash
# p65_cert4_tag.sh — cert4 TAG-BUILT image verification (r47e precedent:
# the tag-built image must prove manifest fork_sha == tag SHA, image-native
# verifies, a certified-default battery, and the p65 face — the rc image's
# battery does not transfer).
set -uo pipefail
R=/root/build
CT=$R/p65/cert4tag
IMG=vllm-arc:v0.30.0-b70-cert4
TAGSHA=b5df28b553ead3abfe03352dc8572a5ee20880f5
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_cert4tag.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

echo "=== [1] IMAGE_MANIFEST: fork_sha must == $TAGSHA ==="
MSHA=$(docker run --rm --entrypoint cat "$IMG" /etc/vllm-arc/IMAGE_MANIFEST | grep -o 'fork_sha=[a-f0-9]*' | cut -d= -f2)
echo "MANIFEST fork_sha=$MSHA"
[ "$MSHA" = "$TAGSHA" ] || { echo "FATAL: fork_sha mismatch"; exit 1; }
docker run --rm --entrypoint cat "$IMG" /etc/vllm-arc/IMAGE_MANIFEST > "$CT/manifest.txt"
echo "MANIFEST_OK"

echo "=== [2] image-native verify: p48 (28) + p65 (10) ==="
docker rm -f cert4tag-verify >/dev/null 2>&1
docker run -d --name cert4tag-verify --privileged --ipc host \
  --entrypoint /bin/bash "$IMG" -c 'sleep 900' >/dev/null
docker cp $R/r47c/p48_verify.py cert4tag-verify:/tmp/p48_verify.py 2>/dev/null \
  || docker cp $R/p65/cert4/p48_verify.py cert4tag-verify:/tmp/p48_verify.py
docker cp $R/p65/cert4/p65_verify.py cert4tag-verify:/tmp/p65_verify.py
docker exec -e LMCACHE_TRACK_USAGE=false cert4tag-verify /opt/venv/bin/python /tmp/p48_verify.py \
  > "$CT/p48_verify.out" 2>&1
tail -2 "$CT/p48_verify.out"
docker exec -e LMCACHE_TRACK_USAGE=false cert4tag-verify /opt/venv/bin/python /tmp/p65_verify.py \
  > "$CT/p65_verify.out" 2>&1
tail -2 "$CT/p65_verify.out"
docker rm -f cert4tag-verify >/dev/null
grep -q ALL_GREEN "$CT/p48_verify.out" || { echo "FATAL: p48 not ALL_GREEN"; exit 1; }
grep -q "10/10 PASS" "$CT/p65_verify.out" || { echo "FATAL: p65 not 10/10"; exit 1; }

echo "=== [3] certified default boot (25-var, NO p65 envs) + run7n t0 ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 bash $R/serve_arc_b70.sh \
  > "$CT/serve_boot.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/serve_boot.out" || { echo "FATAL: serve launch failed"; cat "$CT/serve_boot.out"; exit 1; }
wait_health
echo "DEFER_ENV_ABSENT=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep -c LMC_DEFER' || echo 0) (must be 0)"
python3 $R/r46t/bench_run7n_t.py --temperature 0 > "$CT/k3_t0_run7n.out" 2>&1
grep -h 'aggregate=\|solo200' "$CT/k3_t0_run7n.out"

echo "=== [4] p65 face boot + U64k (fix face on the tag-built image) ==="
docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
docker rm -f arc-test >/dev/null 2>&1; sleep 5
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_defer2.sh > "$CT/boot_p65face.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_p65face.out" || { echo "FATAL: p65 face launch failed"; cat "$CT/boot_p65face.out"; exit 1; }
wait_health
echo "ENV_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep LMC_DEFER')"
cd $CT && P65_FACES="U64k" P65_W2MT=4000 python3 $R/p65/p65_timeline.py $CT/p65_tl_cert4tag_v51.json 2>&1 | tee $CT/p65_sum_cert4tag_v51.txt
grep "prompt sizes\|TTFT" $CT/p65_sum_cert4tag_v51.txt
echo "STORE classes:"
docker exec arc-test sh -c 'grep -o "Stored [0-9]* out of total" /root/serve_full.log | sort | uniq -c | sort -rn | head -6'
docker exec arc-test cat /root/serve_full.log > "$CT/serve_full_p65face.log" 2>/dev/null
echo "P65_CERT4_TAG_GATE: DONE"
