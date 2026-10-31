#!/bin/bash
# p65_gate6.sh — P6.5 r48f: false-prefix-hit reproduction on the CERTIFIED
# posture (no seds: cert3, connector ON, threshold 0, 25-var chain).
#   Evidence driving this leg: r48e (no connector) U64k TTFT 1.75 s for
#   63 784 tok (~36k tok/s = impossible -> ~98% prompt skipped); r48d
#   (connector ON, SDC off) U64k 22.28 s (~2863 tok/s > solo 2036 ->
#   partial skip ~50%). Both were sent post-drain after U32k completed.
#   Harness v4 adds usage cached_tokens capture + first-240-chars content.
#   Faces: U32k (25) + C32k (55, legitimate replay control) + U64k (85,
#   the false-hit face on the shipped posture).
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48f.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

echo "=== [1] boot CERTIFIED posture (no modifications) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 bash $R/serve_arc_b70.sh \
  > $CT/boot_cert_r48f.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_cert_r48f.out || { echo "FATAL: serve launch failed"; cat $CT/boot_cert_r48f.out; exit 1; }
ok=""
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
echo "HEALTH=OK after ~$((i*5))s"

echo "=== [2] capture v4 (U32k + C32k + U64k; usage+content captured) ==="
cd $CT && P65_FACES="U32k,C32k,U64k" python3 $CT/p65_timeline.py $CT/p65_tl_cert_v4.json 2>&1 | tee $CT/p65_sum_cert_v4.txt

echo "=== [3] engine log ==="
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48f.log 2>/dev/null
echo "P65_LEG_R48F: DONE"
