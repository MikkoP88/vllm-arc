#!/bin/bash
# p66_gate2c.sh — P6.6 leg 2 v3: decompose the ~0.4s per-chunked-step
# overhead. v2 lesson: step() dead code under async scheduling (batch
# queue), child print()s never reach serve_full.log => v3 probe image
# (cert4 + patch_p66c.py) instruments the LIVE path and writes to
# /root/p66_steps.log. Two boots: thr2048 (16-step face) + thr none
# (4-step face), solo U32k each, nolmc.
set -uo pipefail
R=/root/build
CT=$R/p66
IMG=vllm-arc:v0.30.0-b70-cert4
PROBE=vllm-arc:v0.30.0-b70-cert4-p66probe2
COREF=/opt/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py
RUNF=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/v1/worker/hpu_model_runner.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p66d.log 2>&1 &
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

echo "=== [1] extract + patch + probe image (p66probe2) ==="
mkdir -p $CT/probe_src/vllm/v1/engine $CT/probe_src/vllm_gaudi/v1/worker
docker rm -f p66x3 >/dev/null 2>&1
docker run -d --name p66x3 --entrypoint sleep $IMG 300 >/dev/null
docker cp p66x3:$COREF $CT/probe_src/vllm/v1/engine/core.py
docker cp p66x3:$RUNF $CT/probe_src/vllm_gaudi/v1/worker/hpu_model_runner.py
docker rm -f p66x3 >/dev/null 2>&1
rm -rf $CT/probe_out
python3 $CT/patch_p66c.py || { echo "FATAL: patcher"; exit 1; }
docker rm -f p66probe2 >/dev/null 2>&1
docker run -d --name p66probe2 --entrypoint sleep $IMG 300 >/dev/null
docker cp $CT/probe_out/core.py p66probe2:$COREF
docker cp $CT/probe_out/hpu_model_runner.py p66probe2:$RUNF
docker exec p66probe2 /opt/venv/bin/python -m py_compile $COREF $RUNF \
  || { echo "FATAL: in-container py_compile"; exit 1; }
docker commit -m "p66 probe v3: live-path step decomposition instrumentation" p66probe2 $PROBE >/dev/null
docker rm -f p66probe2 >/dev/null 2>&1
docker run --rm --entrypoint grep $PROBE -c ARC_P66 $RUNF

for THR in 2048 none; do
  echo "=== boot probe2 thr=$THR (nolmc) ==="
  if [ "$THR" = "none" ]; then
    SRV=$R/serve_arc_b70.sh
  else
    SRV=$R/serve_arc_b70_p66_$THR.sh
    sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold '"$THR"' \\|' \
      $R/serve_arc_b70.sh > "$SRV"
    grep -c "long-prefill-token-threshold $THR" "$SRV" >/dev/null || { echo "FATAL: sed"; exit 1; }
  fi
  docker rm -f arc-test >/dev/null 2>&1
  ARC_IMAGE=$PROBE ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
    bash "$SRV" > "$CT/boot_probe2_$THR.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/boot_probe2_$THR.out" || { echo "FATAL: launch $THR"; cat "$CT/boot_probe2_$THR.out"; exit 1; }
  wait_health
  cd $CT && P66_FACES="U32k" python3 $CT/p66_solo_ttft.py $CT/p66_probe2_thr$THR.json 2>&1 | tee $CT/p66_probe2_thr$THR.txt
  docker cp arc-test:/root/p66_steps.log $CT/p66_steps_thr$THR.txt \
    || echo "WARN: no p66_steps.log for thr=$THR"
  echo "-- p66_steps thr=$THR: $(wc -l < $CT/p66_steps_thr$THR.txt 2>/dev/null || echo 0) lines; tail:"
  tail -6 $CT/p66_steps_thr$THR.txt 2>/dev/null
  docker exec arc-test cat /root/serve_full.log > "$CT/serve_probe2_$THR.log" 2>/dev/null
  docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
  docker rm -f arc-test >/dev/null 2>&1; sleep 5
done
echo "P66_GATE2C: DONE"
