#!/bin/bash
# p66_gate2.sh — P6.6 leg 2: decompose the ~0.4s per-chunked-step overhead.
# Probe image (cert4 + ARC_P66 instrumentation via patch_p66b.py, r48j
# pattern), two boots: thr2048 (16-step face) and thr none (4-step face),
# solo U32k each, nolmc. Output: ARC_P66_STEP / ARC_P66_CHUNK / ARC_P66_RUN
# lines decomposing every step into sched / prep / prefill-block / fwd /
# tail / allreduce count+ms.
set -uo pipefail
R=/root/build
CT=$R/p66
IMG=vllm-arc:v0.30.0-b70-cert4
PROBE=vllm-arc:v0.30.0-b70-cert4-p66probe
COREF=/opt/venv/lib/python3.12/site-packages/vllm/v1/engine/core.py
RUNF=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/v1/worker/hpu_model_runner.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p66c.log 2>&1 &
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

echo "=== [1] extract + patch + probe image ==="
if ! docker image inspect $PROBE >/dev/null 2>&1; then
  mkdir -p $CT/probe_src/vllm/v1/engine $CT/probe_src/vllm_gaudi/v1/worker
  docker rm -f p66x2 >/dev/null 2>&1
  docker run -d --name p66x2 --entrypoint sleep $IMG 300 >/dev/null
  docker cp p66x2:$COREF $CT/probe_src/vllm/v1/engine/core.py
  docker cp p66x2:$RUNF $CT/probe_src/vllm_gaudi/v1/worker/hpu_model_runner.py
  docker rm -f p66x2 >/dev/null 2>&1
  python3 $CT/patch_p66b.py || { echo "FATAL: patcher"; exit 1; }
  docker run -d --name p66probe --entrypoint sleep $IMG 300 >/dev/null
  docker cp $CT/probe_out/core.py p66probe:$COREF
  docker cp $CT/probe_out/hpu_model_runner.py p66probe:$RUNF
  docker exec p66probe /opt/venv/bin/python -m py_compile $COREF $RUNF \
    || { echo "FATAL: in-container py_compile"; exit 1; }
  docker commit -m "p66 probe: step decomposition instrumentation" p66probe $PROBE >/dev/null
  docker rm -f p66probe >/dev/null 2>&1
fi
docker run --rm --entrypoint grep $PROBE -c ARC_P66 $RUNF

for THR in 2048 none; do
  echo "=== boot probe thr=$THR (nolmc) ==="
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
    bash "$SRV" > "$CT/boot_probe_$THR.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/boot_probe_$THR.out" || { echo "FATAL: launch $THR"; cat "$CT/boot_probe_$THR.out"; exit 1; }
  wait_health
  cd $CT && P66_FACES="U32k" python3 $CT/p66_solo_ttft.py $CT/p66_probe_thr$THR.json 2>&1 | tee $CT/p66_probe_thr$THR.txt
  docker exec arc-test sh -c "grep -h ARC_P66_ /root/serve_full.log | tail -60" > $CT/p66_lines_thr$THR.txt
  echo "-- ARC_P66 summary thr=$THR (last face):"
  grep -c "ARC_P66_STEP" $CT/p66_lines_thr$THR.txt
  tail -8 $CT/p66_lines_thr$THR.txt
  docker exec arc-test cat /root/serve_full.log > "$CT/serve_probe_$THR.log" 2>/dev/null
  docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
  docker rm -f arc-test >/dev/null 2>&1; sleep 5
done
echo "P66_GATE2: DONE"
