#!/bin/bash
# p66_gate2e.sh — P6.6 leg 2 v5 DIAGNOSTIC: one boot (thr2048, solo U32k,
# nolmc) to answer WHERE the worker instrumentation vanishes. Probe image
# cert4 + patch_p66e.py (v4 subs + IMPORT markers + Worker/GPUModelRunner
# module-tail wrappers). Expectation table:
#   IMPORT_RUNNER lines w/ worker pids?  no  -> H1 (different module loaded)
#   WRK_EXEC lines present?               no  -> H3 (writes fail in workers)
#   WRK_EXEC yes, EXEC/EXEC2 no           -> H2 (method replaced wholesale)
set -uo pipefail
R=/root/build
CT=$R/p66
IMG=vllm-arc:v0.30.0-b70-cert4
PROBE=vllm-arc:v0.30.0-b70-cert4-p66probe4
SP=/opt/venv/lib/python3.12/site-packages
COREF=$SP/vllm/v1/engine/core.py
RUNF=$SP/vllm/v1/worker/gpu_model_runner.py
WORKF=$SP/vllm/v1/worker/gpu_worker.py
XWORKF=$SP/vllm/v1/worker/xpu_worker.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p66f.log 2>&1 &
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

echo "=== [1] extract + patch + probe image (p66probe4) ==="
mkdir -p $CT/probe_src/vllm/v1/engine $CT/probe_src/vllm/v1/worker
docker rm -f p66x5 >/dev/null 2>&1
docker run -d --name p66x5 --entrypoint sleep $IMG 300 >/dev/null
docker cp p66x5:$COREF  $CT/probe_src/vllm/v1/engine/core.py
docker cp p66x5:$RUNF   $CT/probe_src/vllm/v1/worker/gpu_model_runner.py
docker cp p66x5:$WORKF  $CT/probe_src/vllm/v1/worker/gpu_worker.py
docker cp p66x5:$XWORKF $CT/probe_src/vllm/v1/worker/xpu_worker.py
docker rm -f p66x5 >/dev/null 2>&1
rm -rf $CT/probe_out
python3 $CT/patch_p66e.py || { echo "FATAL: patcher"; exit 1; }
docker rm -f p66probe4 >/dev/null 2>&1
docker run -d --name p66probe4 --entrypoint sleep $IMG 300 >/dev/null
docker cp $CT/probe_out/core.py            p66probe4:$COREF
docker cp $CT/probe_out/gpu_model_runner.py p66probe4:$RUNF
docker cp $CT/probe_out/gpu_worker.py      p66probe4:$WORKF
docker cp $CT/probe_out/xpu_worker.py      p66probe4:$XWORKF
docker exec p66probe4 /opt/venv/bin/python -m py_compile $COREF $RUNF $WORKF $XWORKF \
  || { echo "FATAL: in-container py_compile"; exit 1; }
docker commit -m "p66 probe v5: worker-vanishing diagnostics" p66probe4 $PROBE >/dev/null
docker rm -f p66probe4 >/dev/null 2>&1

echo "=== [2] boot probe4 thr=2048 ==="
SRV=$R/serve_arc_b70_p66_2048.sh
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
  $R/serve_arc_b70.sh > "$SRV"
grep -c "long-prefill-token-threshold 2048" "$SRV" >/dev/null || { echo "FATAL: sed"; exit 1; }
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$PROBE ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash "$SRV" > "$CT/boot_probe4_2048.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_probe4_2048.out" || { echo "FATAL: launch"; cat "$CT/boot_probe4_2048.out"; exit 1; }
wait_health
cd $CT && P66_FACES="U32k" python3 $CT/p66_solo_ttft.py $CT/p66_probe4_thr2048.json 2>&1 | tee $CT/p66_probe4_thr2048.txt
docker cp arc-test:/root/p66_steps.log $CT/p66_steps4_thr2048.txt || echo "WARN: no p66_steps.log"
echo "-- line-type census:"
awk '{for(i=1;i<=NF;i++) if($i ~ /^ARC_P66_/) {print $i; break}}' $CT/p66_steps4_thr2048.txt | sort | uniq -c
echo "-- import markers:"
grep ARC_P66_IMPORT $CT/p66_steps4_thr2048.txt
docker exec arc-test cat /root/serve_full.log > "$CT/serve_probe4_2048.log" 2>/dev/null
docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
docker rm -f arc-test >/dev/null 2>&1; sleep 5
echo "P66_GATE2E: DONE"
