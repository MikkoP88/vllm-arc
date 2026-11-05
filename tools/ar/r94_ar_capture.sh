#!/bin/bash
# r94_ar_capture.sh — P6.22 r94 (AR telemetry capture, cert9 + probe
# posture) + r95 (AR primitives ceiling) in ONE GPU window.
#
# Kit (scp'd from the repo to $R/p622/kit/ before launch):
#   ar_trace.py     vllm_gaudi/xpu/patches/ar_trace.py  (the instrument)
#   platform.py     vllm_gaudi/xpu/platform.py (main: cert9-parity + the
#                   ar_trace tuple line — md5-logged vs image for record)
#   ar_ceiling.py   tools/ar/ar_ceiling.py   (r95 bench)
#   ar_summarize.py tools/ar/ar_summarize.py (offline census fold)
#
# Posture: r67b-comparable (cert9 image, ARC_GRAPH=1, ARC_SPEC=3,
# ARC_LMC=0, mode none) + probe-only deltas, ALL discarded with the
# container: (a) bind-mounted gpu_worker.py with the ONE activities
# CUDA->XPU line (p67 method), (b) bind-mounted platform.py+ar_trace.py,
# (c) ARC_AR_TRACE=1 in the exec env, (d) profiler=torch flags.
# Sequence: health -> U8k warm (unprofiled) -> /start_profile ->
# U32k solo -> U64k solo -> /stop_profile -> flush -> collect JSONL +
# traces -> summarize -> r95 throwaway bench (stock dist AR vs one-shot
# ceiling at the census sizes) -> teardown -> W14 restore by trap.
set -uo pipefail
R=/root/build
CT=$R/p622
KIT=$CT/kit
GW=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py
VG=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu
CCL_BLOCK="export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1 CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072 CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0"

mkdir -p $CT
for f in ar_trace.py platform.py ar_ceiling.py ar_summarize.py; do
  [ -s "$KIT/$f" ] || { echo "FATAL: kit missing $KIT/$f"; exit 1; }
done
md5sum $KIT/*

w14_start() {
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p622.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 20

wait_health() {
  ok=""
  for i in $(seq 1 "$1"); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if docker exec arc-test grep -q "Engine core initialization failed" /root/serve_full.log 2>/dev/null; then
      echo "FATAL: engine init died"; docker exec arc-test tail -30 /root/serve_full.log; exit 1
    fi
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health"; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

echo "=== [1] gpu_worker.py activities patch (p67 method, cert9) ==="
docker run --rm --entrypoint cat vllm-arc:v0.30.0-b70-cert9 $GW > $CT/gpu_worker_xpu.py
N=$(grep -c 'activities=\["CPU", "CUDA"\],' $CT/gpu_worker_xpu.py)
[ "$N" = "1" ] || { echo "FATAL: activities line count=$N (expected 1)"; exit 1; }
sed -i 's|activities=\["CPU", "CUDA"\],|activities=["CPU", "XPU"],|' $CT/gpu_worker_xpu.py
grep -c 'activities=\["CPU", "XPU"\],' $CT/gpu_worker_xpu.py || { echo "FATAL: xpu sed"; exit 1; }
echo "=== [1b] instrument syntax gate (no local python on host) ==="
docker run --rm -v $KIT:/kit:ro --entrypoint /opt/venv/bin/python vllm-arc:v0.30.0-b70-cert9 -c "
import ast; [ast.parse(open(f'/kit/{m}').read(), f'/kit/{m}') for m in ('ar_trace.py','platform.py','ar_ceiling.py','ar_summarize.py')]; print('KIT_SYNTAX_OK')" || { echo "FATAL: kit syntax"; exit 1; }

echo "=== [2] serve variant: profiler flags + 3 mounts + ARC_AR_TRACE ==="
SRV_PR=$R/serve_arc_b70_p622.sh
python3 - "$R/serve_arc_b70.sh" "$SRV_PR" "$CT" "$GW" "$VG" <<'PYEOF'
import sys
src, dst, ct, gwc, vg = sys.argv[1:6]
s = open(src).read()
old = "  $COMPILATION_FLAG \\\n"
new = ("  $COMPILATION_FLAG --profiler-config.profiler=torch "
       "--profiler-config.torch_profiler_dir=/root/prof "
       "--profiler-config.torch_profiler_with_stack=false \\\n")
assert s.count(old) == 1, f"COMPILATION_FLAG usage count={s.count(old)}"
s = s.replace(old, new)
old2 = '    --entrypoint /bin/bash "$ARC_IMAGE" -c \'sleep infinity\'\n'
new2 = (f"    -v {ct}/gpu_worker_xpu.py:{gwc}:ro \\\n"
        f"    -v {ct}/kit/platform.py:{vg}/platform.py:ro \\\n"
        f"    -v {ct}/kit/ar_trace.py:{vg}/patches/ar_trace.py:ro \\\n"
        + old2)
assert s.count(old2) == 1, f"entrypoint line count={s.count(old2)}"
s = s.replace(old2, new2)
old3 = "export ARC_TRACE_FLIP=1\n"
new3 = "export ARC_TRACE_FLIP=1\nexport ARC_AR_TRACE=1\n"
assert s.count(old3) == 1, f"ARC_TRACE_FLIP count={s.count(old3)}"
s = s.replace(old3, new3)
open(dst, "w").write(s)
print("serve p622 variant written (flags + mounts + ARC_AR_TRACE)")
PYEOF
grep -c "profiler-config.profiler=torch" "$SRV_PR" || { echo "FATAL: prof flag edit"; exit 1; }
grep -c "ar_trace.py" "$SRV_PR" || { echo "FATAL: mount edit"; exit 1; }
grep -c "ARC_AR_TRACE=1" "$SRV_PR" || { echo "FATAL: env edit"; exit 1; }
sed -i 's/\r$//' "$SRV_PR"

echo "=== [3] boot cert9 + probe posture ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert9 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash "$SRV_PR" > "$CT/boot_r94.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_r94.out" || { echo "FATAL: launch"; cat "$CT/boot_r94.out"; exit 1; }
wait_health 300
docker exec arc-test grep -c "ar_trace INSTALLED" /root/serve_full.log | tee $CT/ar_trace_install_count.txt
docker exec arc-test cat /models/target/config.json > $CT/model_config.json 2>/dev/null || true

echo "=== [4] warm U8k face (unprofiled) ==="
cd $CT && P66_FACES="U8k" python3 $R/p66/p66_solo_ttft.py $CT/r94_u8k_warm.json 2>&1 | tail -2

echo "=== [5] profile: U32k + U64k faces ==="
curl -sf -X POST http://127.0.0.1:8000/start_profile || { echo "FATAL: start_profile refused"; exit 1; }
P66_FACES="U32k" python3 $R/p66/p66_solo_ttft.py $CT/r94_u32k.json 2>&1 | tee $CT/r94_u32k.txt | tail -3
P66_FACES="U64k" python3 $R/p66/p66_solo_ttft.py $CT/r94_u64k.json 2>&1 | tee $CT/r94_u64k.txt | tail -3
curl -sf -X POST http://127.0.0.1:8000/stop_profile && echo "profiler stopped"

echo "=== [6] flush + collect ==="
for i in $(seq 1 36); do
  sleep 10
  C=$(docker exec arc-test sh -c "ls /root/prof/*.gz 2>/dev/null | wc -l")
  echo "poll$i gz=$C"
  [ "$C" -ge 2 ] && sleep 5 && break
done
docker exec arc-test sh -c "ls -la /root/prof/ /tmp/arc_ar_trace_*" | tee $CT/collect_listing.txt
mkdir -p $CT/traces $CT/census
docker cp arc-test:/root/prof/. $CT/traces/ 2>/dev/null || true
docker exec arc-test sh -c "cat /tmp/arc_ar_trace_*.jsonl" > $CT/census/all_ranks.jsonl 2>/dev/null || echo "FATAL: no census files"
docker exec arc-test cat /root/serve_full.log > $CT/serve_after.log 2>/dev/null
docker exec arc-test grep -c "ar_trace INSTALLED" /root/serve_full.log > $CT/ar_trace_installed_after.txt

echo "=== [7] fold the census ==="
python3 $KIT/ar_summarize.py $CT/census/all_ranks.jsonl --json $CT/census/summary.json | tee $CT/census/summary.txt

echo "=== [8] teardown serve, r95 ceiling bench ==="
docker rm -f arc-test >/dev/null 2>&1; sleep 10
SIZES=$(python3 - $CT/census/summary.json <<'PYEOF'
import json, sys
try:
    rows = json.load(open(sys.argv[1]))["rows"]
except Exception:
    print(""); raise SystemExit
sizes = []
for r in rows:
    if r["op"] == "all_reduce" and r["bytes"] and r["bytes"] >= (1 << 20):
        if r["bytes"] not in sizes:
            sizes.append(r["bytes"])
print(",".join(str(s) for s in sorted(sizes)[:6]))
PYEOF
)
[ -n "$SIZES" ] || SIZES="150994944,76677120,68157440,83886080,15779840,2621440"
echo "r95 sizes: $SIZES"
docker run --rm --privileged --ipc host --shm-size 32g \
  -v $KIT:/kit:ro -v $CT:/work -w /work \
  -e ZE_AFFINITY_MASK=0,1 \
  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert9 -c "
  export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
  $CCL_BLOCK
  /opt/venv/bin/python /kit/ar_ceiling.py --role oneshot --sizes-bytes $SIZES --out /work/r95_oneshot.json &&
  /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node 2 \
    /kit/ar_ceiling.py --role dist --sizes-bytes $SIZES --out /work/r95_dist.json
  " 2>&1 | tee $CT/r95_bench.out
echo "R94_R95_WINDOW: DONE (W14 restore by trap)"
