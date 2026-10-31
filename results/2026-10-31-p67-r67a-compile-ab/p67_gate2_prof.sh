#!/bin/bash
# p67_gate2_prof.sh — P6.7 r67b: op-class prefill capture (E-law leg for
# rung-B/C sizing). ONE U32k solo face profiled with torch-profiler XPU
# device activities, on the STOCK cert4 posture (mode none, zero engine
# change). Probe-only, never shipped, all discarded with the container:
#   1) serve script copy gets (a) + --profiler-config.profiler=torch
#      --profiler-config.torch_profiler_dir=/root/prof (dotted CLI per
#      gpu_worker.py's own error message; the profile router attaches only
#      when profiler is set — profile/api_router.py attach_router), and
#      (b) one extra -v bind mount in the docker run block.
#   2) the bind mount overlays a host-side copy of
#      vllm/v1/worker/gpu_worker.py with ONE line changed:
#      activities=["CPU","CUDA"] -> ["CPU","XPU"] (0.30 hardcodes CUDA;
#      the XPU map key exists — profiler/wrapper.py TorchProfilerActivityMap).
#      Bind mount => zero import race; the file is patched before the
#      container even starts serving.
# Sequence: health -> U8k face run (decode/graph/JIT warm, NOT profiled)
# -> /start_profile -> U32k face run (profiled) -> /stop_profile -> wait
# trace flush -> listing.
set -uo pipefail
R=/root/build
CT=$R/p67
mkdir -p $CT
GW=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_serve_only.sh > $R/w14_restore_p67b.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

wait_health() {
  ok=""
  for i in $(seq 1 "$1"); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if docker exec arc-test grep -q "Engine core initialization failed" /root/serve_full.log 2>/dev/null; then
      echo "FATAL: engine init died"; docker exec arc-test cat /root/serve_full.log > "$CT/serve_prof_dead.log" 2>/dev/null; exit 1
    fi
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health"; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

# --- host-side patched gpu_worker.py (bind-mount source)
docker run --rm --entrypoint cat vllm-arc:v0.30.0-b70-cert4 $GW > $CT/gpu_worker_xpu.py
N=$(grep -c 'activities=\["CPU", "CUDA"\],' $CT/gpu_worker_xpu.py)
[ "$N" = "1" ] || { echo "FATAL: activities line count=$N (expected 1)"; exit 1; }
sed -i 's|activities=\["CPU", "CUDA"\],|activities=["CPU", "XPU"],|' $CT/gpu_worker_xpu.py
grep -c 'activities=\["CPU", "XPU"\],' $CT/gpu_worker_xpu.py || { echo "FATAL: xpu sed"; exit 1; }
echo "gpu_worker_xpu.py: 1-line activities patch ready"

# --- serve script variant: profiler flags + the bind mount
SRV_PR=$R/serve_arc_b70_p67prof.sh
python3 - "$R/serve_arc_b70.sh" "$SRV_PR" "$CT/gpu_worker_xpu.py" "$GW" <<'PYEOF'
import sys
src, dst, patched, gwc = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = open(src).read()
old = "  $COMPILATION_FLAG \\\n"
new = ("  $COMPILATION_FLAG --profiler-config.profiler=torch "
       "--profiler-config.torch_profiler_dir=/root/prof "
       "--profiler-config.torch_profiler_with_stack=false \\\n")
assert s.count(old) == 1, f"COMPILATION_FLAG usage count={s.count(old)}"
s = s.replace(old, new)
old2 = '    --entrypoint /bin/bash "$ARC_IMAGE" -c \'sleep infinity\'\n'
new2 = f'    -v {patched}:{gwc}:ro \\\n' + old2
assert s.count(old2) == 1, f"entrypoint line count={s.count(old2)}"
s = s.replace(old2, new2)
open(dst, "w").write(s)
print("serve prof variant written (flags + mount)")
PYEOF
grep -c "profiler-config.profiler=torch" "$SRV_PR" || { echo "FATAL: prof flag edit"; exit 1; }
grep -c "gpu_worker_xpu.py" "$SRV_PR" || { echo "FATAL: mount edit"; exit 1; }

echo "=== [1] boot cert4 stock + profiler flags + activities mount ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash "$SRV_PR" > "$CT/boot_prof.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_prof.out" || { echo "FATAL: launch"; cat "$CT/boot_prof.out"; exit 1; }
# mounted patch took effect if the in-container file shows XPU:
docker exec arc-test grep -c 'activities=\["CPU", "XPU"\],' $GW || { echo "FATAL: mount not effective"; exit 1; }
echo "activities CUDA->XPU: mounted, verified in-container"

wait_health 300
docker exec arc-test mkdir -p /root/prof
docker exec arc-test cat /root/serve_full.log > "$CT/serve_prof_boot.log" 2>/dev/null

echo "=== [2] warmup face U8k (decode/graph/JIT warm, not profiled) ==="
cd $CT && P66_FACES="U8k" python3 $R/p66/p66_solo_ttft.py $CT/p67_prof_warm.json 2>&1 | tail -2

echo "=== [3] /start_profile + U32k face (profiled) + /stop_profile ==="
curl -sf -X POST http://127.0.0.1:8000/start_profile || { echo "FATAL: start_profile refused"; docker exec arc-test tail -5 /root/serve_full.log; exit 1; }
echo "profiler started"
P66_FACES="U32k" python3 $R/p66/p66_solo_ttft.py $CT/p67_prof_u32k.json 2>&1 | tee $CT/p67_prof_u32k.txt
curl -sf -X POST http://127.0.0.1:8000/stop_profile && echo "profiler stopped"

echo "=== [4] wait trace flush ==="
for i in $(seq 1 36); do
  sleep 10
  C=$(docker exec arc-test sh -c "ls /root/prof/*.gz 2>/dev/null | wc -l")
  echo "poll$i gz=$C"
  [ "$C" -ge 2 ] && sleep 5 && break
done
docker exec arc-test sh -c "ls -la /root/prof/" | tee $CT/prof_dir_listing.txt
docker exec arc-test cat /root/serve_full.log > "$CT/serve_prof_after.log" 2>/dev/null
echo "P67_GATE2: DONE"
