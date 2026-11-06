#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# r105t_cert10_today.sh — P6.26 TTFT-shift discriminator. r105 run-2
# (clean, quiescent host) still shows prefill TTFT faces +5..+15% over
# the r99 refs while every steady-state face sits AT ref (agg 300.48 vs
# 300.39, solo 84.55 vs 84.51, tps 326.43 vs 328.44) and platform.py is
# runtime-inert for the fp8 lane (supports_mx consumers are all
# mxfp4/mxfp8/quark/MoE; supported_quantization is config-validation
# only — grep-proven 2026-11-06). Engine deltas cannot explain it.
# This leg measures the IDENTICAL p65 timeline on cert10 ITSELF, today,
# same host state, same harness:
#   cert10-today ~ r99 refs        -> refs drifted with rig state
#                                       (re-anchor on this leg, doc'd)
#   cert10-today at ref, cert11c high -> real cert11c regression
#                                       (next isolate: cert10 +
#                                       docker-cp platform.py, r103b
#                                       pattern)
# W14 stopped for the window, restored by trap; kill_arc r84 retry law.
set -uo pipefail
R=/root/build
CT=$R/p626/r105t
IMG=vllm-arc:v0.30.0-b70-cert10

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash "$R"/w14_boot.sh fullres > "$R"/w14_restore_p626t.log 2>&1 < /dev/null &
}
trap w14_start EXIT
# Stop W14 only AFTER it is serving (r105t attempt-1 lesson: stopping
# mid-boot risks incomplete device cleanup; and never pkill -f a pattern
# that matches your own ssh cmdline — kill by PID).
for i in $(seq 1 60); do
  docker exec radiance-test curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  sleep 10
done
docker exec radiance-test curl -s -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 \
  && echo "W14 healthy before window" \
  || echo "WARN: W14 not healthy before window (restored from mid-state)"
docker stop radiance-test >/dev/null 2>&1; sleep 20

if pgrep -f "snapshot_download" >/dev/null 2>&1; then
  echo "FATAL: host model download in flight — quiesce the rig first"
  exit 1
fi
# Load-settle guard (attempt-1 lesson: the W14 restore-boot echo ran LA
# ~15 into this leg's boot window): require 1-min LA < 3 before opening
# the arc window, 20-min cap.
for i in $(seq 1 120); do
  LA=$(cut -d, -f1 /proc/loadavg | cut -d. -f1)
  [ "$LA" -lt 3 ] && break
  sleep 10
done
echo "loadavg at window open: $(cat /proc/loadavg)"

mkdir -p $CT/def
cd $CT/def || exit 1
uptime > host_load.txt; cat host_load.txt

kill_arc() {  # r84 retry law
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1 || true
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1 || true
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 10
  docker inspect arc-test >/dev/null 2>&1 && { echo "FATAL: arc-test undeletable"; exit 1; }
}
kill_arc

echo "=== R105T CERT10-TODAY TIMELINE START $(date +%F\ %H:%M:%S) ==="
( ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 \
    bash $R/serve_arc_b70.sh > boot.out 2>&1 )
grep ARC_SERVE_LAUNCHED boot.out || { echo "FATAL: launch failed"; cat boot.out; exit 1; }
ok=""
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
echo "HEALTH=OK after ~$((i*5))s"
docker exec arc-test sh -c "grep -E 'KV cache size|Xe2 posture active' /root/serve_full.log | head -2" > posture.txt || true
cat posture.txt

# Same warm-up shape as r105 [1] before its timeline: the battery ran
# integrity + run7n + spec + tps + longprobe first; replicate a decode
# warm set (~2 min of generation) so cache/allocator state is comparable.
# Engine-alive guard: attempt-1's orphan bench ran against a trap-killed
# engine and banked pure 500s — verify a live completion first.
python3 - > warm_probe.out 2>&1 << 'PYEOF' || true
import json, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 16, "messages": [{"role": "user", "content": "Say OK."}]}
try:
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=120))
    print("WARM_PROBE_OK", r["choices"][0]["finish_reason"])
except Exception as e:
    print("WARM_PROBE_FAIL", e)
PYEOF
cat warm_probe.out
grep -q WARM_PROBE_OK warm_probe.out || { echo "FATAL: engine not serving completions"; exit 1; }
echo "loadavg at timeline warm start: $(cat /proc/loadavg)"
python3 $R/r46t/bench_run7n_t.py --temperature 0 > warm_run7n.out 2>&1 || true
grep -hE 'aggregate=|solo200' warm_run7n.out | head -4
echo "loadavg at timeline start: $(cat /proc/loadavg)"

echo "--- p65 TTFT face (cert10 today) $(date +%H:%M:%S) ---"
# Thermal/power telemetry capture points (r105t2 cooldown leg): bank
# device state around the faces to test the sustained-load throttle
# hypothesis for the prefill-TTFT drift (decode at ref, prefill +6-15%).
if command -v xpu-smi >/dev/null 2>&1; then
  : > thermal_before.txt
  for d in 0 1; do
    xpu-smi dump --device $d --metrics POWER,TEMPERATURE,CLOCK --number 1 2>/dev/null | tail -2 >> thermal_before.txt || true
  done
  cat thermal_before.txt
else
  echo "xpu-smi absent" > thermal_before.txt
fi
python3 $R/p65/p65_timeline.py timeline.json > timeline_stdout.txt 2>&1 || echo "timeline exit=$?"
if command -v xpu-smi >/dev/null 2>&1; then
  : > thermal_after.txt
  for d in 0 1; do
    xpu-smi dump --device $d --metrics POWER,TEMPERATURE,CLOCK --number 1 2>/dev/null | tail -2 >> thermal_after.txt || true
  done
  tail -4 thermal_after.txt
fi
grep -A5 'long-stream faces' timeline_stdout.txt || tail -15 timeline_stdout.txt
grep -A2 'bystander inter-chunk' timeline_stdout.txt | head -3
docker exec arc-test sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' > erraudit.txt 2>/dev/null || true
cat erraudit.txt
docker exec arc-test cat /root/serve_full.log > serve_full.log 2>/dev/null || true
echo "=== R105T COMPLETE $(date +%F\ %H:%M:%S) ==="
