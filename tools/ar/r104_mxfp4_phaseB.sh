#!/bin/bash
# r104_mxfp4_phaseB.sh — P6.26 phase B: both mxfp4 checkpoints on the
# PATCHED engine image (cert11c candidate = cert10 + ArcXpuPlatform quark/
# supports_mx), FULL certified posture (graph FULL_DECODE_ONLY + MTP k3
# probabilistic + LMCache — the incumbent's exact posture, serve_arc_b70.sh
# defaults). Phase A proved the quant path in isolation; phase B proves it
# under the shipped posture and measures the G2 perf faces.
#
# Per leg: boot via serve_arc_b70.sh (ARC_MODEL/ARC_SERVED/ARC_QUANT knobs),
#   - telemetry: kernel-dispatch lines, graph posture line, KV pool
#   - G1: 3-probe t0 quality sample (factual / reasoning / code), spec-gate
#     (arc_spec_gate.py --model <served> — acceptance-ratio + correctness)
#   - G2: p65_timeline full faces (U32k/C32k/U8k/U64k + bystander waves),
#     model-parameterized harness copy (sed on MODEL, r99b precedent)
# W14 stopped for the window, restored by trap (rig law).
set -uo pipefail
R=/root/build
CT=$R/p626/phaseB
IMG=vllm-arc:v0.30.0-b70-cert11c
mkdir -p $CT

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p626b.log 2>&1 < /dev/null &
}
trap w14_start EXIT

docker stop radiance-test >/dev/null 2>&1 || true
sleep 3

docker image inspect $IMG >/dev/null 2>&1 || { echo "FATAL: image $IMG absent"; exit 1; }

kill_arc() {  # r84 retry law
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1 || true
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1 || true
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 5
}

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -50 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

run_leg() {  # $1 tag, $2 model dir, $3 served name
  local TAG=$1 MODEL=$2 SERVED=$3 LEG=$CT/$1
  mkdir -p $LEG
  echo "=== [$TAG] phase-B boot start $(date +%H:%M:%S) ==="
  kill_arc
  ARC_IMAGE=$IMG ARC_NAME=arc-test \
  ARC_MODEL=$MODEL ARC_SERVED=$SERVED ARC_QUANT= \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 \
    bash $R/serve_arc_b70.sh
  wait_health
  docker cp arc-test:/root/serve_full.log $LEG/serve.log >/dev/null 2>&1 || true
  echo "--- [$TAG] telemetry ---"
  grep -n "Using .* for MXFP4 GEMM\|Selected .* for CompressedTensors\|graph posture\|Available KV cache\|quantization=\|Simulated\|supports_mx" $LEG/serve.log | head -10

  echo "--- [$TAG] G1: t0 quality probes $(date +%H:%M:%S) ---"
  python3 - "$SERVED" > $LEG/g1_probes.txt 2>&1 << 'PYEOF' || true
import json, sys, urllib.request
served = sys.argv[1]
probes = [
    ("factual",  "Say the number seven in words, then stop.", 24),
    ("reason",   "If a train leaves at 3pm traveling 60 km/h and another at 4pm traveling 90 km/h the same direction, when does the second catch the first? Answer with just the time.", 96),
    ("code",     "Write a Python one-liner that reverses a string s. Output only the code.", 64),
]
for name, q, mt in probes:
    body = {"model": served, "max_tokens": mt, "temperature": 0,
            "messages": [{"role": "user", "content": q}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=300))
    m = r["choices"][0]["message"]
    txt = (m.get("reasoning") or "") + "|" + (m.get("content") or "")
    u = r["usage"]
    print(f"[{name}] finish={r['choices'][0]['finish_reason']} tok={u['completion_tokens']}")
    print(f"  {txt[:400]!r}")
PYEOF
  cat $LEG/g1_probes.txt

  echo "--- [$TAG] G1: spec gate (k3 probabilistic) $(date +%H:%M:%S) ---"
  python3 $R/arc_spec_gate.py --model "$SERVED" --url http://127.0.0.1:8000 > $LEG/spec_gate.txt 2>&1 || true
  tail -12 $LEG/spec_gate.txt

  echo "--- [$TAG] G2: p65 timeline faces $(date +%H:%M:%S) ---"
  local TL=$R/p65/p65_timeline_p626_$TAG.py
  cp $R/p65/p65_timeline.py $TL
  sed -i "s|^MODEL = .*|MODEL = \"$SERVED\"|" $TL
  grep -q "MODEL = \"$SERVED\"" $TL || { echo "FATAL: harness sed failed"; exit 1; }
  python3 $TL $LEG/timeline.json > $LEG/timeline_stdout.txt 2>&1 || echo "timeline exit=$?"
  grep -A6 "long-stream faces" $LEG/timeline_stdout.txt || tail -20 $LEG/timeline_stdout.txt

  docker exec arc-test cat /root/serve_full.log > $LEG/serve_end.log 2>/dev/null || true
  echo "=== [$TAG] phase-B leg done $(date +%H:%M:%S) ==="
}

run_leg rhmxfp4 /models/qwen3.8-27b-mxfp4xfp8     qwen3.8-27b-mxfp4xfp8
run_leg qurk    /models/qwen3.8-27b-quark-awq-mxfp4 qwen3.8-27b-quark-awq-mxfp4

echo "=== phase-B complete $(date +%H:%M:%S); W14 restoring via trap ==="
