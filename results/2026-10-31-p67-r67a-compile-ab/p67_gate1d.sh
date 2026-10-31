#!/bin/bash
# p67_gate1d.sh — P6.7 r67a pass 2: prefill compile A/B with the GDN capture
# law satisfied. Pass 1 (gate1c) died at boot: our cudagraph_capture_sizes
# ladder topped at 320, but gdn_attn.py:117 hard-caps captured tokens at
# max_num_seqs*(num_spec+1) = 64*4 = 256 (decode_cudagraph_max_bs) — the
# synthetic (bs=64, tokens=320) PIECEWISE capture tripped
# build_for_cudagraph_capture's decode-only assert (gdn_attn.py:537).
# Fix: trim the ladder to 256 + max_cudagraph_capture_size=256 (satisfies
# compilation.py:1145). Real k3 verify = 64*4 = 256 exactly — decode FULL
# graphs unchanged vs cert4. Prefill >256 tokens runs compiled-uncaptured
# (the W14 law: sizes [4..64] max 64 on the same model — captured sizes
# must be decode-shaped; big prefill never replays a cudagraph).
# Boot = cert4 image, CLI-only change. Faces + run7n + audit + stock ref
# A/B as in gate1. Fast-fail watcher: boot death is captured and exits in
# minutes instead of parking 45 min with the log still in the container.
set -uo pipefail
R=/root/build
CT=$R/p67
mkdir -p $CT

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_serve_only.sh > $R/w14_restore_p67d.log 2>&1 < /dev/null &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

wait_health() { # $1 = max polls (5s each); fast-fails on engine-init death
  ok=""
  for i in $(seq 1 "$1"); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if docker exec arc-test grep -q "Engine core initialization failed" /root/serve_full.log 2>/dev/null; then
      echo "FATAL: engine init died after ~$((i*5))s — capturing serve log now"
      docker exec arc-test cat /root/serve_full.log > "$CT/serve_pw_dead.log" 2>/dev/null
      grep -m3 "RuntimeError\|Error" "$CT/serve_pw_dead.log" | head -3
      exit 1
    fi
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in $(( $1 * 5 / 60 )) min"; docker exec arc-test cat /root/serve_full.log > "$CT/serve_pw_dead.log" 2>/dev/null || true; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

SRV_PW=$R/serve_arc_b70_p67pwd.sh
# file bytes carry ESCAPED quotes ({\"mode\":\"none\"}) — sed patterns must
# match them byte-exact (gate1 pass-1 defect). Pass-2 addition: the ladder
# tail ,260..,320 is cut and max_cudagraph_capture_size=256 declared
# (compilation.py:1145 asserts last size == max).
sed -e 's|{\\"mode\\":\\"none\\"|{\\"mode\\":\\"vllm_compile\\"|' \
    -e 's|\\"cudagraph_mode\\":\\"FULL_DECODE_ONLY\\"|\\"cudagraph_mode\\":\\"FULL_AND_PIECEWISE\\"|' \
    -e 's|,260,265,270,275,280,285,290,295,300,305,310,315,320\]|],\\"max_cudagraph_capture_size\\":256|' \
    $R/serve_arc_b70.sh > "$SRV_PW"
grep -c '{\\"mode\\":\\"vllm_compile\\"' "$SRV_PW" >/dev/null || { echo "FATAL: sed mode"; exit 1; }
grep -c '\\"cudagraph_mode\\":\\"FULL_AND_PIECEWISE\\"' "$SRV_PW" >/dev/null || { echo "FATAL: sed cgmode"; exit 1; }
grep -c ',255,256\],\\"max_cudagraph_capture_size\\":256}' "$SRV_PW" >/dev/null || { echo "FATAL: sed sizes/max"; exit 1; }

echo "=== [1] boot cert4 PW combo v2 (sizes<=256 + max_cudagraph_capture_size=256) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash "$SRV_PW" > "$CT/boot_pwd.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_pwd.out" || { echo "FATAL: launch"; cat "$CT/boot_pwd.out"; exit 1; }
wait_health 540
docker exec arc-test cat /root/serve_full.log > "$CT/serve_pwd.log" 2>/dev/null
echo "-- config assertions:"
grep -o "compilation_config={'mode': <CompilationMode.VLLM_COMPILE[^,]*" "$CT/serve_pwd.log" | head -1 || echo "WARN: VLLM_COMPILE mode not in config dump"
grep -o "'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE[^,]*" "$CT/serve_pwd.log" | head -1 || true
grep -o "'max_cudagraph_capture_size': [0-9]*" "$CT/serve_pwd.log" | head -1 || true
grep -c "speculator graph mode coerced" "$CT/serve_pwd.log" >/dev/null && echo "-- speculator coercion: LIVE" || echo "-- speculator coercion: NOT SEEN (check draft path)"
grep "XPU KV cache size" "$CT/serve_pwd.log" || echo "-- KV pool line not found"
grep -o "Capturing CUDA graphs (PIECEWISE): 100%[^ ]*| [0-9]*/[0-9]*" "$CT/serve_pwd.log" | tail -1 || echo "-- piecewise capture bar not parsed (check log)"

echo "=== [2] solo TTFT faces (bars 3.74/17.33/39.19) ==="
cd $CT && P66_FACES="U8k,U32k,U64k" python3 $R/p66/p66_solo_ttft.py $CT/p67_pwd_solo.json 2>&1 | tee $CT/p67_pwd_solo.txt

echo "=== [3] run7n t0 (decode/acceptance + b2 survival) ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 > "$CT/p67_pwd_run7n_t0.out" 2>&1
grep -h 'aggregate=\|solo200\|pos1\|acceptance' "$CT/p67_pwd_run7n_t0.out" | head -10
docker exec arc-test cat /root/serve_full.log > "$CT/serve_pwd_after.log" 2>/dev/null
echo "-- audit: $(grep -c 'WARNING\|ERROR' $CT/serve_pwd_after.log) WARN/ERROR lines"
docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
docker rm -f arc-test >/dev/null 2>&1; sleep 5

echo "=== [4] stock nolmc refresher boot (run7n A/B ref) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash $R/serve_arc_b70.sh > "$CT/boot_ref_d.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_ref_d.out" || { echo "FATAL: launch ref"; exit 1; }
wait_health 240
python3 $R/r46t/bench_run7n_t.py --temperature 0 > "$CT/p67_refd_run7n_t0.out" 2>&1
grep -h 'aggregate=\|solo200\|pos1\|acceptance' "$CT/p67_refd_run7n_t0.out" | head -10
docker exec arc-test cat /root/serve_full.log > "$CT/serve_ref_d.log" 2>/dev/null
echo "P67_GATE1D: DONE"
