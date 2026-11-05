#!/bin/bash
# p621 r86 — W13H ar_quant donor-port A/B/A/B on cert9 (bind-mount probe,
# NO image build). One variable between postures: VLLM_ARC_AR_QUANT on/off
# (the new patches/ar_quant.py, platform.py, envs.py bind-mounted over the
# cert9 image tree). Phases interleaved arq1/off1/arq2/off2 + cf1
# counterfactual (armed, MIN_ELEMS=10^9 -> installed line present, active
# line ABSENT = the sizing knob gates the lane).
#
# Per-phase battery: p65_timeline v5.1 faces (the banked P6.15/P6.20
# metric), r42 integrity+det (determinism WITHIN posture — cross-posture
# hashes are meaningless, the lane is not byte-exact by design), p618
# longprobe full (coherence/garble under e4m3 peer noise), arc spec gate
# (MTP acceptance under noise), run7n t0 (per-posture determinism +
# throughput reference).
#
# Posture gates: ON phases must show BOTH the "patch installed" boot line
# and the post-traffic "all-reduce active" line; OFF/cf phases must show
# NEITHER installed-active pair leak (cf: installed yes, active no).
#
# Evidence: results/2026-11-05-p621-arquant/ (plan P6.21; donor
# /root/build/w13h, md5 b744f090...).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert9
SP=/opt/venv/lib/python3.12/site-packages
CTROOT=$R/p621/r86
FILES=$R/p621/files
mkdir -p "$CTROOT"
cd "$CTROOT"

w14_start() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  (nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p621.log 2>&1 &)
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1
sleep 20

kill_arc() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 10
}

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

# quality print (NOT a hard gate here — the r84 banked lesson: the helper's
# full-probe key-set grep prints benign ZERO-GARBLE FAIL artifact lines on
# files whose garble repr differs; the hard quality gates are the longprobe
# verdict lines + integrity/det + spec below).
garble_print() {
  local lines empty
  lines=$(grep -cE "^\[(essay|code|multi|simple)\]" "$1" 2>/dev/null || true)
  empty=$(grep -c "garble={'cold': \[\], 'rehit': \[\], 'rehit2': \[\]\]}" "$1" 2>/dev/null || true)
  echo "GARBLE-INFO: class_lines=$lines empty_garble=$empty in $(basename "$1")"
}

# $1=tag $2=arq 0/1 $3=minelems (optional, default unset)
phase() {
  local TAG=$1 EV=$2 MINE=${3:-} CT=$CTROOT/$1
  mkdir -p "$CT"
  echo "=================== [phase $TAG] ar_quant=$EV minelems=${MINE:-default} $(date +%H:%M:%S) ==================="
  kill_arc
  docker run -d --name arc-test --privileged --network host --ipc host \
    --shm-size 64g \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v /root/build/v127_stage/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
    -v /root/build/neo_cache:/root/.neo_cache \
    -v /root/build/triton_cache:/root/.triton_cache \
    -v /root/build/inductor_cache:/root/.inductor_cache \
    -v "$FILES/ar_quant.py":$SP/vllm_gaudi/xpu/patches/ar_quant.py \
    -v "$FILES/platform.py":$SP/vllm_gaudi/xpu/platform.py \
    -v "$FILES/envs.py":$SP/vllm_gaudi/envs.py \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --entrypoint /bin/bash "$IMG" -c 'sleep infinity'
  sleep 2
  cp $R/serve_arc_b70.sh "$CT/serve_p621_$TAG.sh"
  sed -i "s|export VLLM_XPU_ENABLE_XPU_GRAPH=1;|export VLLM_XPU_ENABLE_XPU_GRAPH=1; export VLLM_ARC_AR_QUANT=$EV;${MINE:+ export VLLM_ARC_AR_QUANT_MIN_ELEMS=$MINE;}|" "$CT/serve_p621_$TAG.sh"
  grep -q "VLLM_ARC_AR_QUANT=$EV" "$CT/serve_p621_$TAG.sh" || { echo "FATAL: env inject failed"; exit 1; }
  cp "$CT/serve_p621_$TAG.sh" "$CT/serve_used.sh"
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash "$CT/serve_p621_$TAG.sh" > "$CT/boot.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/boot.out" || { echo "FATAL: launch failed"; cat "$CT/boot.out"; exit 1; }
  wait_health | tee "$CT/health.txt"

  # boot-posture gate: the install line fires at register-time (register()
  # precedes model import) — present iff armed.
  docker exec arc-test sh -c "grep -m2 'ar_quant: quantized-transport all-reduce patch installed' /root/serve_full.log" > "$CT/posture_boot.txt" 2>&1 || true
  cat "$CT/posture_boot.txt"
  if [ "$EV" = "1" ]; then
    [ -s "$CT/posture_boot.txt" ] || { echo "FATAL: installed line absent in armed phase"; exit 1; }
  else
    [ -s "$CT/posture_boot.txt" ] && { echo "FATAL: installed line present in unarmed phase — posture leak"; exit 1; } || true
  fi

  echo "--- battery: integrity+det ---"
  python3 $R/r42/r42_integrity.py > "$CT/integrity.out" 2>&1
  I5=$(grep -c "INTEGRITY 5/5 PASS" "$CT/integrity.out" || true)
  D3=$(grep -c "DETERMINISM 3/3 PASS" "$CT/integrity.out" || true)
  echo "A-gates: integrity_5of5=$I5 det_3of3=$D3"
  { [ "$I5" = "1" ] && [ "$D3" = "1" ]; } || echo "A-GATES: FAIL (see integrity.out)"

  echo "--- battery: longprobe full (coherence under e4m3 peer noise) ---"
  python3 $R/p618/p618_longprobe.py p621$TAG full > "$CT/longprobe_full.out" 2>&1
  grep -E "^\[|LONGPROBE" "$CT/longprobe_full.out"
  garble_print "$CT/longprobe_full.out"

  echo "--- battery: spec gate ---"
  python3 $R/arc_spec_gate.py > "$CT/spec_gate.out" 2>&1
  tail -5 "$CT/spec_gate.out"

  echo "--- battery: run7n t0 ---"
  python3 $R/r46t/bench_run7n_t.py --temperature 0 > "$CT/t0_run7n.out" 2>&1
  grep -h 'aggregate=\|solo200' "$CT/t0_run7n.out"

  echo "--- battery: p65 timeline faces ---"
  python3 $R/p65/p65_timeline.py "$CT/timeline.json" > "$CT/timeline_stdout.txt" 2>&1 || echo "timeline exit=$? (see stdout)"
  tail -14 "$CT/timeline_stdout.txt"

  echo "--- post-traffic posture proof + bank ---"
  docker exec arc-test sh -c "grep -m2 'ar_quant: quantized-transport all-reduce active' /root/serve_full.log" > "$CT/posture_active.txt" 2>&1 || true
  cat "$CT/posture_active.txt"
  if [ "$EV" = "1" ] && [ -z "$MINE" ]; then
    [ -s "$CT/posture_active.txt" ] || { echo "FATAL: active line absent in armed phase (lane never engaged)"; exit 1; }
  elif [ -n "$MINE" ]; then
    [ -s "$CT/posture_active.txt" ] && { echo "FATAL: active line present under MIN_ELEMS=1e9 — sizing gate broken"; exit 1; } || echo "CF-OK: installed but inert under MIN_ELEMS=1e9"
  else
    [ -s "$CT/posture_active.txt" ] && { echo "FATAL: active line present in unarmed phase"; exit 1; } || true
  fi
  docker cp arc-test:/root/serve_full.log "$CT/serve_full.log" >/dev/null 2>&1 || true
  docker exec arc-test sh -c "grep -c 'ar_quant: disabled after failure' /root/serve_full.log" > "$CT/failsoft.txt" 2>&1 || true
  cat "$CT/failsoft.txt"
  grep -q "^0$" "$CT/failsoft.txt" || echo "WARN: fail-soft fired in this phase — inspect serve log"
  kill_arc
}

phase arq1 1
phase off1 0
phase arq2 1
phase off2 0
phase cf1  1 1000000000

echo "=================== P621 R86 DONE $(date +%F\ %H:%M:%S) ==================="
