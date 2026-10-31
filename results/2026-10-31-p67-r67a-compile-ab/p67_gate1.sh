#!/bin/bash
# p67_gate1.sh — P6.7 r67a: prefill compile A/B (REVAL of Appendix H law #1
# on 0.30). Combo law (Appendix C b2): compiled prefill must be CAPTURED =>
# mode piecewise + cudagraph_mode FULL_AND_PIECEWISE (W14-style), NEVER
# piecewise x FULL_DECODE_ONLY (known-dead: loose launch -> stream flagged
# recording -> 2nd-request 500). Boot = cert4 image, CLI-only change.
# Faces: solo TTFT U8k/U32k/U64k thr-none nolmc (bars 3.74/17.33/39.19,
# r66c same-session class) + run7n t0 (decode/acceptance + the b2 survival
# gate: hundreds of requests) + WARN/ERROR audit + speculator-coercion
# LIVE check. Then a stock nolmc boot for the same-session run7n A/B ref.
set -uo pipefail
R=/root/build
CT=$R/p67
mkdir -p $CT

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p67a.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

wait_health() { # $1 = max polls (5s each)
  ok=""
  for i in $(seq 1 "$1"); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in $(( $1 * 5 / 60 )) min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

SRV_PW=$R/serve_arc_b70_p67pw.sh
# the flag's JSON lives in the script with ESCAPED quotes ({\"mode\":\"none\"})
# — the pass-1 sed patterns lacked the backslashes and matched NOTHING
# (banked defect; assertions below check the escaped forms byte-exact)
sed -e 's|{\\"mode\\":\\"none\\"|{\\"mode\\":\\"vllm_compile\\"|' \
    -e 's|\\"cudagraph_mode\\":\\"FULL_DECODE_ONLY\\"|\\"cudagraph_mode\\":\\"FULL_AND_PIECEWISE\\"|' \
    $R/serve_arc_b70.sh > "$SRV_PW"
grep -c '{\\"mode\\":\\"vllm_compile\\"' "$SRV_PW" >/dev/null || { echo "FATAL: sed mode"; exit 1; }
grep -c '\\"cudagraph_mode\\":\\"FULL_AND_PIECEWISE\\"' "$SRV_PW" >/dev/null || { echo "FATAL: sed cgmode"; exit 1; }

echo "=== [1] boot cert4 PW combo (piecewise x FULL_AND_PIECEWISE, nolmc, 45-min wait: compile) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash "$SRV_PW" > "$CT/boot_pw.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_pw.out" || { echo "FATAL: launch"; cat "$CT/boot_pw.out"; exit 1; }
wait_health 540
docker exec arc-test cat /root/serve_full.log > "$CT/serve_pw.log" 2>/dev/null
echo "-- config assertions:"
grep -o "compilation_config={'mode': <CompilationMode.VLLM_COMPILE[^,]*" "$CT/serve_pw.log" | head -1 || echo "WARN: PIECEWISE mode not in config dump"
grep -c "speculator graph mode coerced" "$CT/serve_pw.log" >/dev/null && echo "-- speculator coercion: LIVE" || echo "-- speculator coercion: NOT SEEN (check draft path)"
grep "XPU KV cache size" "$CT/serve_pw.log" || echo "-- KV pool line not found"

echo "=== [2] solo TTFT faces (bars 3.74/17.33/39.19) ==="
cd $CT && P66_FACES="U8k,U32k,U64k" python3 $R/p66/p66_solo_ttft.py $CT/p67_pw_solo.json 2>&1 | tee $CT/p67_pw_solo.txt

echo "=== [3] run7n t0 (decode/acceptance + b2 survival) ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 > "$CT/p67_pw_run7n_t0.out" 2>&1
grep -h 'aggregate=\|solo200\|pos1\|acceptance' "$CT/p67_pw_run7n_t0.out" | head -10
docker exec arc-test cat /root/serve_full.log > "$CT/serve_pw_after.log" 2>/dev/null
echo "-- audit: $(grep -c 'WARNING\|ERROR' $CT/serve_pw_after.log) WARN/ERROR lines"
docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
docker rm -f arc-test >/dev/null 2>&1; sleep 5

echo "=== [4] stock nolmc refresher boot (run7n A/B ref) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash $R/serve_arc_b70.sh > "$CT/boot_ref.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_ref.out" || { echo "FATAL: launch ref"; exit 1; }
wait_health 240
python3 $R/r46t/bench_run7n_t.py --temperature 0 > "$CT/p67_ref_run7n_t0.out" 2>&1
grep -h 'aggregate=\|solo200\|pos1\|acceptance' "$CT/p67_ref_run7n_t0.out" | head -10
docker exec arc-test cat /root/serve_full.log > "$CT/serve_ref.log" 2>/dev/null
echo "P67_GATE1: DONE"
