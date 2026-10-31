#!/bin/bash
# p65_gate3.sh — P6.5 r48c: discriminator + fix-face leg.
#   Boot A: certified posture + VLLM_ARC_LOOKUP_ELIDE=65536 (env-only;
#     skips ZMQ lookups on chunked re-admissions; first admissions still
#     walk) — discriminates [i] sync-lookup walk (TTFT recovers) vs [iv]
#     mamba snap (TTFT unchanged ~77 s would need threshold OFF anyway).
#     Faces: U32k + C32k.
#   Boot B: ELIDE + --long-prefill-token-threshold 2048 — the fix face:
#     walk elided + step time bounded. Faces: U32k + U64k + C32k + run7n t0.
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48c.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

boot_and_wait() {  # $1=serve-script $2=bootlog
  docker rm -f arc-test >/dev/null 2>&1
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 bash "$1" > "$2" 2>&1
  grep ARC_SERVE_LAUNCHED "$2" || { echo "FATAL: serve launch failed"; cat "$2"; exit 1; }
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

# ---------- Boot A: ELIDE only ----------
echo "=== [A1] serve copy: certified + ELIDE=65536 ==="
sed -e 's/^export ARC_TRACE_FLIP=1$/export ARC_TRACE_FLIP=1\nexport VLLM_ARC_LOOKUP_ELIDE=65536/' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_elide.sh
grep -c "VLLM_ARC_LOOKUP_ELIDE" $R/serve_arc_b70_elide.sh || { echo "FATAL: sed A failed"; exit 1; }
echo "=== [A2] boot ==="
boot_and_wait $R/serve_arc_b70_elide.sh $CT/boot_elide.out
echo "=== [A3] capture (U32k + C32k) ==="
cd $CT && P65_FACES="U32k,C32k" python3 $CT/p65_timeline.py $CT/p65_tl_elide.json 2>&1 | tee $CT/p65_sum_elide.txt
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48cA.log 2>/dev/null
docker stop arc-test >/dev/null 2>&1; sleep 3

# ---------- Boot B: ELIDE + threshold 2048 ----------
echo "=== [B1] serve copy: ELIDE + threshold 2048 ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
  $R/serve_arc_b70_elide.sh > $R/serve_arc_b70_elide_lpt.sh
grep -c "long-prefill" $R/serve_arc_b70_elide_lpt.sh || { echo "FATAL: sed B failed"; exit 1; }
echo "=== [B2] boot ==="
boot_and_wait $R/serve_arc_b70_elide_lpt.sh $CT/boot_elide_lpt.out
echo "=== [B3] capture (U32k + U64k + C32k) ==="
cd $CT && P65_FACES="U32k,U64k,C32k" python3 $CT/p65_timeline.py $CT/p65_tl_elide_lpt.json 2>&1 | tee $CT/p65_sum_elide_lpt.txt
echo "=== [B4] run7n t0 regression ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_elide_lpt.out
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48cB.log 2>/dev/null
echo "P65_LEG_R48C: DONE"
