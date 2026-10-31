#!/bin/bash
# p65_gate7.sh — P6.5 r48g/r48h: clean-matrix legs with the FIXED harness
# (v5: timer lambda now binds prompts[tg]; the v3/v4 closure bug made every
# face send the LAST face's prompt, so r48b-r48f "fresh" faces after the
# first were same-prompt replays — VERDICT r48f reversal). Two boots:
#   Boot A (r48g): certified + threshold 2048, connector ON.
#     Faces U32k (32k FRESH), C32k (32k replay), U64k (64k FRESH) — the
#     label-correct thr2048+LMC cells that no prior leg has — plus run7n t0.
#     This is also the PRE-FIX baseline for the adapter store-path fix.
#   Boot B (r48h): threshold 2048 + ARC_LMC=0 (connector OFF), face U32k
#     (32k FRESH) — direct [vii-a] isolation at 32k vs boot A's U32k.
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48g.log 2>&1 &
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

echo "=== [A1] serve copy: threshold 2048 (connector ON, certified otherwise) ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_lpt.sh
grep -c "long-prefill" $R/serve_arc_b70_lpt.sh || { echo "FATAL: sed failed"; exit 1; }

echo "=== [A2] boot A ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_lpt.sh > $CT/boot_lpt_r48g.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_lpt_r48g.out || { echo "FATAL: serve launch failed"; cat $CT/boot_lpt_r48g.out; exit 1; }
wait_health
docker exec arc-test grep -m1 -o "long_prefill_token_threshold[^,]*" /root/serve_full.log || true

echo "=== [A3] capture v5 (U32k FRESH + C32k replay + U64k FRESH) ==="
cd $CT && P65_FACES="U32k,C32k,U64k" python3 $CT/p65_timeline.py $CT/p65_tl_lpt_v5.json 2>&1 | tee $CT/p65_sum_lpt_v5.txt
grep "prompt sizes\|TTFT\|prompt_tokens=" $CT/p65_sum_lpt_v5.txt

echo "=== [A4] run7n t0 (pre-fix baseline) ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_lpt_v5.out
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48g.log 2>/dev/null

echo "=== [B1] boot B: threshold 2048 + ARC_LMC=0 ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_nolmc_lpt.sh
grep -c "long-prefill" $R/serve_arc_b70_nolmc_lpt.sh || { echo "FATAL: sed failed"; exit 1; }
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash $R/serve_arc_b70_nolmc_lpt.sh > $CT/boot_nolmc_r48h.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_nolmc_r48h.out || { echo "FATAL: serve launch failed"; cat $CT/boot_nolmc_r48h.out; exit 1; }
docker exec arc-test grep -c "kv_transfer_config\|LMCacheEngine" /root/serve_full.log || echo "CONFIRMED: no connector in engine"
wait_health

echo "=== [B2] capture v5 (U32k FRESH only — [vii-a] isolation at 32k) ==="
cd $CT && P65_FACES="U32k" python3 $CT/p65_timeline.py $CT/p65_tl_nolmc32k_v5.json 2>&1 | tee $CT/p65_sum_nolmc32k_v5.txt
grep "prompt sizes\|TTFT\|prompt_tokens=" $CT/p65_sum_nolmc32k_v5.txt

docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48h.log 2>/dev/null
echo "P65_LEG_R48GH: DONE"
