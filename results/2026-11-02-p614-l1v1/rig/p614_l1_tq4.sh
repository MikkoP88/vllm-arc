#!/bin/bash
# p614_l1_tq4.sh — L1 fix leg (master plan P6.14): turboquant_4bit_nc on
# the p614 candidate image (TQ-aware lagged attn store, commit 50d283f1).
# Direct A/B vs results/2026-11-02-p65-tq4-ab: same rig, same certified
# 25-var chain, same battery; the ONLY variable is the image
# (cert5 -> p614c1 = the fix). fp8 face on the same image is a separate
# leg (default-face battery) — nothing here touches the certified posture.
#
# L1 gates (master plan P6.14):
#   G1 serve log: 0 ERROR lines + 0 Tracebacks (A/B: 112 pairs)
#   G2 kit TQ geometry live: [r43kvfmt] ... HS=268 AND [r43xfer] with
#      H=268 dir=D2H (store path actually transferring; A/B: 0 kit lines,
#      the pool never even reached the kit detector). H2D (retrieve) is
#      reported, not gated: healthy cert5 legs also show no H2D because
#      nothing evicts at this pool headroom (re-hit served internally).
#   G3 External prefix cache hit rate reported (max), informational
#   G4 MemoryObj GC warnings == 0 (A/B: 272 "is being garbage collected
#      with ref_count=1" lines -- downstream of the failed transfers)
#   G5 r42 integrity 5/5 (A/B: 3/5) + determinism 3/3
#   G6 spec gate PASS (MTP acceptance under 4bit KV, >= 0.85)
#   G7 run7n t0 in family: agg#2 within +-5% of 297.19, solo within
#      +-5% of 84.48 (TQ4 A/B was 299.36/85.97 — above family, PASS band)
#   G8 t1.0 xhigh + U32k/U64k bystander gaps re-measured (banked for the
#      verdict; quality disposition = owner call per the A/B reading)
# Trap restores W14 (recreate law: w14_boot.sh fullres ONLY).
set -uo pipefail
R=/root/build
CT=$R/p65/p614/l1_tq4
IMG=vllm-arc:v0.30.0-b70-p614c1
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_p614l1.log 2>&1 &
}
trap w14_start EXIT

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

# 0) pre-existing arc-test law: snapshot + remove
docker inspect arc-test > "$CT/preexisting_arc-test.json" 2>/dev/null || rm -f "$CT/preexisting_arc-test.json"
docker rm -f arc-test >/dev/null 2>&1
docker stop radiance-test >/dev/null 2>&1
sleep 30   # r48h boot-drain law

# 1) one-variable serve script off the p614c1 image (verify 1/0 substitution)
sed 's/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_4bit_nc/' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_tq4.sh
n=$(grep -c "turboquant_4bit_nc" $R/serve_arc_b70_tq4.sh)
f=$(grep -c "kv-cache-dtype fp8_e4m3" $R/serve_arc_b70_tq4.sh)
echo "SUBSTITUTIONS: tq=$n fp8_left=$f (must be 1/0)"
[ "$n" = "1" ] && [ "$f" = "0" ] || { echo "FATAL: sed did not produce a single-variable script"; exit 1; }
diff <(grep -oE "\-\-[a-z-]+ [^\\\\]+|export [A-Z_]+=" $R/serve_arc_b70.sh | head -80) \
     <(grep -oE "\-\-[a-z-]+ [^\\\\]+|export [A-Z_]+=" $R/serve_arc_b70_tq4.sh | head -80) \
  > "$CT/serve_diff.txt" || true
cat "$CT/serve_diff.txt"

# 2) boot + health (image-native adapter = the p614 fix; no volume mounts)
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_tq4.sh > "$CT/boot_tq4.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_tq4.out" || { echo "FATAL: launch failed"; cat "$CT/boot_tq4.out"; exit 1; }
wait_health

# 3) KV pool + posture lines (capacity claim must reproduce: ~757k tokens)
docker exec arc-test sh -c 'grep -E "KV cache size|memory pool|Available KV|TurboQuant|kv_cache_dtype|mamba" /root/serve_full.log | head -20' \
  > "$CT/posture_lines.txt" || true
cat "$CT/posture_lines.txt"

# 4) flip-first (832)
python3 - << 'PYEOF' > "$CT/flip_first.out" 2>&1
import json, time, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 832, "messages": [{"role": "user", "content": "Count from 1 to 20."}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
print("FLIP_FIRST:", r["choices"][0]["finish_reason"], f"{time.perf_counter()-t0:.1f}s")
PYEOF
cat "$CT/flip_first.out"

# 5) run7n t0 + t1.0 xhigh
B=$R/r46t/bench_run7n_t.py
python3 "$B" --temperature 0 > "$CT/tq4_t0_run7n.out" 2>&1
grep -h 'aggregate=\|solo200' "$CT/tq4_t0_run7n.out"
python3 "$B" --temperature 1.0 --thinking xhigh > "$CT/tq4_t1.0_run7n.out" 2>&1
grep -h 'aggregate=\|solo200' "$CT/tq4_t1.0_run7n.out"

# 6) spec gate
python3 $R/arc_spec_gate.py > "$CT/spec_gate.out" 2>&1
tail -6 "$CT/spec_gate.out"

# 7) integrity + determinism
python3 $R/r42/r42_integrity.py > "$CT/integrity.out" 2>&1
tail -3 "$CT/integrity.out"

# 8) long-context faces (U32k + U64k bystander-gap re-measure)
cd $CT && P65_FACES="U32k,U64k" P65_W2MT=4000 python3 $R/p65/p65_timeline.py $CT/p65_tl_tq4_v51.json 2>&1 | tee $CT/p65_sum_tq4_v51.txt
grep "prompt sizes\|TTFT" $CT/p65_sum_tq4_v51.txt

# 9) save engine log + WARN/ERROR audit
docker exec arc-test cat /root/serve_full.log > "$CT/serve_full_tq4.log" 2>/dev/null
grep -E "WARNING|ERROR" "$CT/serve_full_tq4.log" | grep -vE "lmcache.logging|filelock|asyncio" > "$CT/serve_warn_error.txt" || true
wc -l "$CT/serve_warn_error.txt"

# 10) GATES
echo "=== L1 GATES ==="
E=$(grep -c "ERROR" "$CT/serve_full_tq4.log" || true)
T=$(grep -c "Traceback" "$CT/serve_full_tq4.log" || true)
echo "G1 ERROR=$E Traceback=$T (want 0/0)"
KV=$(grep -c "r43kvfmt.*HS=268" "$CT/serve_full_tq4.log" || true)
XD=$(grep -c "r43xfer.*H=268.*dir=D2H" "$CT/serve_full_tq4.log" || true)
XH=$(grep -c "r43xfer.*dir=H2D" "$CT/serve_full_tq4.log" || true)
echo "G2 kvfmt_HS268=$KV (want >0) xfer_H268_D2H=$XD (want >0) xfer_H2D=$XH (informational)"
EXH=$(grep -oE "External prefix cache hit rate: [0-9.]+" "$CT/serve_full_tq4.log" | awk '{if ($NF+0 > m) m = $NF+0} END {print m+0}')
echo "G3 max External prefix cache hit rate = ${EXH}% (informational; cert5 legs also 0.0)"
GC=$(grep -c "is being garbage collected" "$CT/serve_full_tq4.log" || true)
echo "G4 MemoryObj GC-warning lines=$GC (want 0)"
I5=$(grep -c "INTEGRITY 5/5 PASS" "$CT/integrity.out" || true)
D3=$(grep -c "DETERMINISM 3/3 PASS" "$CT/integrity.out" || true)
echo "G5 integrity_5of5=$I5 determinism_3of3=$D3 (want 1/1)"
SP=$(grep -ciE "^SPEC.*(PASS|OK)|accept" "$CT/spec_gate.out" || true)
echo "G6 spec_gate lines=$SP (inspect spec_gate.out)"
AGG2=$(grep -oE "agg4x256#2 wall=[0-9.]+s aggregate=[0-9.]+" "$CT/tq4_t0_run7n.out" | grep -oE "[0-9.]+$" | head -1)
SOLO=$(grep -oE "tps=[0-9.]+" "$CT/tq4_t0_run7n.out" | grep -oE "[0-9.]+" | head -1)
AGG_OK=$(awk -v a="$AGG2" 'BEGIN{print (a>=282.33 && a<=312.05) ? 1 : 0}')
SOLO_OK=$(awk -v s="$SOLO" 'BEGIN{print (s>=80.26 && s<=88.70) ? 1 : 0}')
echo "G7 t0 agg#2=$AGG2 (band 282.33-312.05 ok=$AGG_OK) solo=$SOLO (band 80.26-88.70 ok=$SOLO_OK)"
echo "G8 t1.0 + faces: banked in $CT (verdict-side)"
echo "P614_L1: DONE (W14 restored by trap)"
