#!/bin/bash
# p617f_warm2.sh — P6.17f: fresh-seed cold/warm probe (p70_warm_rehit2.py)
# on the certified probabilistic line. Closes the warm-v1 validity gap:
# v1's t1 rows shared the ladder corpus (r70 VERDICT quarantined them);
# v2 uses disjoint seeds 6157/6159/6161. Boot via the deployed flipped
# serve script with ARC_SPEC_METHOD=probabilistic explicit for the record.
# Rig laws: W14 stopped for the leg, RESTORED on exit (trap); graph=1.
set -u
D=/root/build/p617
B=/root/build
R="$B/r70"
IMG=vllm-arc:v0.30.0-b70-cert6
SERVE="$B/serve_arc_b70.sh"
mkdir -p "$D/f"
ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $*"; }
exec >> "$D/warm2.log" 2>&1

w14_restore() {
  docker rm -f arc-p617f >/dev/null 2>&1 || true
  docker start radiance-test >/dev/null 2>&1 || true
  bash "$B/w14_serve_only.sh" > "$D/w14_restore2.log" 2>&1 || true
  log "W14_RESTORED (radiance-test up)"
}
trap 'rc=$?; log "warm2 exit rc=$rc"; w14_restore' EXIT

wait_health() {
  local deadline=$(( $(date +%s) + 1200 )) c
  while [ "$(date +%s)" -lt "$deadline" ]; do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 \
        http://127.0.0.1:8000/health || true)
    [ "$c" = "200" ] && return 0
    sleep 10
  done
  return 1
}

log "p617f warm2 mini-boot start (img=$IMG method=probabilistic)"
docker stop radiance-test >/dev/null 2>&1 || true
t0=$(date +%s)
ARC_IMAGE=$IMG ARC_NAME=arc-p617f ARC_SPEC=3 ARC_GRAPH=1 \
  ARC_SPEC_METHOD=probabilistic bash "$SERVE" > "$D/f/boot.out" 2>&1
if wait_health; then
  log "arc-p617f health 200 boot=$(( $(date +%s) - t0 ))s"
else
  log "arc-p617f HEALTH TIMEOUT — abort"
  exit 1
fi
# hook audit (same shape as campaign legs)
docker exec arc-p617f ps aux | grep -oF -- "--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}" \
  | sort -u > "$D/f/ps_spec_bytes.txt" 2>/dev/null || true
docker exec arc-p617f grep -c "draft_sample_method" /root/serve_full.log \
  > "$D/f/log_prob_key_count.txt" 2>/dev/null || true
log "hook_audit arc-p617f: ps_bytes=$(wc -l < "$D/f/ps_spec_bytes.txt") log_hits=$(cat "$D/f/log_prob_key_count.txt")"

# first-request survival (rig law: never first-measure on a cold engine face)
c=$(curl -s -o "$D/f/first_resp.json" -w "%{http_code}" -m 300 \
  -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Reply with OK."}]}')
echo "survival http=$c" > "$D/f/survival.txt"

# pre-window of the serve log (tier-2 attribution), probe, post-window
docker exec arc-p617f tail -c 400000 /root/serve_full.log \
  > "$D/f/serve_pre_warm.log" 2>/dev/null || true
python3 "$R/p70_warm_rehit2.py" "$D/f/warm_rehit2.json" \
  > "$D/f/warm_rehit2.txt" 2>&1
echo "warm2_rc=$?" >> "$D/f/warm_rehit2.txt"
docker exec arc-p617f tail -c 400000 /root/serve_full.log \
  > "$D/f/serve_post_warm.log" 2>/dev/null || true

grep -ac "ERROR\|Traceback" "$D/f/serve_full.log" > /dev/null 2>&1
docker cp arc-p617f:/root/serve_full.log "$D/f/serve_full.log" >/dev/null 2>&1 || true
echo "errors=$(grep -ac "ERROR\|Traceback" "$D/f/serve_full.log" || echo 0)" > "$D/f/log_hygiene.txt"
log "p617f warm2 COMPLETE (see f/warm_rehit2.txt)"
