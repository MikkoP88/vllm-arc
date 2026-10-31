#!/bin/bash
# r70_campaign.sh — P6.10: comprehensive cert4 measurement campaign.
#
# Owner directives (2026-10-31): "cert4 has to be measured comprehensively
# and all information has to be exist based on last certificated image not
# from other image" + "all feature, performance and etc... informations
# has to be more detailed, comprehensive and World-class professional
# grade". Every current-record face is re-measured NATIVELY on
# vllm-arc:v0.30.0-b70-cert4 (tag b5df28b5):
#
#   leg a — default k3 boot: first-request survival, run7n t0 battery x2,
#           greedy spec-acceptance gate (cert4-native %), solo prefill
#           TTFT ladder U8k/U16k/U32k/U64k, warm re-hit t1/t2 + warm-hit
#           face WITH cache-tier attribution windows, xhigh temp sweep
#           0.2/0.4/0.6/0.8/1.0 (real-world sampling set), alt face
#           (t0.7/tp0.80/pp1.5)
#   leg b — no-spec boot (ARC_SPEC=0): same three battery faces ->
#           MTP-over-no-spec ratios on cert4
#   leg c — stability soak (default posture, SOAK_HOURS): battery +
#           p1_soak integrity cycles every ~5 min; health, acceptance,
#           call_FAIL, WARN/ERROR audit
#
# Rig laws honored: W14 (radiance-test) stopped for the campaign and
# RESTORED on exit (trap — even on crash; never leave the rig on an arc
# image), VLLM_XPU_ENABLE_XPU_GRAPH=1 (serve script ARC_GRAPH=1), detached
# host-side run (setsid), evidence under /root/build/p70/.
set -u
D=/root/build/p70
B=/root/build                 # harness dir (arc_spec_gate.py, p1_soak.py)
R="$B/r70"                    # campaign kit (this script + probes)
IMG=vllm-arc:v0.30.0-b70-cert4
SERVE="$B/serve_arc_b70_p70.sh"
SOAK_HOURS=${SOAK_HOURS:-4}
mkdir -p "$D"/a "$D"/b "$D"/c
ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $*"; }

w14_restore() {
  docker rm -f arc-r70a arc-r70b arc-r70c >/dev/null 2>&1 || true
  docker start radiance-test >/dev/null 2>&1 || true
  bash "$B/w14_serve_only.sh" > "$D/w14_restore.log" 2>&1 || true
  log "W14_RESTORED (radiance-test up)"
}
trap 'rc=$?; log "campaign exit rc=$rc"; w14_restore' EXIT

wait_health() { # $1=timeout_s
  local deadline=$(( $(date +%s) + ${1:-1200} )) c
  while [ "$(date +%s)" -lt "$deadline" ]; do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 \
        http://127.0.0.1:8000/health || true)
    [ "$c" = "200" ] && return 0
    sleep 10
  done
  return 1
}

# spec-decode + prefix/cache/LMCache-tier counters (tier attribution:
# spec deltas -> acceptance per battery window; cache counters -> which
# prefix tier served the warm faces)
metrics_snap() { # $1=file
  { echo "# $(ts)"
    curl -s -m 10 http://127.0.0.1:8000/metrics | grep -E \
      '^vllm:[a-zA-Z_:]*(spec_decode|prefix|cache|retriev|store|lmcache)[a-zA-Z_:]*[ {]' \
      || true
  } >> "$1"
}

boot_leg() { # $1=container $2=spec $3=legdir
  docker rm -f "$1" >/dev/null 2>&1 || true
  ARC_IMAGE=$IMG ARC_NAME="$1" ARC_SPEC=$2 ARC_GRAPH=1 \
    bash "$SERVE" > "$3/boot.out" 2>&1
  if wait_health 1200; then
    log "$1 health 200 (spec=$2)"
  else
    log "$1 HEALTH TIMEOUT — leg aborts"
    return 1
  fi
}

survival() { # $1=legdir — first request after boot (p12b interlock face)
  local t0 t1 code
  t0=$(date +%s.%N)
  code=$(curl -s -o "$1/first_resp.json" -w "%{http_code}" -m 300 \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Reply with OK."}]}')
  t1=$(date +%s.%N)
  echo "survival http=$code dt=$(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')s" \
    | tee "$1/survival.txt"
}

serve_log_window() { # $1=container $2=outfile — raw tail for offline diff
  docker exec "$1" tail -c 400000 /root/serve_full.log > "$2" 2>/dev/null \
    || true
}

leg_a() {
  log "LEG A: stopping W14 incumbent; boot cert4 default k3"
  docker stop radiance-test >/dev/null 2>&1 || true
  boot_leg arc-r70a 3 "$D/a" || return 1
  survival "$D/a"
  metrics_snap "$D/a/metrics.txt"                 # baseline (post-warmup)
  python3 "$R/bench_run7n_t5.py" > "$D/a/t0_run7n_1.out" 2>&1
  python3 "$R/bench_run7n_t5.py" > "$D/a/t0_run7n_2.out" 2>&1
  metrics_snap "$D/a/metrics.txt"
  python3 "$B/arc_spec_gate.py" --container arc-r70a --strict-target \
    --json-out "$D/a/spec_gate.json" > "$D/a/spec_gate.out" 2>&1
  echo "spec_gate_rc=$?" >> "$D/a/spec_gate.out"
  metrics_snap "$D/a/metrics.txt"
  python3 "$R/p70_solo_ttft.py" "$D/a/solo_ttft.json" > "$D/a/solo_ttft.txt" 2>&1
  metrics_snap "$D/a/metrics.txt"
  serve_log_window arc-r70a "$D/a/serve_pre_warm.log"
  python3 "$R/p70_warm_rehit.py" "$D/a/warm_rehit.json" > "$D/a/warm_rehit.txt" 2>&1
  serve_log_window arc-r70a "$D/a/serve_post_warm.log"
  metrics_snap "$D/a/metrics.txt"
  for T in 0.2 0.4 0.6 0.8 1.0; do
    python3 "$R/bench_run7n_t5.py" --temperature "$T" --thinking xhigh \
      > "$D/a/xhigh_t${T}.out" 2>&1
    metrics_snap "$D/a/metrics.txt"
    sleep 3
  done
  python3 "$R/bench_run7n_t5.py" --temperature 0.7 --sampling alt \
    > "$D/a/alt.out" 2>&1
  metrics_snap "$D/a/metrics.txt"
  docker cp arc-r70a:/root/serve_full.log "$D/a/serve_full.log" \
    >/dev/null 2>&1 || true
  docker rm -f arc-r70a >/dev/null 2>&1 || true
  log "LEG A done"
}

leg_b() {
  log "LEG B: boot cert4 no-spec (ARC_SPEC=0)"
  boot_leg arc-r70b 0 "$D/b" || return 1
  survival "$D/b"
  python3 "$B/arc_spec_gate.py" --container arc-r70b --expect-off \
    > "$D/b/spec_off.out" 2>&1
  echo "rc=$?" >> "$D/b/spec_off.out"
  python3 "$R/bench_run7n_t5.py" > "$D/b/t0_run7n.out" 2>&1
  python3 "$R/bench_run7n_t5.py" --temperature 1.0 --thinking xhigh \
    > "$D/b/xhigh_t1.0.out" 2>&1
  python3 "$R/bench_run7n_t5.py" --temperature 0.7 --sampling alt \
    > "$D/b/alt.out" 2>&1
  docker cp arc-r70b:/root/serve_full.log "$D/b/serve_full.log" \
    >/dev/null 2>&1 || true
  docker rm -f arc-r70b >/dev/null 2>&1 || true
  log "LEG B done"
}

leg_c() {
  log "LEG C: boot cert4 default k3 — soak ${SOAK_HOURS}h"
  boot_leg arc-r70c 3 "$D/c" || return 1
  survival "$D/c"
  metrics_snap "$D/c/metrics.txt"
  local deadline=$(( $(date +%s) + SOAK_HOURS * 3600 )) cycle=0 fails=0 integ_fail=0
  : > "$D/c/soak_log.txt"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    cycle=$((cycle + 1))
    local cf
    cf=$(printf %03d "$cycle")
    python3 "$R/bench_run7n_t5.py" > "$D/c/cycle_${cf}.out" 2>&1
    grep -q call_FAIL "$D/c/cycle_${cf}.out" && fails=$((fails + 1))
    python3 "$B/p1_soak.py" > "$D/c/integ_${cf}.out" 2>&1 \
      || integ_fail=$((integ_fail + 1))
    local h
    h=$(curl -s -o /dev/null -w "%{http_code}" -m 5 \
        http://127.0.0.1:8000/health || true)
    echo "cycle=$cycle health=$h $(ts)" >> "$D/c/soak_log.txt"
    metrics_snap "$D/c/metrics.txt"
    sleep 240
  done
  # close-out: final gate + full engine log + WARN/ERROR audit
  python3 "$B/arc_spec_gate.py" --container arc-r70c --strict-target \
    --json-out "$D/c/spec_gate_final.json" > "$D/c/spec_gate_final.out" 2>&1
  echo "rc=$?" >> "$D/c/spec_gate_final.out"
  docker cp arc-r70c:/root/serve_full.log "$D/c/serve_full.log" \
    >/dev/null 2>&1 || true
  {
    echo "cycles=$cycle batt_fail_cycles=$fails integ_fail_cycles=$integ_fail hours=$SOAK_HOURS"
    echo "engine_errors=$(grep -acE ' ERROR ' "$D/c/serve_full.log" || true)"
    echo "engine_warns=$(grep -acE ' WARN ' "$D/c/serve_full.log" || true)"
  } > "$D/c/soak_summary.txt"
  docker rm -f arc-r70c >/dev/null 2>&1 || true
  log "LEG C done: $(cat "$D/c/soak_summary.txt" | tr '\n' ' ')"
}

log "r70 campaign start (img=$IMG soak=${SOAK_HOURS}h)"
leg_a
leg_b
leg_c
log "r70 CAMPAIGN COMPLETE"
