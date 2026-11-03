#!/bin/bash
# p617_campaign.sh — P6.17: prob-lane certification campaign.
#
# Owner directives (2026-11-03): "new valid certificated line has to be
# ARC_SPEC_METHOD=probabilistic and all performance values has to come from
# ARC_SPEC_METHOD=probabilistic lane, if some missing run certificate
# performance testings" + "clean old not needed informations without
# ARC_SPEC_METHOD=probabilistic lanes" (prob lane is the superior lane).
#
# Every current-record face re-measured NATIVELY on
# vllm-arc:v0.30.0-b70-cert6 under ARC_SPEC_METHOD=probabilistic via the
# CANONICAL deployed serve script (/root/build/serve_arc_b70.sh, the p616c
# hook — no seds, env-var form, the exact recommended-line bytes):
#
#   leg a — fresh boot, spec k3 prob: first-request survival, run7n t0 x2,
#           strict spec gate, integrity det 3/3 + warm 5/5, golden 5/5,
#           solo prefill TTFT ladder U8k/U16k/U32k/U64k + solo pair, warm
#           re-hit t1/t2 with cache-tier attribution windows, xhigh temp
#           sweep 0.2/0.4/0.6/0.8/1.0, alt face (t0.7/tp0.80/pp1.5),
#           garbage suite v2, hook audit (ps cmdline + serve log prob key)
#   leg b — second fresh boot (family): survival, run7n t0 x2, xhigh t1.0
#           x2, strict spec gate, integrity det 3/3
#   leg c — no-spec boot (ARC_SPEC=0): survival, spec-off gate, run7n t0,
#           xhigh t1.0, alt face -> MTP-over-no-spec ratios prob-lane
#
# Rig laws honored: W14 (radiance-test) stopped for the campaign and
# RESTORED on exit (trap — even on crash; never leave the rig on an arc
# image), VLLM_XPU_ENABLE_XPU_GRAPH=1 (ARC_GRAPH=1), detached host-side
# run (setsid), evidence under /root/build/p617/.
set -u
D=/root/build/p617
B=/root/build                 # harness dir (arc_spec_gate.py, serve script)
R="$B/r70"                    # battery kit (bench/solo/warm probes)
IMG=vllm-arc:v0.30.0-b70-cert6
SERVE="$B/serve_arc_b70.sh"   # canonical deployed script (p616c-validated hook)
mkdir -p "$D"/a "$D"/b "$D"/c
ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $*"; }

w14_restore() {
  docker rm -f arc-p617a arc-p617b arc-p617c >/dev/null 2>&1 || true
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

metrics_snap() { # $1=file
  { echo "# $(ts)"
    curl -s -m 10 http://127.0.0.1:8000/metrics | grep -E \
      '^vllm:[a-zA-Z_:]*(spec_decode|prefix|cache|retriev|store|lmcache)[a-zA-Z_:]*[ {]' \
      || true
  } >> "$1"
}

boot_leg() { # $1=container $2=spec $3=legdir $4=method(""=omit env)
  docker rm -f "$1" >/dev/null 2>&1 || true
  local t0=$(date +%s)
  if [ -n "$4" ]; then
    ARC_IMAGE=$IMG ARC_NAME="$1" ARC_SPEC=$2 ARC_GRAPH=1 \
      ARC_SPEC_METHOD="$4" bash "$SERVE" > "$3/boot.out" 2>&1
  else
    ARC_IMAGE=$IMG ARC_NAME="$1" ARC_SPEC=$2 ARC_GRAPH=1 \
      bash "$SERVE" > "$3/boot.out" 2>&1
  fi
  if wait_health 1200; then
    log "$1 health 200 (spec=$2 method=${4:-unset}) boot=$(( $(date +%s) - t0 ))s"
  else
    log "$1 HEALTH TIMEOUT — leg aborts"
    return 1
  fi
}

survival() { # $1=legdir — first request after boot
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

hook_audit() { # $1=container $2=legdir — engine cmdline + log prob-key proof
  docker exec "$1" ps aux | grep -F "vllm serve" | grep -vF grep \
    > "$2/ps_engine.txt" 2>/dev/null || true
  docker exec "$1" ps aux | grep -oF -- "--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}" \
    | sort -u > "$2/ps_spec_bytes.txt" 2>/dev/null || true
  docker exec "$1" grep -c "draft_sample_method" /root/serve_full.log \
    > "$2/log_prob_key_count.txt" 2>/dev/null || true
  log "hook_audit $1: ps_spec_bytes=$(wc -l < "$2/ps_spec_bytes.txt" 2>/dev/null || echo 0) log_hits=$(cat "$2/log_prob_key_count.txt" 2>/dev/null || echo NA)"
}

leg_a() {
  log "LEG A: stopping W14 incumbent; boot cert6 k3 PROBABILISTIC"
  docker stop radiance-test >/dev/null 2>&1 || true
  boot_leg arc-p617a 3 "$D/a" probabilistic || return 1
  survival "$D/a"
  hook_audit arc-p617a "$D/a"
  metrics_snap "$D/a/metrics.txt"
  python3 "$R/bench_run7n_t5.py" > "$D/a/t0_run7n_1.out" 2>&1
  python3 "$R/bench_run7n_t5.py" > "$D/a/t0_run7n_2.out" 2>&1
  metrics_snap "$D/a/metrics.txt"
  python3 "$B/arc_spec_gate.py" --container arc-p617a --strict-target \
    --json-out "$D/a/spec_gate.json" > "$D/a/spec_gate.out" 2>&1
  echo "spec_gate_rc=$?" >> "$D/a/spec_gate.out"
  metrics_snap "$D/a/metrics.txt"
  python3 "$B/r42/r42_integrity.py" > "$D/a/integrity.out" 2>&1
  echo "integrity_rc=$?" >> "$D/a/integrity.out"
  python3 "$B/p25/p32_golden.py" --run "$D/a/golden_run.json" \
    --ref "$B/p25/golden_ref.json" > "$D/a/golden.out" 2>&1
  echo "golden_rc=$?" >> "$D/a/golden.out"
  metrics_snap "$D/a/metrics.txt"
  python3 "$R/p70_solo_ttft.py" "$D/a/solo_ttft.json" > "$D/a/solo_ttft.txt" 2>&1
  metrics_snap "$D/a/metrics.txt"
  serve_log_window arc-p617a "$D/a/serve_pre_warm.log"
  python3 "$R/p70_warm_rehit.py" "$D/a/warm_rehit.json" > "$D/a/warm_rehit.txt" 2>&1
  serve_log_window arc-p617a "$D/a/serve_post_warm.log"
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
  python3 "$B/r46t/arc_garbage_probe2.py" > "$D/a/garbage.out" 2>&1
  echo "garbage_rc=$?" >> "$D/a/garbage.out"
  docker cp arc-p617a:/root/serve_full.log "$D/a/serve_full.log" \
    >/dev/null 2>&1 || true
  docker rm -f arc-p617a >/dev/null 2>&1 || true
  log "LEG A done"
}

leg_b() {
  log "LEG B: second fresh boot cert6 k3 PROBABILISTIC (family)"
  boot_leg arc-p617b 3 "$D/b" probabilistic || return 1
  survival "$D/b"
  hook_audit arc-p617b "$D/b"
  python3 "$R/bench_run7n_t5.py" > "$D/b/t0_run7n_1.out" 2>&1
  python3 "$R/bench_run7n_t5.py" > "$D/b/t0_run7n_2.out" 2>&1
  python3 "$R/bench_run7n_t5.py" --temperature 1.0 --thinking xhigh \
    > "$D/b/xhigh_t1.0_1.out" 2>&1
  python3 "$R/bench_run7n_t5.py" --temperature 1.0 --thinking xhigh \
    > "$D/b/xhigh_t1.0_2.out" 2>&1
  python3 "$B/arc_spec_gate.py" --container arc-p617b --strict-target \
    --json-out "$D/b/spec_gate.json" > "$D/b/spec_gate.out" 2>&1
  echo "spec_gate_rc=$?" >> "$D/b/spec_gate.out"
  python3 "$B/r42/r42_integrity.py" > "$D/b/integrity.out" 2>&1
  echo "integrity_rc=$?" >> "$D/b/integrity.out"
  docker cp arc-p617b:/root/serve_full.log "$D/b/serve_full.log" \
    >/dev/null 2>&1 || true
  docker rm -f arc-p617b >/dev/null 2>&1 || true
  log "LEG B done"
}

leg_c() {
  log "LEG C: boot cert6 no-spec (ARC_SPEC=0) — MTP-over-no-spec denominators"
  boot_leg arc-p617c 0 "$D/c" "" || return 1
  survival "$D/c"
  python3 "$B/arc_spec_gate.py" --container arc-p617c --expect-off \
    > "$D/c/spec_off.out" 2>&1
  echo "rc=$?" >> "$D/c/spec_off.out"
  python3 "$R/bench_run7n_t5.py" > "$D/c/t0_run7n.out" 2>&1
  python3 "$R/bench_run7n_t5.py" --temperature 1.0 --thinking xhigh \
    > "$D/c/xhigh_t1.0.out" 2>&1
  python3 "$R/bench_run7n_t5.py" --temperature 0.7 --sampling alt \
    > "$D/c/alt.out" 2>&1
  docker cp arc-p617c:/root/serve_full.log "$D/c/serve_full.log" \
    >/dev/null 2>&1 || true
  docker rm -f arc-p617c >/dev/null 2>&1 || true
  log "LEG C done"
}

serve_log_window() { # $1=container $2=outfile — raw tail for offline diff
  docker exec "$1" tail -c 400000 /root/serve_full.log > "$2" 2>/dev/null \
    || true
}

log "p617 campaign start (img=$IMG method=probabilistic script=$SERVE)"
leg_a
leg_b
leg_c
log "p617 CAMPAIGN COMPLETE"
