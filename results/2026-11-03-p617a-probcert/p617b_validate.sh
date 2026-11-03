#!/bin/bash
# p617b_validate.sh — P6.17 phase 2: flipped-default validation.
#
# After the p617 legs (a/b/c) certified the probabilistic lane on the
# p616-hook script (env-var form), this validates the FLIPPED serve
# script whose DEFAULT (unset) posture is probabilistic:
#   phase 1 — deploy flip: canonical -> serve_arc_b70_p616.sh (provenance
#             backup), flipped bytes -> serve_arc_b70.sh (sha-gated)
#   phase 2 — flag probe 5/5 via PATH-stubbed docker (construction test on
#             the REAL script): unset=prob bytes, probabilistic=prob bytes,
#             junk=prob bytes, greedy=old certified bytes (no draft key),
#             ARC_SPEC=0=no spec flag at all
#   phase 3 — Boot 4 arc-p617d DEFAULT PATH (no env): survival, hook audit
#             (ps bytes must EQUAL leg a's certified prob bytes), run7n t0,
#             xhigh t1.0 x2 (reproduction attempt for the leg-B degraded
#             window), strict spec gate, integrity, golden, log hygiene
#   phase 4 — Boot 5 arc-p617e ARC_SPEC_METHOD=greedy OPT-OUT: hook audit
#             (ps bytes = the pre-P6.16 certified greedy bytes, no draft
#             key), run7n t0 (in-family vs greedy record), strict spec gate
#   phase 5 — W14 restore + health-200 verify (rig law)
set -u
D=/root/build/p617
B=/root/build
R="$B/r70"
IMG=vllm-arc:v0.30.0-b70-cert6
SERVE="$B/serve_arc_b70.sh"
FLIP_SHA_FILE="$D/flip.sha256"
mkdir -p "$D"/d "$D"/e "$D"/probe
ts() { date +%H:%M:%S; }
log() { echo "[$(ts)] $*"; }

w14_restore() {
  docker rm -f arc-p617d arc-p617e >/dev/null 2>&1 || true
  docker start radiance-test >/dev/null 2>&1 || true
  bash "$B/w14_serve_only.sh" > "$D/w14_restore2.log" 2>&1 || true
  local deadline=$(( $(date +%s) + 1200 )) c
  while [ "$(date +%s)" -lt "$deadline" ]; do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health || true)
    [ "$c" = 200 ] && { log "W14_RESTORED health=200"; return 0; }
    sleep 10
  done
  log "W14 RESTORE HEALTH FAIL (post-restore probe != 200)"
  return 1
}
trap 'rc=$?; log "validate exit rc=$rc"; w14_restore' EXIT

wait_health() {
  local deadline=$(( $(date +%s) + ${1:-1200} )) c
  while [ "$(date +%s)" -lt "$deadline" ]; do
    c=$(curl -s -o /dev/null/ -w "%{http_code}" -m 5 \
        http://127.0.0.1:8000/health || true)
    [ "$c" = "200" ] && return 0
    sleep 10
  done
  return 1
}

PROB_BYTES="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}'"
GREEDY_BYTES="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'"

# ---------- phase 1: deploy flip ----------
if [ ! -f "$FLIP_SHA_FILE" ]; then
  log "PHASE1 FAIL: $FLIP_SHA_FILE missing (deploy step incomplete)"; exit 1
fi
FLIP_WANT=$(cut -d" " -f1 "$FLIP_SHA_FILE")
FLIP_GOT=$(sha256sum "$B/serve_arc_b70.sh" | cut -d" " -f1)
if [ "$FLIP_GOT" != "$FLIP_WANT" ]; then
  log "PHASE1 FAIL: deployed sha $FLIP_GOT != expected $FLIP_WANT"; exit 1
fi
log "PHASE1 OK: flipped script in place (sha ${FLIP_GOT:0:12}); p616 original backed up"

# ---------- phase 2: flag probe (stub docker, real script) ----------
mkdir -p /tmp/stub
cat > /tmp/stub/docker <<'STUB'
#!/bin/bash
echo "DOCKER-STUB: $*" >> /tmp/stub/docker.log
case "$1" in
  image) echo "sha256:1111111111111111111111111111111111111111111111111111111111111111";;
  *) exit 0;;
esac
exit 0
STUB
chmod +x /tmp/stub/docker
probe_one() { # $1=label $2=env-value ("" = unset)
  rm -f /tmp/stub/docker.log
  if [ -n "$2" ]; then
    ARC_SPEC=3 ARC_GRAPH=1 ARC_PORT=8123 ARC_SPEC_METHOD="$2" \
      PATH=/tmp/stub:$PATH bash "$SERVE" >/dev/null 2>&1
  else
    ARC_SPEC=3 ARC_GRAPH=1 ARC_PORT=8123 \
      env -u ARC_SPEC_METHOD PATH=/tmp/stub:$PATH bash "$SERVE" >/dev/null 2>&1
  fi
  grep -ao -- "--speculative-config '[^']*'" /tmp/stub/docker.log | head -1 || true
}
P=0; F=0
chk() { # $1=label $2=got $3=want
  if [ "$2" = "$3" ]; then P=$((P+1)); log "probe $1: PASS"; else F=$((F+1)); log "probe $1: FAIL got='$2' want='$3'"; fi
}
V=$(probe_one unset "");        chk "unset->prob"        "$V" "$PROB_BYTES"
V=$(probe_one probabilistic "probabilistic"); chk "probabilistic->prob" "$V" "$PROB_BYTES"
V=$(probe_one junk "junk");     chk "junk->prob(default)" "$V" "$PROB_BYTES"
V=$(probe_one greedy "greedy"); chk "greedy->old-bytes"  "$V" "$GREEDY_BYTES"
rm -f /tmp/stub/docker.log
env -u ARC_SPEC_METHOD ARC_SPEC=0 ARC_GRAPH=1 ARC_PORT=8123 \
  PATH=/tmp/stub:$PATH bash "$SERVE" >/dev/null 2>&1
if grep -aq "speculative-config" /tmp/stub/docker.log; then
  F=$((F+1)); log "probe spec0->noflag: FAIL (flag present)"
else P=$((P+1)); log "probe spec0->noflag: PASS"; fi
{ echo "flag_probe pass=$P fail=$F"; } > "$D/probe/flag_probe.txt"
log "PHASE2 flag probe: $P pass / $F fail"
[ "$F" = 0 ] || { log "PHASE2 FAIL — aborting before boots"; exit 2; }

# ---------- phase 3: Boot 4 default path ----------
hook_audit() { # $1=container $2=legdir
  docker exec "$1" ps aux | grep -oF -- "--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}" \
    | sort -u > "$2/ps_spec_bytes.txt" 2>/dev/null || true
  docker exec "$1" grep -c "draft_sample_method" /root/serve_full.log \
    > "$2/log_prob_key_count.txt" 2>/dev/null || true
}
survival() { # $1=legdir
  local t0 t1 code
  t0=$(date +%s.%N)
  code=$(curl -s -o "$1/first_resp.json" -w "%{http_code}" -m 300 \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","max_tokens":16,"temperature":0,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Reply with OK."}]}')
  t1=$(date +%s.%N)
  echo "survival http=$code dt=$(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')s" | tee "$1/survival.txt"
}
log "PHASE3: stop W14; boot cert6 DEFAULT PATH (no ARC_SPEC_METHOD)"
docker stop radiance-test >/dev/null 2>&1 || true
docker rm -f arc-p617d >/dev/null 2>&1 || true
T0=$(date +%s)
ARC_IMAGE=$IMG ARC_NAME=arc-p617d ARC_SPEC=3 ARC_GRAPH=1 \
  bash "$SERVE" > "$D/d/boot.out" 2>&1
if wait_health 1200; then log "arc-p617d health 200 default-path boot=$(( $(date +%s) - T0 ))s"; else log "arc-p617d HEALTH TIMEOUT"; exit 3; fi
survival "$D/d"
hook_audit arc-p617d "$D/d"
if cmp -s "$D/d/ps_spec_bytes.txt" "$D/a/ps_spec_bytes.txt"; then
  log "hook_audit d==a: DEFAULT PATH bytes IDENTICAL to leg-a env-form bytes"
  echo "PS_BYTES_EQUAL_D_A=yes" >> "$D/d/ps_spec_bytes.txt"
else
  log "hook_audit MISMATCH d vs a"; echo "PS_BYTES_EQUAL_D_A=NO" >> "$D/d/ps_spec_bytes.txt"
fi
python3 "$R/bench_run7n_t5.py" > "$D/d/t0_run7n.out" 2>&1
python3 "$R/bench_run7n_t5.py" --temperature 1.0 --thinking xhigh \
  > "$D/d/xhigh_t1.0_1.out" 2>&1
python3 "$R/bench_run7n_t5.py" --temperature 1.0 --thinking xhigh \
  > "$D/d/xhigh_t1.0_2.out" 2>&1
python3 "$B/arc_spec_gate.py" --container arc-p617d --strict-target \
  --json-out "$D/d/spec_gate.json" > "$D/d/spec_gate.out" 2>&1
echo "spec_gate_rc=$?" >> "$D/d/spec_gate.out"
python3 "$B/r42/r42_integrity.py" > "$D/d/integrity.out" 2>&1
echo "integrity_rc=$?" >> "$D/d/integrity.out"
python3 "$B/p25/p32_golden.py" --out "$D/d/golden_run.json" > "$D/d/golden_cap.out" 2>&1
python3 "$B/p25/p32_golden.py" --run "$D/d/golden_run.json" --ref "$B/p25/golden_ref.json" > "$D/d/golden.out" 2>&1
echo "golden_rc=$?" >> "$D/d/golden.out"
docker cp arc-p617d:/root/serve_full.log "$D/d/serve_full.log" >/dev/null 2>&1 || true
echo "errors=$(grep -acE ' ERROR |Traceback' "$D/d/serve_full.log" || true)" > "$D/d/log_hygiene.txt"
docker rm -f arc-p617d >/dev/null 2>&1 || true
log "PHASE3 done: $(cat "$D/d/log_hygiene.txt")"

# ---------- phase 4: Boot 5 greedy opt-out ----------
log "PHASE4: boot cert6 ARC_SPEC_METHOD=greedy (opt-out historical lane)"
docker rm -f arc-p617e >/dev/null 2>&1 || true
T0=$(date +%s)
ARC_IMAGE=$IMG ARC_NAME=arc-p617e ARC_SPEC=3 ARC_GRAPH=1 ARC_SPEC_METHOD=greedy \
  bash "$SERVE" > "$D/e/boot.out" 2>&1
if wait_health 1200; then log "arc-p617e health 200 greedy boot=$(( $(date +%s) - T0 ))s"; else log "arc-p617e HEALTH TIMEOUT"; exit 4; fi
survival "$D/e"
docker exec arc-p617e ps aux | grep -oF -- "--speculative-config {\"method\":\"mtp\",\"num_speculative_tokens\":3}" \
  | sort -u > "$D/e/ps_spec_bytes.txt" 2>/dev/null || true
docker exec arc-p617e ps aux | grep -c "draft_sample_method" \
  > "$D/e/ps_draft_key_count.txt" 2>/dev/null || echo 0 > "$D/e/ps_draft_key_count.txt"
log "greedy opt-out audit: bytes=$(cat "$D/e/ps_spec_bytes.txt" 2>/dev/null | head -1) draft_key_count=$(cat "$D/e/ps_draft_key_count.txt")"
python3 "$R/bench_run7n_t5.py" > "$D/e/t0_run7n.out" 2>&1
python3 "$B/arc_spec_gate.py" --container arc-p617e --strict-target \
  --json-out "$D/e/spec_gate.json" > "$D/e/spec_gate.out" 2>&1
echo "spec_gate_rc=$?" >> "$D/e/spec_gate.out"
docker cp arc-p617e:/root/serve_full.log "$D/e/serve_full.log" >/dev/null 2>&1 || true
echo "errors=$(grep -acE ' ERROR |Traceback' "$D/e/serve_full.log" || true)" > "$D/e/log_hygiene.txt"
docker rm -f arc-p617e >/dev/null 2>&1 || true
log "PHASE4 done: $(cat "$D/e/log_hygiene.txt")"

log "p617b VALIDATE COMPLETE"
