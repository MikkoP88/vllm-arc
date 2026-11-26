#!/bin/bash
# r246b_lmc_gate_e2e.sh -- P6.47 P-D validation leg (e2e half): the
# LMC x MX boot gate proven on the REAL serve path (branch
# feat/arc-pd-lmc-mx-boot-gate; the offline probe half is r246_lmc_
# gate_offline.sh). Two boots on the qwen MX lane (stock cert28 MX
# engine -- no mx_kv overlay needed; the P-D files mount over
# site-packages via the serve script's env-optional ARC_PD_* mounts):
#
#   pd_refuse  ARC_LMC=1 forced over mxfp4 (the EXACT r240b confound
#              posture) -> the boot MUST die at connector construction
#              with ARC_LMC_MX_REFUSED in the serve log (the r241
#              posture law made structural; the serve script's WARN
#              still radiates, the engine now refuses). Outcome
#              GATE_FIRED = pass; NO_REFUSAL = fail (the gate leaked).
#   pd_clean   ARC_LMC=0 + mxfp4, P-D files mounted -> HEALTH_OK,
#              the wheel-hook install line (factory <-
#              patches.lmc_mx_boot_gate OK), a real face (p139_conc
#              ctx 8192 n1 mt 64, n_ok>0 semantic gate), zero LMCache
#              lines (posture law -- aborts on any).
#
# Faces are validation-class, not perf: the r241 bank owns the MX
# grids; this leg proves the GATE, not the lane. Refuses: cert28
# image, staged md5s (same table as r246), the ARC_PD_* mount lines
# present in serve_arc_b70.sh (the e2e half depends on them), no arc
# leg in flight, port check. W14 quiesce before, EXIT-trap fullres
# restore after (r24x law). Artifacts -> /root/build/p246b/{cells}.
# Exit 0 = both cells meet their expectations.
#
# RUN HISTORY: run 1 (rev 1, 13:19:34Z) pd_refuse NO_REFUSAL was a
# RUNNER defect, never a gate verdict: the marker poll broke at
# iteration 1 on the not-yet-created container (the serve script
# reaches docker run seconds after nohup), banked a 3-second serve
# log, declared NO_REFUSAL, and pkilled a healthy booting engine --
# the connector constructs minutes into init, after model load, so
# the refusal never had its window. Rev 2 waits for the container,
# then polls marker/health/EngineCore liveness (seen-once-then-gone
# = engine death -> flush -> final grep). pd_clean (rev 1) ran the
# old code but is cell-independent of the defect (boot_wait always
# waited for health); rev 2 reruns both cells for one coherent bank.
# Run 2 (rev 2, 13:30:43Z): pd_clean PASS (HEALTH_OK 13:41:11,
# gate_hook_installed=4, lmc_lines=0, CONC_OK n_ok=1) -- pd_refuse
# GENUINELY LEAKED: the gate fired on the real serve path (posture
# lines at worker AND scheduler sites) but read kv_dtype=None --
# vllm 0.30 renamed CacheConfig.kv_cache_dtype -> cache_dtype
# (config/cache.py:111, no legacy alias; the CLI kwarg and the
# AttentionSelectorConfig tuple kept the old name) and the gate rev
# 1 read the pre-rename attr. Gate rev 2 reads both surfaces
# (legacy first, then the 0.30 field). This rev 3 reruns both
# cells over gate rev 2.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
P=$R/p246b
SP=$R/p246/stage
CNAME=arc-r246b
QM="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_FP8=1 ARC_SPEC=3"
MX="ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1"
PDM="ARC_PD_PLATFORM=$SP/platform.py ARC_PD_ENVS=$SP/envs.py ARC_PD_GATE=$SP/lmc_mx_boot_gate.py"
declare -A MD5=( \
  [platform.py]=0733f0b8c7dd2fb89422bafa56ec35f5 \
  [envs.py]=31ca4ac7cfcd996aed6446d4b24c2bcd \
  [lmc_mx_boot_gate.py]=ffb1ef11b17bf7b99987c84940c16360 )
FAILED=""

fail() { echo "R246B_ABORT: $*"; }

for f in platform.py envs.py lmc_mx_boot_gate.py; do
  [ -f "$SP/$f" ] || { fail "staged $f missing (run the r246 ship step)"; exit 1; }
  got=$(md5sum "$SP/$f" | cut -d" " -f1)
  [ "$got" = "${MD5[$f]}" ] || { fail "staged $f md5 $got != ${MD5[$f]}"; exit 1; }
done
echo "R246B_STAGE_OK"
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
grep -q 'ARC_PD_PLATFORM' $R/serve_arc_b70.sh || { fail "serve_arc_b70.sh lacks the ARC_PD_* mounts (apply the P-D extension first)"; exit 1; }
[ -f "$R/p139_conc.py" ] || { fail "p139_conc.py missing"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    fail "port 8000 held by an unknown holder"; exit 3
  fi
fi
echo "R246B refuses passed"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R246B_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p246b.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    docker rm -f "$1" >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "R246B_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec $CNAME grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r246b:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r246b:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r246b:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R246B_WARN_PORT8000_STILL_HELD"
sleep 10

# ============ cell 1: pd_refuse -- the r240b confound path ============
CD=$P/pd_refuse
mkdir -p "$CD"
docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
echo "=== [r246b:pd_refuse] boot $(date +%H:%M:%S) (ARC_LMC=1 over mxfp4 -- refusal REQUIRED) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $QM $MX ARC_LMC=1 $PDM \
  bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
# poll the serve log for the refusal. The connector constructs minutes
# into engine init (after config/device/model load), so this is a slow
# poll, not a fast one -- and the container must EXIST before absence
# can mean anything: the serve script takes seconds to reach docker
# run, and run 1 broke at iteration 1 on the not-yet-created
# container, banked a 3-second log, and pkilled a healthy booting
# engine (NO_REFUSAL was a runner defect, never a gate verdict).
# Order per poll: marker -> health (200 over the confound posture =
# the gate truly leaked) -> EngineCore liveness (seen-once-then-gone =
# engine died; flush window, final marker grep). Cap 150 x 3s + slack
# ~= 9 min > the ~6 min LMC=1 MX boot window. The refusal ValueError
# kills the CLI; its traceback lands in serve_full.log.
i=0; SEEN=0
while [ $i -lt 30 ]; do
  docker inspect $CNAME >/dev/null 2>&1 && { SEEN=1; break; }
  sleep 3; i=$((i+1))
done
if [ "$SEEN" != "1" ]; then
  echo "R246B_ABORT_pd_refuse: container never appeared (90 s)"
  echo "BOOT_FAIL" > "$CD/outcome.txt"
  FAILED="$FAILED pd_refuse"
  echo "R246B_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi
i=0; REF=0; HEALTH=0; CORE_SEEN=0; ENG_DEAD=0
while [ $i -lt 150 ]; do
  n=$(gcount "ARC_LMC_MX_REFUSED")
  [ "${n:-0}" -gt 0 ] && { REF=1; break; }
  if curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    HEALTH=1; break
  fi
  if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then CORE_SEEN=1
  elif [ "$CORE_SEEN" = "1" ]; then
    ENG_DEAD=1; sleep 8
    n=$(gcount "ARC_LMC_MX_REFUSED"); [ "${n:-0}" -gt 0 ] && REF=1
    break
  fi
  sleep 3; i=$((i+1))
done
sleep 2
n=$(gcount "ARC_LMC_MX_REFUSED")
[ "${n:-0}" -gt 0 ] && REF=1
docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
echo "tracebacks=$(gcount 'Traceback (most recent call last)')" > "$CD/posture.txt"
echo "lmc_mx_refused=$n" >> "$CD/posture.txt"
echo "engine_dead=$ENG_DEAD health_up=$HEALTH" >> "$CD/posture.txt"
# the serve script's own WARN line (boot.txt -- script stdout, not the
# engine log) proves the confound posture was armed before the refusal
echo "warn_line=$(grep -c 'forced on ARC_KVDTYPE' "$CD/boot.txt" 2>/dev/null || echo 0)" >> "$CD/posture.txt"
cat "$CD/posture.txt"
if [ "$REF" = "1" ]; then
  echo "GATE_FIRED_pd_refuse (refusal count=$n)"
  echo "GATE_FIRED" > "$CD/outcome.txt"
else
  echo "NO_REFUSAL_pd_refuse -- the gate leaked (confound posture reached the engine; health=$HEALTH engine_dead=$ENG_DEAD)"
  echo "NO_REFUSAL" > "$CD/outcome.txt"
  FAILED="$FAILED pd_refuse"
fi
docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 5
docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
sleep 3

# ============ cell 2: pd_clean -- the lawful posture ============
CD=$P/pd_clean
mkdir -p "$CD"
echo "=== [r246b:pd_clean] boot $(date +%H:%M:%S) (ARC_LMC=0 mxfp4) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $QM $MX ARC_LMC=0 $PDM \
  bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
if ! boot_wait pd_clean; then
  echo "R246B_BOOT_FAIL_pd_clean -- banking log"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
  echo "BOOT_FAIL" > "$CD/outcome.txt"
  FAILED="$FAILED pd_clean"
  echo "R246B_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi
echo "HEALTH_OK_r246b_pd_clean $(date +%H:%M:%S)"
grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
HOOKN=$(gcount "patches.lmc_mx_boot_gate")
echo "gate_hook_installed=$HOOKN" >> "$CD/posture.txt"
echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
LMCN=$(gcount 'LMCache')
echo "lmc_lines=$LMCN" >> "$CD/posture.txt"
cat "$CD/posture.txt"
if [ "${LMCN:-1}" -gt 0 ]; then
  echo "R246B_POSTURE_DEFECT_pd_clean lmc_lines=$LMCN -- ABORTING (r241 posture law)"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 5
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
  echo "POSTURE_DEFECT" > "$CD/outcome.txt"
  exit 3
fi
if [ "${HOOKN:-0}" -lt 1 ]; then
  echo "GATE_NOT_INSTALLED_pd_clean -- the wheel-hook line is missing (overlay leak)"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 5
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
  echo "GATE_NOT_INSTALLED" > "$CD/outcome.txt"
  FAILED="$FAILED pd_clean"
  echo "R246B_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi
# warmup + one real face (n_ok>0 semantic gate, r245 lesson)
curl -s -m 120 -o "$CD/warmup.json" -w '%{http_code}' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > "$CD/warmup.code"
WRC=$(cat "$CD/warmup.code")
if [ "$WRC" != "200" ]; then
  echo "WARMUP_DEAD_pd_clean (code=$WRC)"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 5
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
  echo "WARMUP_DEAD" > "$CD/outcome.txt"
  FAILED="$FAILED pd_clean"
  echo "R246B_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi
docker cp $R/p139_conc.py $CNAME:/root/p139_conc.py >/dev/null 2>&1
docker exec $CNAME rm -f /root/p139_conc_pd_clean.json
docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py 8192 64 1 pd_clean qwen3.8-27b-fp8 0.0
i=0
while [ $i -lt 60 ]; do
  docker exec $CNAME test -f /root/p139_conc_pd_clean.json && break
  sleep 5; i=$((i+1))
done
docker cp $CNAME:/root/p139_conc_pd_clean.json "$CD/" >/dev/null 2>&1 || true
NOK=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$CD/p139_conc_pd_clean.json" 2>/dev/null || echo 0)
docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 8
docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
if [ "${NOK:-0}" -le 0 ]; then
  echo "CONC_DEAD_pd_clean (n_ok=${NOK:-?})"
  echo "CONC_DEAD" > "$CD/outcome.txt"
  FAILED="$FAILED pd_clean"
else
  echo "CONC_OK_pd_clean n_ok=$NOK"
  echo "CONC_OK" > "$CD/outcome.txt"
fi

cd $P
sha256sum pd_refuse/* pd_clean/* 2>/dev/null > SHA256SUMS || true
echo "R246B_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -z "$FAILED" ] || exit 1
exit 0
