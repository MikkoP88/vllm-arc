#!/bin/bash
# r268b_bisect.sh -- P6.47 P-D (r268b): the request-bisection leg.
#
# THE r268 ANOMALY: the trace boot (e2 posture byte-identical: cert29,
# gemma g3 skin, kv mxfp8 + native, ARC_GRAPH=0, serve args diff = none
# of 18 keys) was CLEAN on 3 small sequential raw requests while e2 was
# RED from its FIRST request (324x 400-nan; gate battery 'Paris our
# our' at ptok=40). The r267 canary proved the kernel/store math exact
# offline at the engine's true lanes, so the corruption is request-
# CONDITIONAL wiring. This leg runs ONE plain-image c3-posture boot
# (NO overlay -- the r268 overlay is proven behaviorally inert by
# diff, removing the variable entirely) and bisects the request axes:
#   A  prompt length (raw ladder ptok ~6..~120, logprobs=1 as the nan
#      detector)
#   B  the exact e2 request shapes sequential (gate-battery chat at
#      ptok=40; the card chat with the full param set -- the e2 first
#      400)
#   C  concurrency (the probe fires --workers 8; the clean r268
#      workload was strictly sequential) -- 2-way and 8-way bursts,
#      raw and chat
#   D  poisoning recheck (does the clean control STAY clean?)
#   E  determinism (re-run the first RED twice)
# then, on the SAME boot, the FULL p266 matrix probe with the exact
# e2 PARGS (--workers 8 --cells chat,raw,nll,gate --chat-cells t0,card)
# -- the same-boot arbiter: ladder-clean + probe-RED = the trigger
# lives inside the probe's request set; both clean = boot-
# nondeterminism (re-run until RED).
# Ladder driver: p268b_ladder.py (host python3 stdlib, threads for the
# bursts) -> out/ladder.json.  Artifacts ->
# /root/build/p647/r268b_bisect/{out,logs}.  Exit 0 = leg banked.
# W14 quiesce before, clean-recipe trap restore after (r262 law).
# Launch (ship + launch in SEPARATE ssh calls, md5-verified ship):
#   mkdir -p /root/build/p647/r268b_bisect/logs
#   nohup setsid bash /root/build/r268b/r268b_bisect.sh \
#     > /root/build/p647/r268b_bisect/logs/launch.log 2>&1 &
set -uo pipefail
R=/root/build
P=$R/p647/r268b_bisect
S=$R/r268b
R266=$R/r266
IMG=vllm-arc:v0.30.0-b70-cert29
CNAME=arc-r268b

[ -f "$S/p268b_ladder.py" ] || { echo "R268B_NO_LADDER (ship first)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R268B_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R268B_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R268B_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R268B_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  bash $R/r262_w14_restore.sh > $P/logs/w14_restore_r268b.log 2>&1
  rc=$?
  grep -E "^W14_" $P/logs/w14_restore_r268b.log | tail -3 || true
  echo "W14_RESTORE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$P/logs/w14_restore_r268b.log"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out
echo "R268B_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) image=$IMG ladder_md5=$(md5sum $S/p268b_ladder.py | cut -d' ' -f1)"

echo "=== [r268b] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R268B_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r268b] boot c3 posture PLAIN IMAGE (no overlay) $(date +%H:%M:%S) ==="
# exact e2 env (the r266 runner's e2 leg): no ARC_MXKV_OVERLAY, no
# ARC_MXKV_TRACE -- the plain image posture that was RED 4/4 boots.
nohup env \
  ARC_IMAGE=$IMG ARC_NAME=$CNAME \
  ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
  ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 \
  ARC_RPC_TIMEOUT=1200 ARC_MAXLEN=40960 ARC_FP8=0 \
  ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja \
  ARC_RPARSER= ARC_TPARSER= \
  ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 \
  bash $R266/serve_arc_b70.sh > $P/logs/boot.out 2>&1 &
BOOTPID=$!
B=0
while [ $B -lt 60 ]; do
  grep -q "ARC_SERVE_LAUNCHED" $P/logs/boot.out 2>/dev/null && break
  kill -0 $BOOTPID 2>/dev/null || break
  sleep 2; B=$((B+1))
done
if ! grep -q "ARC_SERVE_LAUNCHED" $P/logs/boot.out 2>/dev/null; then
  echo "R268B_BOOT_FAIL (no serve line after $((B*2))s)"; tail -20 $P/logs/boot.out; exit 3
fi
grep -E "ARC_SERVE_LAUNCHED" $P/logs/boot.out | head -1 | cut -c1-220

echo "=== [r268b] health poll (identity-verified) ==="
H=0; HEALTH=""
while [ $H -lt 1500 ]; do
  M=$(curl -s --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null || true)
  echo "$M" | grep -q "gemma4-31b" && { HEALTH="$M"; break; }
  sleep 3; H=$((H+1))
done
[ -n "$HEALTH" ] || { echo "R268B_HEALTH_FAIL (no gemma4-31b identity after $((H*3))s)"; exit 4; }
echo "HEALTH_OK_r268b $(date -u +%H:%M:%SZ) (identity-verified)"

echo "=== [r268b] the bisection ladder (A length / B e2 shapes / C concurrency / D recheck / E determinism) ==="
python3 $S/p268b_ladder.py $P/out/ladder.json > $P/out/ladder.out 2>&1
LRC=$?
cat $P/out/ladder.out
grep -E "^\[p268b\] VERDICT|^\[p268b\] red_steps" $P/out/ladder.out

echo "=== [r268b] the FULL matrix probe on the SAME boot (e2 PARGS) $(date +%H:%M:%S) ==="
docker cp $R266/p266_matrix_probe.py $CNAME:/root/p266_matrix_probe.py >/dev/null 2>&1
docker cp $R/chat_template_gemma4.jinja $CNAME:/root/tpl_g3.jinja >/dev/null 2>&1
docker cp $R266/chat_template_gemma4_official.jinja $CNAME:/root/tpl_g4.jinja >/dev/null 2>&1
PROBE_TIMEOUT=3000
timeout "$PROBE_TIMEOUT" docker exec $CNAME /opt/venv/bin/python /root/p266_matrix_probe.py \
  --model gemma4-31b --tokenizer /models/target --out /root/p266_out --tag r268b_bisect --served g3 \
  --g3 /root/tpl_g3.jinja --g4 /root/tpl_g4.jinja --workers 8 \
  --cells chat,raw,nll,gate --chat-cells t0,card \
  > $P/out/matrix.out 2>&1
PRC=$?
echo "probe done $(date +%H:%M:%S) rc=$PRC"
docker cp $CNAME:/root/p266_out/matrix_r268b_bisect.json $P/out/ >/dev/null 2>&1 || echo "MATRIX_JSON_MISSING"
grep -E "^(PROMPT_ORACLE|FORMAT|chat |raw |    ptok_ok|gate264|    \[gate|NLL |P266_MATRIX_DONE)" $P/out/matrix.out | cut -c1-330 || true

echo "=== [r268b] bank + verdict ==="
docker cp $CNAME:/root/serve_full.log $P/out/serve.log >/dev/null 2>&1 || echo "R268B_SERVE_LOG_MISSING"
docker rm -f $CNAME >/dev/null 2>&1

LADDER_VERDICT=$(grep -E "^\[p268b\] VERDICT" $P/out/ladder.out | tail -1 | sed 's/.*VERDICT: //')
PROBE_400=$(python3 -c "
import json
try:
    j = json.load(open('$P/out/matrix_r268b_bisect.json'))
    n = sum(1 for r in j.get('chat', []) + j.get('raw', []) if str(r.get('http')) == '400')
    print(n)
except Exception:
    print(-1)
" 2>/dev/null || echo -1)
echo "R268B_LADDER $LADDER_VERDICT"
echo "R268B_PROBE http400_chat_raw=$PROBE_400 rc=$PRC"
if [ "$PROBE_400" = "-1" ]; then
  echo "R268B_VERDICT PROBE_INCONCLUSIVE (matrix json missing -- inspect bank)"
elif [ "$PROBE_400" -gt 0 ]; then
  echo "R268B_VERDICT PROBE_RED on the same boot (ladder=$LADDER_VERDICT) -- trigger inside the probe request set"
else
  echo "R268B_VERDICT BOTH_CLEAN on this boot (ladder=$LADDER_VERDICT) -- boot-nondeterminism: re-run until RED"
fi

echo "R268B_DONE rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
