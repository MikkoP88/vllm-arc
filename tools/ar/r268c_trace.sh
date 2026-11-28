#!/bin/bash
# r268c_trace.sh -- P6.47 P-D (r268c): the trace discriminator leg.
#
# THE r268b VERDICT this leg follows: on ONE plain-image c3-posture boot
# the ladder was 14/33 RED (A_r12 sequential-plain RED at ptok=122; B2/C3
# chats RED; C1/C2 2-way bursts RED) while the FULL p266 probe on the SAME
# boot was 323/324 RED (every raw cell incl. t0 temp=0 and NON-crossing
# g4 short prompts) -- the corruption is request-conditional ENGINE
# WIRING, and the probe's own request set is uniformly inside the broken
# class. The kernels/store math are proven exact for correct metadata
# (r267), so the defect is in the metadata VALUES the engine hands the
# kernels for the broken request class. This leg boots the SAME posture
# with the r268c_overlay (engine.py _TRACE hardcoded True -- the fix for
# the r268 dead letter: serve_arc_b70.sh's curated export list never
# passes ARC_MXKV_TRACE into the container) and fires a SEQUENTIAL
# minimal discriminator set (p268c_disc.py): T0 clean control / T1 the
# A_r12 RED / T2 the gate chat ('our' class) / T3 the card chat (400
# class) / T4+T5 the probe's own byte-exact templated raws (non-crossing
# g4 short + crossing g3) / T6 the C1 2-way burst. The overlay prints
# store smap + decode bt/stride/seq_lens per call (MXKVTRACE lines in
# serve.log); T0's values are the CORRECT reference and any divergence
# on a RED request NAMES the wrong metadata value directly.
# Artifacts -> /root/build/p647/r268c_trace/{out,logs}.  Exit 0 = banked.
# W14 quiesce before, clean-recipe trap restore after (r262 law).
# Launch (ship + launch in SEPARATE ssh calls, md5-verified ship):
#   mkdir -p /root/build/p647/r268c_trace/logs
#   nohup setsid bash /root/build/r268c/r268c_trace.sh \
#     > /root/build/p647/r268c_trace/logs/launch.log 2>&1 &
set -uo pipefail
R=/root/build
P=$R/p647/r268c_trace
S=$R/r268c
R266=$R/r266
IMG=vllm-arc:v0.30.0-b70-cert29
CNAME=arc-r268c

[ -f "$S/p268c_disc.py" ] || { echo "R268C_NO_DISC (ship first)"; exit 1; }
[ -d "$R/r268c_overlay" ] || { echo "R268C_NO_OVERLAY (build r268c_overlay first)"; exit 1; }
OVMD5=$(md5sum $R/r268c_overlay/engine.py | cut -d' ' -f1)
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R268C_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R268C_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R268C_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R268C_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  bash $R/r262_w14_restore.sh > $P/logs/w14_restore_r268c.log 2>&1
  rc=$?
  grep -E "^W14_" $P/logs/w14_restore_r268c.log | tail -3 || true
  echo "W14_RESTORE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$P/logs/w14_restore_r268c.log"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out
echo "R268C_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) image=$IMG overlay_engine_md5=$OVMD5 disc_md5=$(md5sum $S/p268c_disc.py | cut -d' ' -f1)"

echo "=== [r268c] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R268C_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r268c] boot c3 posture + r268c_trace overlay (eager) $(date +%H:%M:%S) ==="
# e2/c3 posture (the r268b runner's env) + the trace overlay. _TRACE is
# hardcoded True INSIDE the overlay engine.py, so no env var needs to
# survive serve_arc_b70.sh's curated export list (the r268 dead letter).
nohup env \
  ARC_IMAGE=$IMG ARC_NAME=$CNAME \
  ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
  ARC_QUANT=fp8 ARC_GMU=0.8 ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 \
  ARC_RPC_TIMEOUT=1200 ARC_MAXLEN=40960 ARC_FP8=0 \
  ARC_TEMPLATE=/root/build/chat_template_gemma4.jinja \
  ARC_RPARSER= ARC_TPARSER= \
  ARC_KVDTYPE=mxfp8 ARC_MXKV_NATIVE=1 \
  ARC_MXKV_OVERLAY=$R/r268c_overlay \
  bash $R266/serve_arc_b70.sh > $P/logs/boot.out 2>&1 &
BOOTPID=$!
B=0
while [ $B -lt 60 ]; do
  grep -q "ARC_SERVE_LAUNCHED" $P/logs/boot.out 2>/dev/null && break
  kill -0 $BOOTPID 2>/dev/null || break
  sleep 2; B=$((B+1))
done
if ! grep -q "ARC_SERVE_LAUNCHED" $P/logs/boot.out 2>/dev/null; then
  echo "R268C_BOOT_FAIL (no serve line after $((B*2))s)"; tail -20 $P/logs/boot.out; exit 3
fi
grep -E "ARC_SERVE_LAUNCHED" $P/logs/boot.out | head -1 | cut -c1-220

echo "=== [r268c] health poll (identity-verified) ==="
H=0; HEALTH=""
while [ $H -lt 1500 ]; do
  M=$(curl -s --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null || true)
  echo "$M" | grep -q "gemma4-31b" && { HEALTH="$M"; break; }
  sleep 3; H=$((H+1))
done
[ -n "$HEALTH" ] || { echo "R268C_HEALTH_FAIL (no gemma4-31b identity after $((H*3))s)"; exit 4; }
echo "HEALTH_OK_r268c $(date -u +%H:%M:%SZ) (identity-verified)"

echo "=== [r268c] the trace discriminator set (T0 clean ref / T1-T6 the red classes) ==="
python3 $S/p268c_disc.py $P/out/disc.json > $P/out/disc.out 2>&1
DRC=$?
cat $P/out/disc.out
grep -E "^\[p268c\] VERDICT" $P/out/disc.out || true

echo "=== [r268c] bank: serve.log + trace extraction ==="
docker cp $CNAME:/root/serve_full.log $P/out/serve.log >/dev/null 2>&1 || echo "R268C_SERVE_LOG_MISSING"
docker rm -f $CNAME >/dev/null 2>&1
TRACE_N=$(grep -c "MXKVTRACE" $P/out/serve.log 2>/dev/null || echo 0)
TRACE_STORE=$(grep -c "MXKVTRACE store" $P/out/serve.log 2>/dev/null || echo 0)
TRACE_DECX=$(grep -c "MXKVTRACE decx" $P/out/serve.log 2>/dev/null || echo 0)
grep "MXKVTRACE" $P/out/serve.log > $P/out/trace_disc.txt 2>/dev/null || true
echo "trace lines: $TRACE_N total, $TRACE_STORE store, $TRACE_DECX decx"
DISC_VERDICT=$(grep -E "^\[p268c\] VERDICT" $P/out/disc.out | tail -1 | sed 's/.*VERDICT: //')
echo "R268C_DISC $DISC_VERDICT"
echo "R268C_TRACE n=$TRACE_N store=$TRACE_STORE decx=$TRACE_DECX"
if [ "$TRACE_N" -gt 0 ] 2>/dev/null; then
  echo "R268C_VERDICT TRACE_LIVE (n=$TRACE_N) -- analyze decx/smap vs the disc reds"
else
  echo "R268C_VERDICT TRACE_DEAD (0 lines) -- the dead letter persists; inspect overlay mount"
fi

echo "R268C_DONE rc=0 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
