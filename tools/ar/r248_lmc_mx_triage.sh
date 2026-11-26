#!/bin/bash
# r248_lmc_mx_triage.sh -- P6.47 P-H1 (fix phase): the LMC x MX
# interop triage leg. Owner directive: comprehensive telemetry capture
# points to LOCATE the root issue before fixing. The P-D gate is
# BYPASSED (VLLM_ARC_LMC_MX_GATE=0, the A/B-evidence knob) over the
# exact r240b death posture (ARC_LMC=1 + mxfp4) on cert28 + the P-D
# overlay trio, with TWO diag instruments mounted over the lmcint kit
# (env-optional serve mounts, the P-D pattern):
#   lmc_kvfmt_detector.py  census of every discover() call: input
#                          pools (shape/dtype/strides), arc-guard
#                          outcome, the format stock discovery chose.
#   vllm_v1_adapter.py     register_kv_caches pool census, the KVGATE
#                          rows + engine kv_shape/kv_dtype, and every
#                          _arc_attn_pool_view call: which branch
#                          (tq / r43-fp8 / RAW-FALLTHROUGH) + the
#                          geometry in vs out.
# Faces: (1) clean-store face c8192 n1 mt64 -- the r240b lanes served
#        8K faces with stores running (banks the diag telemetry of a
#        WORKING store); (2) the exact r240b death face c32768 n4
#        mt600 -- the cell that killed kvm4 (RPC watchdog wedge) and
#        kvmk (UR_RESULT_ERROR_DEVICE_LOST). If the face hangs while
#        workers live, wedge-time py-spy dumps (host /usr/local/bin/
#        py-spy docker-cp'd in -- the r108w method) bank the store-
#        lock stack. Evidence either way: survival = the wedge needs
#        more load; death = the instrumented crash.
# Expected from the code read (the hypothesis this leg confirms):
#   the MX pool (NB, BS, Hk, slot) uint8 has NO _arc_attn_pool_view
#   branch (not TQ, axis-1 != 2) -> RAW fall-through, while the
#   engine metadata builds standard-geometry kv_shape=(L,2,chunk,
#   Hkv,HS) with kv_heads*head_size rows -> the store misaddresses
#   the pool (the r108w wedge class).
# Refuses: cert28 image, staged md5s (P-D trio + 2 diag), the serve
# script's ARC_PD_* + ARC_PD_LMCDIAG_* mount lines, no arc leg in
# flight, port check. W14 quiesce before, EXIT-trap fullres restore
# after. Artifacts -> /root/build/p248/. Exit 0 = TRIAGE_CAPTURED
# (the leg banks evidence; pass/fail is the ANALYSIS, not the boot).
#
# RUN HISTORY: run 1 (rev 1, 16:24:39Z) BOOT_DEATH at 16:27:32Z with
# ZERO diag lines -- the P-D gate itself fired: the runner exported
# VLLM_ARC_LMC_MX_GATE=0 but the serve script's docker-exec env block
# never passed host VLLM_* vars, so the engine saw the default (gate
# ON) and refused the connector construction (as designed -- the
# r246b pd_refuse signature, "Engine core initialization failed").
# Fix: serve patch rev 2 adds the ${VLLM_ARC_LMC_MX_GATE:+export ...}
# pass-through (unset = certified gate-ON posture byte-identical).
# This rev 2 also banks SHA256SUMS on the boot-death path (run 1
# exited before the sums). Attempt-2 artifacts overwrite the bank;
# run 1's evidence = this header + the banked boot-death serve.log.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
P=$R/p248
SP=$R/p248/stage
CNAME=arc-r248
QM="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_FP8=1 ARC_SPEC=3"
MX="ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1"
PDM="ARC_PD_PLATFORM=$SP/platform.py ARC_PD_ENVS=$SP/envs.py ARC_PD_GATE=$SP/lmc_mx_boot_gate.py"
DIAG="ARC_PD_LMCDIAG_DET=$SP/lmc_kvfmt_detector.py ARC_PD_LMCDIAG_AD=$SP/vllm_v1_adapter.py"
declare -A MD5=( \
  [platform.py]=0733f0b8c7dd2fb89422bafa56ec35f5 \
  [envs.py]=31ca4ac7cfcd996aed6446d4b24c2bcd \
  [lmc_mx_boot_gate.py]=ffb1ef11b17bf7b99987c84940c16360 \
  [lmc_kvfmt_detector.py]=43c5ddbbb3ded4c33b798ba10823c321 \
  [vllm_v1_adapter.py]=d393d6434a22879af0bfd5a33e27bab9 )

fail() { echo "R248_ABORT: $*"; }

for f in platform.py envs.py lmc_mx_boot_gate.py lmc_kvfmt_detector.py vllm_v1_adapter.py; do
  [ -f "$SP/$f" ] || { fail "staged $f missing (run the r248 ship step)"; exit 1; }
  got=$(md5sum "$SP/$f" | cut -d" " -f1)
  [ "$got" = "${MD5[$f]}" ] || { fail "staged $f md5 $got != ${MD5[$f]}"; exit 1; }
done
echo "R248_STAGE_OK"
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
for pat in ARC_PD_PLATFORM ARC_PD_LMCDIAG_DET ARC_PD_LMCDIAG_AD; do
  grep -q "$pat" $R/serve_arc_b70.sh || { fail "serve_arc_b70.sh lacks the $pat mount (apply the r248 serve extension first)"; exit 1; }
done
[ -f "$R/p139_conc.py" ] || { fail "p139_conc.py missing"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  fail "arc leg already in flight"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    fail "port 8000 held by an unknown holder"; exit 3
  fi
fi
echo "R248 refuses passed"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R248_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p248.log 2>&1 &
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
  echo "R248_FAIL_STALE_CONTAINER $1"; return 1
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r248] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r248] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r248] ENGINE_DEAD poll=$i"; break; }
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
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R248_WARN_PORT8000_STILL_HELD"
sleep 10

mkdir -p "$P"
docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
echo "=== [r248:triage] boot $(date +%H:%M:%S) (ARC_LMC=1 over mxfp4, GATE=0, diag mounts live) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $QM $MX ARC_LMC=1 $PDM $DIAG \
  VLLM_ARC_LMC_MX_GATE=0 \
  bash $R/serve_arc_b70.sh > "$P/boot.txt" 2>&1 &

if ! boot_wait; then
  echo "R248_BOOT_FAIL -- banking the instrumented log anyway (the boot death IS triage evidence)"
  docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -30 /root/serve_full.log 2>/dev/null
  echo "BOOT_DEATH" > "$P/outcome.txt"
  cd $P
  sha256sum boot.txt serve.log outcome.txt 2>/dev/null > SHA256SUMS || true
  echo "R248_DONE TRIAGE_CAPTURED=boot_death $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi
echo "R248_HEALTH_OK $(date +%H:%M:%S)"
grep -h "ARC_SERVE_LAUNCHED" "$P/boot.txt" | tail -1 > "$P/posture.txt"
GATE0=$(gcount "VLLM_ARC_LMC_MX_GATE")
LMCN=$(gcount 'LMCache')
echo "gate_bypass_env=$GATE0 lmc_lines=$LMCN" >> "$P/posture.txt"
echo "warn_line=$(grep -c 'forced on ARC_KVDTYPE' "$P/boot.txt" 2>/dev/null || echo 0)" >> "$P/posture.txt"

# ---- warmup + face 1: the clean-store face (8K n1) ----
curl -s -m 120 -o "$P/warmup.json" -w '%{http_code}' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > "$P/warmup.code"
docker cp $R/p139_conc.py $CNAME:/root/p139_conc.py >/dev/null 2>&1
docker exec $CNAME rm -f /root/p139_conc_r248f1.json
docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py 8192 64 1 r248f1 qwen3.8-27b-fp8 0.0
i=0
while [ $i -lt 90 ]; do
  docker exec $CNAME test -f /root/p139_conc_r248f1.json && break
  if ! docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then
    echo "R248_F1_ENGINE_DEAD poll=$i"; break
  fi
  sleep 5; i=$((i+1))
done
docker cp $CNAME:/root/p139_conc_r248f1.json "$P/" >/dev/null 2>&1 || true
N1=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$P/p139_conc_r248f1.json" 2>/dev/null || echo 0)
echo "face1_n_ok=${N1:-0} store_lines=$(gcount 'Stored')" >> "$P/posture.txt"
docker exec $CNAME grep 'r248diag' /root/serve_full.log > "$P/diag_face1.txt" 2>/dev/null || true
echo "=== [r248] face1 done n_ok=${N1:-0}; diag lines: $(wc -l < "$P/diag_face1.txt" 2>/dev/null || echo 0) ==="

# ---- face 2: the exact r240b death face (32K n4 mt600) ----
docker exec $CNAME rm -f /root/p139_conc_r248f2.json
docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py 32768 600 4 r248f2 qwen3.8-27b-fp8 0.0
echo "=== [r248] death face launched $(date +%H:%M:%S) (c32768 n4 mt600) ==="
i=0; F2=0; SPY=0; DEAD=0
while [ $i -lt 180 ]; do
  if docker exec $CNAME test -f /root/p139_conc_r248f2.json 2>/dev/null; then F2=1; break; fi
  if ! docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then
    DEAD=1; echo "R248_F2_ENGINE_DEAD poll=$i (ts $(date +%H:%M:%S))"; break
  fi
  # wedge window: face hung > 7 min with workers ALIVE -> py-spy the
  # store-lock stack (the r108w method; host py-spy docker-cp'd in)
  if [ $i -ge 84 ] && [ "$SPY" = "0" ]; then
    SPY=1
    docker cp /usr/local/bin/py-spy $CNAME:/tmp/py-spy >/dev/null 2>&1 || true
    for wp in $(docker exec $CNAME pgrep -f 'EngineCore|Worker_TP' 2>/dev/null); do
      docker exec $CNAME /tmp/py-spy dump --pid "$wp" \
        > "$P/pyspy_$wp.txt" 2>&1 || true
    done
    echo "R248_PYSPY_DUMPED pids=$(docker exec $CNAME pgrep -f 'EngineCore|Worker_TP' 2>/dev/null | tr '\n' ' ')"
  fi
  sleep 5; i=$((i+1))
done
docker cp $CNAME:/root/p139_conc_r248f2.json "$P/" >/dev/null 2>&1 || true
N2=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$P/p139_conc_r248f2.json" 2>/dev/null || echo 0)
sleep 8
docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
docker exec $CNAME grep 'r248diag' /root/serve_full.log > "$P/diag_full.txt" 2>/dev/null || true
TB=$(gcount "Traceback (most recent call last)")
TO=$(gcount "TimeoutError")
DL=$(gcount "UR_RESULT_ERROR_DEVICE_LOST")
ST=$(gcount "Stored")
cat >> "$P/posture.txt" << EOF
face2_n_ok=${N2:-0} face2_done=$F2 engine_dead=$DEAD pyspy_dumped=$SPY
tracebacks=$TB timeout_errors=$TO device_lost=$DL store_lines=$ST
EOF
cat "$P/posture.txt"

# ---- outcome classification (analysis-class, not boot-class) ----
if [ "$DEAD" = "1" ]; then
  echo "DEATH_REPRODUCED (engine died under the instrumented posture)"
  echo "DEATH_REPRODUCED" > "$P/outcome.txt"
elif [ "$F2" = "1" ] && [ "${N2:-0}" -gt 0 ]; then
  echo "FACE2_SURVIVED n_ok=$N2 (the wedge needs more load than one cell -- telemetry still banked)"
  echo "FACE2_SURVIVED" > "$P/outcome.txt"
else
  echo "FACE2_HUNG_NO_ENGINE_DEATH (wedge without death -- pyspy stacks are the evidence)"
  echo "FACE2_HUNG" > "$P/outcome.txt"
fi
docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 5
docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true

cd $P
sha256sum boot.txt serve.log posture.txt outcome.txt diag_face1.txt diag_full.txt \
  pyspy_*.txt warmup.json warmup.code p139_conc_r248f1.json p139_conc_r248f2.json \
  2>/dev/null > SHA256SUMS || true
echo "R248_DONE TRIAGE_CAPTURED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
