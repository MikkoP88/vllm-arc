#!/bin/bash
# r249_lmc_mx_interop_e2e.sh -- P6.47 P-H2/P-H3 (fix phase): the
# LMC x MX interop e2e validation over the PATCHED adapter (the
# P-H2 fix: _init_lmcache_engine classifies mxfp4/mxfp8/mxk8v4 as
# packed-row pools -- kv_dtype=uint8, row=H_kv*slot_bytes(D,fmt),
# kv_shape=(L,2,chunk,1,row//2), hidden=row//2 -- mirroring the p614
# TQ template with the slot width from mx_layout.slot_bytes; this
# aligns the transfer metadata with the pool view _arc_attn_pool_view
# already hands the connector; the r248 triage proved the 512-vs-544
# B/token disagreement misaddressed every transfer: the r168 re-hit
# garble + the r240b sustained-sweep wedge/device-lost).
#
# Posture: cert28 + P-D overlay trio + the PATCHED repo adapter
# mounted over the lmcint kit (the ARC_PD_LMCDIAG_AD env-optional
# serve mount); gate bypassed for the leg (VLLM_ARC_LMC_MX_GATE=0
# via the serve-script pass-through); ARC_LMC=1 over mxfp4 (the
# r240b death posture).
#
# Gates (in order; every outcome banked):
#   G1 PACKED_METADATA  the stock adapter init log line
#      "use mla: False, kv shape: (17, 2, 1600, 1, 272)" (the
#      packed row, H_kv*slot=544 B/token split (2,272)) -- NOT the
#      old "(17, 2, 1600, 2, 256)" standard geometry. The line IS
#      the metadata oracle readout (r248 banked it at 512 B/token).
#   G2 CLEAN_FACE      c8192 n1 mt64 serves n_ok=1 (the r248-leg
#      baseline; stores running).
#   G3 REHIT_AA        fixed-prompt generation twice on one boot:
#      pass 1 stores (chunks of 1600); pass 2 must HIT -- proof is
#      retrieval-log delta (grep -ic etriev, case-insensitive to
#      cover Retrieved/retrieving/retrieval) OR the functional
#      fallback (pass-2 wall time < 50% of pass 1: prefill skipped
#      by the restore) -- and the two completion TEXTS must be
#      byte-equal (the corrupt-restore class surfaces as garbled
#      generation -- the r168 REP-LOOP signature; byte-true restore
#      reproduces the exact state).
#   G4 DEATH_SEQUENCE  the r240b kvm4 kill sequence on one boot:
#      c32768 n1 -> same-tag re-hit -> n2 -> n4 (mt600), every face
#      n_ok full, no wedge, no DEVICE_LOST, no watchdog timeout.
#   G5 CLEAN_TRANSFERS zero tracebacks and zero LMCache ERROR lines
#      in the full serve log (run-1 law: gates 1-4 passed while the
#      lagged-store path threw 96 shape-mismatch RuntimeErrors that
#      the catch-all swallowed as "cached chunk absent").
# Exit 0 = all four gates green.
#
# RUN HISTORY: run 1 (adapter md5 eab50dd7, 16:54:38Z) -- all four
# gates GREEN (kv shape (17,2,1600,1,272); re-hit A/A byte-equal with
# 3.3x wall drop; the full r240b death sequence n_ok 1/1/2/4 with
# zero wedge/device-lost) BUT the posture caught 96 "Lagged attn
# store failed" RuntimeErrors: the lagged from_gpu path builds its
# tmp_gpu_buffer from hidden_dim_size, which the P-H2 edit left at
# the standard num_kv_head*head_size=512 for MX (_packed = _tq did
# not cover the MX branch) -- tensor a 272 (kv_shape-derived obj) vs
# b 512 at dim 3; the sync path stored byte-true throughout. Fix
# (r249b, adapter md5 bd7065c7): _packed = _tq or bool(_mx_fmt).
# Run-1 artifacts banked at run1/ alongside this leg's. Post-fix
# expectations: tracebacks=0, lmc_error_lines=0, store_lines UP (the
# lagged drops vanish), [r43xfer] H=272, kv-shape line unchanged.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert28
P=$R/p249
SP=$R/p249/stage
CNAME=arc-r249
QM="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_FP8=1 ARC_SPEC=3"
MX="ARC_KVDTYPE=mxfp4 ARC_MXKV_NATIVE=1"
PDM="ARC_PD_PLATFORM=$SP/platform.py ARC_PD_ENVS=$SP/envs.py ARC_PD_GATE=$SP/lmc_mx_boot_gate.py"
ADAPT="ARC_PD_LMCDIAG_AD=$SP/vllm_v1_adapter.py"
declare -A MD5=( \
  [platform.py]=0733f0b8c7dd2fb89422bafa56ec35f5 \
  [envs.py]=31ca4ac7cfcd996aed6446d4b24c2bcd \
  [lmc_mx_boot_gate.py]=ffb1ef11b17bf7b99987c84940c16360 \
  [vllm_v1_adapter.py]=bd7065c79fe14053c9c2a39fa6179226 )
FAILED=""

fail() { echo "R249_ABORT: $*"; }

for f in platform.py envs.py lmc_mx_boot_gate.py vllm_v1_adapter.py; do
  [ -f "$SP/$f" ] || { fail "staged $f missing (run the r249 ship step)"; exit 1; }
  got=$(md5sum "$SP/$f" | cut -d" " -f1)
  [ "$got" = "${MD5[$f]}" ] || { fail "staged $f md5 $got != ${MD5[$f]}"; exit 1; }
done
echo "R249_STAGE_OK"
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
for pat in ARC_PD_GATE ARC_PD_LMCDIAG_AD VLLM_ARC_LMC_MX_GATE; do
  grep -q "$pat" $R/serve_arc_b70.sh || { fail "serve_arc_b70.sh lacks the $pat line (apply the r248 serve patch)"; exit 1; }
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
echo "R249 refuses passed"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R249_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p249.log 2>&1 &
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
  echo "R249_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec $CNAME grep -ic "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&2; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r249] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r249] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r249] ENGINE_DEAD poll=$i"; break; }
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
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R249_WARN_PORT8000_STILL_HELD"
sleep 10

mkdir -p "$P"
docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true
echo "=== [r249:interop] boot $(date +%H:%M:%S) (ARC_LMC=1 + mxfp4 + PATCHED adapter, GATE=0) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME $QM $MX ARC_LMC=1 $PDM $ADAPT \
  VLLM_ARC_LMC_MX_GATE=0 \
  bash $R/serve_arc_b70.sh > "$P/boot.txt" 2>&1 &

if ! boot_wait; then
  echo "R249_BOOT_FAIL -- banking the log"
  docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -30 /root/serve_full.log 2>/dev/null
  echo "BOOT_FAIL" > "$P/outcome.txt"
  cd $P
  sha256sum boot.txt serve.log outcome.txt 2>/dev/null > SHA256SUMS || true
  echo "R249_DONE failed=[boot] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 1
fi
echo "R249_HEALTH_OK $(date +%H:%M:%S)"

# ---- G1: the packed-metadata oracle readout (stock adapter init log) ----
docker exec $CNAME grep -im1 "kv shape" /root/serve_full.log > "$P/kvshape_line.txt" 2>/dev/null || true
if grep -q "(17, 2, 1600, 1, 272)" "$P/kvshape_line.txt" 2>/dev/null; then
  echo "G1_PACKED_METADATA_OK"
  echo "G1_PACKED_METADATA_OK" > "$P/g1.txt"
elif grep -q "(17, 2, 1600, 2, 256)" "$P/kvshape_line.txt" 2>/dev/null; then
  echo "G1_FAIL_OLD_STANDARD_GEOMETRY (the patched adapter did not take the MX branch)"
  echo "G1_FAIL_OLD_GEOMETRY" > "$P/g1.txt"
  FAILED="$FAILED g1"
else
  echo "G1_FAIL_UNEXPECTED (see kvshape_line.txt)"
  echo "G1_FAIL_UNEXPECTED" > "$P/g1.txt"
  FAILED="$FAILED g1"
fi
cat "$P/kvshape_line.txt" 2>/dev/null

# warmup
curl -s -m 120 -o "$P/warmup.json" -w '%{http_code}' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"qwen3.8-27b-fp8\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > "$P/warmup.code"

# ---- G2: the clean face ----
docker cp $R/p139_conc.py $CNAME:/root/p139_conc.py >/dev/null 2>&1
docker exec $CNAME rm -f /root/p139_conc_r249a.json
docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py 8192 64 1 r249a qwen3.8-27b-fp8 0.0
i=0
while [ $i -lt 90 ]; do
  docker exec $CNAME test -f /root/p139_conc_r249a.json 2>/dev/null && break
  if ! docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then echo "R249_G2_ENGINE_DEAD"; break; fi
  sleep 5; i=$((i+1))
done
docker cp $CNAME:/root/p139_conc_r249a.json "$P/" >/dev/null 2>&1 || true
NA=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$P/p139_conc_r249a.json" 2>/dev/null || echo 0)
if [ "${NA:-0}" -ge 1 ]; then
  echo "G2_CLEAN_FACE_OK n_ok=$NA store_lines=$(gcount 'Stored')"
  echo "G2_OK n_ok=$NA" > "$P/g2.txt"
else
  echo "G2_FAIL n_ok=${NA:-?}"
  echo "G2_FAIL n_ok=${NA:-?}" > "$P/g2.txt"
  FAILED="$FAILED g2"
fi

# ---- G3: the fixed-prompt re-hit A/A (the byte-true restore oracle) ----
PROMPT_JSON="$P/aaprompt.json"
python3 - "$PROMPT_JSON" << 'PYEOF'
import json, sys
# a long deterministic prompt (~30K chars / several 1600-token chunks):
# the store path banks full chunks, the re-hit retrieves them above
# min_retrieve_tokens=1024; same bytes both passes, greedy temp 0.
words = ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta"]
prompt = " ".join(f"({i}) the {words[i % len(words)]} sequence continues {i} * {i+1};" for i in range(700))
json.dump({"model": "qwen3.8-27b-fp8", "prompt": prompt,
           "max_tokens": 64, "temperature": 0, "ignore_eos": True}, open(sys.argv[1], "w"))
PYEOF
docker cp "$PROMPT_JSON" $CNAME:/root/aaprompt.json >/dev/null 2>&1
T1=$(curl -s -m 600 -o "$P/aa1.json" -w '%{time_total}' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' -d @"$PROMPT_JSON")
# RET_BASE: retrieval-line count AFTER pass 1 (stores only; the boot
# config dumps dominate here) -- the delta vs post-pass-2 is the hit
RET_BASE=$(gcount "etriev")
sleep 10
T2=$(curl -s -m 600 -o "$P/aa2.json" -w '%{time_total}' http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' -d @"$PROMPT_JSON")
sleep 2
RET2=$(gcount "etriev")
X1=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choices'][0]['text'])" "$P/aa1.json" 2>/dev/null || echo AA1_EMPTY)
X2=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['choices'][0]['text'])" "$P/aa2.json" 2>/dev/null || echo AA2_EMPTY)
echo "$X1" > "$P/aa1.text"; echo "$X2" > "$P/aa2.text"
echo "t1=$T1 t2=$T2 retrieve_before=$RET_BASE retrieve_after=$RET2" | tee "$P/aa_times.txt"
DELTA=$(( ${RET2:-0} - ${RET_BASE:-0} ))
SPEEDUP=$(python3 -c "print(1 if float('${T2:-999}') < 0.5 * float('${T1:-1}') else 0)" 2>/dev/null || echo 0)
if [ "$X1" = "$X2" ] && [ "$X1" != "AA1_EMPTY" ]; then
  if [ "$DELTA" -gt 0 ]; then
    echo "G3_REHIT_AA_OK retrieve_delta=$DELTA texts_byte_equal=yes t1=$T1 t2=$T2"
    echo "G3_OK retrieve_delta=$DELTA t1=$T1 t2=$T2" > "$P/g3.txt"
  elif [ "$SPEEDUP" = "1" ]; then
    echo "G3_REHIT_AA_OK_QUIETLOG (texts equal + wall ${T2}s < 50% of ${T1}s = functional hit; retrieval log quiet)"
    echo "G3_OK_QUIETLOG t1=$T1 t2=$T2" > "$P/g3.txt"
  else
    echo "G3_FAIL_NO_HIT_PROOF retrieve_delta=$DELTA t1=$T1 t2=$T2 (texts equal but no retrieval/TTFT signal)"
    echo "G3_FAIL_NO_HIT_PROOF $DELTA" > "$P/g3.txt"
    FAILED="$FAILED g3"
  fi
elif [ "$DELTA" -gt 0 ]; then
  echo "G3_FAIL_TEXT_DRIFT (retrieved but generations differ -- corrupt restore)"
  echo "G3_FAIL_TEXT_DRIFT" > "$P/g3.txt"
  FAILED="$FAILED g3"
else
  echo "G3_FAIL texts_differ_no_retrieve (no hit happened: t1=$T1 t2=$T2 delta=$DELTA)"
  echo "G3_FAIL_NO_HIT texts_differ" > "$P/g3.txt"
  FAILED="$FAILED g3"
fi

# ---- G4: the r240b death sequence (c32768: n1 -> rehit -> n2 -> n4) ----
G4="G4_OK"
RUN=0
for spec in "1 r249d_n1" "1 r249d_n1" "2 r249d_n2" "4 r249d_n4"; do
  set -- $spec; NN=$1; TAG=$2; RUN=$((RUN+1))
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py 32768 600 $NN $TAG qwen3.8-27b-fp8 0.0
  i=0; DONE=0
  while [ $i -lt 170 ]; do
    if docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then DONE=1; break; fi
    if ! docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then
      echo "R249_G4_ENGINE_DEAD tag=$TAG poll=$i"; DONE=0; break
    fi
    sleep 5; i=$((i+1))
  done
  # same tag twice = the re-hit cell; bank to run-indexed names (basename
  # collision law: pick2 must not overwrite pick1)
  docker cp $CNAME:/root/p139_conc_$TAG.json "$P/p139_conc_${TAG}_run${RUN}.json" >/dev/null 2>&1 || true
  if [ "$DONE" = "1" ]; then
    NOK=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_ok'])" "$P/p139_conc_${TAG}_run${RUN}.json" 2>/dev/null || echo 0)
    if [ "${NOK:-0}" -ge "$NN" ]; then
      echo "G4 cell $TAG run$RUN n_ok=$NOK OK"
    else
      echo "G4 cell $TAG run$RUN PARTIAL n_ok=${NOK:-?}/$NN"
      G4="G4_FAIL_partial"
    fi
  else
    echo "G4 cell $TAG run$RUN HUNG (no JSON in $((170*5))s)"
    G4="G4_FAIL_hung_${TAG}_run${RUN}"
    break
  fi
done
TO=$(gcount "TimeoutError")
DL=$(gcount "UR_RESULT_ERROR_DEVICE_LOST")
if [ "$G4" = "G4_OK" ] && [ "${TO:-0}" = "0" ] && [ "${DL:-0}" = "0" ]; then
  echo "G4_DEATH_SEQUENCE_CLEAN (no wedge, no device-lost, no watchdog)"
  echo "G4_OK timeouts=$TO device_lost=$DL" > "$P/g4.txt"
else
  echo "G4_FAIL ($G4 timeouts=$TO device_lost=$DL)"
  echo "G4_FAIL $G4 timeouts=$TO device_lost=$DL" > "$P/g4.txt"
  FAILED="$FAILED g4"
fi

docker cp $CNAME:/root/serve_full.log "$P/serve.log" >/dev/null 2>&1 || true

# ---- G5: clean transfers (run-1 law: gates 1-4 can pass while the
# lagged-store path error-storms -- the catch-all swallows them) ----
TB=$(gcount "Traceback (most recent call last)")
LE=$(gcount "LMCache ERROR")
if [ "${TB:-0}" = "0" ] && [ "${LE:-0}" = "0" ]; then
  echo "G5_CLEAN_TRANSFERS_OK (zero tracebacks, zero LMCache ERROR lines)"
  echo "G5_OK tracebacks=$TB lmc_errors=$LE" > "$P/g5.txt"
else
  echo "G5_FAIL tracebacks=$TB lmc_error_lines=$LE (transfer-path exceptions -- see serve.log)"
  echo "G5_FAIL tracebacks=$TB lmc_errors=$LE" > "$P/g5.txt"
  FAILED="$FAILED g5"
fi

{
  echo "store_lines=$(gcount 'Stored')"
  echo "retrieve_lines=$(gcount 'etriev')"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
  echo "timeout_errors=$TO device_lost=$DL"
  echo "lmc_error_lines=$(gcount 'LMCache ERROR')"
} > "$P/posture.txt"
cat "$P/posture.txt"
docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 5
docker rm -f $CNAME >/dev/null 2>&1; verify_removed $CNAME || true

echo "$FAILED" > "$P/failed.txt"
if [ -z "$FAILED" ]; then echo "INTEROP_PASS" > "$P/outcome.txt"; else echo "INTEROP_FAIL" > "$P/outcome.txt"; fi
cd $P
sha256sum boot.txt serve.log posture.txt outcome.txt failed.txt kvshape_line.txt \
  g1.txt g2.txt g3.txt g4.txt g5.txt warmup.json warmup.code aa1.json aa2.json aa1.text aa2.text \
  aa_times.txt aaprompt.json p139_conc_r249a.json p139_conc_r249d_*_run*.json \
  2>/dev/null > SHA256SUMS || true
echo "R249_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -z "$FAILED" ] || exit 1
exit 0
