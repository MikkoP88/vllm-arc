#!/bin/bash
# r229a_fmt_replay.sh — P6.47 cert27 format-replay legs: the
# payload-replay gate class extended to mxfp8/mxfp4 per the r225
# coverage note (the ReduceSplitK combine fix serves all three MX
# lanes identically; r225t/r226a proved mxk8v4, this leg proves the
# other two on their OWN engine tensors).
#
# Cert27 bakes BOTH the fixed wheel and the r224a ARC_MXKV_AB
# instrument, so the r226a /proc argv+environ dance is unnecessary:
# ONE env-armed boot (eager per the pageable-H2D capture law,
# ARC_MXKV_AB=1 N=60 MINSL=1000) banks payloads at the native
# dispatch; then harvest FIRST (r226a first-run law), class-scan
# (max_sl <= nbt*64 -> COVERED; the 96-block bank cap), and score the
# first 8 COVERED payloads with the fmt-parametric q-cast-matched
# scorer (each lane vs the fp64 walk of ITS OWN input).
#
# Refuses: image cert27 + manifest cert27_fork_sha == $1; scorer+iso
# shipped; fmt in {mxfp8, mxfp4}.
#
# Usage: r229a_fmt_replay.sh <CERT27_FORK_SHA> <mxfp8|mxfp4>
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert27
FORK_SHA="${1:?usage: r229a_fmt_replay.sh <CERT27_FORK_SHA> <mxfp8|mxfp4>}"
FMT="${2:?usage: r229a_fmt_replay.sh <CERT27_FORK_SHA> <mxfp8|mxfp4>}"
case "$FMT" in
  mxfp8) SUFF=a ;;
  mxfp4) SUFF=b ;;
  *) echo "R229_ABORT: fmt must be mxfp8 or mxfp4 (got $FMT)"; exit 1 ;;
esac
CNAME=arc-r229$SUFF
BANK=$R/p647/r229$SUFF
SCORER=$R/r229_tools/r229_replay_fmt.py
ISOLIB=$R/r225_tools/r225j_iso.py
ANALYZER=$R/r224b_ab_analyze.py

mkdir -p "$BANK"
cd "$BANK" || exit 2

fail() { echo "R229${SUFF}_ABORT: $*"; }

# ---- refuses (staged-ctx law) ----
docker image inspect "$IMG" >/dev/null 2>&1 || { fail "image $IMG missing"; exit 1; }
MF_SHA=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "grep -o 'cert27_fork_sha=[0-9a-f]*' /etc/vllm-arc/IMAGE_MANIFEST | head -1 | cut -d= -f2")
[ "$MF_SHA" = "$FORK_SHA" ] || { fail "manifest fork_sha $MF_SHA != $FORK_SHA"; exit 1; }
for f in "$SCORER" "$ISOLIB" "$ANALYZER"; do
  [ -f "$f" ] || { fail "tool not shipped ($f)"; exit 1; }
done
grep -q "R229F_CELL" "$SCORER" || { fail "scorer lacks the cell verdict (wrong file?)"; exit 1; }
SO_MD5=$(docker run --rm --entrypoint /bin/bash "$IMG" -c \
  "/opt/venv/bin/python -c \"import glob, hashlib; ps=glob.glob('/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libmqa_logits*.so'); print(hashlib.md5(open(ps[0],'rb').read()).hexdigest())\"" 2>/dev/null) || SO_MD5=""
[ "$SO_MD5" = "190e261edc79b3b497d8d8e801e65757" ] \
  || { fail "image libmqa_logits .so md5 $SO_MD5 != build25 pin"; exit 1; }
echo "R229${SUFF} refuses passed (fork_sha=$FORK_SHA fmt=$FMT so_md5=$SO_MD5)"

RESTORE_DONE=0
finish() {
  if [ "$RESTORE_DONE" = 1 ]; then return; fi
  RESTORE_DONE=1
  docker rm -f "$CNAME" >/dev/null 2>&1 || true
  echo "R229${SUFF}: restoring W14 ($(date +%H:%M:%S))"
  nohup bash /root/build/w14_boot.sh fullres > /root/build/w14_restore.log 2>&1 < /dev/null &
  for i in $(seq 1 40); do
    c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
    [ "$c" = "200" ] && { echo "R229${SUFF}: W14 HEALTH 200 ($(date +%H:%M:%S))"; return; }
    sleep 30
  done
  echo "R229${SUFF}: WARN W14 not 200 after restore poll (last=$c)"
}
trap finish EXIT

# ---- 1. arc window ----
docker stop radiance-test >/dev/null 2>&1 || true
sleep 5

# ---- 2. ONE env-armed boot (instrument + fixed wheel baked) ----
echo "=== boot $IMG fmt=$FMT native+AB eager $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=0 ARC_SPEC=3 ARC_FP8=1 \
  ARC_KVDTYPE=$FMT ARC_MXKV_NATIVE=1 \
  ARC_MXKV_AB=1 ARC_MXKV_AB_N=60 ARC_MXKV_AB_MINSL=1000 \
  bash $R/serve_arc_b70.sh > "$BANK/boot.txt" 2>&1 &
SEEN=0
for i in $(seq 1 100); do
  sleep 15
  c=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health)
  [ "$c" = "200" ] && { SEEN=1; break; }
  if ! docker inspect "$CNAME" >/dev/null 2>&1; then
    break
  fi
done
if [ "$SEEN" != 1 ]; then
  echo "GATE_FAIL_BOOT (last=$c)"
  docker cp $CNAME:/root/serve_full.log "$BANK/serve_bootfail.log" >/dev/null 2>&1 || true
  exit 1
fi
echo "HEALTH_OK fmt=$FMT $(date +%H:%M:%S)"
{
  echo "mx_backend_lines=$(docker exec $CNAME grep -c 'Using MX KV attention backend' /root/serve_full.log 2>/dev/null || echo 0)"
  echo "native_lines=$(docker exec $CNAME grep -c 'MXKV native call' /root/serve_full.log 2>/dev/null || echo 0)"
  echo "ab_armed=$(docker exec $CNAME grep -c 'MXKV AB' /root/serve_full.log 2>/dev/null || echo 0)"
} > "$BANK/markers.txt"
cat "$BANK/markers.txt"
grep -Eq "native_lines=[1-9]" "$BANK/markers.txt" \
  || { fail "no native telemetry for fmt=$FMT"; exit 1; }

# ---- 3. traffic: small gen, then essay-class long prompt (r226 bytes) ----
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup r229","max_tokens":16,"temperature":0}' \
  > "$BANK/gen_small.json" || true
sleep 5
python3 - <<'PYEOF' > "$BANK/essay_prompt.json"
import json
base = ("The archive room held rows of ledgers, each spine numbered by hand, and "
        "every evening the archivist walked the aisles recording which volumes had "
        "been consulted that day, a ritual older than the building itself. ")
prompt = {"model": "qwen3.8-27b-fp8", "prompt": base * 130, "max_tokens": 96,
          "temperature": 0}
print(json.dumps(prompt))
PYEOF
curl -s -m 300 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d @"$BANK/essay_prompt.json" > "$BANK/gen_essay.json" || true
sleep 10

# ---- 4. harvest FIRST, class scan, then strict scoring ----
docker exec $CNAME grep -E "MXKV AB" /root/serve_full.log > "$BANK/ab_lines.txt" 2>/dev/null || true
echo "=== AB verdict lines ($(wc -l < "$BANK/ab_lines.txt" 2>/dev/null || echo 0)) ==="
docker exec $CNAME sh -c 'ls -la /tmp/mxkv_ab.*.pt 2>/dev/null' > "$BANK/payload_inventory.txt" || true
for f in $(docker exec $CNAME sh -c 'ls /tmp/mxkv_ab.*.pt 2>/dev/null'); do
  docker cp "$CNAME:$f" "$BANK/" >/dev/null 2>&1 || true
done
echo "harvested payloads: $(ls "$BANK"/mxkv_ab.*.pt 2>/dev/null | wc -l)"
docker cp $CNAME:/root/serve_full.log "$BANK/serve.log" >/dev/null 2>&1 || true

# payload-class scan (docker exec needs -i for the heredoc; r224b law)
docker exec -i $CNAME /opt/venv/bin/python - <<'PYEOF' > "$BANK/payload_classes.txt" 2>&1
import glob
import torch
for f in sorted(glob.glob("/tmp/mxkv_ab.*.pt")):
    p = torch.load(f, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    msl = int(p["sl"][:r].max())
    nbt = int(p["bt_head"].numel())
    print(f"CLASS {f} fmt={p['fmt']} max_sl={msl} nbt={nbt} "
          f"{'COVERED' if msl <= nbt * 64 else 'TRUNCATED'}")
PYEOF
grep -q "^CLASS" "$BANK/payload_classes.txt" || { fail "payload class scan empty"; exit 1; }
cat "$BANK/payload_classes.txt"
NCOV=$(grep -c " COVERED$" "$BANK/payload_classes.txt" || true)
NTRU=$(grep -c " TRUNCATED$" "$BANK/payload_classes.txt" || true)
echo "payload classes: covered=$NCOV truncated=$NTRU"
grep -q "fmt=$FMT " "$BANK/payload_classes.txt" \
  || { fail "banked payloads are not fmt=$FMT"; exit 1; }

# THE VERDICT: fmt-parametric q-cast-matched scorer, first 8 COVERED,
# strict (a crashed cell aborts — never pass by shrinking the set)
docker cp "$SCORER" "$CNAME:/tmp/r229_replay_fmt.py" >/dev/null 2>&1 || true
docker cp "$ISOLIB" "$CNAME:/tmp/r225j_iso.py" >/dev/null 2>&1 || true
SLOG=$BANK/r229_score.log
: > "$SLOG"
ninv=0
for f in $(grep " COVERED$" "$BANK/payload_classes.txt" | awk '{print $2}' | head -8); do
  ninv=$((ninv+1))
  echo "=== R229 score $f ($(date +%H:%M:%S)) ===" | tee -a "$SLOG"
  docker exec $CNAME env PYTHONUNBUFFERED=1 /opt/venv/bin/python -u \
    /tmp/r229_replay_fmt.py "$f" >> "$SLOG" 2>&1 \
    || { fail "scorer rc=$? on $f"; exit 1; }
done
cat "$SLOG"
nscored=$(grep -c "^R229F_CELL" "$SLOG" || true)
npassed=$(grep -c "^R229F_CELL .* PASS$" "$SLOG" || true)
echo "R229${SUFF}_INVOKED=$ninv SCORED=$nscored PASSED=$npassed"
[ "$ninv" -eq 8 ] || { fail "fewer than 8 COVERED payloads scored ($ninv; covered=$NCOV)"; exit 1; }
[ "$nscored" -eq "$ninv" ] || { fail "scorer cells $nscored != invocations $ninv (crashed cell)"; exit 1; }
[ "$npassed" = "$nscored" ] || { fail "q-cast-matched scoring FAILED ($npassed/$nscored)"; exit 1; }

# lane-gap analyzer rollup (raw lanes; q-cast sensitivity expected)
docker cp "$ANALYZER" "$CNAME:/tmp/r224b_ab_analyze.py" >/dev/null 2>&1 || true
docker exec $CNAME /opt/venv/bin/python /tmp/r224b_ab_analyze.py /tmp > "$BANK/r224b_report.txt" 2>&1 || true
tail -5 "$BANK/r224b_report.txt" || true

echo "R229${SUFF}_LEG_DONE $(date +%H:%M:%S) fmt=$FMT payloads=$(ls "$BANK"/mxkv_ab.*.pt 2>/dev/null | wc -l) score=$npassed/$nscored"
# ---- 5. teardown + W14 restore via the EXIT trap ----
exit 0
