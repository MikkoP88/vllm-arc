#!/bin/bash
# r267_nan_canary.sh -- P6.47 P-D (r267): offline NaN-canary of the
# r266 engine MX-KV decode onset, inside cert29 with the IMAGE'S OWN
# bytes (NO pip install, NO engine boot — unlike r242, which force-
# reinstalled the build5 wheel: here the whole point is fidelity to
# what the r266 legs ran; the r266 runner sets no ARC_MXKV_OVERLAY,
# so /opt/venv/.../site-packages IS the engine posture and the canary
# resolves mx_kv from there, md5-provenance-printed at start).
#
# The evidence chain this canary decides:
#   r266 c3/c4/c5 (KV-MX, all three fmts) are content-RED with a
#   token-exact onset — token 1 correct, 'our' attractor from token 2,
#   324/323/324 x HTTP-400 nan-JSON, prefill NLL CLEAN (0.9131 vs b2
#   0.9151), zero DEVICE_LOST; weight-MX at KV fp8 is clean (w1/w2) ->
#   fmt-independent, confined to the KV-MX store/decode path.  The
#   offline authorities (r242 D=512p64, r245a D=512p128, r243 W=1024
#   windowed) are all GREEN but hand-quantized their caches and never
#   ran the Triton store as producer interleaved with decodes, never
#   at the gemma cells' divergent k/v strides, and mxfp8 x W=1024
#   long ctx was never run (r243 validated mxfp4 only).
# Faces: stepwise (THE onset face: bulk T=1024 prefill rows, then 24
#   incremental 1-row decode-step stores each immediately decoded
#   over all rows + the pointed-row bit-law audit K AND V vs mx_quant
#   + sentinel mutation), widewin (mxfp8/mxk8v4/mxfp4 at ctx
#   8192/32768/40960, W=1024, d_blind), autosplit (6 interleaved
#   requests, auto/forced8/s1 split plans, both cells), uninit (0xFF
#   + empty pool decode census — EVIDENCE for the wrong-row NaN
#   signature), ctrl (the certified qwen p16 lane regression guard).
# Cells: slide D=256 page64 HK=2 G=8 W=1024; glob D=512 page128 HK=2
#   G=8 windowless; ctrl D=256 page16 HK=2 G=6.
# Gates: non-finite = FAIL (the NaN canary); else the r198 face law
#   (d<=2e-3 or bf16-rounded 1-ulp; cos>=0.99999); store-audit is the
#   bit law (same source, same decode_kv -> identical values, tol
#   1e-6).  PASS = canary clean -> the engine defect is WIRING
#   (metadata values / XPU-graph capture-replay), pointed by batch-2's
#   e1/e2 eager pair.  FAIL = reproduced offline -> store/kernel math.
# W14 quiesce before, clean-recipe trap restore after (r262 law; the
# batch-1 r266 restore failure class — w14_boot fullres + docker start
# cannot clear the JIT-warmup hang).  Artifacts ->
# /root/build/p647/r267_canary/{out,logs}.  Exit 0 = canary PASS.
# Launch (ship + launch in SEPARATE ssh calls, md5-verified ship):
#   mkdir -p /root/build/p647/r267_canary/logs
#   nohup setsid bash /root/build/r267/r267_nan_canary.sh \
#     > /root/build/p647/r267_canary/logs/launch.log 2>&1 &
set -uo pipefail
R=/root/build
P=$R/p647/r267_canary
S=$R/r267
IMG=vllm-arc:v0.30.0-b70-cert29
CNAME=arc-r267

[ -f "$S/p267_nan_canary.py" ] || { echo "R267_NO_HARNESS -- ship first"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R267_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R267_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R267_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R267_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  bash $R/r262_w14_restore.sh > $P/logs/w14_restore_r267.log 2>&1
  rc=$?
  grep -E "^W14_" $P/logs/w14_restore_r267.log | tail -3 || true
  echo "W14_RESTORE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) log=$P/logs/w14_restore_r267.log"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out
echo "R267_BEGIN $(date -u +%Y-%m-%dT%H:%M:%SZ) image=$IMG harness_md5=$(md5sum $S/p267_nan_canary.py | cut -d' ' -f1)"

echo "=== [r267] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R267_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r267] offline canary in $IMG (image's own bytes) $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$S":/ship:ro -v "$P/out":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /ship && /opt/venv/bin/python p267_nan_canary.py /out" \
  > $P/logs/r267.out 2>&1
rc=$?
echo "=== [r267] canary exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[p267\]|^  ok |^  FAIL|^  note" $P/logs/r267.out > $P/out/r267_summary.txt || true
cat $P/out/r267_summary.txt

echo "R267_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
