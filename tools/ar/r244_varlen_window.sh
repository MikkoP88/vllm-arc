#!/bin/bash
# r244_varlen_window.sh -- P6.47 P-C2 (r244): offline XPU validation of
# the TQ-LANE PREFILL sliding-window fix (the python half of P-C2) on
# the STOCK cert28 image venv — no arc7mx wheel: the varlen prefill op
# is untouched upstream surface (only the MX paged-decode kernel needed
# the P-C2 twin). The probe imports flash_attn_varlen_func by the
# engine's exact chain (fa_utils -> _xpu_ops -> wheel interface) and
# gates the window pair (W-1, 0) against fp64 window-masked references:
# stock (windowless regression guard), exact (W=64 binding + the
# d_blind window-blind discriminator), nonbinding (n <= W twin law),
# boundary (63/64/65 at W=64; 1023/1024/1025 at W=1024), telemetry
# (ctx 8192 windowed vs windowless). Gates: the p244 probe's face law
# (d<=2e-3 or bf16-rounded 1-ulp match; cos>=0.99999; explicit
# blindness gate). SAFE to run while the vxk-arc3-builder compile is
# live (CPU-only foreign container, untouched). W14 quiesce before,
# trap restore after (rig law; fullres recipe). Artifacts ->
# /root/build/p244/{out,logs}. Exit 0 = r244 offline PASS.
set -uo pipefail
R=/root/build
P=$R/p244
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert28
CNAME=arc-r244

[ -f "$W/tools/ar/p147_mxkv/p244_varlen_window.py" ] || { echo "R244_NO_HARNESS -- ship first"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R244_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R244_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R244_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R244_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (docker start + serve_only can never
  # restore the actor-recreated radiance-test -- r185b/r186 law)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p244.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r244] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R244_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r244] stock-venv varlen window harness in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p244_varlen_window.py /out" \
  > $P/logs/r244.out 2>&1
rc=$?
echo "=== [r244] harness exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^  ok |^  FAIL|^  note|VERDICT|telemetry/" $P/logs/r244.out > $P/out/r244_summary.txt || true
cat $P/out/r244_summary.txt

cd $P/out
sha256sum r244_summary.txt p244_varlen.json $P/logs/r244.out 2>/dev/null > SHA256SUMS || true
echo "R244_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
