#!/bin/bash
# r148_mxkv_legb1.sh -- P6.47 leg B part 1: in-container admission +
# routing probe for MX KV dtypes (p148_admission.py, one-shot python,
# NO serve). Gates: platform, literal, admit, dtype_map, mode_alias,
# routing, stamp, slot (264/136), page, tq_spec_fix carry-over,
# engine_import. W14 quiesce before, trap restore after (rig law).
# Files arrive via the p148 ship step (md5-verified tar); this script
# only runs. Artifacts -> /root/build/p148/{out,logs}. Exit 0 = PASS.
set -uo pipefail
R=/root/build
P=$R/p148
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert23

w14_restore() {
  docker rm -f arc-p148a >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p148a.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r148:legB1] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R148_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p148a >/dev/null 2>&1 || true

echo "=== [r148:legB1] admission probe in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p148a \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e PYTHONPATH=/work \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p148_mxkv/p148_admission.py" \
  > $P/logs/legB1.out 2>&1
rc=$?
echo "=== [r148:legB1] probe exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^VERDICT" $P/logs/legB1.out > $P/out/legB1_summary.txt || true
cat $P/out/legB1_summary.txt

cd $P/out
sha256sum legB1_summary.txt $P/logs/legB1.out 2>/dev/null > SHA256SUMS || true
echo "R148_LEGB1_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
