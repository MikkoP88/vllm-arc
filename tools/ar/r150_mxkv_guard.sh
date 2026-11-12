#!/bin/bash
# r150_mxkv_guard.sh -- P6.47 leg B: negative-slot guard proof for
# triton_mx_store (the spec-corruption root-cause fix). Runs the guard
# probe (3 on-device assertions, tools/ar/p148_mxkv/store_guard_probe.py)
# + re-runs the leg-B1 admission probe (wiring must be unchanged by the
# kernel edit) in one cert23 container against the /work shadow. W14
# quiesce before, restore after (rig law).
# Artifacts -> /root/build/p150/{out,logs}. Exit 0 = GUARD-PASS +
# admission unchanged.
set -uo pipefail
R=/root/build
P=$R/p150
W=$R/p148/work
IMG=vllm-arc:v0.30.0-b70-cert23

w14_restore() {
  docker rm -f arc-p150a >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p150.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r150] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5
docker rm -f arc-p150a >/dev/null 2>&1 || true

echo "=== [r150] guard probe + admission re-run $(date +%H:%M:%S) ==="
docker run --rm --name arc-p150a \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e PYTHONPATH=/work \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p148_mxkv/store_guard_probe.py > /out/guard_probe.out 2>&1; rc1=\$?; /opt/venv/bin/python tools/ar/p148_mxkv/p148_admission.py > /out/admission_rerun.txt 2>&1; rc2=\$?; echo rc1=\$rc1 rc2=\$rc2; exit \$((rc1+rc2))" \
  > $P/logs/guard.out 2>&1
rc=$?
grep "^rc1=" $P/logs/guard.out || true
grep -E "^guard_|^VERDICT" $P/out/guard_probe.out 2>/dev/null || true
echo "admission faces ok: $(grep -cE '^\[face:.*ok' $P/out/admission_rerun.txt 2>/dev/null || echo 0)"
grep -E "^\[face:.*FAIL|^VERDICT" $P/out/admission_rerun.txt 2>/dev/null || true

cp $P/logs/guard.out $P/out/ 2>/dev/null || true
cd $P/out && sha256sum guard_probe.out admission_rerun.txt guard.out > SHA256SUMS 2>/dev/null || true
if [ $rc -eq 0 ] && grep -q "VERDICT GUARD-PASS" $P/out/guard_probe.out 2>/dev/null; then
  echo "R150_GUARD_PASS $(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
  echo "R150_GUARD_FAIL rc=$rc"
fi
[ $rc -eq 0 ]
