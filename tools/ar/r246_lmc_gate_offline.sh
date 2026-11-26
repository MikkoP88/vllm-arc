#!/bin/bash
# r246_lmc_gate_offline.sh -- P6.47 P-D validation leg (probe class):
# the LMC x MX boot gate (patches/lmc_mx_boot_gate.py, branch
# feat/arc-pd-lmc-mx-boot-gate) proven inside cert28 with the three
# P-D files overlaid over site-packages. NO engine boot -- the gate
# fires at connector construction, so the probe (p246_lmc_gate_probe.py,
# mounted read-only) drives the REAL wiring: platform
# _install_wheel_hooks -> meta-path finder -> factory import ->
# exec-wrap -> KVConnectorFactory.create_connector. Ten faces: hook
# fired; refusal over mxfp4/mxfp8/mxk8v4 (ARC_LMC_MX_REFUSED); the real
# registry resolves LMCacheConnectorV1 LMC-named (detection basis,
# resolve-only); a non-LMC connector over MX passes (scope); LMC over
# checkpoint_fp8 passes (the certified lane untouched); the
# VLLM_ARC_LMC_MX_GATE=0 bypass logs gate=bypassed; the stock
# None-config ValueError intact; idempotent second _install.
# The serve-path proof (gate files overlaid, real MX boot + faces)
# rides the P-D e2e leg (r246b) over the serve script's ARC_PD_*
# mounts (applied 2026-11-26 13:09Z).
# Ship step (separate ssh call, md5-verified both sides) stages the
# four files at /root/build/p246/stage BEFORE this runner starts:
#   platform.py            0733f0b8c7dd2fb89422bafa56ec35f5
#   envs.py                31ca4ac7cfcd996aed6446d4b24c2bcd
#   lmc_mx_boot_gate.py    1267b826a214953c9e7130e804f615f5
#   p246_lmc_gate_probe.py 05c0385bd08562acc77549bfe64545a5 (rev 3)
# RUN HISTORY: run 1 (rev 1, 13:10:59Z) gate core green on the real
# factory (refusal x3 dtypes, wiring, idempotence, stock behavior);
# 4 FAILs = stub gaps (scheduler_config, kv_connector_module_path).
# run 2 (rev 2, 13:14:50Z) 7/10; last stub gap = engine_id. rev 3
# adds it. Logs preserved in the r246 rig logs.
# (branch blob md5s at 71ff9cb4). W14 quiesce before, trap restore
# after (r24x law). Artifacts -> /root/build/p246/{out,logs}.
# Exit 0 = P-D probe PASS.
set -uo pipefail
R=/root/build
P=$R/p246
SP=$P/stage
IMG=vllm-arc:v0.30.0-b70-cert28
CNAME=arc-r246
declare -A MD5=( \
  [platform.py]=0733f0b8c7dd2fb89422bafa56ec35f5 \
  [envs.py]=31ca4ac7cfcd996aed6446d4b24c2bcd \
  [lmc_mx_boot_gate.py]=1267b826a214953c9e7130e804f615f5 \
  [p246_lmc_gate_probe.py]=05c0385bd08562acc77549bfe64545a5 )

for f in platform.py envs.py lmc_mx_boot_gate.py p246_lmc_gate_probe.py; do
  [ -f "$SP/$f" ] || { echo "R246_NO_STAGED_FILE $f (run the ship step)"; exit 1; }
  got=$(md5sum "$SP/$f" | cut -d" " -f1)
  [ "$got" = "${MD5[$f]}" ] || { echo "R246_MD5_MISMATCH $f $got != ${MD5[$f]}"; exit 1; }
done
echo "R246_STAGE_OK md5s verified"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R246_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R246_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R246_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R246_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (docker start + serve_only can never
  # restore the actor-recreated radiance-test -- r185b/r186 law)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p246.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r246] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R246_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r246] probe in $IMG + P-D overlay $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$SP/platform.py":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py:ro \
  -v "$SP/envs.py":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/envs.py:ro \
  -v "$SP/lmc_mx_boot_gate.py":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmc_mx_boot_gate.py:ro \
  -v "$SP/p246_lmc_gate_probe.py":/root/p246_lmc_gate_probe.py:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/python /root/p246_lmc_gate_probe.py" \
  > $P/logs/r246.out 2>&1
rc=$?
echo "=== [r246] probe exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^== p246|VERDICT" $P/logs/r246.out > $P/out/r246_summary.txt || true
cat $P/out/r246_summary.txt

cd $P/out
sha256sum r246_summary.txt $P/logs/r246.out 2>/dev/null > SHA256SUMS || true
echo "R246_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
