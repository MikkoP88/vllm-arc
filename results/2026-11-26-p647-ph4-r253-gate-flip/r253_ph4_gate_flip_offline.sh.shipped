#!/bin/bash
# r253_ph4_gate_flip_offline.sh -- P6.47 P-H4 validation leg (probe
# class): the posture-aware LMC x MX gate flip proven inside cert28
# with the five P-H4 files overlaid over site-packages. NO engine
# boot -- the gate fires at connector construction, so the probe
# (tools/ar/p147_mxkv/p253_ph4_gate_flip_probe.py, mounted
# read-only) drives the REAL wiring (the p246 pattern, 11/11 at
# P-D): platform _install_wheel_hooks -> meta-path finder -> factory
# import -> exec-wrap -> KVConnectorFactory.create_connector.
# 15 faces: the real-registry bootstrap (run-1 lesson: the kit path
# must NOT be imported bare -- the p41b exec-wrap installs the lmcint
# src-override finder at registry resolution, and the vendored
# lmcache_integration/__init__ dies on lmcache 0.5.5 without it),
# interop-live pass-through x3 dtypes (gate=interop-live,
# the r249 packed-row interop admitted with telemetry), stock-adapter
# refusal x3 + the legacy kv_cache_dtype surface (the P-D posture
# stands when ARC_LMC_MX_INTEROP is absent -- pre-r249 installs),
# the real registry resolve, non-LMC-over-MX scope, LMC-over-non-MX
# lawful lane, the knob bypass with interop=absent noted, stock
# None-config behavior, idempotent second _install. The probe
# toggles the capability marker ON THE REAL MODULE to drive both
# postures in one container.
# Ship step (separate ssh call, md5-verified both sides) stages the
# five files at /root/build/p253_stage BEFORE this runner starts:
#   platform.py            0733f0b8c7dd2fb89422bafa56ec35f5 (P-D blob)
#   envs.py                31ca4ac7cfcd996aed6446d4b24c2bcd (P-D blob)
#   lmc_mx_boot_gate.py    04408d2e12cbdd8587cac4633624b7bc (P-H4 flip)
#   vllm_v1_adapter.py     7ca7cff4e1e04d4e0ea73573b1234664 (P-H2 fix +
#                          the ARC_LMC_MX_INTEROP capability marker)
#   p253_ph4_gate_flip_probe.py 1561daf77a0ec7009561d2aa3c1ff6f5 (rev 2:
#                          R_bootstrap face -- resolution before kit)
# W14 quiesce before, trap restore after (r24x law; stop+start +
# serve_only -- the r252-proven restore for a stop-quiesced W14).
# Artifacts -> /root/build/p253/{out,logs}. Exit 0 = P-H4 probe PASS.
set -uo pipefail
R=/root/build
P=$R/p253
SP=$R/p253_stage
IMG=vllm-arc:v0.30.0-b70-cert28
CNAME=arc-r253
declare -A MD5=( \
  [platform.py]=0733f0b8c7dd2fb89422bafa56ec35f5 \
  [envs.py]=31ca4ac7cfcd996aed6446d4b24c2bcd \
  [lmc_mx_boot_gate.py]=04408d2e12cbdd8587cac4633624b7bc \
  [vllm_v1_adapter.py]=7ca7cff4e1e04d4e0ea73573b1234664 \
  [p253_ph4_gate_flip_probe.py]=1561daf77a0ec7009561d2aa3c1ff6f5 )

for f in platform.py envs.py lmc_mx_boot_gate.py vllm_v1_adapter.py \
         p253_ph4_gate_flip_probe.py; do
  [ -f "$SP/$f" ] || { echo "R253_NO_STAGED_FILE $f (run the ship step)"; exit 1; }
  got=$(md5sum "$SP/$f" | cut -d" " -f1)
  [ "$got" = "${MD5[$f]}" ] || { echo "R253_MD5_MISMATCH $f $got != ${MD5[$f]}"; exit 1; }
done
echo "R253_STAGE_OK md5s verified"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R253_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R253_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R253_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p253.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r253] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R253_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r253] probe in $IMG + P-H4 overlay $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$SP/platform.py":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py:ro \
  -v "$SP/envs.py":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/envs.py:ro \
  -v "$SP/lmc_mx_boot_gate.py":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmc_mx_boot_gate.py:ro \
  -v "$SP/vllm_v1_adapter.py":/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py:ro \
  -v "$SP/p253_ph4_gate_flip_probe.py":/root/p253_ph4_gate_flip_probe.py:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/python /root/p253_ph4_gate_flip_probe.py" \
  > $P/logs/r253.out 2>&1
rc=$?
echo "=== [r253] probe exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^== p253|VERDICT" $P/logs/r253.out > $P/out/r253_summary.txt || true
cat $P/out/r253_summary.txt

cd $P/out
sha256sum r253_summary.txt $P/logs/r253.out 2>/dev/null > SHA256SUMS || true
echo "R253_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
