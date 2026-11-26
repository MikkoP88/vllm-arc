#!/bin/bash
# r247_mx_surface.sh -- P6.47 P-E validation leg (probe class): the
# pinned-stack MX surface census (p247_mx_surface_probe.py) inside
# cert28. CPU-SAFE by design: no --device, no host network, no W14
# quiesce (the census imports torch/torchao and introspects APIs --
# no device init, no engine boot, no port binds; W14 stays up
# untouched -- the rig laws govern engine legs, and this probe never
# touches the GPU). Banks the census to /root/build/p247/out.
# Ship step (separate ssh call, md5-verified both sides) stages
# the probe at /root/build/p247/stage BEFORE this runner starts:
#   p247_mx_surface_probe.py <ship-time md5>
# Refuses: cert28 image present, staged file md5, no arc leg in
# flight (name-safety only). Exit 0 = CENSUS_DONE.
set -uo pipefail
R=/root/build
P=$R/p247
SP=$P/stage
IMG=vllm-arc:v0.30.0-b70-cert28
MD5_PROBE=78ad4a49b3d3cfb97403c62c9e2729a9
declare -A MD5=( [p247_mx_surface_probe.py]=$MD5_PROBE )

for f in p247_mx_surface_probe.py; do
  [ -f "$SP/$f" ] || { echo "R247_NO_STAGED_FILE $f (run the ship step)"; exit 1; }
  got=$(md5sum "$SP/$f" | cut -d" " -f1)
  [ "$got" = "${MD5[$f]}" ] || { echo "R247_MD5_MISMATCH $f $got != ${MD5[$f]}"; exit 1; }
done
echo "R247_STAGE_OK md5s verified"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R247_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
  echo "R247_BUSY (arc leg in flight -- stopped corpses are expected)"; exit 2
fi

mkdir -p $P/out $P/logs
echo "=== [r247] MX surface census in $IMG $(date +%H:%M:%S) (W14 untouched) ==="
docker run --rm --name arc-r247c \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$SP/p247_mx_surface_probe.py":/root/p247_mx_surface_probe.py:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/python /root/p247_mx_surface_probe.py" \
  > $P/logs/r247.out 2>&1
rc=$?
echo "=== [r247] census exit rc=$rc $(date +%H:%M:%S) ==="
cat $P/logs/r247.out
cd $P/out
sha256sum $P/logs/r247.out > SHA256SUMS 2>/dev/null || true
echo "R247_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
