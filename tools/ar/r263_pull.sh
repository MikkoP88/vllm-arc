#!/bin/bash
# r263_pull.sh -- run LOCALLY (git-bash) after the r263 cert29 rebuild
# completes on the rig (R263_IMAGE_OK in /root/build/logs/
# r263_build.log). Collects rig-side into /root/build/p647/r263_build/:
#   build.log           the run-2 log (the coherent one: R263_IMAGE_OK)
#   build_run1.log      the run-1 log (valid image, false-negative final
#                       check -- the runner read the manifest's LAST line)
#   IMAGE_MANIFEST      the rebuilt image's manifest file, verbatim
#   manifest_tail.txt   its last five lines
#   images.txt          the cert28 / cert29 / cert29-rc1 image ids
#   stage_md5.txt       the six staged vllm_gaudi files (md5, far side)
#   script_md5.txt      the build runner shipped to the rig
# and pulls that dir into results/2026-11-27-p647-r263-cert29-rebuild/.
# Redacts rig + client host literals byte-exact over the TEXT artifacts
# (perl, NOT sed -- MSYS sed injects CRLF). Writes the data-file
# SHA256SUMS (VERDICT.md is convention-excluded, written after pull).
# Idempotent (VERDICT.md guard). Pull-only. Read-only on the rig apart
# from the collection dir: one short CPU-only `docker run --rm` (cat the
# manifest) -- no GPU, no ports, so it is safe beside a live gate leg.
# Rig address from the caller's environment (law 8: never in-tree):
#   ARC_RIG=<user>@<rig-host> tools/ar/r263_pull.sh
set -euo pipefail
export SSH_ASKPASS=/tmp/askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0
RIG=${ARC_RIG:?set ARC_RIG=<user>@<rig-host> before running}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)/results

B=2026-11-27-p647-r263-cert29-rebuild
SRC=/root/build/p647/r263_build
IMG=vllm-arc:v0.30.0-b70-cert29

if [ -f "$ROOT/$B/VERDICT.md" ]; then
  echo "== $B already banked (VERDICT present) -- nothing to do"
  exit 0
fi

echo "== verifying the rebuild is complete (R263_IMAGE_OK)"
OK=$(ssh "$RIG" 'grep -c "^R263_IMAGE_OK " /root/build/logs/r263_build.log 2>/dev/null || true')
[ "${OK:-0}" -ge 1 ] || { echo "R263_NOT_BUILT (no R263_IMAGE_OK line yet) -- wait"; exit 2; }

echo "== collecting rig-side into $SRC"
ssh "$RIG" "set -e; mkdir -p $SRC
  cp /root/build/logs/r263_build.log $SRC/build.log
  cp /root/build/logs/r263_build_run1.log $SRC/build_run1.log
  docker run --rm --entrypoint cat $IMG /etc/vllm-arc/IMAGE_MANIFEST > $SRC/IMAGE_MANIFEST
  tail -5 $SRC/IMAGE_MANIFEST > $SRC/manifest_tail.txt
  docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}} {{.Size}}' \
    | grep -E '^vllm-arc:v0\.30\.0-b70-cert(28|29|29-rc1) ' > $SRC/images.txt
  ( cd /root/build/cert29stage_r263 && md5sum envs.py platform.py lmc_mx_boot_gate.py vllm_v1_adapter.py engine.py native_decode.py ) > $SRC/stage_md5.txt
  md5sum /root/build/r263_build_cert29.sh > $SRC/script_md5.txt
  wc -l $SRC/* | tail -1"

echo "== pulling $SRC -> $B"
mkdir -p "$ROOT/$B"
scp -r -q "$RIG:$SRC/." "$ROOT/$B/"
ls -la "$ROOT/$B" | sed -n '1,20p'

cd "$ROOT/$B"
echo "== byte-exact redaction (text artifacts): $B"
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -print0 |
  xargs -0 perl -pi -e 's/10\.20\.3\.65/10.20.3.REDACTED/g; s/10\.100\.8\.6/10.100.8.REDACTED/g'
find . -type f ! -name "SHA256SUMS*" ! -name "VERDICT.md" -exec sha256sum {} \; | sort -k2 > SHA256SUMS
echo "banked: $(find . -type f | wc -l) files -> $ROOT/$B"

echo "== semantic audit: the rebuild"
grep -nE "^R263_|^CERT29_WHEEL" build.log | cut -c1-160 || true
echo "-- run 1 (superseded log, false-negative final check):"
grep -nE "^R263_|^CERT29_WHEEL" build_run1.log | cut -c1-160 || true
echo "-- manifest tail:"; cut -c1-160 manifest_tail.txt
echo "-- images:"; cat images.txt
echo "-- stage md5s:"; cat stage_md5.txt
echo "== address audit over the bank (must be empty):"
grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}' . 2>/dev/null | grep -v '^./SHA256SUMS$' || echo "   clean"
