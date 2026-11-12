#!/bin/bash
# r151_mxkv_dumpdiff.sh -- P6.47 leg B: the spec-vs-eager KV-pool
# ground-truth differential (the instrument that closes the corruption
# hunt: every self-consistency xcheck passed while text corrupted, so
# the wrong bytes must be IN the pool — only an eager-run byte
# comparison can name them). Boots the r149 flow twice (SPEC=1 then
# SPEC=0, mxfp8, TRACE off, DUMP=1), fetches each boot's post-battery
# pool dump, then byte-diffs them in a cert23 container
# (tools/ar/p148_mxkv/dump_diff.py).
# Artifacts -> /root/build/p151. Exit 0 = differential produced.
set -uo pipefail
R=/root/build
P=$R/p151
W=$R/p148/work

mkdir -p $P/logs $P/out

echo "=== [r151] phase 1: SPEC=1 boot + dump $(date +%H:%M:%S) ==="
env FMT=mxfp8 SPEC=1 GRAPH=0 MAXLEN=32768 TRACE=0 DUMP=1 TAG=trace12spec \
  bash $R/p149_boot.sh > $P/logs/spec.out 2>&1
rc1=$?
grep -E "PROMPT_OK|POOL_DUMP|R149_" $P/logs/spec.out || true
rm -rf $P/out/spec_dumps; cp -r $R/p149/trace12spec/dumps $P/out/spec_dumps 2>/dev/null || rc1=99

if [ $rc1 -ne 0 ]; then
  echo "R151_FAIL_SPEC_PHASE rc=$rc1"; exit 1
fi

echo "=== [r151] phase 2: SPEC=0 boot + dump $(date +%H:%M:%S) ==="
env FMT=mxfp8 SPEC=0 GRAPH=0 MAXLEN=32768 TRACE=0 DUMP=1 TAG=trace12eager \
  bash $R/p149_boot.sh > $P/logs/eager.out 2>&1
rc2=$?
grep -E "PROMPT_OK|POOL_DUMP|R149_" $P/logs/eager.out || true
rm -rf $P/out/eager_dumps; cp -r $R/p149/trace12eager/dumps $P/out/eager_dumps 2>/dev/null || rc2=99

if [ $rc2 -ne 0 ]; then
  echo "R151_FAIL_EAGER_PHASE rc=$rc2"; exit 1
fi

echo "=== [r151] differential $(date +%H:%M:%S) ==="
docker run --rm --name arc-p151a \
  -e PYTHONPATH=/work \
  -v "$W":/work -v "$P/out":/out \
  --entrypoint /bin/bash \
  vllm-arc:v0.30.0-b70-cert23 -c \
  "cd /work && /opt/venv/bin/python tools/ar/p148_mxkv/dump_diff.py /out/spec_dumps /out/eager_dumps" \
  > $P/out/dump_diff.txt 2>&1
cat $P/out/dump_diff.txt

cd $P/out && find spec_dumps eager_dumps -type f | sort | xargs sha256sum > SHA256SUMS 2>/dev/null || true
sha256sum dump_diff.txt >> SHA256SUMS 2>/dev/null || true
grep -q "VERDICT" $P/out/dump_diff.txt && echo "R151_DIFF_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
