#!/bin/bash
# r152_mxkv_layer0.sh -- P6.47 leg B: layer-pinned pool time series.
# r151's "wholesale clobber" pair was INVALID: the two spec verify
# payloads had different impl ids = different layers (each layer owns a
# separate pool tensor at the same slot ids; ~97% byte differ is the
# EXPECTED cross-layer baseline). This leg re-runs the two DUMP boots
# with the r152 instrument (ARC_MXKV_DUMP_ORD=0: only the first
# attention impl consumes sentinels; payloads carry "ord") and runs the
# series comparator:
#   in-boot  A(prefill)-vs-B(deep verify) at prompt positions -- bytes
#            there must never change after prefill; LARGE = a
#            mid-generation writer rewrites live prompt K/V;
#   cross-boot spec-verify vs eager-prefill (both ord 0) + decoded
#            cosine (content-swap vs corrupted transform).
# Artifacts -> /root/build/p152. Exit 0 = series produced.
set -uo pipefail
R=/root/build
P=$R/p152
W=$R/p148/work

mkdir -p $P/logs $P/out

echo "=== [r152] engine md5 $(date +%H:%M:%S) ==="
md5sum $W/vllm_gaudi/xpu/patches/mx_kv/engine.py $W/tools/ar/p148_mxkv/dump_layer0_series.py
grep -q "ARC_MXKV_DUMP_ORD" $W/vllm_gaudi/xpu/patches/mx_kv/engine.py || { echo "R152_FAIL_STALE_ENGINE"; exit 1; }
grep -q "ARC_MXKV_DUMP_FILTER" $W/vllm_gaudi/xpu/patches/mx_kv/engine.py || { echo "R152_FAIL_STALE_ENGINE_NOFILTER"; exit 1; }

echo "=== [r152] phase 1: SPEC=1 layer0 dump boot $(date +%H:%M:%S) ==="
env FMT=mxfp8 SPEC=1 GRAPH=0 MAXLEN=32768 TRACE=0 DUMP=1 TAG=trace18spec \
  bash $R/p149_boot.sh > $P/logs/spec.out 2>&1
rc1=$?
grep -E "PROMPT_OK|POOL_DUMP|R149_" $P/logs/spec.out || true
rm -rf $P/out/spec_dumps; cp -r $R/p149/trace18spec/dumps $P/out/spec_dumps 2>/dev/null || rc1=99
if [ $rc1 -ne 0 ]; then echo "R152_FAIL_SPEC_PHASE rc=$rc1"; exit 1; fi

echo "=== [r152] phase 2: SPEC=0 layer0 dump boot $(date +%H:%M:%S) ==="
env FMT=mxfp8 SPEC=0 GRAPH=0 MAXLEN=32768 TRACE=0 DUMP=1 TAG=trace18eager \
  bash $R/p149_boot.sh > $P/logs/eager.out 2>&1
rc2=$?
grep -E "PROMPT_OK|POOL_DUMP|R149_" $P/logs/eager.out || true
rm -rf $P/out/eager_dumps; cp -r $R/p149/trace18eager/dumps $P/out/eager_dumps 2>/dev/null || rc2=99
if [ $rc2 -ne 0 ]; then echo "R152_FAIL_EAGER_PHASE rc=$rc2"; exit 1; fi

echo "=== [r152] layer0 series $(date +%H:%M:%S) ==="
docker run --rm --name arc-p152a \
  -e PYTHONPATH=/work \
  -v "$W":/work -v "$P/out":/out \
  --entrypoint /bin/bash \
  vllm-arc:v0.30.0-b70-cert23 -c \
  "cd /work && /opt/venv/bin/python tools/ar/p148_mxkv/dump_layer0_series.py /out/spec_dumps /out/eager_dumps" \
  > $P/out/layer0_series.txt 2>&1
cat $P/out/layer0_series.txt

cd $P/out && find spec_dumps eager_dumps -type f | sort | xargs sha256sum > SHA256SUMS 2>/dev/null || true
sha256sum layer0_series.txt >> SHA256SUMS 2>/dev/null || true
grep -q "A-vs-B" $P/out/layer0_series.txt && echo "R152_SERIES_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
