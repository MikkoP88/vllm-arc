#!/bin/bash
# r196_build_cert25zc.sh -- P6.47: build vllm-arc:v0.30.0-b70-cert25zc
# (cert25zb + the Phase 7 closing rung: TILE_FLOOR launcher default 4->2
# + the G-adaptive mxfp4 verify-decoder dispatch + the dormant v7
# research kernel + the one-time decoder dispatch log). The ctx tree
# (engine.py / triton_mx_decode.py / __init__.py from fork HEAD) is
# staged by scp before this runs; the Dockerfile is written here so the
# build face ships as one tracked file. Refuses unless the staged tree
# carries the content markers AND the evidence banks PASSed (r193 sweep
# / r195 e2e confirmation / r188 B1 knee). FORK_SHA must be passed
# explicitly (cert-release-sequence law: manifest fork_sha = the content
# commit -- dd747150 for this line).
set -euo pipefail
R=/root/build
CTX=$R/cert25zcctx
FORK_SHA=${1:?usage: r196_build_cert25zc.sh <FORK_SHA>}

SP=$CTX
# --- staged-tree content refuses ---
grep -q 'os.environ.get("ARC_MXKV_V5_TFLOOR", "2") or 2' "$SP/triton_mx_decode.py" \
  || { echo "R196_STALE_TREE (TILE_FLOOR launcher default not 2)"; exit 1; }
grep -q 'n_tiles // TILE_FLOOR' "$SP/triton_mx_decode.py" \
  || { echo "R196_STALE_TREE (no TILE_FLOOR-floored split rule)"; exit 1; }
grep -q 'os.environ.get("ARC_MXKV_V7", "0") == "1"' "$SP/triton_mx_decode.py" \
  || { echo "R196_STALE_TREE (v7 kernel not default-off)"; exit 1; }
grep -q "_V5_G_MIN" "$SP/engine.py" \
  || { echo "R196_STALE_TREE (no G-adaptive knee)"; exit 1; }
grep -q '"g-adaptive"' "$SP/engine.py" \
  || { echo "R196_STALE_TREE (no g-adaptive log label)"; exit 1; }
# zb inheritance must be intact in the staged tree
grep -q 'ARC_MXKV_K2", "1") != "0"' "$SP/triton_mx_decode.py" \
  || { echo "R196_STALE_TREE (K2 not default-on)"; exit 1; }
grep -q 'ARC_MXKV_K3", "1") != "0"' "$SP/triton_mx_decode.py" \
  || { echo "R196_STALE_TREE (K3 not default-on)"; exit 1; }
grep -q "mx_decode_rows_dot" "$SP/__init__.py" \
  || { echo "R196_STALE_TREE (no package export)"; exit 1; }
[ -f "$SP/engine.py" ] && [ -f "$SP/triton_mx_decode.py" ] && [ -f "$SP/__init__.py" ] \
  || { echo "R196_MISSING staged py files"; exit 1; }
# --- evidence refuses ---
grep -q "TFL_PASS" "$R/p647/tflA26/tfloor.txt" \
  || { echo "R196_NO_SWEEP_EVIDENCE (r193 bank)"; exit 1; }
grep -q "ACCEPT_KNEE_PASS" "$R/p647/tfe2A26/tfe2.txt" \
  || { echo "R196_NO_E2E_EVIDENCE (r195 bank)"; exit 1; }
grep -q "ACCEPT_KNEE_PASS" "$R/p647/flipA25/flipv5.txt" \
  || { echo "R196_NO_B1_EVIDENCE (r188 bank)"; exit 1; }
docker image inspect vllm-arc:v0.30.0-b70-cert25zb >/dev/null 2>&1 \
  || { echo "R196_NO_BASE (cert25zb)"; exit 1; }

mkdir -p "$CTX"
cat > "$CTX/Dockerfile" <<'DOCKERFILE_EOF'
# SPDX-License-Identifier: Apache-2.0
# Dockerfile.cert25zc -- P6.47 cert25zc image: cert25zb + the Phase 7
# closing rung. (1) TILE_FLOOR launcher default 4 -> 2: deep splits
# REMOVE serial bt->slot->gather chain tiles rather than pipelining them
# (r192 toolchain verdict: Triton-on-XPU sinks gather loads to use
# sites; program-order prefetch not honored; num_stages no-op -- the
# in-language prefetch toolbox is empty). r193 sweep banked tf2/sp256 as
# the universal kernel winner; r195 e2e confirmed on cert25zb with the
# single env delta: m8q n1 16K 75->70, 32K 106->95 (-10.4%); mkq -4..-2
# at 16-32K; m4q byte-identical (v2 lane, TFLOOR-exempt by design);
# ACCEPT_KNEE_PASS x3, 24/24 CONC_OK. The kernel constexpr stays 4
# (certified-signature freeze); the ARC_MXKV_V5_TFLOOR env default is
# the flip site. (2) G-adaptive mxfp4 verify-decoder dispatch
# (engine.py): v2 below the r188-B1 knee, v5 at/above
# (ARC_MXKV_V5_G_MIN, default 8, mxfp4-only). B1 banked the blanket
# flip refuted (n1 +8..14 ms -- the v2 batch grid out-parallelizes v5's
# at G=1) but n8 long-ctx decisive (32K ss 1978->1703, -14%; aggregate
# p50 1551->696). Capture-lawful: the split-batch request count is a
# host-known shape at plan-build time and the plan cache freezes the
# choice per shape. (3) The BN law stands post-K2/K3: r194 re-sweep
# refuted BN=64 (0.36-0.69x; SMEM doubles -> occupancy halves) and
# BN=128 is SMEM-impossible (147456 B required vs 131072 limit).
# (4) The v7 raw-tile double-buffer kernel ships DORMANT
# (ARC_MXKV_V7=1 to enable; r192 perf-refuted 0.61-0.95x, bitwise-clean
# d=0 -- kept for the record). One-time decoder dispatch log line
# (fmt+decoder+source: env-override/env-flag/g-adaptive/format-default).
# v2/v3/v4/v6 lanes otherwise byte-identical to cert25zb. Built by
# r196_build_cert25zc.sh with FORK_SHA=<content commit>.
FROM vllm-arc:v0.30.0-b70-cert25zb

ARG FORK_SHA=unknown
ARG BUILT=unknown

COPY engine.py /tmp/mx_kv_engine.py
COPY triton_mx_decode.py /tmp/mx_kv_triton_mx_decode.py
COPY __init__.py /tmp/mx_kv_init.py

RUN set -euxo pipefail; \
    SP=/opt/venv/lib/python3.12/site-packages/vllm_gaudi; \
    cp /tmp/mx_kv_engine.py           $SP/xpu/patches/mx_kv/engine.py; \
    cp /tmp/mx_kv_triton_mx_decode.py $SP/xpu/patches/mx_kv/triton_mx_decode.py; \
    cp /tmp/mx_kv_init.py             $SP/xpu/patches/mx_kv/__init__.py; \
    rm -rf $SP/xpu/patches/__pycache__/mx_kv; \
    rm -f /tmp/mx_kv_engine.py /tmp/mx_kv_triton_mx_decode.py /tmp/mx_kv_init.py; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import inspect;\
       import vllm_gaudi.xpu.patches.mx_kv.engine as e;\
       import vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode as t;\
       import vllm_gaudi.xpu.patches.mx_kv as pkg;\
       tsrc = inspect.getsource(t);\
       esrc = inspect.getsource(e);\
       isrc = inspect.getsource(pkg);\
       assert '_mx_decode_combine' in tsrc, 'v6 chunked combine kernel';\
       assert '_V5_SPLIT_DEFAULT' in tsrc, 'v6 split cap line';\
       assert 'ARC_MXKV_V5_SPLITS' in tsrc, 'v6 split env knob';\
       assert 'ARC_MXKV_V5_BN' in tsrc, 'v6 BN env knob';\
       assert 'n_tiles // TILE_FLOOR' in tsrc, 'v6 adaptive split rule, TILE_FLOOR-floored';\
       assert 'TILE_FLOOR: tl.constexpr = 4' in tsrc, 'kernel constexpr stays 4 (signature freeze)';\
       assert t._V5_SPLIT_DEFAULT == 256, 'v6 split cap 256';\
       assert t._V5_TUNE == {'mxfp8': (32, 4, 2, False), 'mxfp4': (32, 8, 2, False), 'mxk8v4': (32, 8, 2, True)}, 'A14 BN=32 sweep winner tune';\
       assert t._MX_TUNE == {'mxfp8': (32, 4, 2, False), 'mxfp4': (32, 8, 2, False), 'mxk8v4': (32, 8, 2, True)}, 'A9 scalar tune unchanged';\
       assert '_DECODER = os.environ.get(\"ARC_MXKV_DECODER\", \"v2\")' in tsrc, 'per-row knob default v2 unchanged';\
       assert '_f16_scale' in tsrc, 'K2 packed-f16 scale helper';\
       assert '_e2m1_byte' in tsrc, 'K3 e2m1 byte-map helper';\
       assert 'F16OUT' in tsrc and 'E2M1FAST' in tsrc, 'K2/K3 kernel constexprs';\
       assert 'os.environ.get(\"ARC_MXKV_K2\", \"1\") != \"0\"' in tsrc, 'K2 lane default-on';\
       assert 'os.environ.get(\"ARC_MXKV_K3\", \"1\") != \"0\"' in tsrc, 'K3 lane default-on';\
       assert 'os.environ.get(\"ARC_MXKV_V5_TFLOOR\", \"2\") or 2' in tsrc, 'r195 TILE_FLOOR launcher default 2';\
       assert '_mx_decode_split_dot2' in tsrc, 'v7 research kernel present';\
       assert 'os.environ.get(\"ARC_MXKV_V7\", \"0\") == \"1\"' in tsrc, 'v7 dormant default-off';\
       assert '_V5_DEFAULT' in esrc, 'v5 default flip unchanged';\
       assert 'ARC_MXKV_VERIFY_DECODER' in esrc, 'verify-decoder override env unchanged';\
       assert '_V5_G_MIN' in esrc, 'G-adaptive knee';\
       assert e._V5_G_MIN == 8, 'G-adaptive knee default 8 (the n8 face)';\
       assert 'g-adaptive' in esrc, 'G-adaptive dispatch log label';\
       assert not __import__('re').search(r'query\.float\(\)', esrc), 'native-q decode path';\
       assert 'mx_decode_rows_dot' in isrc, 'package export unchanged';\
       print('CERT25ZC_DECODE_LINE_LIVE tf2 default + G-adaptive + v7 dormant OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert25zc=P6.47 decode-superiority line part 6 -- Phase 7 closing rung: (1) TILE_FLOOR launcher default 4->2 (ARC_MXKV_V5_TFLOOR read at launch; kernel constexpr stays 4 -- certified-signature freeze): deep splits REMOVE serial bt->slot->gather chain tiles rather than pipelining them (r192: Triton-on-XPU sinks gather loads to use sites, program-order prefetch not honored, num_stages no-op -- in-language prefetch toolbox empty); r193 sweep banked tf2/sp256 as the universal kernel winner; r195 e2e confirmed on cert25zb with the single env delta: m8q n1 16K 75->70, 32K 106->95 (-10.4%), mkq -4..-2 at 16-32K, m4q byte-identical (v2 lane null by design), ACCEPT_KNEE_PASS x3, 24/24 CONC_OK. (2) G-adaptive mxfp4 verify-decoder dispatch (engine.py): v2 below the r188-B1 knee, v5 at/above -- ARC_MXKV_V5_G_MIN (default 8, mxfp4-only); B1 banked the blanket flip refuted (n1 +8..14 ms; the v2 batch grid out-parallelizes v5's at G=1 and the bt_groups plan-build tax does not amortize) but n8 long-ctx decisive (32K ss 1978->1703, -14%; aggregate p50 1551->696); capture-lawful -- the split-batch request count is a host-known shape at plan-build and the plan cache freezes the choice per shape. (3) BN law stands post-K2/K3: r194 refuted BN=64 (0.36-0.69x; SMEM doubles -> occupancy halves -> the work-floor punishes) and BN=128 is SMEM-impossible (147456 B vs 131072 B limit). (4) v7 raw-tile double-buffer kernel ships DORMANT (ARC_MXKV_V7=1 to enable; r192 perf-refuted 0.61-0.95x, bitwise-clean d=0 -- kept for the record). One-time decoder dispatch log (fmt+decoder+source: env-override/env-flag/g-adaptive/format-default). v2/v3/v4/v6 lanes otherwise byte-identical to cert25zb."; \
      echo "cert25zc_fork_sha=${FORK_SHA}"; \
      echo "cert25zc_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p647; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p647 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
DOCKERFILE_EOF

BUILT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "=== [r196] building vllm-arc:v0.30.0-b70-cert25zc FORK_SHA=$FORK_SHA BUILT=$BUILT ==="
docker build \
  --build-arg FORK_SHA="$FORK_SHA" \
  --build-arg BUILT="$BUILT" \
  -t vllm-arc:v0.30.0-b70-cert25zc \
  "$CTX"
echo "R196_CERT25ZC_BUILT $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker run --rm --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert25zc -c \
  'tail -4 /etc/vllm-arc/IMAGE_MANIFEST'
