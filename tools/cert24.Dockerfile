# SPDX-License-Identifier: Apache-2.0
# cert24.Dockerfile — vllm-arc:v0.30.0-b70-cert24 = cert23 + the P6.47
# MXFP4/MXFP8/MXK8V4 KV-cache lane: `--kv-cache-dtype mxfp4|mxfp8|mxk8v4`
# (OCP microscaling per-32-element E8M0 scales; mxfp8 slot 528B @D=256
# [K e4m3][Ks][V e4m3][Vs], mxk8v4 slot 400B [K e4m3][Ks][V e2m1][Vs],
# mxfp4 slot 272B e2m1 nibbles) as its OWN family — enum member MXKV
# routed by ArcXpuPlatform.get_attn_backend_cls
# (2026-11-11: MXFP4/MXFP8 are their own family, not part of
# TURBOQUANT; the admission alias in mx_kv_mode.py is a bounded internal
# bridge onto TQ-class page plumbing only). mxk8v4 added 2026-11-12
# (commit 3b92b9f8) against the r154 run1 m4q verdict: 4-bit K degrades
# long-generation retrieval into a stable REP-LOOP attractor under churn
# while m8q stays byte-stable — 8-bit keys are the retrieval bottleneck
# (wheel doctrine P6.15 B6); mxk8v4 is the TRUE turboquant_k8v4 analogue.
#
# ROOT-CAUSE PROVENANCE (trace20-28, results/2026-11-12 banks on rig +
# plan §P6.47): the first store kernel shared keys' token+head strides
# for BOTH loads; in-engine key and value arrive with INDEPENDENT
# layouts (upstream law — the wheel's int4_per_token_head.py takes
# stride_val_tok/stride_val_head). Stored-V rel-L2 was 42-80% with K
# clean at 2-3% -> spec/eager text corruption on every MX posture. Fix
# = separate stride_v_n/stride_v_h params + defensive .contiguous()
# (triton_mx_store.py); regression guard tools/ar/p148_mxkv/
# store_vstride_probe.py (transpose-view V, bidirectionally proven).
# Validated live on rig /work shadow, ALL FOUR postures: mxfp8-eager
# texts == native-fp8 BYTE-EXACT (7/209/236); mxfp8-spec 7/209/239
# (3-char p3 tail); mxfp4 spec==eager BYTE-EXACT (7/203/239 both);
# every lane answers '\n\nParis' (trace25-28, plan bank 2026-11-12).
#
# Construction law (cert11+ overlay precedent, minimal delta): this
# layer adds the NEW mx_kv/ package (7 files) + 2 patches-level modules
# (mx_dtype_map.py, mx_kv_mode.py) + the cumulative tracked platform.py
# (adds MXKV enum-member injection, CacheDtype literal admission, MX
# dtype admission fallback, get_attn_backend_cls routing). No wheel, no
# serve-line change (ARC_KVDTYPE already exposed at
# serve_arc_b70.sh:361 — champion chain boots MX natively).
#
# Gate: none needed beyond dtype choice — the lane engages ONLY via
# --kv-cache-dtype mxfp4|mxfp8|mxk8v4 (default fp8_e4m3 posture
# unchanged, byte-identical serve line; r154 cert24 gate proves
# default-off neutrality + every MX leg incl. mkq/mkg).
#
# Inputs (cert7 reproducibility law):
#   - base: certified cert23 (IMAGE_MANIFEST rows carry forward)
#   - repo (tracked): vllm_gaudi/xpu/patches/mx_kv/{__init__,engine,
#     mx_dequant,mx_layout,mx_quant,triton_mx_decode,triton_mx_store}.py
#     + vllm_gaudi/xpu/patches/{mx_dtype_map,mx_kv_mode}.py +
#     vllm_gaudi/xpu/platform.py from the P6.47 content commit
#     (FORK_SHA below)
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + the ten overlay files, flat):
#   docker build -f cert24.Dockerfile \
#     --build-arg FORK_SHA=<content commit sha> \
#     --build-arg BUILT=<utc ts> \
#     -t vllm-arc:v0.30.0-b70-cert24 .
ARG BASE=vllm-arc:v0.30.0-b70-cert23
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
COPY __init__.py /tmp/mx_kv___init__.py
COPY engine.py /tmp/mx_kv_engine.py
COPY mx_dequant.py /tmp/mx_kv_mx_dequant.py
COPY mx_layout.py /tmp/mx_kv_mx_layout.py
COPY mx_quant.py /tmp/mx_kv_mx_quant.py
COPY triton_mx_decode.py /tmp/mx_kv_triton_mx_decode.py
COPY triton_mx_store.py /tmp/mx_kv_triton_mx_store.py
COPY mx_dtype_map.py /tmp/mx_dtype_map.py
COPY mx_kv_mode.py /tmp/mx_kv_mode.py
COPY platform.py /tmp/platform.py
RUN set -euxo pipefail; \
    SP=/opt/venv/lib/python3.12/site-packages/vllm_gaudi; \
    mkdir -p $SP/xpu/patches/mx_kv; \
    cp /tmp/mx_kv___init__.py       $SP/xpu/patches/mx_kv/__init__.py; \
    cp /tmp/mx_kv_engine.py         $SP/xpu/patches/mx_kv/engine.py; \
    cp /tmp/mx_kv_mx_dequant.py     $SP/xpu/patches/mx_kv/mx_dequant.py; \
    cp /tmp/mx_kv_mx_layout.py      $SP/xpu/patches/mx_kv/mx_layout.py; \
    cp /tmp/mx_kv_mx_quant.py       $SP/xpu/patches/mx_kv/mx_quant.py; \
    cp /tmp/mx_kv_triton_mx_decode.py $SP/xpu/patches/mx_kv/triton_mx_decode.py; \
    cp /tmp/mx_kv_triton_mx_store.py  $SP/xpu/patches/mx_kv/triton_mx_store.py; \
    cp /tmp/mx_dtype_map.py         $SP/xpu/patches/mx_dtype_map.py; \
    cp /tmp/mx_kv_mode.py           $SP/xpu/patches/mx_kv_mode.py; \
    cp /tmp/platform.py             $SP/xpu/platform.py; \
    rm -f $SP/xpu/__pycache__/platform.cpython-312.pyc; \
    rm -f $SP/xpu/patches/__pycache__/mx_dtype_map.cpython-312.pyc; \
    rm -f $SP/xpu/patches/__pycache__/mx_kv_mode.cpython-312.pyc; \
    rm -rf $SP/xpu/patches/__pycache__/mx_kv; \
    rm -f /tmp/mx_kv_*.py /tmp/mx_dtype_map.py /tmp/mx_kv_mode.py /tmp/platform.py; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import inspect; \
       import vllm_gaudi.xpu.patches.mx_kv.engine as e; \
       assert hasattr(e, 'MxAttentionBackend'), 'MX backend'; \
       import vllm_gaudi.xpu.patches.mx_kv.triton_mx_store as s; \
       ssrc = inspect.getsource(s); \
       assert 'stride_v_n' in ssrc and 'stride_v_h' in ssrc, 'v-stride fix'; \
       assert 'K4' in ssrc and 'V4' in ssrc, 'mxk8v4 kernel legs'; \
       import vllm_gaudi.xpu.patches.mx_kv.mx_layout as lay; \
       assert lay.slot_bytes(256, 'mxfp8') == 528 and lay.slot_bytes(256, 'mxfp4') == 272 \
           and lay.slot_bytes(256, 'mxk8v4') == 400, 'slot law'; \
       import vllm_gaudi.xpu.patches.mx_dtype_map as d; \
       import vllm_gaudi.xpu.patches.mx_kv_mode as m; \
       assert 'mxk8v4' in inspect.getsource(d), 'dtype map'; \
       assert 'mxk8v4' in inspect.getsource(m), 'mode alias'; \
       import vllm_gaudi.xpu.patches.mx_kv.engine as eng; \
       assert len(eng.MX_KV_DTYPES) == 3 and 'mxk8v4' in eng.MX_KV_DTYPES, 'engine dtypes'; \
       assert eng.MX_TQ_INIT_ALIAS['mxk8v4'] == 'turboquant_k8v4', 'tq alias'; \
       p = open('$SP/xpu/platform.py', encoding='utf-8').read(); \
       assert 'mx_kv.engine.MxAttentionBackend' in p and 'MXKV' in p, 'platform routing'; \
       assert 'mxfp4' in p and 'mxfp8' in p and 'mxk8v4' in p, 'dtype admission'; \
       print('CERT24_MXKV_LIVE engine+store+layout+platform OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert24=P6.47 MXFP4/MXFP8/MXK8V4 KV-cache lane (--kv-cache-dtype mxfp4|mxfp8|mxk8v4, own MXKV family enum routed via ArcXpuPlatform): patches/mx_kv/{__init__,engine,mx_dequant,mx_layout,mx_quant,triton_mx_decode,triton_mx_store}.py + patches/{mx_dtype_map,mx_kv_mode}.py + platform.py; OCP-MX E8M0 per-32 scales, mxfp8 528B/mxk8v4 400B/mxfp4 272B slots @D=256; SEPARATE-V-STRIDE STORE LAW (trace24 root cause: shared k-strides stored V from wrong addresses, rel-L2 42-80% -> spec/eager text corruption; fix validated all 4 postures trace25-28, mxfp8-eager == native byte-exact 7/209/236, mxfp4 spec==eager 7/203/239; guard store_vstride_probe.py); mxk8v4 = e4m3 K + e2m1 V mixed slot (K4/V4 kernel legs) added against the r154 m4q 4-bit-K REP-LOOP quality limit (turboquant_k8v4 doctrine); telemetry env ARC_MXKV_TRACE/ARC_MXKV_DUMP default SILENT"; \
      echo "cert24_fork_sha=${FORK_SHA}"; \
      echo "cert24_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p647; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p647 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
