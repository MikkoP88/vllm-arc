# cert17: cert16 + ARC P6.28 lmcint kit hardening (detector BS floor +
# two-head-size fail-loud store gate)
#
# Layer 3 of the G2 long-prefill freeze, root-caused in r108w (banked:
# results/2026-11-06-p628-gemma4/phaseC/cert16_boot/): with the warm-gate
# cache-key fix live (cert16), probe98 still 500'd — wedge-time py-spy of
# both ranks shows the MainThreads parked at wait_for_save ->
# _lmc_lock_guard while the arc-lmc-store worker holds _lmc_store_lock
# stuck ACTIVE in lmcache's stock multi_layer_kv_transfer python fallback
# (mem_kernels.py:155 blocking copy), reached through our wrapper's _ORIG
# passthrough. The kit TWO_HS fast path never ran once (r43xfer=0 all
# boot): gemma-4 boots block_size=64 and the kit detector's BS>=256 guard
# rejected the adapter's (NB,64,1,2,kvdim) views, dropping discovery to
# stock lmcache, which misaddresses the per-token-interleaved hybrid pool.
#
# Two kit fixes (files under vllm_gaudi/xpu/patches/lmcint/, overlaid on
# the installed vllm_gaudi tree — md5-verified byte-identical base):
#   1. lmc_kvfmt_detector.py: BS floor 256 -> 16 (the vLLU block-size
#      minimum). The old bound was overfit to the r43 model's page size
#      and rejected every small-block engine; p614 had already documented
#      the same trap on the 4bit leg.
#   2. vllm_v1_adapter.py: [ARC-P628-LMC-KVGATE] fail-loud uniform-KV-row
#      gate at register_kv_caches. gemma-4 is a two-head-size attention
#      hybrid (50 sliding hd256 -> 2048/rank + 10 full hd512 -> 1024/rank)
#      and the KV_2LTD memory object is single-hidden-size by design, so
#      the store can never be byte-true — the gate refuses at BOOT with a
#      clear message instead of wedging at the first request inside the
#      300s execute-model watchdog. Uniform models (and equal-row-width
#      groups, byte-true by construction) are unaffected. The certified
#      gemma-4 posture is ARC_LMC=0; r108y proved LMC=0 removes the
#      freeze on cert16.
#
# Supersedes cert16; chain cert12 -> cert16 -> cert17.
#
# Build (rig):  docker build -f cert17.Dockerfile \
#                 --build-arg FORK_SHA=<sha> --build-arg BUILT=<utc> \
#                 -t vllm-arc:v0.30.0-b70-cert17 <ctx with the 2 kit files>
ARG BASE=vllm-arc:v0.30.0-b70-cert16
FROM ${BASE}

COPY lmc_kvfmt_detector.py vllm_v1_adapter.py /tmp/cert17kit/

RUN KIT=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmcint && \
    cp /tmp/cert17kit/lmc_kvfmt_detector.py "$KIT/lmc_kvfmt_detector.py" && \
    cp /tmp/cert17kit/vllm_v1_adapter.py "$KIT/vllm_v1_adapter.py" && \
    rm -f "$KIT"/__pycache__/lmc_kvfmt_detector.cpython-312.pyc \
          "$KIT"/__pycache__/vllm_v1_adapter.cpython-312.pyc && \
    /opt/venv/bin/python3 -c "import py_compile; py_compile.compile('$KIT/lmc_kvfmt_detector.py', doraise=True); py_compile.compile('$KIT/vllm_v1_adapter.py', doraise=True); d = open('$KIT/lmc_kvfmt_detector.py', encoding='utf-8').read(); a = open('$KIT/vllm_v1_adapter.py', encoding='utf-8').read(); assert d.count('int(t.shape[1]) >= 16') == 1 and d.count('int(t.shape[2]) >= 16') == 1, 'detector BS floor 16 missing'; assert 'int(t.shape[1]) >= 256' not in d and 'int(t.shape[2]) >= 256' not in d, 'overfit BS bound still present'; assert a.count('[ARC-P628-LMC-KVGATE]') == 1, 'kvgate marker'; assert 'ARC_LMC=0 for this model' in a, 'gate guidance text'; print('CERT17_KIT_HARDENING_LIVE')" && \
    rm -rf /tmp/cert17kit && \
    MF=/etc/vllm-arc/IMAGE_MANIFEST && \
    echo "cert17=P6.28 cert16 + lmcint kit hardening (detector BS floor 256->16 so small-block engines take the byte-true TWO_HS path instead of stock lmcache; [ARC-P628-LMC-KVGATE] uniform-KV-row gate fails loudly at boot on two-head-size attention hybrids like gemma-4 whose KV_2LTD store is structurally unsupported — r108w wedge root cause; certified gemma posture ARC_LMC=0)" >> "$MF" && \
    echo "cert17_fork_sha=${FORK_SHA}" >> "$MF" && \
    echo "cert17_built=${BUILT}" >> "$MF"
