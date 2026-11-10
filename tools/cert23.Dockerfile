# SPDX-License-Identifier: Apache-2.0
# cert23.Dockerfile — vllm-arc:v0.30.0-b70-cert23 = cert22 + the P6.35
# graph-exec double-buffer: torch 2.13 XPU command-graph replay on the SAME
# exec object blocks host-side until that exec's prior launch retires
# (p135i bench: same-graph replay 13,957 us in-flight vs 8-12 us across two
# execs sharing one capture pool alternating) — the root cause of the
# leg-A "29.3 ms device-idle" decode gaps and the 24 ms XPUGraph.replay
# walls. Fix: capture every graph TWICE into the manager's shared pool and
# alternate replays per desc (swap-technique wrappers on
# CudaGraphManager.capture / .run_fullgraph; second-capture failure
# degrades to stock single-buffer semantics with a WARN, never a boot
# death). Validated leg J (results/2026-11-10-p635-telemetry/legJ/):
# XRP replay max 24 ms -> 0.37 ms, 12 XRP ids alternate x59 over 119
# steps, engine serial work 40.8 -> 12.7 ms/step, device utilization
# 27.7% -> ~80%, boot +29 s (6 extra "arc-db" capture passes), zero
# extra activation memory (shared pool). Decode step wall is unchanged
# (leg K: the AsyncOutput copy_event wait waits out REAL device work —
# the fp8 pipeline floor ~40 ms; posture levers address that).
#
# Construction law (cert11+ overlay precedent, minimal delta): this
# layer changes EXACTLY: vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py
# (new), vllm_gaudi/xpu/platform.py (one entry),
# vllm_gaudi/envs.py (one registry entry). No wheel, no serve-line env.
#
# Gate: VLLM_ARC_GRAPH_EXEC_DB (default 1 = ACTIVE; docs/envs.md row —
# consumer patches/graph_exec_doublebuffer.py) — the only kill-switch is
# `=0` (stock same-exec replay semantics), a serve-ENV knob outside the
# 25-var chain like ARC_ATTN_FA2.
#
# Inputs (cert7 reproducibility law):
#   - base: certified cert22 (IMAGE_MANIFEST rows carry forward)
#   - repo (tracked): vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py +
#     vllm_gaudi/xpu/platform.py + vllm_gaudi/envs.py from the P6.35 D1
#     content commit (FORK_SHA below)
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + the three overlay files):
#   docker build -f cert23.Dockerfile \
#     --build-arg FORK_SHA=<content commit sha> \
#     --build-arg BUILT=<utc ts> \
#     -t vllm-arc:v0.30.0-b70-cert23 .
ARG BASE=vllm-arc:v0.30.0-b70-cert22
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
COPY graph_exec_doublebuffer.py /tmp/graph_exec_doublebuffer.py
COPY platform.py /tmp/platform.py
COPY envs.py /tmp/envs.py
RUN set -euxo pipefail; \
    cp /tmp/graph_exec_doublebuffer.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/graph_exec_doublebuffer.py; \
    cp /tmp/platform.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py; \
    cp /tmp/envs.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/envs.py; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/__pycache__/platform.cpython-312.pyc; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/__pycache__/graph_exec_doublebuffer.cpython-312.pyc; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/__pycache__/envs.cpython-312.pyc; \
    rm -f /tmp/graph_exec_doublebuffer.py /tmp/platform.py /tmp/envs.py; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import vllm_gaudi.xpu.patches.graph_exec_doublebuffer as g; \
       assert hasattr(g, '_install') and hasattr(g, '_INSTALLED'); \
       import inspect; \
       src = inspect.getsource(g); \
       assert '_arc_db_g1' in src and 'VLLM_ARC_GRAPH_EXEC_DB' in src, 'db core / gate'; \
       p = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py', encoding='utf-8').read(); \
       assert p.count('graph_exec_doublebuffer') >= 1, 'platform registration'; \
       e = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/envs.py', encoding='utf-8').read(); \
       assert 'VLLM_ARC_GRAPH_EXEC_DB' in e, 'envs registration'; \
       print('CERT23_GRAPH_EXEC_DB_LIVE patch+registration OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert23=P6.35 D1 graph-exec double-buffer: patches/graph_exec_doublebuffer.py captures every XPU graph twice into the shared pool and alternates replays per desc (torch 2.13 same-exec replay blocks on the prior launch — p135i bench; VLLM_ARC_GRAPH_EXEC_DB gate, default 1, =0 kill-switch) + platform/envs entries; XRP replay 24ms->0.37ms, engine serial 40.8->12.7ms/step, util ~80%, boot +29s (leg J, results/2026-11-10-p635-telemetry/)"; \
      echo "cert23_fork_sha=${FORK_SHA}"; \
      echo "cert23_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p635; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p635 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
