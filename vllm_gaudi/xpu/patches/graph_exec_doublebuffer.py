# SPDX-License-Identifier: Apache-2.0
"""Graph-exec double-buffering for XPU command graphs (P6.35-D1).

torch 2.13's ``XPUGraph.replay()`` on the SAME exec object blocks the
calling thread until that exec's prior launch retires (measured 13,957 µs
in-flight vs 8–12 µs for two execs sharing one memory pool alternating —
``results/2026-11-10-p635-telemetry/`` bench p135i B7/B8; per-node cost,
CCL-in-graph, peer rendezvous and queue-depth throttling all falsified by
p135d/e/g/h). With core 0.30's async scheduling (auto-on) the engine runs
one step ahead, so every step's verify replay lands on the same ~6k-node
exec still executing the previous step — the call blocks ≈ the graph's
own device time (~23–24 ms; XRP walls, leg F), fully serializing host
and device (27.7% device busy).

Fix: capture every graph TWICE into the manager's ONE shared pool and
ALTERNATE replays per descriptor — launches never block. Validated live
on the dual-B70 rig (leg J, gate2c): XRP max/step 24 ms → 0.37 ms with
perfect G1/G2 alternation (12 XRP ids × 59 over 119 steady steps),
engine serial work 40.8 → 12.7 ms/step, device utilization 27.7% →
~80%. Cost: one extra capture pass per manager (+29 s boot at the
certified qwen3.8-27b posture, 3 managers × 2 TP workers); the shared
pool means no extra activation memory.

Both ``ModelCudaGraphManager`` and ``SpeculatorCudaGraphManager``
funnel through the base ``CudaGraphManager.capture``/``run_fullgraph``
(the Model subclass only adds post-replay hidden-state reads over
``super()``), so wrapping the base class covers every manager.
``run_fullgraph`` swaps the selected exec into ``self.graphs[desc]``
around the ORIGINAL call (no body copy — subclass overrides and future
base drift keep working) and restores it in ``finally``.

Gated by ``VLLM_ARC_GRAPH_EXEC_DB`` (default on; 0 restores stock
single-exec replays). XPU-only: CUDA's graph launch is already async —
the same-exec blocking is XPU-specific — so the install is a no-op
elsewhere.

Provenance: plan §P6.35 legs F/J/K (2026-11-10); fix shape p135i B8.
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc")

_INSTALLED = False


def _install() -> None:
    """Wrap CudaGraphManager capture/run_fullgraph in place (idempotent,
    per-process). Installed from the platform hooks like the other
    ``_XPU_PATCH_MODULES`` (config time + per-worker init_device)."""
    global _INSTALLED
    if _INSTALLED:
        return
    if not os.environ.get("VLLM_ARC_GRAPH_EXEC_DB", "1").lower() in (
            "1", "true"):
        logger.info("ArcXpuPlatform: graph-exec double-buffer DISABLED "
                    "(VLLM_ARC_GRAPH_EXEC_DB=0).")
        _INSTALLED = True
        return
    try:
        import torch

        if not torch.xpu.is_available():
            # CUDA graph launch is async; the same-exec block is XPU-only.
            _INSTALLED = True
            return
    except Exception:  # noqa: BLE001 — never break manager import
        _INSTALLED = True
        return

    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    _orig_capture = CudaGraphManager.capture
    _orig_run_fullgraph = CudaGraphManager.run_fullgraph

    def capture(self, create_forward_fn,
                progress_bar_desc: str = "Capturing CUDA graphs"):
        _orig_capture(self, create_forward_fn, progress_bar_desc)
        graphs_g1 = dict(self.graphs)
        if not graphs_g1:
            return
        # Second pass into the SAME shared pool: clear the dict so the
        # base per-desc assert (``desc not in self.graphs``) holds, then
        # re-run the untouched base capture. On failure, restore the
        # first set and keep serving single-buffered (perf patch: degrade
        # loudly, never take the engine down).
        self.graphs = {}
        self._graphs_captured = False
        try:
            _orig_capture(self, create_forward_fn,
                          progress_bar_desc + " [arc-db]")
        except BaseException as e:  # noqa: BLE001
            self.graphs = graphs_g1
            self._graphs_captured = True
            logger.warning(
                "ArcXpuPlatform: graph-exec double-buffer second capture "
                "failed (%r) — continuing single-buffered (stock replay "
                "semantics).", e)
            return
        self._arc_db_g1 = graphs_g1
        self._arc_db_toggle = {}
        logger.info(
            "ArcXpuPlatform: graph-exec double-buffer captured %d graphs "
            "x2 (shared pool; same-exec replay blocking avoided — P6.35 "
            "D1).", len(graphs_g1))

    def run_fullgraph(self, desc):
        graphs_g1 = getattr(self, "_arc_db_g1", None)
        if graphs_g1 is None or desc not in graphs_g1:
            return _orig_run_fullgraph(self, desc)
        toggle = self._arc_db_toggle
        toggle[desc] = not toggle.get(desc, False)
        exec_g2 = self.graphs[desc]
        self.graphs[desc] = graphs_g1[desc] if toggle[desc] else exec_g2
        try:
            return _orig_run_fullgraph(self, desc)
        finally:
            self.graphs[desc] = exec_g2

    CudaGraphManager.capture = capture
    CudaGraphManager.run_fullgraph = run_fullgraph
    _INSTALLED = True
    logger.info(
        "ArcXpuPlatform: graph-exec double-buffer patch installed "
        "(pid %d, CudaGraphManager capture/run_fullgraph wrapped).",
        os.getpid())
