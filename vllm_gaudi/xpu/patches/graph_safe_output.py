"""XPU graph-safe worker output copies (defect-b family, plan Appendix C).

vLLM 0.30's ``AsyncOutput``/``AsyncPoolingOutput`` (v1/worker/gpu/
async_utils.py) enqueue non-blocking D2H copies on a side stream, record a
blocking event there, and the executor's output thread synchronizes that
event to know the host buffers are filled. On XPU, torch-xpu graphs are L0
command graphs and associate events recorded while a command graph is open
with that graph; synchronizing such an event raises

    RuntimeError: wait method cannot be used for an event associated
    with a command graph

and kills the engine on the FIRST served step (repro 2026-10-27,
results/2026-10-27-p2r1-graph-only/: boot + FULL_DECODE_ONLY capture succeed,
first /v1/completions returns 200, then Worker_TP0 dies in
``enqueue_output -> get_output -> copy_event.synchronize``).

Fix posture (graph-safe copy, plan Appendix C fix direction): keep upstream's
copy logic untouched, then take a full device synchronize as the barrier and
neutralize the event for ``get_output``. All values are identical — the host
arrays are provably complete once ``torch.xpu.synchronize()`` returns — only
the completion mechanism changes. Inside an active graph capture the device
sync is illegal, so we fail safe to upstream behavior there (capture-time
handles demonstrably synchronize fine; boot captures completed 27/27 before
the repro death).

Provenance: error family + fix direction = plan Appendix C defect (b);
capture-probe fail-safe law = D-line waveD patch_gdn_metadata_026.py
(``_iexp_is_capturing``, results/2026-10-18-waveD-anchor-analysis.log).
"""

from __future__ import annotations

import torch


class _ArcNoOpEvent:
    """Drop-in barrier for an already-satisfied copy: synchronize is free."""

    __slots__ = ()

    def synchronize(self) -> None:  # noqa: D102 — upstream callsite parity
        return None


_INSTALLED = False


def _is_capturing() -> bool:
    """True when an XPU graph capture is active (fail-safe: assume yes).

    Mirrors D-line ``_iexp_is_capturing`` (waveD law): a probe that errors is
    treated as capturing so nothing graph-unsafe runs on a guess.
    """
    try:
        probe = getattr(torch.xpu, "is_current_stream_capturing", None)
        if probe is not None:
            return bool(probe())
    except Exception:  # noqa: BLE001 — cannot prove not-capturing
        return True
    return False


def _install() -> None:
    """Patch the async output classes in-place (idempotent, per-process).

    Call sites (belt and suspenders — each vllm process must patch its own
    classes): plugin ``register()`` (early, best-effort), platform
    ``check_and_update_config`` (config time, APIServer + EngineCore), and
    platform ``check_if_supports_dtype`` (per-worker from
    ``XPUWorker.init_device`` — the guaranteed post-import point).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.worker.gpu import async_utils as au

    _orig_async_output_init = au.AsyncOutput.__init__
    _orig_pooling_init = au.AsyncPoolingOutput.__init__

    def _graph_safe_init(orig_init):
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if _is_capturing():
                # Inside a capture window a device sync is illegal; upstream
                # behavior is correct there (capture-time handles sync fine).
                return
            torch.xpu.synchronize()
            # Host buffers are complete; the recorded event may carry a
            # command-graph association and must never be waited on.
            self.copy_event = _ArcNoOpEvent()

        return __init__

    au.AsyncOutput.__init__ = _graph_safe_init(_orig_async_output_init)
    au.AsyncPoolingOutput.__init__ = _graph_safe_init(_orig_pooling_init)
    _INSTALLED = True
