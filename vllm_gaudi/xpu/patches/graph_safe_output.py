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
sync is illegal, so the wrapper fails safe to upstream behavior there; the
``get_output`` wrapper below then self-heals if such a capture-time event
ever carries a command-graph association at wait time (device-sync barrier +
retry, once).

Visibility law: every install, apply, defer, and self-heal logs (rate-limited)
under ``vllm.platforms.arc`` so rig legs can prove which process patched and
which branch ran — the p2r2 leg died with this defect while all three install
hooks passed silently.

Provenance: error family + fix direction = plan Appendix C defect (b);
capture-probe fail-safe law = D-line waveD patch_gdn_metadata_026.py
(``_iexp_is_capturing``, results/2026-10-18-waveD-anchor-analysis.log).
"""

from __future__ import annotations

import os
import traceback

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc")

# The L0 error text this patch exists to neutralize (see module docstring).
_COMMAND_GRAPH_ERR = "command graph"

# Rate-limited first-occurrence logging: leg logs must show the branch that
# ran without per-step spam (thousands of steps per leg).
_LOG_BUDGET = 3
_applied_logs = 0
_deferred_logs = 0
_probe_err_logs = 0
_healed_logs = 0


class _ArcNoOpEvent:
    """Drop-in barrier for an already-satisfied copy: synchronize is free."""

    __slots__ = ()

    def synchronize(self) -> None:  # noqa: D102 — upstream callsite parity
        return None


_INSTALLED = False


def _capture_state() -> tuple[bool, Exception | None]:
    """(capturing, probe_error) — fail-safe: assume capturing on any doubt.

    Mirrors D-line ``_iexp_is_capturing`` (waveD law): a probe that errors is
    treated as capturing so nothing graph-unsafe runs on a guess. Only
    ``torch.xpu.is_current_stream_capturing`` is consulted — the
    ``torch.cuda.`` alias raises "dummy base class" on this torch-xpu build
    (rig-verified 2026-10-27).
    """
    try:
        probe = getattr(torch.xpu, "is_current_stream_capturing", None)
        if probe is None:
            return True, RuntimeError(
                "torch.xpu.is_current_stream_capturing unavailable")
        return bool(probe()), None
    except Exception as e:  # noqa: BLE001 — cannot prove not-capturing
        return True, e


def _is_capturing() -> bool:
    """True when an XPU graph capture is active (fail-safe: assume yes)."""
    return _capture_state()[0]


def _install() -> None:
    """Patch the async output classes in-place (idempotent, per-process).

    Call sites (belt and suspenders — each vllm process must patch its own
    classes): platform ``check_and_update_config`` (config time, APIServer +
    EngineCore), platform ``check_if_supports_dtype`` (per-worker from
    ``XPUWorker.init_device`` — the guaranteed post-import point), and plugin
    ``register()`` (earliest, best-effort — the ``vllm.v1.worker.gpu`` import
    is circular mid platform-resolution and is expected to fail there; the
    platform hooks retry it post-import).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.worker.gpu import async_utils as au

    _orig_async_output_init = au.AsyncOutput.__init__
    _orig_pooling_init = au.AsyncPoolingOutput.__init__
    _orig_async_output_get = au.AsyncOutput.get_output
    _orig_pooling_get = au.AsyncPoolingOutput.get_output

    def _graph_safe_init(orig_init):
        def __init__(self, *args, **kwargs):
            global _applied_logs, _deferred_logs, _probe_err_logs
            orig_init(self, *args, **kwargs)
            capturing, probe_err = _capture_state()
            if capturing:
                # Inside a capture window a device sync is illegal; upstream
                # behavior is correct there. If the capture-time event ever
                # turns out poisoned at wait time, the get_output wrapper
                # below self-heals instead of killing the engine.
                if probe_err is not None:
                    if _probe_err_logs < _LOG_BUDGET:
                        _probe_err_logs += 1
                        logger.warning(
                            "ArcXpuPlatform: graph-safe output probe failed "
                            "(%r) — deferred to upstream event (fail-safe "
                            "law).", probe_err)
                elif _deferred_logs < _LOG_BUDGET:
                    _deferred_logs += 1
                    logger.warning(
                        "ArcXpuPlatform: graph-safe output deferred to "
                        "upstream (capture window active). Stack at this "
                        "moment:\n%s",
                        "".join(traceback.format_stack()[-18:]))
                return
            torch.xpu.synchronize()
            # Host buffers are complete; the recorded event may carry a
            # command-graph association and must never be waited on.
            self.copy_event = _ArcNoOpEvent()
            if _applied_logs < _LOG_BUDGET:
                _applied_logs += 1
                logger.info(
                    "ArcXpuPlatform: graph-safe output applied — device-sync "
                    "barrier + no-op copy event.")

        return __init__

    def _graph_safe_get_output(orig_get_output):
        def get_output(self):
            global _healed_logs
            try:
                return orig_get_output(self)
            except RuntimeError as e:
                if _COMMAND_GRAPH_ERR not in str(e):
                    raise
                # An unpatched (capture-created) event carried a command-graph
                # association. The copy itself has long since completed —
                # take the device-sync barrier once and retry; the retry's
                # synchronize is the no-op above or an already-satisfied
                # upstream event. Log FIRST (p2r3b lesson: a sync that raises
                # mid-capture would otherwise swallow the diagnosis).
                if _healed_logs < _LOG_BUDGET:
                    _healed_logs += 1
                    logger.warning(
                        "ArcXpuPlatform: graph-safe output self-heal engaged "
                        "for a command-graph-associated copy event: %s", e)
                try:
                    torch.xpu.synchronize()
                except RuntimeError as sync_err:
                    # The barrier itself is illegal while a capture window is
                    # STILL open (p2r3b: "wait cannot be called for a queue
                    # which is recording to a command graph") — nothing can
                    # wait here. Surface the original error.
                    logger.error(
                        "ArcXpuPlatform: graph-safe output self-heal barrier "
                        "illegal (capture still open): %r", sync_err)
                    raise e from sync_err
                self.copy_event = _ArcNoOpEvent()
                return orig_get_output(self)

        return get_output

    au.AsyncOutput.__init__ = _graph_safe_init(_orig_async_output_init)
    au.AsyncPoolingOutput.__init__ = _graph_safe_init(_orig_pooling_init)
    au.AsyncOutput.get_output = _graph_safe_get_output(_orig_async_output_get)
    au.AsyncPoolingOutput.get_output = _graph_safe_get_output(
        _orig_pooling_get)
    _INSTALLED = True
    logger.info(
        "ArcXpuPlatform: graph-safe output patch installed (pid %d, "
        "AsyncOutput/AsyncPoolingOutput __init__ + get_output wrapped).",
        os.getpid())
