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

ZOMBIE ANSWER (2026-10-27 p2-graph leg + device probes,
results/2026-10-27-p2-graph-defectb-answer/): the discrimination canary ran at
serve time on the p13 image — ``capturing(cur)=True capturing(fresh)=False``
on BOTH TP workers, ``canary[executed=False]`` (fill never executed; read-back
showed uninitialized memory). The recording flag is per-stream, only the
worker's serving stream is poisoned, and kernels enqueued there genuinely
never run: a TRUE ZOMBIE, not a stale boolean. Full-boot sequence proves the
poison enters during graph capture (27/27 sizes, "finished in 7 secs", no
errors) and is already present at the first serve-time ``sample_tokens``
(before any replay). Bare-process probes could NOT reproduce the leak with
trivial captures (clean single + pool-shared + side-stream captures all end
``capturing=False`` with working execution and correct replay), but DID
reproduce the sibling poison: any event recorded inside a capture is
permanently graph-associated (``ev.synchronize()`` outside raises the exact
defect-b text). The leak therefore needs the real captured runnable (CCL /
GDN / sampler content) on this torch 2.13.0+xpu + oneAPI 2026.0 stack.

Fix law (this module, p12b): clear the zombie at every boundary where a swap
cannot be undone — after each capture context teardown
(``torch.xpu.graph.__exit__``; capture accounting via
``XPUGraph.capture_begin``/``capture_end``: a recording flag with none open
= leaked recorder), before and after each ``replay``, and as a last-resort
alarm at serve time. A stuck recorder is swapped for a fresh stream
(probe-validated recovery: fresh-stream kernels execute; the swap is
serve-transparent). The serve-time
alarm also FAILS the in-flight request loudly — its forward was enqueued on
the dead queue, so its outputs are garbage, and the integrity law forbids
returning unverifiable tokens. capture-time events that later carry a
command-graph association remain covered by the ``get_output`` self-heal.

Provenance: error family + fix direction = plan Appendix C defect (b);
capture-probe fail-safe law = D-line waveD patch_gdn_metadata_026.py
(``_iexp_is_capturing``, results/2026-10-18-waveD-anchor-analysis.log);
zombie discrimination + fresh-stream recovery law = rig probes
2026-10-27/28 (probe_zombie{,2}.py, results/2026-10-27-p2-graph-defectb-answer/).
"""

from __future__ import annotations

import os

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

# --- capture accounting + zombie hygiene (p12b, see module docstring) ---
# Number of XPUGraph captures currently open in this process (captures are
# strictly sequential — torch documents one at a time). A recording flag on
# the current stream with none open = leaked recorder = zombie.
_captures_open = 0
_dezombie_logs = 0


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


def _stream_ids() -> str:
    """Identity line for hygiene logs: which streams exist and their queue
    ids — identifies the poisoned stream class (default vs side capture vs
    runner stream) when the alarm fires."""
    try:
        parts = [f"cur={torch.xpu.current_stream()}"]
        try:
            parts.append(f"default={torch.xpu.default_stream()}")
        except Exception:  # noqa: BLE001
            parts.append("default=ERR")
        side = getattr(torch.xpu.graph, "default_capture_stream", None)
        parts.append(f"side={side}")
        return " ".join(parts)
    except Exception as e:  # noqa: BLE001
        return f"ids ERR {e!r}"


_WATCHDOG_STARTED = False


def _ensure_watchdog() -> None:
    """Start the onset watchdog exactly once per process, and only where a
    GPU context already exists — probing streams initializes an XPU context,
    which CPU-only processes (APIServer, EngineCore) must not pay for."""
    global _WATCHDOG_STARTED
    if _WATCHDOG_STARTED:
        return
    if not (torch.xpu.is_available() and torch.xpu.is_initialized()):
        return
    _WATCHDOG_STARTED = True
    _start_watchdog()


def _start_watchdog() -> None:
    """Onset detector (p12b round 2): poll the default and side capture
    streams for a recording flag with no capture open, and log the FIRST
    transition with a timestamp — bracketing the flip against vllm's own
    log timeline (warmup vs first request vs quiet window) to pin the
    poisoning operation. Thread-local current streams mean the watchdog
    can only speak for default/side streams; worker-thread streams are
    identified by id at the alarm itself. Daemon, self-terminating on
    error, transitions logged with a budget.
    """
    import threading
    import time as _time

    def _probe(stream) -> bool:
        me = torch.xpu.current_stream()
        torch.xpu.set_stream(stream)
        try:
            return _capture_state()[0]
        finally:
            torch.xpu.set_stream(me)

    state = {"default": False, "side": False, "logs": 0}

    def _run():
        try:
            side = torch.xpu.graph.default_capture_stream
        except Exception:  # noqa: BLE001
            side = None
        while state["logs"] < 8:
            try:
                if _captures_open > 0:
                    _time.sleep(0.1)
                    continue
                dnow = _probe(torch.xpu.default_stream())
                if dnow and not state["default"]:
                    state["default"] = True
                    state["logs"] += 1
                    logger.warning(
                        "ArcXpuPlatform: ARC_GRAPH_ZOMBIE_ONSET default "
                        "stream now recording (no capture open) at %s "
                        "(ids: %s)", _time.strftime("%H:%M:%S"), _stream_ids())
                snow = side is not None and _probe(side)
                if snow and not state["side"]:
                    state["side"] = True
                    state["logs"] += 1
                    logger.warning(
                        "ArcXpuPlatform: ARC_GRAPH_ZOMBIE_ONSET side capture "
                        "stream now recording (no capture open) at %s (ids: "
                        "%s)", _time.strftime("%H:%M:%S"), _stream_ids())
            except Exception:  # noqa: BLE001 — watchdog must never disturb
                return
            _time.sleep(0.1)

    t = threading.Thread(target=_run, name="arc-graph-watchdog", daemon=True)
    t.start()


def _zombie_evidence() -> str:
    """Recovery proof for the hygiene log: fill on the (post-swap) current
    stream, give queued work a second, read back via a clean stream."""
    try:
        import time as _time

        c = torch.full((4,), 123.456, device=torch.xpu.current_device())
        _time.sleep(1.0)
        with torch.xpu.stream(torch.xpu.Stream()):
            v = c.to("cpu")
        return (f"post-swap canary[executed={bool((v == 123.456).all())} "
                f"vals={v.tolist()}]")
    except Exception as e:  # noqa: BLE001 — evidence only, never fatal
        return f"post-swap canary[ERR {e!r}]"


def _dezombie(tag: str) -> bool:
    """Clear a leaked command-graph recorder: swap the poisoned current
    stream for a fresh one. Only acts when the zombie conditions hold
    (flag says recording, probe healthy, no capture open by accounting).
    Probe-validated 2026-10-27: fresh streams are clean, their kernels
    execute, and the swap is serve-transparent. Returns True when swapped.
    """
    global _dezombie_logs
    capturing, probe_err = _capture_state()
    if not capturing or probe_err is not None or _captures_open > 0:
        return False
    try:
        dead = torch.xpu.current_stream()
        torch.xpu.set_stream(torch.xpu.Stream())
        if _dezombie_logs < _LOG_BUDGET:
            _dezombie_logs += 1
            logger.warning(
                "ArcXpuPlatform: graph stream hygiene (%s): current stream "
                "%s was left recording with no capture open — swapped to a "
                "fresh stream (post-swap capturing=%s, %s; ids: %s).",
                tag, dead, _capture_state()[0], _zombie_evidence(),
                _stream_ids())
        else:
            logger.warning(
                "ArcXpuPlatform: graph stream hygiene (%s): zombie stream "
                "swapped (post-swap capturing=%s).",
                tag, _capture_state()[0])
        return True
    except Exception as e:  # noqa: BLE001 — never mask the caller's path
        logger.error(
            "ArcXpuPlatform: graph stream hygiene (%s) FAILED: %r", tag, e)
        return False


def _hygiene(tag: str) -> None:
    """Post-boundary zombie check (call after each capture_end / replay)."""
    _dezombie(tag)


def _wrap_xpu_graph() -> None:
    """Instrument XPU graph boundaries (p12b zombie law, module docstring):
    keep the capture-open account on ``XPUGraph.capture_begin/capture_end``
    and run stream hygiene where a swap cannot be undone — after the graph
    context fully tears down (``torch.xpu.graph.__exit__`` restores the
    pre-capture current stream AFTER ``capture_end``, so a swap made inside
    ``capture_end`` would be silently reverted) and before/after each
    ``replay``. Applies to every graph use in the process — vllm's
    CUDAGraphWrapper (``torch.cuda.graph`` aliased to ``torch.xpu.graph``)
    and the speculator graph manager share these classes.
    """
    global _captures_open
    cls = torch.xpu.XPUGraph
    gcls = torch.xpu.graph
    if getattr(cls, "_arc_hygiene", False):
        return
    orig_begin = cls.capture_begin
    orig_end = cls.capture_end
    orig_replay = cls.replay
    orig_gexit = gcls.__exit__

    def _arc_capture_begin(self, *args, **kwargs):
        global _captures_open
        _captures_open += 1
        return orig_begin(self, *args, **kwargs)

    def _arc_capture_end(self, *args, **kwargs):
        global _captures_open
        try:
            return orig_end(self, *args, **kwargs)
        finally:
            # Accounting only (decrement even on a failed close — that
            # capture is over): the hygiene swap itself runs at graph_exit /
            # replay / serve boundaries, never here (see docstring above).
            _captures_open = max(0, _captures_open - 1)

    def _arc_replay(self, *args, **kwargs):
        # Pre-check: a replay enqueued on a zombie stream is lost together
        # with this decode step's outputs — clear the recorder FIRST.
        _hygiene("replay-entry")
        out = orig_replay(self, *args, **kwargs)
        _hygiene("replay")
        return out

    def _arc_graph_exit(self, *args):
        try:
            return orig_gexit(self, *args)
        finally:
            # Capture context fully torn down (capture_end returned or
            # raised, pre-capture stream restored): a still-recording flag
            # with no open capture is the leaked recorder — swap now;
            # nothing after this point overrides the current stream.
            _hygiene("graph_exit")

    cls.capture_begin = _arc_capture_begin
    cls.capture_end = _arc_capture_end
    cls.replay = _arc_replay
    cls._arc_hygiene = True
    gcls.__exit__ = _arc_graph_exit
    logger.info(
        "ArcXpuPlatform: XPU graph boundary hygiene installed (pid %d, "
        "XPUGraph capture_begin/capture_end/replay + graph context __exit__ "
        "wrapped).", os.getpid())


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
            # A GPU worker mid-inference: guaranteed-initialized context for
            # the onset watchdog (lazy backstop for installs pre-device-init).
            _ensure_watchdog()
            # Probe BEFORE the copy is enqueued: a zombie current stream must
            # be cleared before upstream records the copy event on it.
            capturing, probe_err = _capture_state()
            if capturing and probe_err is None and _captures_open == 0:
                # Zombie alarm (p12b): no capture is open by accounting, yet
                # the current stream reports recording. The RPC sample_tokens
                # path never sits inside a genuine capture, so this is the
                # leaked recorder. The in-flight request's forward was
                # enqueued on the dead queue and its outputs are garbage —
                # clear the zombie, then fail loudly (integrity law: never
                # return unverifiable tokens). Unreachable when the boundary
                # hygiene above did its job; this is the last-resort alarm.
                _dezombie("serve-time output alarm")
                raise RuntimeError(
                    "ArcXpuPlatform: zombie graph stream (recorder left open, "
                    "no capture active) cleared at serve time; the in-flight "
                    "request's outputs are unreliable — retry the request")
            orig_init(self, *args, **kwargs)
            if capturing:
                # Genuine capture window (or probe-doubt fail-safe): a device
                # sync is illegal inside capture; upstream behavior stands.
                # Capture-created events that later carry a command-graph
                # association are handled by the get_output self-heal below.
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
                        "upstream (genuine capture window active, "
                        "captures_open=%d).", _captures_open)
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
                    # wait here. Clear any leaked recorder so whatever
                    # survives us gets a live stream, then surface the
                    # original error.
                    _dezombie("get_output barrier illegal")
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
    # Boundary hygiene must be in place before any capture runs (worker init
    # captures follow the install hooks); idempotent per-process.
    _wrap_xpu_graph()
    _ensure_watchdog()
    _INSTALLED = True
    logger.info(
        "ArcXpuPlatform: graph-safe output patch installed (pid %d, "
        "AsyncOutput/AsyncPoolingOutput __init__ + get_output wrapped).",
        os.getpid())
