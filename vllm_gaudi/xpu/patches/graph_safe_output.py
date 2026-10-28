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

ROUND-2/3 EVIDENCE (2026-10-28, results/2026-10-28-p2-graph-p12b-r1r2/ and
rig legs arc-p2-graph-20261028-{002040,005232}): a request is fully served
clean (vllm warmup AND the first real request both returned 200 with clean
probes), then the flip lands BETWEEN requests — the fixed watchdog caught the
onset on the device-default stream at 00:59:52, two seconds BEFORE the next
request's alarm (00:59:54) and ~1s after the previous response. The worker
serves on the device default stream (onset id == alarm's poisoned id). The
poison is not one stream: the async copy stream (created at module import,
pre-flip) ALSO probes recording post-flip — every PRE-flip stream is dead,
only post-flip-created streams are live. Drain-path deaths showed
``torch.xpu.synchronize()`` (device-wide) refuses forever while any abandoned
queue records: barrier law — per-stream waits tolerate the zombie, device
syncs do not. Dezombie (swap current for a fresh stream) validated in
production three times (post-swap canaries all ``executed=True``).

FIX LAW (round 4): never enqueue on, wait for, or record events on a dead
stream. ``torch.xpu.set_stream`` transparently substitutes a cached live
replacement when its target probes recording with no capture open (the
``torch.cuda.set_stream`` alias is rebound too whenever the donor alias law
bound it to the same original — async_utils' stream ctx calls the cuda
name); ``Stream.wait_stream`` skips dead arguments (a dead queue produces
nothing worth waiting for); ``Event.record`` reroutes a dead argument to the
current stream; ``GPUModelRunner.execute_model`` entry hygiene clears a
worker thread still parked on the poisoned default BEFORE any forward is
enqueued (threads start on the device default, and thread-local currents mean
one dezombie does not fix another thread). The healthy-path output copy keeps
upstream's own blocking event — recorded on a live stream outside any
capture, therefore waitable — instead of the device-wide sync. The capture
accounting, boundary hygiene and serve-time alarm (raise: integrity law) stay
as verifiers for the unobserved mid-request flip class.

ROUND-4 RESULT (leg arc-p2-graph-20261028-013037): every guard worked as
designed — onset 01:37:57 armed them, ARC_GRAPH_SUBSTITUTE / WAIT_SKIP fired
on the next request's copy path — but the engine still died at the
serve-time alarm: ``execute_model`` and ``sample_tokens`` are SEPARATE RPCs
on DIFFERENT executor threads. The forward ran on a live thread (entry
hygiene held), yet the construction RPC ran on a thread still parked on the
dead default — a thread that never transitions never passes a guard — and
the alarm raised on a perfectly healthy request.

ROUND-5 LAW: entry hygiene on every worker-RPC entry that touches the GPU
(``execute_model`` AND ``sample_tokens`` — thread-local currents make
per-entry clearing mandatory), plus a step-poison flag: the execute_model
EXIT probe marks the step when the flip lands mid-forward (entry was clean,
exit dead — kernels may be lost) and the next sample_tokens raises on it
(integrity law: unverifiable outputs are never returned).

ROUND-5 RESULT (leg arc-p2-graph-20261028-020249): all machinery fired —
onset 02:10:10, ``execute_model-exit (mid-forward flip)`` dezombie on both
workers 02:10:11 (the flip landed between the clean entry probe and the exit
probe — genuinely MID-FORWARD this time; earlier rounds caught it between
requests — the flip timing inside the post-completion window varies), the
step-poison flag raised at sample_tokens, no garbage served, engine died.

ROUND-6 LAW (the lost-step retry): kernels enqueued on a queue before it
flipped NEVER execute (canary law) — so a mid-forward flip loses the ENTIRE
step atomically: no KV writes, no sampler RNG effects, no partial state. The
execute_model wrap therefore CLEARS the recorder and RE-RUNS the same call
once on the live stream; the retry's outputs are correct and the flip is
serve-transparent. A second consecutive exit-poison (double flip in one
step) gives up loudly. Symmetrically, a dead current at the AsyncOutput
construction can no longer mean a lost forward (the exit probe verified the
forward completed on a live stream): dezombie and PROCEED unless the poison
flag is set — the alarm stays only for the double-flip corner.

Provenance: error family + fix direction = plan Appendix C defect (b);
capture-probe fail-safe law = D-line waveD patch_gdn_metadata_026.py
(``_iexp_is_capturing``, results/2026-10-18-waveD-anchor-analysis.log);
zombie discrimination + fresh-stream recovery law = rig probes
2026-10-27/28 (probe_zombie{,2}.py, results/2026-10-27-p2-graph-defectb-answer/);
between-requests flip + all-pre-flip-streams-dead + barrier law = p12b
rounds 1-3 legs (results/2026-10-28-p2-graph-p12b-r1r2/, rig legs
arc-p2-graph-20261028-*).
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

# --- stream-reroute state (round 4) ---
# Armed by the first zombie evidence (watchdog onset, dezombie action, or a
# guard detection). Pre-flip this is False and every guard below is a single
# boolean check — zero probing on the hot path.
_zombie_ever = False
# Dead stream -> its live replacement (one per dead stream, so ordering
# among things rerouted from the same dead stream is preserved).
_LIVE_REPLACEMENTS: dict[int, torch.xpu.Stream] = {}
_SUBST_LOGS = 0
_WAIT_SKIP_LOGS = 0
_RECORD_REROUTE_LOGS = 0
_SYNC_TRIP_LOGS = 0
# The unwrapped set_stream, captured at install: internal switches (watchdog
# side-probe, _stream_poisoned) must bypass the substitution shim.
_ORIG_SET_STREAM = None
# Set by the execute_model EXIT probe when the flip landed mid-forward
# (entry clean, exit dead): the step's kernels may be lost, so the following
# sample_tokens fails the request loudly (integrity law, round 5).
_poisoned_step = False


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
    ids — names the poisoned stream class when the alarm fires. NOTE:
    ``torch.xpu.default_stream()`` raises on this build (round-2 leg,
    ``default=ERR``) — a fresh thread's own current IS the device default,
    which is what the watchdog probes instead."""
    try:
        side = getattr(torch.xpu.graph, "default_capture_stream", None)
        return f"cur={torch.xpu.current_stream()} side={side}"
    except Exception as e:  # noqa: BLE001
        return f"ids ERR {e!r}"


def _note_zombie() -> None:
    """Arm the stream-reroute guards (idempotent; see round-4 fix law)."""
    global _zombie_ever
    _zombie_ever = True


def _stream_poisoned(s) -> bool:
    """Probe a specific stream object by switching to it (thread-local,
    unwrapped set/restore — must bypass the substitution shim). False while a
    capture is open (the capture stream legitimately records then)."""
    if _captures_open > 0:
        return False
    set_ = _ORIG_SET_STREAM or torch.xpu.set_stream
    me = torch.xpu.current_stream()
    if me == s:
        return _capture_state()[0]
    set_(s)
    try:
        return _capture_state()[0]
    finally:
        set_(me)


def _replacement_for(dead) -> torch.xpu.Stream:
    """Cached live replacement for a dead stream (created post-flip = live,
    probe-validated; one per dead stream preserves rerouted ordering)."""
    key = id(dead)
    live = _LIVE_REPLACEMENTS.get(key)
    if live is None:
        live = torch.xpu.Stream()
        _LIVE_REPLACEMENTS[key] = live
    return live


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
    """Onset detector (p12b round 3): poll for a recording flag with no
    capture open and log the first transition with a timestamp — bracketing
    the flip against vllm's log timeline. This thread's OWN current stream
    is the device default (fresh threads start on the default; the broken
    ``torch.xpu.default_stream()`` API is never called), so an onset here
    means the DEFAULT stream — the worker's serving stream if serving runs
    on the default — is the poisoned one. The side capture stream is
    probed via an explicit thread-local switch (unwrapped set_stream). On
    onset the reroute guards are armed immediately (round 4): every later
    switch onto any dead stream is substituted before the next request can
    enqueue on it. Daemon; stops after the transition budget.
    """
    import threading
    import time as _time

    state = {"own": False, "side": False, "logs": 0}

    def _run():
        try:
            side = torch.xpu.graph.default_capture_stream
        except Exception:  # noqa: BLE001
            side = None
        set_ = _ORIG_SET_STREAM or torch.xpu.set_stream
        while state["logs"] < 8:
            try:
                if _captures_open > 0:
                    _time.sleep(0.1)
                    continue
                if not state["own"] and _capture_state()[0]:
                    state["own"] = True
                    state["logs"] += 1
                    _note_zombie()
                    logger.warning(
                        "ArcXpuPlatform: ARC_GRAPH_ZOMBIE_ONSET this "
                        "thread's current (device default) stream now "
                        "RECORDING with no capture open at %s — reroute "
                        "guards ARMED (ids: %s)",
                        _time.strftime("%H:%M:%S"), _stream_ids())
                if side is not None and not state["side"]:
                    me = torch.xpu.current_stream()
                    set_(side)
                    try:
                        snow = _capture_state()[0]
                    finally:
                        set_(me)
                    if snow:
                        state["side"] = True
                        state["logs"] += 1
                        _note_zombie()
                        logger.warning(
                            "ArcXpuPlatform: ARC_GRAPH_ZOMBIE_ONSET side "
                            "capture stream now RECORDING with no capture "
                            "open at %s (ids: %s)",
                            _time.strftime("%H:%M:%S"), _stream_ids())
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
    _note_zombie()
    try:
        dead = torch.xpu.current_stream()
        set_ = _ORIG_SET_STREAM or torch.xpu.set_stream
        set_(_replacement_for(dead))
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
    ``replay``. Post-round-4 the set_stream shim covers these transitions
    too; the wraps stay as verifiers for replays that bypass set_stream.
    Applies to every graph use in the process — vllm's CUDAGraphWrapper
    (``torch.cuda.graph`` aliased to ``torch.xpu.graph``) and the
    speculator graph manager share these classes.
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


def _wrap_stream_apis() -> None:
    """Round-4 fix law (module docstring): functional guards on the stream/
    event APIs so nothing is ever enqueued on, waited from, or recorded onto
    a dead (recording-with-no-capture) stream.

    - ``torch.xpu.set_stream(s)``: if s probes dead post-switch, switch to
      its cached live replacement instead (and rebind the ``torch.cuda``
      alias when the donor alias law bound it to the same original —
      async_utils' stream ctx switches via the cuda name).
    - ``Stream.wait_stream(s)``: skip dead arguments — a dead queue never
      progresses, so waiting on it could only hang; it also produces no
      outputs anyone needs ordered.
    - ``Event.record(s)``: reroute a dead argument to the current stream so
      the event actually fires when the (rerouted) copies complete.
    - ``torch.xpu.synchronize``: log-only tripwire kept from round 3 (a
      device-wide sync is forever illegal once a queue is abandoned).

    All guards arm on ``_zombie_ever`` only — before the flip they cost one
    boolean check; genuine captures (``_captures_open > 0``) are exempt.
    """
    global _ORIG_SET_STREAM, _SUBST_LOGS, _WAIT_SKIP_LOGS
    global _RECORD_REROUTE_LOGS, _SYNC_TRIP_LOGS
    import traceback

    _ORIG_SET_STREAM = torch.xpu.set_stream
    orig_set = _ORIG_SET_STREAM
    orig_wait = torch.xpu.Stream.wait_stream
    orig_record = torch.xpu.Event.record
    orig_sync = torch.xpu.synchronize

    def _arc_set_stream(s):
        global _SUBST_LOGS
        orig_set(s)
        if _zombie_ever and _captures_open == 0:
            try:
                if _capture_state()[0]:
                    live = _replacement_for(s)
                    orig_set(live)
                    if _SUBST_LOGS < _LOG_BUDGET:
                        _SUBST_LOGS += 1
                        logger.warning(
                            "ArcXpuPlatform: ARC_GRAPH_SUBSTITUTE set_stream "
                            "target %s probes recording (no capture open) — "
                            "rerouted to live %s (ids: %s).",
                            s, live, _stream_ids())
            except Exception:  # noqa: BLE001 — never block a stream switch
                pass

    def _arc_wait_stream(self, s=None):
        global _WAIT_SKIP_LOGS
        if _zombie_ever and s is not None and _captures_open == 0:
            try:
                if _stream_poisoned(s):
                    if _WAIT_SKIP_LOGS < _LOG_BUDGET:
                        _WAIT_SKIP_LOGS += 1
                        logger.warning(
                            "ArcXpuPlatform: ARC_GRAPH_WAIT_SKIP "
                            "wait_stream argument %s probes recording (no "
                            "capture open) — skip (dead queue never "
                            "progresses; ids: %s).", s, _stream_ids())
                    return None
            except Exception:  # noqa: BLE001 — fall through to upstream
                pass
        return orig_wait(self, s)

    def _arc_event_record(self, stream=None):
        global _RECORD_REROUTE_LOGS
        if _zombie_ever and stream is not None and _captures_open == 0:
            try:
                if _stream_poisoned(stream):
                    if _RECORD_REROUTE_LOGS < _LOG_BUDGET:
                        _RECORD_REROUTE_LOGS += 1
                        logger.warning(
                            "ArcXpuPlatform: ARC_GRAPH_RECORD_REROUTE "
                            "Event.record argument %s probes recording (no "
                            "capture open) — recording on current stream "
                            "instead (ids: %s).", stream, _stream_ids())
                    stream = None
            except Exception:  # noqa: BLE001 — fall through to upstream
                pass
        return orig_record(self, stream)

    def _arc_xpu_sync(*args, **kwargs):
        global _SYNC_TRIP_LOGS
        if (_zombie_ever and _captures_open == 0
                and _capture_state()[0] and _SYNC_TRIP_LOGS < _LOG_BUDGET):
            _SYNC_TRIP_LOGS += 1
            logger.warning(
                "ArcXpuPlatform: ARC_GRAPH_TRIPWIRE xpu.synchronize called "
                "with zombie recording flag (device-wide barriers stay "
                "illegal while a queue is abandoned; ids: %s). Caller:\n%s",
                _stream_ids(), "".join(traceback.format_stack()[-6:]))
        return orig_sync(*args, **kwargs)

    torch.xpu.set_stream = _arc_set_stream
    torch.xpu.Stream.wait_stream = _arc_wait_stream
    torch.xpu.Event.record = _arc_event_record
    torch.xpu.synchronize = _arc_xpu_sync
    # The donor wheel's _torch_cuda_wrapper aliases bound torch.cuda.* to
    # the ORIGINAL torch.xpu functions at import time, so cuda-named calls
    # bypass the wraps above — rebind when the alias demonstrably points at
    # the same original (identity check against the pre-wrap object).
    if getattr(torch.cuda, "set_stream", None) is orig_set:
        torch.cuda.set_stream = _arc_set_stream
    logger.info(
        "ArcXpuPlatform: zombie stream-reroute guards installed (pid %d, "
        "set_stream substitution + wait_stream skip + Event.record reroute "
        "+ synchronize tripwire).", os.getpid())


def _wrap_runner_entry() -> bool:
    """Round-4/5 save points on ``GPUModelRunner`` (XPUModelRunner subclasses
    it without overriding these methods — wheel-verified 2026-10-28):

    - ``execute_model`` ENTRY hygiene: worker threads start life parked on
      the device default stream — the poisoned one — and thread-local
      currents mean a dezombie on one thread does not fix another. Probing
      (and clearing) at every entry guarantees no forward is ever enqueued
      on a dead queue, even for the first RPC on a fresh thread.
    - ``execute_model`` EXIT probe + lost-step RETRY (rounds 5/6): entry was
      clean but the current is dead at exit → the flip landed MID-FORWARD.
      Kernels enqueued before the flip never execute, so the whole step was
      lost atomically (no KV writes, no partial state) — clear the recorder
      and RE-RUN the same call once on the live stream. A second consecutive
      exit-poison sets the step-poison flag and the next sample_tokens fails
      the request loudly (integrity law).
    - ``sample_tokens`` ENTRY hygiene (round 5, the round-4 lesson):
      execute_model and sample_tokens are SEPARATE RPCs on DIFFERENT
      executor threads — a construction RPC on a thread parked on the dead
      default never transitions, so it never passes a set_stream guard.
      Clearing here keeps the construction probe healthy so the alarm stays
      a real integrity tripwire instead of a thread-locals artifact.
    """
    global _poisoned_step
    try:
        from vllm.v1.worker.gpu import model_runner as mr
    except Exception as e:  # noqa: BLE001 — circular import at register()
        logger.warning(
            "ArcXpuPlatform: runner-entry hygiene install deferred (%r) — "
            "platform hooks retry post-import.", e)
        return False
    if getattr(mr.GPUModelRunner, "_arc_entry_hygiene", False):
        return True
    orig_exec = mr.GPUModelRunner.execute_model
    orig_sample = mr.GPUModelRunner.sample_tokens

    def _arc_execute_model(self, *args, **kwargs):
        global _poisoned_step
        _hygiene("execute_model-entry")

        def _flip_poisoned() -> bool:
            if _captures_open > 0:
                return False
            capturing, probe_err = _capture_state()
            return capturing and probe_err is None

        for attempt in (1, 2):
            try:
                out = orig_exec(self, *args, **kwargs)
            except RuntimeError as e:
                # Raise-flavored flip: a mid-forward op errored with the
                # recording-queue text. Same lost-step law — clear, retry.
                if (attempt == 1 and _captures_open == 0
                        and (_COMMAND_GRAPH_ERR in str(e)
                             or "recording" in str(e))):
                    _note_zombie()
                    _dezombie(
                        "execute_model-raise (mid-forward flip, retrying)")
                    continue
                raise
            if not _flip_poisoned():
                # Healthy exit — includes capture contexts (boot captures),
                # where probing is meaningless and upstream stands.
                return out
            # The recorder was opened DURING this forward: every kernel of
            # the attempt was enqueued on the now-dead queue and none of
            # them executed — the step was lost atomically. Clear and retry.
            _note_zombie()
            if attempt == 1:
                _dezombie("execute_model-exit (mid-forward flip, retrying)")
                continue
            _dezombie("execute_model-exit (mid-forward flip, retry failed)")
            _poisoned_step = True
            return out
        return out  # unreachable

    def _arc_sample_tokens(self, *args, **kwargs):
        global _poisoned_step
        if _poisoned_step:
            _poisoned_step = False
            _dezombie("sample_tokens-entry (poisoned forward)")
            raise RuntimeError(
                "ArcXpuPlatform: zombie graph stream flipped mid-forward "
                "twice in one step — the outputs were enqueued on a dead "
                "queue and are unreliable (integrity law: retry the "
                "request)")
        _hygiene("sample_tokens-entry")
        return orig_sample(self, *args, **kwargs)

    mr.GPUModelRunner.execute_model = _arc_execute_model
    mr.GPUModelRunner.sample_tokens = _arc_sample_tokens
    mr.GPUModelRunner._arc_entry_hygiene = True
    logger.info(
        "ArcXpuPlatform: runner-entry hygiene installed (pid %d, "
        "GPUModelRunner.execute_model + sample_tokens wrapped).", os.getpid())
    return True


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
            global _poisoned_step
            # A GPU worker mid-inference: guaranteed-initialized context for
            # the onset watchdog (lazy backstop for installs pre-device-init).
            _ensure_watchdog()
            # Probe BEFORE the copy is enqueued: a zombie current stream must
            # be cleared before upstream records the copy event on it.
            capturing, probe_err = _capture_state()
            if capturing and probe_err is None and _captures_open == 0:
                # Dead current at construction (round 6 law): the forward of
                # THIS output was already verified complete on a live stream
                # by the execute_model exit probe — so the flip landed AFTER
                # it (or this thread was parked on the default; the
                # sample_tokens entry hygiene normally clears that first).
                # Clear the recorder and PROCEED: the outputs are
                # trustworthy. Raise only when the step-poison flag says a
                # retry already failed (double flip in one step).
                _dezombie("serve-time output dezombie")
                if _poisoned_step:
                    _poisoned_step = False
                    raise RuntimeError(
                        "ArcXpuPlatform: zombie graph stream (recorder left "
                        "open, no capture active) cleared at serve time; the "
                        "in-flight request's outputs are unreliable "
                        "(integrity law: retry the request)")
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
            # Healthy path (round 4): upstream's own blocking event is the
            # barrier — the stream-reroute guards above make the copy ctx
            # and the event record land on a LIVE stream outside any
            # capture, so the event is waitable. The round-1/2 device-wide
            # synchronize is gone: it is forever illegal once a poisoned
            # queue has been abandoned (drain-path deaths, round 2/3).
            if _applied_logs < _LOG_BUDGET:
                _applied_logs += 1
                logger.info(
                    "ArcXpuPlatform: graph-safe output applied — reroute "
                    "guards active, upstream copy event retained.")

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
                return orig_get_output(self)

        return get_output

    au.AsyncOutput.__init__ = _graph_safe_init(_orig_async_output_init)
    au.AsyncPoolingOutput.__init__ = _graph_safe_init(_orig_pooling_init)
    au.AsyncOutput.get_output = _graph_safe_get_output(_orig_async_output_get)
    au.AsyncPoolingOutput.get_output = _graph_safe_get_output(
        _orig_pooling_get)
    # Boundary hygiene must be in place before any capture runs (worker init
    # captures follow the install hooks); idempotent per-process. The
    # stream-reroute guards (round 4) and the runner-entry save point
    # install alongside; the watchdog arms them at onset.
    _wrap_xpu_graph()
    _wrap_stream_apis()
    _wrap_runner_entry()
    _ensure_watchdog()
    _INSTALLED = True
    logger.info(
        "ArcXpuPlatform: graph-safe output patch installed (pid %d, "
        "AsyncOutput/AsyncPoolingOutput __init__ + get_output wrapped).",
        os.getpid())
