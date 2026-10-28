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

ROUND-6 RESULT (leg arc-p2-graph-20261028-023623): the retry fired
("mid-forward flip, retrying", both workers) — and attempt 2 ALSO exited
poisoned: the "double flip" raise killed the engine. The substitution ids
show WHY: the dezombie's own fresh stream (created at 02:43:46) was the
TARGET of a substitution one second later — dead already. This is not a
one-time flip but a POISON STORM: a multi-second window (~3+ s observed)
during which every queue dies, existing and freshly created alike. No leg
has ever outlived the storm, so whether it is time-bounded is unknown.

ROUND-7 LAW (storm-wait): on the first exit-poison, do not retry
immediately — poll freshly-created streams (200 ms cadence) until one
survives, up to 15 s, then retry the forward on THAT stream. If a stream
survives, the storm is bounded: log its measured duration and serve on. If
none survives in 15 s, the storm is permanent on this stack: fail loudly
with that verdict (definitive evidence for the posture decision).
Corpse-proof cache: a replacement created mid-storm is itself dead — every
``_LIVE_REPLACEMENTS`` hit is re-probed and evicted if recording, and a
dezombie whose swapped-to stream dies under it evicts + storm-waits in
place (otherwise the first healthy post-storm request's copy path would
substitute onto a corpse and its copy_event would never fire). r6 raw also
showed streams dead AT CREATION (retry canary executed=False, garbage
vals). CORRECTION (results/2026-10-28-p2-graph-p12b-r6): the r6 raises
were the SHUTDOWN-path torch.accelerator.synchronize (barrier law,
post-mortem noise), NOT construction-path waits — the wait_stream caller
reroute stays as law-correct hardening (a dead waiter's ordering is void),
but the storm itself is what killed the retry, and the storm-wait is the
answer to it.

ROUND-7 RESULT (leg arc-p2-graph-20261028-031336, image 02961955b15a):
ARC_GRAPH_STORM_ENDED fired at 0.0 s on every poll — fresh streams are
ALWAYS clean at creation (and stay clean while unused: the cache-hit
replacement probed clean + canary OK a full second after creation), but
every stream that RAN A FORWARD was recording at that forward's exit —
three fresh streams, three on-use flips within ~1 s of use. THERE IS NO
TIME-BOUNDED STORM: the first flip puts the runtime into a PERMANENT
state where any queue that enqueues real model work is hijacked into
command-graph recording mid-forward. Waiting cannot produce a servable
stream; stream-layer recovery is impossible post-trigger. POSTURE
DECISION (results/2026-10-28-p2-graph-p12b-r7): the graph lane is
abandoned for serving — P2 runs graphs-OFF eager
(VLLM_XPU_ENABLE_XPU_GRAPH unset → platforms/xpu.py:301 → NONE, no
capture sessions, the poison cannot arm). This module's layers stay as
proven integrity keepers should graphs ever be re-enabled; on the eager
lane they are dormant (guards arm only on the first flip, which never
comes without graphs).

ROUND-8 LAW — ROOT CAUSE (champion control leg 2026-10-28 + code diff):
the D champion (lsv-test, vllm 0.26.1.dev0+g568afb3a1, torch 2.12.0+xpu)
serves FULL_DECODE_ONLY graphs + MTP k4 on THIS rig today — the lane is
viable on this hardware+driver; the poison is OUR stack. The diff:
their capture is ``torch.cuda.graph(graph, self.pool)`` — NO stream arg →
torch.xpu captures on its own ``default_capture_stream`` (the side
stream). Our 0.30 wrapper passes ``stream=self._capture_stream(desc)``
which returns ``current_stream()`` — the worker's device DEFAULT — and on
torch 2.13.0+xpu the capture stream stays graph-associated after
capture_end ("poison enters during capture",
results/2026-10-27-p2-graph-defectb-answer: 27/27 captures finish clean,
nothing runs in between, yet the default is recording-dead at the first
real op). FIX: ``CudaGraphManager._capture_stream`` is wrapped to return
``torch.xpu.graph.default_capture_stream`` — captures never touch the
serving default (champion discipline; replays were always fine: k1k 200
in every poisoned round). Prefix-caching OFF also flipped
(results/2026-10-28-p2-graph-prefixoff) — trigger was never the prefix
write, it was the capture stream choice all along.

ROUND-8b LAW — THE LAZY-INIT HOLE (r8 leg INVALID, r8b probe): the round-8
wrapper was a NO-OP for capture #1. ``torch.xpu.graph.default_capture_stream``
is a class attribute lazily initialized INSIDE ``graph.__init__``
(torch/xpu/graphs.py:167-183) — it is None until the first ``graph()``
context entry, and vllm evaluates ``stream=self._capture_stream(desc)``
BEFORE that entry. So capture #1 read None, fell through to upstream
(current_stream = the device default) and poisoned the serving default at
boot; captures #2..84 rode the side stream. The champion passes NO stream
arg, so torch's own lazy init fires inside ``__init__`` and capture #1
rides the side stream from the start. r9 fix: the wrapper EAGERLY
initializes the class attribute (constructs ``torch.xpu.Stream()`` and
assigns it) exactly as torch's own lazy init would — champion parity for
every capture. Bare-torch probe matrix (r8b): toy graphs never poison any
stream (capture-on-default included, same-stream capture+replay clean) —
the leak needs the real captured runnable (CCL, capture-time events,
pool-shared 84-graph boot), matching the p13 note; single stream world at
serve confirmed (xpu_model_runner.py:60 rebinds torch.cuda.graph →
torch.xpu.graph, stream= honored). results/2026-10-28-p2-graph-p12b-r8/.

ROUND-9..11 LAW — RUNTIME-SIDE TRIGGER (r9/r10/r11 legs): with ALL captures
on the side stream (r9 verified engaged), UR_L0_USE_IMMEDIATE_COMMANDLISTS=0
(r10), and --async-scheduling (r11), the flip STILL lands — first
generation completes 200, then the device default reads recording in the
COMPLETION window and everything after dies. Serve-level champion deltas
are exhausted; the trigger lives in the vllm-0.30/torch-2.13 completion
path (AsyncOutput copy / cleanup — the one region that runs once per
request and never in boot warmup). r12 = onset TRACER (ARC_TRACE_FLIP=1):
10 ms watchdog polls, all-thread python stack dump at the flip moment,
capture-state transition logging at every hygiene boundary. Diagnostic
legs only — inert by default. results/2026-10-28-p2-graph-p12b-r{9,10-icl0,11-async}/.

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
    probe-validated; one per dead stream preserves rerouted ordering).
    Round 7: a replacement created DURING the poison storm can itself have
    flipped (r6: the dezombie stream was a substitution target 1s later) —
    re-probe every cache hit and evict corpses; a stream created after the
    storm ends is live, so the cache self-heals on first post-storm use."""
    key = id(dead)
    live = _LIVE_REPLACEMENTS.get(key)
    if live is not None and _stream_poisoned(live):
        _LIVE_REPLACEMENTS.pop(key, None)
        logger.warning(
            "ArcXpuPlatform: ARC_GRAPH_STORM cached replacement %s for dead "
            "%s probes recording — evicted, creating fresh (ids: %s).",
            live, dead, _stream_ids())
        live = None
    if live is None:
        live = torch.xpu.Stream()
        _LIVE_REPLACEMENTS[key] = live
    return live


_WATCHDOG_STARTED = False

# r12 onset tracer (diagnostic legs only; inert unless ARC_TRACE_FLIP=1):
# 10 ms watchdog polls + all-thread python stack dump the moment the
# recording flag is first observed, and capture-state transition logging at
# every hygiene boundary — names the frame executing at the flip.
_TRACE_FLIP = os.environ.get("ARC_TRACE_FLIP", "") == "1"
_TRACE_LAST = None
_TRACE_LOGS = 0
_TRACE_BUDGET = 400
_TRACE_KERNELS_OK = False


def _dump_threads(marker: str) -> None:
    """All-thread python stacks at the flip moment (r12 tracer)."""
    import sys
    import threading
    import traceback
    try:
        frames = sys._current_frames()
        names = {t.ident: t.name for t in threading.enumerate()}
        parts = []
        for tid, frame in frames.items():
            stack = traceback.format_stack(frame)[-10:]
            parts.append(f"[{names.get(tid, tid)}]\n" + "".join(stack))
        logger.warning(
            "ARC_GRAPH_ONSET_STACKS %s begin\n%s\nARC_GRAPH_ONSET_STACKS end",
            marker, "\n".join(parts))
    except Exception as e:  # noqa: BLE001 — tracer must never disturb
        logger.warning("ARC_GRAPH_ONSET_STACKS %s dump failed %r", marker, e)


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
            _wrap_kernel_launch_tracer()  # r12b: before any compiled run
        except Exception:  # noqa: BLE001 — tracer must never disturb
            pass
        try:
            side = torch.xpu.graph.default_capture_stream
        except Exception:  # noqa: BLE001
            side = None
        set_ = _ORIG_SET_STREAM or torch.xpu.set_stream
        poll = 0.01 if _TRACE_FLIP else 0.1
        while state["logs"] < 8:
            try:
                if _captures_open > 0:
                    _time.sleep(poll)
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
                    if _TRACE_FLIP:
                        _dump_threads("onset-default")
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
                        if _TRACE_FLIP:
                            _dump_threads("onset-side")
            except Exception:  # noqa: BLE001 — watchdog must never disturb
                return
            _time.sleep(poll)

    t = threading.Thread(target=_run, name="arc-graph-watchdog", daemon=True)
    t.start()


def _wrap_kernel_launch_tracer() -> None:
    """r12b tracer: probe capture-state around EVERY triton kernel launch
    — names the kernel whose launch flips the serving stream
    (ARC_TRACE_FLIP legs only; inert otherwise). r12 pinned the flip
    inside a piecewise-compiled callable (extern_kernels.mm on the stack,
    output_code.py:763); toy compiles never flip, so the real kernels are
    the remaining suspects. NOTE: never assign onto torch.ops entries —
    the r12b leg's aten.mm.out wrap corrupted the op registry
    ("Expected qualname ... but got _traced_mm") and killed boot; an
    aten-op flip between two triton kernels still shows as FLIP at the
    next triton-pre."""
    if not _TRACE_FLIP or _zombie_ever:
        return
    global _TRACE_KERNELS_OK
    if _TRACE_KERNELS_OK:
        return
    wrapped = []

    def _probe(tag):
        if _TRACE_LOGS >= _TRACE_BUDGET or _zombie_ever or _captures_open > 0:
            return
        try:
            rec = _capture_state()[0]
        except Exception:  # noqa: BLE001
            return
        if rec:
            logger.warning(
                "ArcXpuPlatform: ARC_GRAPH_TRACE FLIP at %s — stream now "
                "recording (captures_open=%d)", tag, _captures_open)

    try:
        from torch._inductor.runtime import triton_heuristics as _th

        orig_run = _th.CachingAutotuner.run

        def _traced_run(self, *args, **kwargs):
            _probe("triton-pre")
            out = orig_run(self, *args, **kwargs)
            _probe("triton-post")
            return out

        _th.CachingAutotuner.run = _traced_run
        wrapped.append("triton:CachingAutotuner.run")
    except Exception as e:  # noqa: BLE001 — optional probe
        logger.warning("ArcXpuPlatform: tracer triton wrap skipped %r", e)

    logger.info(
        "ArcXpuPlatform: r12b kernel-launch tracer installed (%s).",
        ", ".join(wrapped))
    _TRACE_KERNELS_OK = bool(wrapped)


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
        if _capture_state()[0]:
            # Round 7: the swapped-to stream died too — cache race or a
            # mid-storm entry. Evict it and wait out the storm on fresh
            # streams; the survivor is left current. If the storm is
            # permanent the current stays dead and the caller's integrity
            # probes trip (fail loud, posture evidence).
            _LIVE_REPLACEMENTS.pop(id(dead), None)
            logger.warning(
                "ArcXpuPlatform: ARC_GRAPH_STORM dezombie target died under "
                "us (storm, %s) — waiting out the storm.", tag)
            _wait_out_storm()
        return True
    except Exception as e:  # noqa: BLE001 — never mask the caller's path
        logger.error(
            "ArcXpuPlatform: graph stream hygiene (%s) FAILED: %r", tag, e)
        return False


def _hygiene(tag: str) -> None:
    """Post-boundary zombie check (call after each capture_end / replay)."""
    if _TRACE_FLIP:  # r12 tracer: log capture-state TRANSITIONS only
        global _TRACE_LAST, _TRACE_LOGS
        if _TRACE_LOGS < _TRACE_BUDGET:
            now = _captures_open > 0 or _capture_state()[0]
            if now != _TRACE_LAST:
                _TRACE_LAST = now
                _TRACE_LOGS += 1
                logger.warning(
                    "ArcXpuPlatform: ARC_GRAPH_TRACE state->%s at %s "
                    "(captures_open=%d)", now, tag, _captures_open)
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
        if _zombie_ever and _captures_open == 0:
            try:
                if s is not None and _stream_poisoned(s):
                    if _WAIT_SKIP_LOGS < _LOG_BUDGET:
                        _WAIT_SKIP_LOGS += 1
                        logger.warning(
                            "ArcXpuPlatform: ARC_GRAPH_WAIT_SKIP "
                            "wait_stream argument %s probes recording (no "
                            "capture open) — skip (dead queue never "
                            "progresses; ids: %s).", s, _stream_ids())
                    return None
                if _stream_poisoned(self):
                    # r6 leg lines 467/491: a wait issued ON a dead queue
                    # ("dead.wait_stream(live)") raises "wait cannot be
                    # called for a queue which is recording" and killed the
                    # engine — the argument guard cannot see this. The dead
                    # waiter's ordering is void (nothing it enqueued will
                    # ever execute), so let the thread's live current wait
                    # instead.
                    live = torch.xpu.current_stream()
                    if live is not self:
                        if _WAIT_SKIP_LOGS < _LOG_BUDGET:
                            _WAIT_SKIP_LOGS += 1
                            logger.warning(
                                "ArcXpuPlatform: ARC_GRAPH_WAIT_REROUTE "
                                "wait_stream caller %s probes recording (no "
                                "capture open) — waiting from current %s "
                                "instead (ids: %s).", self, live,
                                _stream_ids())
                        return orig_wait(live, s)
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


def _wait_out_storm(deadline_s: float = 15.0):
    """Round-7 storm-wait: poll freshly-created streams until one survives
    (probes clean), then leave it current and return it. Returns None when
    the deadline passes — every queue dying for that long means the poison
    is permanent on this stack (posture-decision verdict, module docstring).
    Fresh-stream creation during the storm is safe: r6 substitutions created
    and rerouted streams mid-storm without device errors.
    """
    import time as _time
    set_ = _ORIG_SET_STREAM or torch.xpu.set_stream
    t0 = _time.monotonic()
    last = None
    while _time.monotonic() - t0 < deadline_s:
        s = torch.xpu.Stream()
        set_(s)
        capturing, probe_err = _capture_state()
        if not capturing and probe_err is None:
            logger.warning(
                "ArcXpuPlatform: ARC_GRAPH_STORM_ENDED a fresh stream "
                "survived after %.1fs — retrying the lost step on it "
                "(ids: cur=%s).", _time.monotonic() - t0, s)
            return s
        last = s
        _time.sleep(0.2)
    logger.error(
        "ArcXpuPlatform: ARC_GRAPH_STORM_PERMANENT no fresh stream survived "
        "%.1fs — every queue dies on this stack (last probe: %s).",
        deadline_s, last)
    return None


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

        for attempt in (1, 2, 3):
            try:
                out = orig_exec(self, *args, **kwargs)
            except RuntimeError as e:
                # Raise-flavored flip: a mid-forward op errored with the
                # recording-queue text. Same lost-step law — storm-wait,
                # then retry.
                if (attempt < 3 and _captures_open == 0
                        and (_COMMAND_GRAPH_ERR in str(e)
                             or "recording" in str(e))):
                    _note_zombie()
                    if _wait_out_storm() is None:
                        _poisoned_step = True
                        raise
                    continue
                raise
            if not _flip_poisoned():
                # Healthy exit — includes capture contexts (boot captures),
                # where probing is meaningless and upstream stands.
                return out
            # The recorder was opened DURING this forward: every kernel of
            # the attempt was enqueued on the now-dead queue and none of
            # them executed — the step was lost atomically. Wait out the
            # poison storm, then re-run the same call on the surviving
            # stream (round 7 — an immediate retry just dies with it).
            _note_zombie()
            if attempt < 3:
                _dezombie("execute_model-exit (mid-forward flip, "
                          "storm-waiting)")
                if _wait_out_storm() is None:
                    _poisoned_step = True
                    return out
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


def _wrap_capture_stream() -> bool:
    """Round-8 root-cause fix: capture on torch.xpu's OWN capture stream,
    never the worker's current (device default). Champion discipline (D
    v133 / vllm 0.26.1: ``torch.cuda.graph(graph, pool)`` with no stream
    arg) — their captures ride ``torch.xpu.graph.default_capture_stream``
    (the side stream) and the serving default stays clean through boot and
    serving (champion control leg 2026-10-28: graphs + MTP survive on this
    rig). Our 0.30 wrapper passes ``stream=current_stream()`` and on torch
    2.13.0+xpu the capture stream stays graph-associated after
    capture_end — the device default is recording-dead at boot capture
    time and the first real op on it kills serving
    (results/2026-10-27-p2-graph-defectb-answer). Replays are unaffected
    (k1k 200 in every poisoned round).
    """
    try:
        from vllm.v1.worker.gpu import cudagraph_utils as cgu
    except Exception as e:  # noqa: BLE001 — circular import at register()
        logger.warning(
            "ArcXpuPlatform: capture-stream fix install deferred (%r) — "
            "platform hooks retry post-import.", e)
        return False
    manager = getattr(cgu, "CudaGraphManager", None)
    if manager is None or getattr(manager, "_arc_capture_stream", False):
        return True

    def _arc_capture_stream(self, desc=None):
        side = getattr(torch.xpu.graph, "default_capture_stream", None)
        if side is None:
            # Round-8b hole: the class attr is None until the first
            # graph() context entry, and this method runs BEFORE that
            # entry (vllm builds the stream= arg with it) — capture #1
            # would fall through to the device default and poison the
            # serving stream at boot. Eager init, identical to torch's
            # own lazy init in graph.__init__.
            side = torch.xpu.Stream()
            torch.xpu.graph.default_capture_stream = side
        return side

    manager._capture_stream = _arc_capture_stream
    manager._arc_capture_stream = True
    logger.info(
        "ArcXpuPlatform: capture-stream fix installed (pid %d, captures "
        "ride torch.xpu.graph.default_capture_stream, EAGERLY INITED "
        "before capture #1 — the serving default never enters capture; "
        "champion 0.26.1 discipline, r9).", os.getpid())
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
    _wrap_capture_stream()
    _ensure_watchdog()
    _INSTALLED = True
    logger.info(
        "ArcXpuPlatform: graph-safe output patch installed (pid %d, "
        "AsyncOutput/AsyncPoolingOutput __init__ + get_output wrapped).",
        os.getpid())
