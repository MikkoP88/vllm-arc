# VERDICT — p12b round 6: the poison is a STORM — every queue dies for a multi-second window, fresh ones included

## What the leg proved (image ef5f7812c2bf, fork 5e9b2217)

The round-6 lost-step retry fired exactly as designed and its attempt-2
failure is the discovery: the recorder flip is not one event — there is a
POISON WINDOW (≥3 s observed) during which every queue on the device dies:
all pre-flip streams AND every freshly-created one. Two independent proofs
in the same second:

1. **Substitution target**: the dezombie stream created 02:43:46 (canary
   executed=True at swap) was probed recording at 02:43:47 when threads
   tried to enter it — lifetime under one second.
2. **Dead at creation**: the retry-failed dezombie's brand-new replacement
   probed recording immediately post-swap with canary executed=False and
   garbage values — kernels enqueued on it never ran.

Whether the window is time-bounded is THE open question; round 7's
survivor-poll answers it by measurement.

## Corrected kill mechanism (raw re-read, this doc)

Earlier analysis blamed a wait issued on a dead queue escaping through the
AsyncOutput construction path. The raw tracebacks (log lines 450-495) show
both `wait cannot be called for a queue which is recording` raises sit in
the SHUTDOWN path — `model_runner.shutdown()` → `torch.accelerator.synchronize()`
(model_runner.py:2241) — firing AFTER the engine was already dead. The
actual kill chain: storm kills the retry stream → exit-poison attempt 2 →
`_poisoned_step=True` → sample_tokens integrity raise → 500s → engine down;
the shutdown sync raise is post-mortem noise (barrier law: a device-wide
sync can never succeed again once a queue is abandoned — r2 evidence).

The wait_stream caller-reroute kept in 986b2c49 remains law-correct
hardening (a dead waiter's ordering is void; nothing it enqueued will ever
execute), but it is not the evidenced r6 killer.

## Integrity law — held through six rounds

k1k 200 at 02:43:44; every later request failed honestly (500 via the
poison-flag raise). No garbage was ever served in any round.

## Round 7 design (commits f5d451b8, c52a035d, 986b2c49 — image building)

- `_wait_out_storm()`: poll a fresh stream every 200 ms (create + orig-set
  + capture-flag probe) up to 15 s; first survivor is left current and the
  lost step is re-run on it. `ARC_GRAPH_STORM_ENDED` logs the measured
  storm duration; `ARC_GRAPH_STORM_PERMANENT` (no survivor in 15 s) sets
  the step-poison flag and fails the request loud.
- Retry loop widened to three attempts, each retry gated by a storm-wait;
  raise-flavored flips ("recording" text) take the same path.
- Corpse-proof cache: every `_LIVE_REPLACEMENTS` hit re-probed, recording
  entries evicted (a mid-storm replacement is dead; post-storm fresh is
  live — the cache self-heals). Without this the first healthy post-storm
  request's copy path substitutes onto a corpse → copy_event never fires →
  get_output hang.
- `_dezombie` storm-waits in place when its swap target dies under it.

Readout: ARC_GRAPH_STORM_ENDED (+ duration) → bounded storm, gates run for
the first time on the graph lane. ARC_GRAPH_STORM_PERMANENT → pivot P2 to
the graphs-OFF eager posture (VLLM_XPU_ENABLE_XPU_GRAPH unset →
platforms/xpu.py:301 → cudagraph NONE, the W14-incumbent-immune lane) with
this directory as the definitive posture evidence; optional root-cause
diagnostic first: prefix-caching-off leg (the storm begins right after the
first REAL request completes — the prefix-cache write is the differentiator
vs the clean dummy-batch warmup; enable_prefix_caching=True in current
args).

Raws: rig leg dir arc-p2-graph-20261028-023623/serve/serve_full.log
(excerpt: r6_excerpt.txt here).
