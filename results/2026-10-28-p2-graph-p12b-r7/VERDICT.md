# VERDICT — p12b round 7: no storm — the poison is PERMANENT and flips queues ON USE; stream-layer recovery is impossible; P2 pivots to eager

## The measurement (image 02961955b15a, fork 986b2c49, leg arc-p2-graph-20261028-031336)

Round 7's storm-wait was built to answer one question: is the multi-second
kill window time-bounded? The answer came back in the first four seconds:

| time (TP0) | event |
|---|---|
| 03:20:56 | ARC_GRAPH_ZOMBIE_ONSET (device default) — guards armed |
| 03:20:58 | attempt-1 exit-poison → dezombie (canary OK) → storm-wait → **STORM_ENDED 0.0 s** → retry |
| 03:20:59 | attempt-2 exit-poison (the 0.0 s survivor, used, now recording) → dezombie → **STORM_ENDED 0.0 s** → retry |
| 03:21:00 | attempt-3 exit-poison → "retry failed" → poison flag → integrity 500s → teardown |

`STORM_ENDED after 0.0 s` on every poll is the decisive datum: fresh
streams are ALWAYS clean at creation. The unused cached replacement
(created 03:20:58, cache-hit-probed 03:20:59, no work in between) also
stayed clean with canary executed=True. But every stream that RAN A
FORWARD was recording at that forward's exit — three fresh streams, three
on-use flips, each within ~1 s of use.

## The corrected law (supersedes the round-6 "storm" model)

The first flip (seconds after the first real generation completes) puts
the runtime into a PERMANENT state: any queue that enqueues real model
work is hijacked into command-graph recording mid-forward. Consequences:

- Waiting cannot help — an unused survivor is only clean until used.
- Fresh-stream retry loops cannot converge (r6 r7 both exhausted retries).
- Substitution/dezombie guards keep integrity (no garbage ever served —
  held through all seven rounds) but cannot restore serving.
- The state does not decay: 3 attempts over ~4 s plus all earlier rounds'
  evidence show no recovery; nothing in any leg ever executed on a flipped
  queue again (canary law).

Root cause sits below the stream layer (L0 immediate-command-graph state
corrupted by/graph-leaked from the boot captures, with the first real
request's completion — prefix-cache write is the leading differentiator vs
the clean dummy warmup — as the trigger). Fixing it in-process is out of
scope for p12b; the lane decision is what p12b owed the plan.

## POSTURE DECISION (the pre-committed contingency)

P2 runs on the graphs-OFF eager posture: VLLM_XPU_ENABLE_XPU_GRAPH unset →
platforms/xpu.py:301 → cudagraph NONE — no XPUGraph objects, no capture
sessions, no recording queues; the poison mechanism cannot arm. This is
the W14-incumbent-immune lane. Contract targets (agg ≥ 274.41, solo ≥
79.74, TPOT c8 ≤ 60 ms, warm-hit ≤ 0.75 s, prefill parity) get MEASURED on
eager; gates then decide. If eager falls short of the contract, the
remaining graph-lane lead is the prefix-cache-write trigger — one
diagnostic leg with --no-enable-prefix-caching on the GRAPH lane (serve
flag, host-side script, no image rebuild) to prove/refute the trigger and
scope a surgical fix.

p12b disposition: the hygiene/reroute/storm-wait layers stay in the tree —
they are proven integrity keepers (fail-loud, no garbage) if graphs are
ever re-enabled; on the eager lane they are dormant (no graphs → no
flips → guards never arm; r1-r7 logs confirm all guards are `_zombie_ever`
-armed only post-flip).

Raws: rig leg dir arc-p2-graph-20261028-031336/serve/serve_full.log
(excerpt r7_excerpt.txt).
