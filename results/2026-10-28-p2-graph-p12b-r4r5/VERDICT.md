# VERDICT — p12b rounds 4+5: the fix converges — every guard fires; two death classes closed, one opened, one remains

## Round 4 — stream-reroute fix works; thread-locals kill it (leg arc-p2-graph-20261028-013037)

Image 7495f59ae025 (bb0606e4). The round-4 guard layer performed exactly as
designed on its first live poison:

- 01:37:57 `ARC_GRAPH_ZOMBIE_ONSET` (device default) — reroute guards ARMED
- 01:37:59 next request: `ARC_GRAPH_SUBSTITUTE` (set_stream target = the dead
  default → live replacement) and `ARC_GRAPH_WAIT_SKIP` (wait_stream dead
  argument skipped) on BOTH workers — the copy path rerouted correctly
- Device-sync raises down from 6 (r3) to 2 — the healthy-path device-wide
  sync removal works

Death: the serve-time alarm still fired. execute_model and sample_tokens are
SEPARATE RPCs served by DIFFERENT executor threads. The forward ran on a live
thread (its entry hygiene held — no dezombie log needed), but the
AsyncOutput-construction RPC ran on a thread still parked on the dead default
since thread birth: it never calls set_stream, so no guard can intercept it,
and its construction probe tripped the alarm on a perfectly healthy request.

## Round 5 — entry hygiene everywhere + poison flag; the flip shows its mid-forward face (leg arc-p2-graph-20261028-020249)

Image 471baa7e8d7c (f4a5671f). sample_tokens entry hygiene added; the
execute_model EXIT probe flags mid-forward flips (entry clean + exit dead),
and the flagged step's sample_tokens fails loudly (integrity law).

What happened: onset 02:10:10; the p31 forward entered clean at ~02:10:10,
the flip landed DURING it, and the exit probe caught it at 02:10:11 on BOTH
workers ("execute_model-exit (mid-forward flip)" + dezombie + canary
executed=True). The poison raise killed the request (k1k had already
returned 200; the p31 POSTs got 500s). No garbage was ever served — the
integrity law held through every round.

Flip timing inside the ~seconds-wide post-completion window VARIES: rounds
3/4 caught it between requests; round 5 caught it mid-forward. A complete
fix must survive both timings.

## Round 6 — the lost-step retry (locked in, building at bank time)

Canary law (all rounds): kernels enqueued on a queue before it flipped NEVER
execute — a mid-forward flip loses the ENTIRE step atomically: no KV-cache
writes, no sampler RNG effects, no partial state. Therefore the step can be
re-run cleanly: execute_model clears the recorder and retries the same call
once on the live stream (also catching raise-flavored flips whose error text
mentions the recording queue); a second consecutive exit-poison sets the
poison flag and sample_tokens fails the request. Symmetrically, a dead
current at the AsyncOutput construction can no longer mean a lost forward
(the exit probe verified the forward completed on a live stream): dezombie
and PROCEED — the raise survives only for the double-flip corner.

Expected round-6 readout: k1k 200, p31 survives the flip with one retry log
line, gates + benches run for the first time on the graph lane.

Raws: rig leg dirs `arc-p2-graph-20261028-{013037,020249}/serve/serve_full.log`.
