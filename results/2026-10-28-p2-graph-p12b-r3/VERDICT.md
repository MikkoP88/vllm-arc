# VERDICT — p12b round 3: onset pinned — the flip lands BETWEEN requests

Leg `arc-p2-graph-20261028-005232`, image `vllm-arc:v0.30.0-b70-p12b`
(318064fddd52, fork 90aee3b3). The round-3 instrumentation (fixed watchdog
probing its own thread's current = device default; post-flip tripwires on
synchronize/Event.record/Stream.wait_*) produced the timeline that fixes the
design of the real fix (round 4, bb0606e4).

## Timeline (TP0, pid 782)

| time | event |
|---|---|
| 00:54:03 | patches installed in workers (captures follow) |
| 00:58:41–46 | 3 budgeted clean constructions per worker — vllm's internal warmup request (probe False, device sync OK) |
| 00:59:12 | init engine complete (299 s, compilation 199 s) |
| 00:59:24 | server routes up; health 200 |
| ~00:59:51 | k1k sanity `POST /v1/completions 200 OK` (its constructions ran silently — budget spent — and clean) |
| **00:59:52** | **ARC_GRAPH_ZOMBIE_ONSET** — watchdog's own current (device default) probes recording, no capture open. ids: cur=0x325574c0 side=0x324004c0 |
| 00:59:54 | next request: serve-time alarm on both workers. TP0 poisoned current = 0x325574c0 — **the same stream the watchdog flagged** |

The flip therefore lands BETWEEN requests: after a fully-served completion
(warmup AND k1k both clean), before the next forward. TP1's watchdog logged no
onset — its flip landed <100 ms before its alarm (single onset line total).
The alarm → dezombie → canary executed=True → our integrity RuntimeError →
engine death: exactly as designed; the leg then churned refusals (matrix
killed, evidence banked).

## New laws from the tripwire ids

1. **The worker serves on the device DEFAULT stream** — onset id == alarm id
   on TP0. Threads start parked on the default; a between-requests flip
   poisons exactly where the next forward will enqueue.
2. **All PRE-flip streams die, not just the serving stream.** The first
   tripwire on each worker is `Event.record` with cur = the async COPY stream
   (created at module import — pre-flip; e.g. TP1 0x45b5a750, distinct from
   the poisoned main 0x4568fe70 and the post-swap fresh 0x4557b000): the copy
   stream ALSO probes recording post-flip. Post-flip-created streams probe
   clean (all dezombie canaries). Practical consequence: upstream's D2H copy
   path is dead too after a flip — a fix must reroute the copy stream, not
   just the main stream.
3. **Device-wide barriers never recover.** The drain-path constructions
   (vllm executing queued RPCs while the engine dies) probed clean on their
   fresh current yet died at `torch.xpu.synchronize()` — "wait cannot be
   called for a queue which is recording" — 6 raises. The abandoned poisoned
   queues keep their L0 recording state and block every device-wide sync
   forever. Per-stream waits tolerate them (the dezombie canary's D2H copy on
   a clean stream succeeds).

## Fix design locked by this evidence (round 4, bb0606e4)

Never enqueue on, wait for, or record onto a dead stream:
- `torch.xpu.set_stream` substitution of dead targets (cached per-dead-stream
  live replacement; `torch.cuda.set_stream` alias rebound when the donor
  alias law bound it to the same original — async_utils' ctx switches via the
  cuda name),
- `Stream.wait_stream` dead-argument skip, `Event.record` dead-argument
  reroute,
- `GPUModelRunner.execute_model` entry hygiene (worker threads parked on the
  poisoned default get cleared before any forward enqueue; thread-local
  currents mean one dezombie does not fix another thread),
- healthy-path output copy keeps upstream's blocking event — recorded on a
  live stream outside capture, waitable — instead of the device-wide sync.

Open risk carried to the round-4 leg: whether an L0 graph REPLAY enqueues on
the current (substituted, live) stream or on its stored capture queue (dead).
If the latter, decode outputs post-flip would be garbage and the correctness
gates (k1k token sanity, p31 acceptance) will catch it.

Raws: rig leg dir `arc-p2-graph-20261028-005232/serve/serve_full.log`
(matrix killed at refusals; container log docker-cp'd before recreate).
