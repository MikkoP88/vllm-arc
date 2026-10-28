# VERDICT — prefix caching EXONERATED: the flip fires with graphs ON + prefix OFF; champion-control leg launched

## The leg (arc-p2-graph-20261028-040309, image 02961955b15a, GRAPH=1 PREFIX=0)

Trigger-isolation: graphs ON (FULL_DECODE_ONLY, capture sizes 1..320),
`--no-enable-prefix-caching` — config dump confirms
`enable_prefix_caching=False`. Serve log: 14× 200 OK, then the SAME
permanent on-use flip:

- 04:09:11 ARC_GRAPH_ZOMBIE_ONSET (device default)
- 04:09:12-13 dezombie + ARC_GRAPH_STORM_ENDED 0.0 s ×2 per worker
  (fresh streams clean at creation, flipped on use — r7 law reproduced)
- 04:09:14 engine death (input dump; matrix refused)

**The prefix-cache write is NOT the trigger.** Remaining differentiators
between the clean boot warmup (profile_run dummy batches serve hundreds of
steps without flipping) and the first real request: scheduler-driven KV
block churn, the AsyncOutput copy path (copy stream/event — real requests
only), chunked-prefill of real prompt lengths, request-completion cleanup.

## The decisive reframe (from the champion's own script)

`/root/build/serve_lmc_v133.sh` (D champion, agg 274.41) runs on THIS rig:
FULL_DECODE_ONLY graphs via `--compilation-config` (no env var) + MTP k4 +
`--async-scheduling` + LMCache kv-transfer, on the iexperimental wheel
(vllm 0.26.1.dev0+g568afb3a1.d20261018.xpu in lsv-test). The graph lane is
PROVEN viable on this hardware+driver — **the poison is a fork-stack
artifact** (arc3 wheel, our xpu layer, or our boot chain), not a driver
dead-end. W14 incumbent: no graphs at all (eager) — consistent with its
own bar.

Control leg launched at 04:15: lsv-test (vllm-iexperimental:v1.0.0) +
champion script verbatim. Readout: if it survives real requests today →
environment unchanged, bug is ours → wheel/stack diff becomes the critical
path; if it flips too → the rig environment changed since the champion
runs (driver/torch update) → escalate as environment regression.

Raws: this directory (serve_full.log has the full flip cascade; matrix
env/summary were not written — leg killed at post-death refusals before
artifact banking; rig leg dir arc-p2-graph-20261028-040309).
