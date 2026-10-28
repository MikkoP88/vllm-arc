# VERDICT — r9: eager side-stream capture engaged for ALL 84 captures and the default STILL flipped at the first request — capture-stream theory DISPROVEN; the poison is runtime-side

## The leg

arc-graph-20261028-052336, image f790fb4adee0 (b8bf1431: `_arc_capture_stream`
eagerly inits `torch.xpu.graph.default_capture_stream` before capture #1).
Boot: captures 84 sizes in 8 s, both workers, install marker with
EAGERLY INITED in the log; single `_capture_stream` definition in
CudaGraphManager (line 390) — ModelCudaGraphManager and
SpeculatorCudaGraphManager inherit it, so the wrap covered every capture.
Champion capture discipline finally reached: ALL captures on the side
stream, serving default never entered capture.

## Result — flip anyway, with a NEW signature

- 05:30:57 ARC_GRAPH_ZOMBIE_ONSET on the device default, first k1k request
  in flight (28 s after server start).
- NEW: the flip hits MID-FORWARD — every request 500 from the first
  (r3-r8 served k1k 200 through the poisoned window); sample_tokens
  integrity guard raised "flipped mid-forward twice in one step".
- **The capture-stream choice is NOT the mechanism.** r8's hole is fixed,
  r9 is a valid test, and the theory died cleanly.

## What survives as the mechanism (post-r9 model)

The poison enters at RUNTIME, during the first real request — the first
moment that (a) a graph REPLAY follows eager kernels on the same queue
(boot warmup never interleaves eager prefill with replay), and/or (b) any
queue-level oneAPI 2026 submission-path difference engages. Bare-torch
probes are exhausted: toy graphs stay clean through capture-on-default,
same-stream capture+replay, eager→replay alternation, pool-shared double
replays (icl_probe.py shapes c/d/e). The repro needs the real captured
runnable — CCL collectives and the real kernel mix.

Prime suspect: **immediate command lists**. The UR L0 adapter exports
UR_L0_USE_IMMEDIATE_COMMANDLISTS / SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_
COMMANDLISTS (strings-verified in libur_adapter_level_zero.so.0). An L0
immediate command list implements launches by appending into a command
graph — a graph launch on such a queue plausibly leaves the queue flagged
"recording" (the exact onset reading), and a default-on in oneAPI 2026
would explain why the 2025.3 champion is immune on the same rig.

## r10 (next): the discriminating knob leg

serve_arc_b70.sh gains `ARC_ICL` — passes
`-e UR_L0_USE_IMMEDIATE_COMMANDLISTS=$ARC_ICL` to the serve exec ONLY
when set (default unset = runtime default, byte-identical posture).
Leg: ARC_LEG=graph GRAPH=1 SPEC=0 **ARC_ICL=0** (classic command queues).
- flip gone → runtime submission path is the mechanism; the fix is one
  env var (law-6 citation: UR env var + this validating leg);
- flip stays → ICL exonerated; next isolations: --async-scheduling
  posture leg (champion has it, we never did), then stack escalation with
  the control evidence.

Raws: serve_full.log, boot.txt, matrix.log, icl_probe.py (rig leg dir
arc-graph-20261028-052336).
