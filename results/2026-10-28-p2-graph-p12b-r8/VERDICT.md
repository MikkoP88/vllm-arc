# VERDICT — r8 INVALID as a disproof: the capture-stream fix never engaged for capture #1 (lazy-init hole); r9 = eager side-stream init

## What happened in the leg

arc-p2-graph-20261028-043949, image b1d3b34e1eb7 (94e39c3d, "captures ride
the side stream"): the flip fired AGAIN at the first real request —
04:49:01 ARC_GRAPH_ZOMBIE_ONSET on the device default with the first
/v1/completions in flight, both workers, same death cascade as r7
(engine exit 04:49:05). Taken at face value this disproved the
capture-stream theory.

## Why that reading is wrong — the wrapper hole (found post-hoc, r8b probe)

`torch.xpu.graph.default_capture_stream` is a CLASS attribute lazily
initialized inside `graph.__init__` (torch/xpu/graphs.py:167-183): it is
**None until the first `torch.xpu.graph()` context entry**. The round-8
wrapper reads it inside `_capture_stream()` — which vllm evaluates to
BUILD the `stream=` argument, i.e. BEFORE the first context entry:

- capture #1: `default_capture_stream` is None → wrapper falls through to
  upstream `_capture_stream()` → **the device DEFAULT captured graph #1**
  → default poisoned at boot per the p13 real-graph law
  (results/2026-10-27-p2-graph-defectb-answer)
- captures #2..84: the attribute now exists → side stream (as intended)

The champion (0.26.1) passes NO stream arg, so torch's own lazy init runs
inside `__init__` and **capture #1 rides the side stream from the start**.
r8 therefore never reproduced champion discipline — it reproduced upstream
poisoning once, then champion discipline 83 times, and died exactly at the
first real request as every poisoned round did. The onset timing is
byte-for-byte the r3-r7 signature; nothing in r8 contradicts the theory.

## r8b bare-torch probe matrix (probe_r8b.log)

Toy graphs never poison anything — capture on side, capture ON THE DEFAULT,
same-stream capture+replay: every probe clean, replays and plain kernels
fine. The poison needs the real captured runnable (CCL collectives, events
recorded inside capture, pool-shared 84-graph boot) — consistent with the
p13 note. Also confirmed: single stream world at serve
(xpu_model_runner.py:60 rebinds torch.cuda.graph → torch.xpu.graph;
stream= honored), so there is no cuda/xpu split to blame.

## Standing model (after r8/r8b)

1. boot captures leave the CAPTURE stream graph-associated on this stack
   (real-model scale only);
2. capture #1 on the device default = the serving default poisoned from
   boot;
3. first real op on it (first request's step/AsyncOutput) raises defect-b /
   trips the onset watchdog;
4. champion discipline (ALL captures on the side stream, from #1) leaves
   the serving default clean — proven viable on this rig by the champion
   control leg.

## Fix (r9, commit follows this banking)

Eager-init in `_arc_capture_stream`: if `default_capture_stream` is None,
construct `torch.xpu.Stream()` and assign it to the class attribute —
exactly what torch's own lazy init does, just BEFORE capture #1 reads it.
Then every capture rides the side stream; champion parity for all 84.

Raws: serve_full.log, boot.txt, gates.txt, k1k_sanity.txt, spec_gate.txt
(rig leg dir arc-p2-graph-20261028-043949 + replay_probe.py banked there
too).
