# VERDICT — r12b INVALID (tracer bug, not the defect): assigning a plain function onto torch.ops.aten.mm.out corrupts the op registry — boot death at profile_run compile; lesson recorded, r12c rebuilt triton-only

## The leg

arc-graph-trace2-20261028, image efd14b88388c (fork 6c25db06 — kernel
tracer wrapping both `CachingAutotuner.run` and `torch.ops.aten.mm.out`).
ARC_TRACE=1. Boot died at 0 requests: profile_run compile failed with

```
Expected `qualname` to be of the form "namespace::name", but got
_wrap_kernel_launch_tracer.<locals>._traced_mm
```

Root cause: the tracer did `torch.ops.aten.mm.out = _traced_mm` — plain
function assignment onto a torch.ops entry rewrites the op registry
object; torch's op-resolution machinery then chokes on the qualname.
Triton wrap installed fine on both ranks before death.

## Lesson (recorded in graph_safe_output.py docstring)

**Never assign onto torch.ops entries.** Wrap module-level Python
callables (CachingAutotuner.run) only — and an aten-op flip between two
triton kernels still shows as FLIP at the next triton-pre probe, so the
aten wrap was unnecessary anyway.

## r12c (fix)

cbb6937e dropped the mm wrap entirely (triton-only tracer). Leg
arc-graph-trace3 booted clean and pinpointed the flip — see
`../2026-10-28-p2-graph-p12b-r12c-trace3/VERDICT.md`.

Raws: serve_death.log (rig leg dir arc-graph-trace2-20261028).
