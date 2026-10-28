# VERDICT — r12 TRACER NAMES THE FLIP FRAME: inside a piecewise-compiled prefill callable (inductor AOT cache, extern_kernels.mm) — the poison is the torch.compile prefill path, not graphs, not capture, not runtime env

## The leg

arc-graph-trace-20261028-061226, image 171654f579a6 (fork 07403f46),
ARC_TRACE=1 → ARC_TRACE_FLIP=1 in the workers: 10 ms watchdog polls,
all-thread stack dump at onset, state-transition logging at hygiene
boundaries. Same posture and same flip as r9-r11 (first request 200,
onset 06:19:46, death after).

## The captured frame (stack_extract.txt + serve_full.log)

MainThread (the worker's serving thread) at the flip moment:

```
vllm/compilation/piecewise_backend.py:380 __call__
  → range_entry.runnable(*args)
torch/_inductor/standalone_compile.py:321 __call__
torch/_dynamo/aot_compile_types.py:221 / eval_frame.py:1446
torch/_functorch/aot_autograd.py:1277 → runtime_wrappers/subclass_codegen
torch/_inductor/output_code.py:763 __call__
torch_aot_cache/i7/ci7x45...py:362 call
  → extern_kernels.mm(buf1, reinterpret_tensor(arg6_1, (5120, 48), ...), out=buf12)
```

**The first real request's PREFILL runs vllm 0.30's default
piecewise-compiled callable (inductor backend, triton-xpu kernels +
extern aten mm) on the device default stream — and during that callable
the stream flips to recording.** vllm 0.30 enables piecewise compilation
by default (compilation_config in the dump: backend inductor,
cudagraph_mode FULL_DECODE_ONLY ⇒ prefill compiled piecewise, decode
captured FULL). The champion 0.26.1 on torch 2.12 did NOT compile on XPU
— its prefill is pure eager — which is why it never flips on this rig.

Also explains every earlier observation: boot warmup runs dummy shapes
(compile cache warm/clean); the FIRST REAL SHAPE triggers the compiled
runnable at first request; replays themselves were always fine (k1k 200
in every round). COFACTOR VERIFIED: the eager-pivot leg (GRAPH=0) ALSO
compiled (init 296 s, compilation 199 s, same piecewise backend) and
served 315 requests CLEAN — compiled prefill alone does not flip; the
flip needs the compiled callable running WITH the graphs posture live
(captured graphs + pool in the process). Suspect surface narrows to what
the graphs posture changes around that callable: the graph-private
memory pool (the mm's out=buf12 allocation path), the runner's stream
setup in graph mode, or a capture-time event/kernel association firing at
first real-shape execution. r12b kernel-launch tracer discriminates.

## Eliminations standing (r1-r11)

capture stream (r8/r9), ICL (r10), async-scheduling (r11), prefix
(earlier), toy-scale compile+run (clean — needs the real kernels or the
graphs-posture context).

## r12b (building): kernel-level pinpoint

Tracer extension (fork 6c25db06): capture-state probes around EVERY
triton kernel launch (CachingAutotuner.run) and aten mm.out. The FLIP
log line names the kernel whose launch flips the stream — or shows the
flip lands inside the extern mm. Candidate fixes after r12b: disable
piecewise compilation on XPU (vllm compilation-config mode=NONE —
prefill eager like the champion; graphs still capture decode), or the
specific inductor/triton knob the kernel implicates.

Raws: serve_full.log (full stacks), stack_extract.txt, boot.txt (rig leg
dir arc-graph-trace-20261028-061226).
