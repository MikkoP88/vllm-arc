# VERDICT — p2r3b-graph-safe-v2 (2026-10-27)

**LEG: FAIL (defect-b still open — but root cause now bounded to one open question).**

## Posture

image=vllm-arc:v0.30.0-b70-p12 (fork fb368de2) — graph=1 (FULL_DECODE_ONLY,
VLLM_XPU_ENABLE_XPU_GRAPH=1), spec=0, lmc=0, TP2 privileged oneCCL, GMU 0.8.
Fresh container (harness freshness law — see below).

## What this leg PROVED (all four gates of the p12 instrumentation passed)

1. **Patch installs in every process** — serve log lines `graph_safe_output
   patch installed (pid …)`: APIServer pid 13, Worker 780 (TP0), Worker 781
   (TP1) at init_device, EngineCore pid 541. register()-time install fails
   with exactly the predicted circular-import errors (visible now, benign).
2. **The graph-safe apply path works**: eager warmup steps between boot
   captures log `applied — device-sync barrier + no-op copy event` (15:09:49–53)
   and complete cleanly.
3. **The death is NOT an event-sync problem we can wait around**: first real
   step (step_counter=0, 832-tok prefill, POST /v1/completions 200 → stream
   fail) runs with a capture window GENUINELY open — `deferred (capture
   window active)` at AsyncOutput creation, then in the worker output thread:
   `wait method cannot be used for an event associated with a command graph`
   AND our self-heal barrier `torch.xpu.synchronize()` raising `wait cannot
   be called for a queue which is recording to a command graph`. A recording
   queue is open at that moment in BOTH workers.
4. `capture_model()` ended CLEANLY (its trailing `torch.accelerator.
   synchronize()` calls succeeded; boot reached healthy), `dispatch()` has no
   lazy-capture path (eager fallback), and the graph-memory profiling pass
   did not run. So the recording state appeared between boot completion and
   the first real step — or is opened BY the first real step.
5. Isolated repro (same container): one XPUGraph capture/end on a side
   stream leaves NO residual capture state; events sync fine. The poison
   needs the full engine context.

## Harness bug found + fixed this leg (banked separately)

serve_arc_b70.sh only created the serve container when MISSING — p2r3
(stale container) ran old-code and its "repro" was void. Fixed by the
container freshness law (recreate on image mismatch; ARC_CONTAINER_FRESH
line in boot.txt). p2r3 artifacts void; p2r3b is the first clean v2-code leg.

## Next (p2r3c, build 17ae84a7)

The deferred branch now dumps the live stack: which code path holds the
capture window at the first real step (suspects: inductor combo-kernel
autotune-on-first-input opening an XPU graph mid-forward; residual state
from a queue the boot syncs cannot wait on). Self-heal barrier is guarded
(original error surfaces, no masking).

## Artifacts

boot.txt (ARC_CONTAINER_FRESH line), gates.txt (all platform gates PASS),
k1k_sanity.txt (cells refused — engine died pre-bench), serve_full.log
(full boot + death tracebacks + all patch-state lines), env.txt, ps.txt,
summary.txt.
