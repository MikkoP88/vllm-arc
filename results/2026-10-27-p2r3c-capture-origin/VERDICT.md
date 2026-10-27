# VERDICT — p2r3c-capture-origin (2026-10-27)

**LEG: diagnostic (hot-patch overlay, image vllm-arc:v0.30.0-b70-p12
ce0ee65f3766 @ fork 17ae84a7). Defect-b death reproduced; capture-window
ORIGIN still unresolved after this leg — but the candidate set was cut to
the engine context (all isolated repros clean).**

## Posture

graph=1 (FULL_DECODE_ONLY), spec=0, lmc=0, TP2 privileged oneCCL, GMU 0.8.
Container hot-patch overlay (docker cp of graph_safe_output.py with the
defer-branch stack dump) — diagnostic only, not bankable image evidence.

## Findings

1. Defer branch fired at the first real request with an 18-frame stack dump:
   AsyncOutput created in the sample_tokens RPC path (execute_model /
   sample_tokens split-step), worker main compute stream — the capture
   window is NOT held by the boot campaign code on the stack.
2. `capture_model()` tail `torch.accelerator.synchronize()` calls SUCCEEDED
   at boot end — at that moment NO queue was recording (a recording queue
   would have raised). Boot 27/27 FULL captures clean.
3. `dispatch()` (cudagraph_utils.py) has no lazy-capture path — uncaptured
   sizes fall back to eager; "lazy capturing" is a comment, not code.
4. Graph-memory profiling pass did not run.
5. Inductor autotune-with-cudagraphs falsified earlier (benchmark gpu-with-
   cudagraph exists but benchmark_with_cudagraphs defaults False and is
   never enabled; combo-kernel benchmark is plain timing).
6. Isolated repros ALL clean (arc-test container): single XPUGraph
   capture/end on side stream; 27-graph pooled campaign STAGE 1/2; torchrun
   TP2 xccl campaign with allreduce inside/between captures + interleaved
   copy-stream events (repro_ccl_graph.py, repro_layers.py,
   repro_tp2_campaign.py). STAGE 3 blocked by VllmConfig guard (skipped).
   The poison needs the full engine context — prime remaining suspect: the
   torch.compile'd model artifact path (VLLM_COMPILE/AOT) inside captures.

## Harness note

serve logs for this leg were overwritten by later serves in the same
container (p2r3d/e) before mirroring — leg evidence is this VERDICT +
in-session analysis; the raw-log mirror law was violated for this leg and
is restored from p2r3e onward (serve logs mirrored at leg end).

## Next (p2r3d)

Fresh-stream probe at the defer moment: is the recording flag per-stream
or device-global? Decides whether fresh-stream copies are a viable fix.
