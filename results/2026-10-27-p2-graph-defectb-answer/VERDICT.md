# VERDICT — defect-b zombie question ANSWERED (p2-graph leg on p13 image)

**Leg**: `arc-p2-graph-20261027-224809` — image `vllm-arc:v0.30.0-b70-p13`
(fork 8b91f3a4, arc3 wheel), TP2, graph=1 spec=0 lmc=0, gmu=0.8 mnbt=8192
seqs=64, `VLLM_XPU_ENABLE_XPU_GRAPH=1`, FULL_DECODE_ONLY, 27 sizes captured
(≤ max_num_seqs 64).

**Outcome: leg FAIL (expected) — and the armed discrimination question is
ANSWERED with device-level proof. This is the evidence leg for the p12b fix
(`feat/arc-p12b-graph-stream-hygiene`).**

## The answer

The graph-safe-output canary (p12 instrument) fired at the FIRST serve-time
`sample_tokens` on BOTH TP workers:

```
current_stream=torch.xpu.Stream(device=xpu:N sycl_queue=0x…)
capturing(cur)=True  capturing(fresh stream)=False
canary[executed=False vals=[758946.5625 ×4]]   (TP0 later: [0,0,0,1.4e-45])
```

1. **Per-stream, not device-global**: a brand-new stream reports
   `capturing=False`; only the worker's serving stream is flagged.
2. **True zombie, not a stale boolean**: a fill kernel enqueued on the
   flagged stream NEVER EXECUTES (read-back = uninitialized memory, not the
   filled 123.456). Kernels there are lost, so forward outputs are garbage —
   per the p12 decision law, fresh-stream output copies alone are NOT a fix;
   the recorder itself must be cleared.
3. **Poison enters during capture**: boot captured 27/27 sizes "finished in
   7 secs" with zero errors; nothing GPU-touching ran between capture end
   (22:55:08) and the first request (22:55:48) — yet the stream was already
   recording-dead at the first `sample_tokens`, BEFORE any replay.
4. **Death chain** (matches Appendix C defect-b text exactly):
   canary defer (:168) → upstream `copy_event.synchronize()` raises "wait
   method cannot be used for an event associated with a command graph" →
   self-heal (:204) → `torch.xpu.synchronize()` raises "wait cannot be
   called for a queue which is recording" (barrier illegal, :214) →
   EngineCore death → every later request connection-refused. dmesg: 0 GPU
   resets (pure software state; device self-clean).

## Device probes (probe_zombie.py / probe_zombie2.py, in-container, torch
2.13.0+xpu, oneAPI 2026.0)

- Bare capture on the current/default stream (vllm's style), side-stream
  capture, and a single pool-shared capture ALL end `capturing=False` with
  working execution and correct replay — the leak is NOT bare torch.xpu.graph
  mechanics; it needs the real 27-size captured runnable (CCL/GDN/sampler
  content) on this stack.
- **Event-poison class reproduced in miniature**: an event recorded INSIDE a
  capture is permanently graph-associated — `ev.synchronize()` outside
  raises the exact defect-b text. This is the sibling mechanism (any
  capture-time event is unwaitable forever).
- **Recovery validated**: `torch.xpu.set_stream(torch.xpu.Stream())` after a
  zombie leaves a clean current stream whose kernels execute — the swap is
  serve-transparent. This is the p12b dezombie law.
- Side observation: 2nd pool-shared capture with python-held tensor refs
  trips `XPUCachingAllocator.cpp:1330 use_count > 0` (torch-xpu allocator
  assert). vllm's wrapper weak-refs outputs so the real boot's 27 captures
  were clean; recorded as a stack hazard, not our defect.

## Context facts banked on the way

- W14 incumbent (radiance) does NOT use `VLLM_XPU_ENABLE_XPU_GRAPH` — it
  runs `--compilation-config cudagraph_mode=FULL_AND_PIECEWISE` (sizes
  [4,8,16,32,64], MTP k3). The XPU-graph-manager path is the line-D
  iexperimental champion posture; the radiance vllm wheel itself warns
  "XPU Graph support is experimental and currently only supports single-GPU
  execution" (xpu.py:308) — we force it on TP2 by design (D-line champion).
- The donor's `_torch_cuda_wrapper` stream aliases (torch.cuda.set_stream →
  torch.xpu.set_stream etc.) are present in our image's wheel and never
  restored — the "zombie-strike pair" alias law is fully live at serve time.
  The zombie survives it: the aliases switch streams; they cannot resurrect
  a queue that a capture left recording.

## Fix (branch feat/arc-p12b-graph-stream-hygiene)

Clear the zombie at every boundary where a swap cannot be undone — capture
accounting (`XPUGraph.capture_begin/end` wraps; a recording flag with zero
open captures = leaked recorder), hygiene swap after each capture CONTEXT
teardown (`torch.xpu.graph.__exit__` — torch restores the pre-capture
current stream AFTER `capture_end`, so a swap made inside `capture_end`
would be silently reverted) and before/after each `replay`, dezombie-before-
reraise in the get_output barrier-illegal path, and a last-resort serve-time
alarm that clears the stream then fails the in-flight request loudly (its
outputs are garbage; integrity law forbids returning them). Boundary logs
(`graph stream hygiene (graph_exit|replay*)`) pin the leak's origin
empirically in the next leg while the engine survives.

Files: `vllm_gaudi/xpu/patches/graph_safe_output.py` (all layers);
leg raws in this dir (`serve/serve_full.log`, `serve/death_traceback.log`,
`probe{1,2}_zombie.log`, `probe_zombie{,2}.py`, `gates.txt`,
`k1k_sanity.txt`, `spec_gate.txt`, `boot.txt`).
