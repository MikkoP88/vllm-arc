# P6.6 r66b — per-chunked-step overhead decomposed (capture leg 2)

**Date**: 2026-10-31 · **Phase**: P6.6 (plan §12 item 8, [viii] lane)
**Posture**: `vllm-arc:v0.30.0-b70-cert4` + probe images (docker-commit,
r48j pattern), fresh boot, `ARC_LMC=0`, graph law, MTP k3, greedy, solo
U32k face (32553 tok), unique seeds. Probe lines written to
`/root/p66_steps.log` inside the container (child-process `print()`s never
reach serve_full.log — v2 lesson, below).

## Headline

The r66a per-chunked-step overhead c ≈ 0.35–0.44 s is **not scheduler,
not host-side prep, not allreduce, not sampling** — it is the **device
execution time of the chunk itself, measured through the queue-drain wait
that lands in the mamba/GDN `preprocess_state` launch window of the next
step**. Prefill on this stack is **device-bound at ~34 % MFU** (1664-token
chunks) rising with chunk size (7488-token chunks ≈ 2.2–2.4 tok/ms vs 1.85).
The thr2048 solo cost (+24–27 %) = small-M device inefficiency + boundary
count; there is no host-side glue to cut.

## The map correction (the hard-won part)

The arc plugin (`vllm_gaudi.xpu:register` → `ArcXpuPlatform`) extends
stock `vllm.platforms.xpu.XPUPlatform` and boots the **stock vllm XPU
stack**. Async scheduling ⇒ `max_concurrent_batches=2` ⇒ the live engine
loop is `step_with_batch_queue` (core.py), and the live runner is the
**V2 model runner** (`vllm/v1/worker/gpu/model_runner.py`, "Using V2 Model
Runner" — xpu_worker.py:169, `VLLM_USE_V2_MODEL_RUNNER`). Two files that
look authoritative are DEAD under this posture:

- `vllm_gaudi/v1/worker/hpu_model_runner.py` (HpuModelAdapter) — the HPU
  plugin's runner; not imported by workers under arc.
- `vllm/v1/worker/gpu_model_runner.py` (v1 GPUModelRunner) — dead when
  the V2 runner is selected.

Probe iterations (all banked): v2 patched hpu runner + dead `step()` —
zero lines; v3+ patched live `step_with_batch_queue` + file-writes —
engine lines only; v5 census (IMPORT markers + Worker-class wrappers)
partitioned the hypotheses: workers DO import the patched files
(markers, correct `__file__`), `Worker.execute_model/sample_tokens` DO
run (WRK lines) but never reach the patched runner ⇒ third class; serve
log "Using V2 Model Runner" settled it. v6 = full capture on the V2
runner; v7/v8 = sub-brackets down to the single 0.8 s call.

## The decomposition (v6, thr2048 U32k face; 1664-token chunks)

Per chunk step (worker rank, mid-face; both TP ranks agree within 1 %):

| Component | ms | Source |
|---|---|---|
| engine `scheduler.schedule()` | **0.3** | ARC_P66_SCHED |
| runner host prep total (`t0→forward`) | 780–1020 | ARC_P66_EXEC prep_ms |
| — request-state block | 0.8 | st_ms (v7) |
| — batch-state gather + dispatch | 0.1 | disp_ms (v7) |
| — prepare_inputs | 0.7 | pinp_ms (v8) |
| — prepare_attn (block tables + slots) | 0.2 | pattn_ms (v8) |
| — **mamba/GDN `preprocess_state`** | **780–1018** | pmam_ms (v8) |
| — attn-metadata build | 3.8 | meta_ms (v7) |
| forward **enqueue** (eager, async) | 127 (flat) | fwd_ms |
| sampling tail (`sample_tokens`) | 73–107 | ARC_P66_SAMPLE |
| TP2 allreduce (129 ops) | ~28 | ar_ms |
| engine step wall (matches WAIT ≈ EXEC total) | 917–1228 | STEP/WAIT |

`preprocess_state` is **two GPU kernel launches** (triton
`preprocess_mamba_align_fused_kernel` + `run_fused_precopy`,
mamba_hybrid.py:183) — launching kernels cannot cost 0.8 s of CPU. The
wall is the **device queue draining** (the previous chunk's kernels,
enqueued in the 127 ms window, executing for ~0.8–1.0 s). Confirmation:
the **first** face chunk (empty queue) shows `pmam=0.2 ms`; the
`fwd_ms` is flat 127 ms regardless of context while `pmam` grows 780→1018
linearly with context (device per-token cost grows with attention KV —
matches the r66a falling-rate curve).

thr=none (MNBT 8192 ⇒ 7488-token chunks): prep (== queue-drain) 3118 →
3695 ms per chunk, fwd enqueue 130 ms — same mechanism, better per-token
device rate (2.2–2.4 tok/ms). Boot-warmup 6656-token step (fresh short
context): prep 15 ms — the queue-drain is context-dependent device work,
not a fixed cost.

Device-side per-token rate ⇒ MFU: 1664-tok chunks ≈ 1.85 tok/ms ≈ 99
TFLOPS fleet ≈ **34 %** of the 294 TFLOPS fp16 GEMM ceiling (devprobe
r66a); larger chunks approach ~40 %. The device, not the host, is the
prefill bottleneck; the 31–40 % MFU gap is attention over growing KV +
W8A16 dequant + non-GEMM ops + per-layer launch/dispatch — kernel-level
territory (plan rung 3), not glue.

## Improvement ladder (updated by evidence)

| Rung | Lever | Status after r66b |
|---|---|---|
| 1 (knob) | thr trade-off table | r66a (documented) |
| 1b (knob) | **MNBT 16384** — bigger chunks amortize boundary + raise M | gate3 A/B (r66c, this session) |
| 2 (glue) | cut host-side per-step overhead | **closed as empty**: host prep ≤ 10 ms, sched 0.3 ms, allreduce 28 ms (3 %), sample ≤ 107 ms — nothing material to cut |
| 3 (structural) | raise device MFU: prefill compile (NONE→PIECEWISE), attention kernel work | deferred, A/B-gated; the only remaining large lever |

## Probe-instrumentation laws learned (for future legs)

1. Child processes (EngineCore, TP workers) under this stack do NOT
   deliver `print()` to serve_full.log — probes must write to a file
   (O_APPEND line-atomic; `/root/<name>.log` in-container, docker cp out).
2. Async scheduling makes core.py `step()` dead code — instrument
   `_process_engine_step` (wraps `step_fn`) or `step_with_batch_queue`.
3. The runner is the V2 file; verify live-class assumptions with an
   IMPORT-marker probe before instrumenting bodies (v5 pattern).
4. `inspect.getsource` reads the file, not the imported module — verify
   patches with `hasattr(module, '<marker>')` on the import instead.
5. XPU eager forward is async: a wall-clock bracket around the forward
   call measures **enqueue**, and the device time resurfaces at the next
   blocking launch (here: preprocess_state) — do not read fwd_ms as
   compute time.

## Artifacts

- `p66_steps5_thr{2048,none}.txt` — v6 full capture (EXEC/SAMPLE/STEP/
  SCHED/WAIT/WRK + markers)
- `p66_steps6_thr2048.txt` — v7 sub-spans (st/disp/pin/meta/res)
- `p66_steps7_thr2048.txt` — v8 final brackets (pinp/pattn/**pmam**)
- `p66_steps_v3_*.txt`, `p66_steps4_*` — engine-side-only captures (v3/v5)
- `patch_p66{b..h}.py` + `p66_gate2*.sh` — the probe ladder, each stage
  exactly as run; `gate2*.log` — leg logs (incl. the v2 FATAL anchors)
- `p66_probe*_thr*.txt/json` — solo TTFTs per probe boot (TTFT stable
  across probe versions: 21.66/21.60/21.56/21.50/21.57/21.45 — probe
  overhead ≈ 0.5 % max, and identical thr=none 17.44–17.49)
- `serve_probe5_*.log`, `serve_probe4_2048.log` — engine logs (config
  dump, "Using V2 Model Runner", hygiene install lines)
- rig: `/root/build/p66/` (all of the above + boot logs)
