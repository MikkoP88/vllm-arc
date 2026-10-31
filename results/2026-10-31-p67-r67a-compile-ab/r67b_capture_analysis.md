# P6.7 r67b — U32k prefill op-class capture (torch-profiler XPU trace)

**Date**: 2026-10-31 · **Phase**: P6.7 rungs B/C sizing (E-law capture leg)
**Posture**: stock cert4 image, mode none, `ARC_LMC=0`, graph law, MTP k3,
greedy. Profiler tooling (probe-only, discarded with the container):
`--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/root/prof
--profiler-config.torch_profiler_with_stack=false` + a one-line in-container
bind-mounted fix of `vllm/v1/worker/gpu_worker.py` activities
`["CPU","CUDA"]` → `["CPU","XPU"]` (0.30 hardcodes CUDA; the XPU key exists in
`profiler/wrapper.py TorchProfilerActivityMap`). Script: `p67_gate2_prof.sh`.

**Window**: U8k warm face unprofiled first (decode/graph/JIT warm), then
`/start_profile` → harness internal warmup + **U32k solo face (TTFT 17.53 s,
1857 t/s — vs the 17.33 s same-face bar, +1.2% profiler overhead)** + 32
decode tokens → `/stop_profile`. Traces: rank0/rank1 `.pt.trace.json.gz`
(6.3 MB each, banked). All numbers below are **rank0 device kernel time**
(kineto `cat=kernel`), 17,091 events, total **15.382 s** device busy over a
21.79 s span (**71.9% device occupancy** — prefill is device-bound, matching
r66b).

## Op-class table (rank0, by total device time)

| Class | Time | Share | Evidence |
|---|---|---|---|
| GEMM | 6.105 s | 39.7% | `gemm_kernel` ×2196 (generic symbol; **External-id verified: 849/902 big calls launched by `_xpu_C::fp8_gemm_w8a16`**, 37 by `aten::mm` — the 39.7% IS the p44 W8A16 dequant→fp16-GEMM path) |
| TP2 allreduce family | ~4.67 s | ~30.4% | `allreduce_large_su_ring_write_multi_kernel` 2.251 s ×792 + `ReduceCopyKernel` 2.260 s ×792 (**External-id verified: 685/718 big calls launched by `c10d::allreduce_`** — the reduce half of each allreduce) + `oneccl_invoke_barrier` 0.151 s |
| FA-layer attention | 3.122 s | 20.3% | 102 big (>5 ms, ~31 ms avg) `compat::experimental::detail::KernelFunctor` events, **External-id-linked to `_vllm_fa2_C::varlen_fwd`** — the SYCL-compat symbol hides the FA2 varlen kernel name |
| GDN chunk-scan | 0.724 s | 4.7% | `gdn::ChunkFwdOKernel` 254 ms + `gdn::tiled_kernel_launcher` 240 ms + `ChunkComputeWUKernel` 124 ms + `ChunkComputeAKernel` 60 ms + `ChunkInverseOptKernel` 21 ms (×336 each) |
| norms | 0.39 s | 2.5% | `fused_add_rms_norm` 311 ms + `layer_norm_fwd` 58 ms + multi-row 15 ms |
| act | 0.21 s | 1.3% | `act_and_mul_vec` 207 ms |
| elementwise misc | ~0.25 s | ~1.6% | `VectorizedElementwiseKernel` 137 ms + misc (excl. ReduceCopy) |
| indexer/gather/rope | 0.015 s | 0.1% | topk/gather/mrope/`reshape_and_cache` families |
| memcpy/memset | 0.286 s | — | (excluded from kernel total) |

Call-count sanity: GDN ×336 = 48 layers × 7 profiled steps; allreduce ×792 ≈
64 layers × ~12 fwd calls in window (5 U32k chunks + warmup + decode steps);
FA2 big-calls 102 ≈ 16 FA layers × ~6.4.

## What this overturns and what it confirms

1. **The radiance QSA-whale premise is FALSE on our stack.** Radiance's
   d:qsa splitk = 46% of their device step (their qwen4_exp QSA kernel,
   32-split partial flood). Our stock 0.30 DSA path: the whole
   indexer+gather+rope machinery = **0.1%**. There is no splitk flood to
   kill. An ESIMD "QSA" graft aimed at that whale has no target here.
2. **But the owner's FA-layer fusion rung is still material**: the 16 FA
   layers' attention = **20.3%** via `_vllm_fa2_C::varlen_fwd`. The radiance
   W14 ESIMD kernel's ROLE (sparse attention over selected context) maps to
   THIS site — but the contract differs materially: ours is GQA
   (24 q-heads / 4 kv-heads per pre-TP model, head_dim 256, fp8-e4m3 KV,
   PAGE 1664) vs radiance's MQA-1/qHeads-12/width-2051. A port is a kernel
   rework, not an adapter.
3. **GDN chunk-scan fusion = 4.7% ceiling** — the owner's second fusion
   target is marginal on its own.
4. **Rung C (fp8 XMX GEMM) is the whale: 39.7%** — plus, if direct-fp8 GEMM
   also removes dequant work currently fused into/eager around the GEMMs,
   the addressable slice grows further.
5. **NEW, unnamed lever: TP2 allreduce = ~30.4%** of device time at U32k
   (ring+reduce+barrier). Larger than FA, second only to GEMM. Not in the
   owner's ladder; reported as the top recommendation beyond B/C.

## Ladder consequence (decision input)

- Rung C first (whale, and contingent only on an on-image kernel existing).
- Rung B narrowed to the FA2-varlen site (20.3%); GDN fusion (4.7%) likely
  not worth kernel effort on its own — fold into the FA rung's image if the
  ESIMD build happens anyway, else drop with evidence.
- Allreduce: document as a recommended follow-up rung (needs its own
  capture of message sizes; not owner-named).

## Verdict

Capture banked; no engine change this leg. Rung A (compile) already closed
negative (see VERDICT.md). Ladder re-scoped by evidence above.

## Artifacts

- `rank0.…pt.trace.json.gz` / `rank1.…pt.trace.json.gz` — raw kineto traces
- `profiler_out_0.txt` / `profiler_out_1.txt` — wrapper CPU-time tables
- `p67_prof_u32k.{txt,json}` / `p67_prof_warm.json` — face outputs (TTFT
  17.53 s profiled, U8k warm 3.74 s)
- `gate2.log` / `boot_prof.out` / `serve_prof_boot.log` / `serve_prof_after.log`
- `gpu_worker_xpu.py` + `serve_arc_b70_p67prof.sh` — the probe edits (exact
  bytes as run)
- rig: `/root/build/p67/` (same names + `prof_dir/`)
