# P6.7 ladder verdict — rungs A and C CLOSED NEGATIVE, rung B re-scoped

**Date**: 2026-10-31 · owner ladder directive (plan §12 P6.7 LADDER
EXPANSION): rung A compile, rung B ESIMD FA fusion + GDN chunk-scan, rung C
direct fp8 XMX GEMM — "included if none of regressions".

## Rung A — prefill compile REVAL: CLOSED NEGATIVE (see VERDICT.md)

Law #1 acceptance gate fired on 0.30: MTP acceptance 3.40/3.57 → 2.21/2.32
(−35%), decode t/s −39%, prefill flat. `mode` stays NONE. En-route config
law banked: GDN decode-only capture cap = `max_num_seqs × (k+1)` (64×4=256);
capture ladders must respect it under FULL_AND_PIECEWISE.

## Rung C — direct fp8 XMX GEMM: CLOSED NEGATIVE on banked + new evidence

The owner's condition was "where an on-image kernel exists". It exists:
`XPUFp8BlockScaledMMKernel` (vllm/model_executor/kernels/linear/scaled_mm/
xpu.py:192) → `torch.ops._xpu_C.fp8_gemm` — native oneDNN block-scaled
W8A8, exactly our 128×128 checkpoint format. But it is NOT unexploited:

- The banked r40 microbench (results/2026-10-29-p4-r40-forward-gemm-capture,
  cited in `vllm_gaudi/xpu/patches/fp8_block_w8a16.py`) measured the stock
  direct-fp8 path (per-token-group input quant + `_xpu_C.fp8_gemm`) vs
  `fp8_gemm_w8a16` at the six per-rank TP2 projections of this model:
  **w8a16 wins 2.2–2.4× at prefill Ms and 3–8× at decode Ms** (direct-fp8
  runs 48–268 GB/s effective — the W8A8 path is the slow one on this
  oneDNN/build), with strictly better numerics (no activation-quant error).
- Today's r67b capture confirms the deployed state: the 39.7% GEMM slice IS
  `_xpu_C::fp8_gemm_w8a16` (849/902 big calls External-id verified) — i.e.
  the prefill GEMM whale already runs the FASTEST on-image kernel. There is
  no on-image kernel that beats it; "kill the W8A16 dequant overhead" has no
  on-image replacement. A custom (ESIMD) fp8 GEMM would be new-kernel
  territory, not rung C's "where an on-image kernel exists".

## Rung B — ESIMD FA-layer fusion + GDN chunk-scan: RE-SCOPED, open (kernel project)

The capture overturns the premise AND re-aims the target:

1. The radiance QSA-whale premise is false here: indexer/gather/rope = 0.1%
   of rank0 device time; there is no splitk flood on stock 0.30. The
   radiance W14 kernel (`qsa_ops.so`, sparse logical-indices contract) has
   no drop-in target.
2. The 16 FA layers are **DENSE** attention (qwen3_5.py has no sparse
   indexer; the DeepSelect import warning is an unused module), running
   `_vllm_fa2_C::varlen_fwd` through `vllm/vllm_flash_attn` (SYCL-compat
   build) via `FlashAttentionBackend` — **3.12 s = 20.3% of rank0 device
   time at U32k**, ~31 ms × 102 calls. Estimated achieved efficiency at
   U32k ≈ 5–7% compute MFU (2·causal-FLOP model, 12 q-heads/rank, d=256)
   — plausibly KV-bandwidth-bound; share grows ~quadratically with context
   (consistent with r66a's rate decay 2178 → 1666 t/s 8k→64k).
3. GDN chunk-scan (5 `gdn::Chunk*` kernels ×336) = 4.7% — marginal alone.
4. NEW unlisted lever: TP2 allreduce family ≈ **30.4%** of device time at
   U32k (ring 2.25 s + ReduceCopy 2.26 s → `c10d::allreduce_` verified +
   barrier 0.15 s). Second-largest slice; not owner-named — recorded as the
   top recommendation beyond the named rungs (needs its own message-size
   capture; candidates: one-shot for small chunks, overlap, fused AR+norm).

**Rung-B port spec (next branch, kernel project — NOT a config leg):**
target the dense FA site — ESIMD flash-class kernel, GQA 24q/4kv per model
(12q/2kv per rank at TP2), head_dim 256, q/out fp16 (serve dtype), KV paged
fp8-e4m3 PAGE=1664 with per-layer descales, causal var-len chunked prefill.
Structural donors from W14: lsc block loads + SLM staging + online softmax +
one-work-item-per-(row,kv_head,split) + capture-growth/loader/GRF laws. The
sparse→dense delta (drop logical-indices walk; full page range + causal) is
smaller than the GQA+fp16 delta. Build per W14_QSA_ESIMD_PLAN.md (DPC++
2026.2, esimd containers on rig). Realistic ceiling if 1.5–2× over FA2
varlen: U32k TTFT −8…−12%; U64k larger (share grows).

## Phase end-state

No engine change landed (A/C negative, B not started). **cert4 remains the
certified image; no cert5 from P6.7 rungs A/C.** Evidence complete for the
re-scope; docs updated alongside this file.
