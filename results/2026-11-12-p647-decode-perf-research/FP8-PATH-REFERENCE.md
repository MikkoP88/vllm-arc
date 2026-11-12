# FP8 KV reference-path analysis — why the baseline is fast and always clean

**Date:** 2026-11-12 · **Leg:** decode-perf research (A10 line) · **Owner directive:** "do
deep analyze end-to-end --kv-cache-dtype FP8 path why it has so good performance and is
always corruption free even this hardware do not support native FP8."

This document maps the ctrl lane's fp8_e4m3 KV path end-to-end on cert25x-class images
and derives the property list the MX lane must match. It is the reference side of the
A10 telemetry attribution: every MX-vs-ctrl delta should trace to a property the fp8
path has and the MX path lacks — or the delta is not structural and must be re-measured.

## 1. What actually serves the ctrl lane

Banked serve logs (cert24/cert25 ctrl legs, qwen champion chain):

```text
[flash_attn.py:1116] Using FlashAttention version 2
[xpu.py:199] Using Flash Attention backend.
```

No TQ, no MX routing — the stock vllm_gaudi FlashAttention v2 XPU backend with
`kv_cache_dtype=fp8_e4m3` admitted natively (`supported_kv_cache_dtypes` includes
`fp8`/`fp8_e4m3`, gated by `flash_attn_supports_kv_cache_dtype`).

## 2. The end-to-end path

| Stage | What runs | Kernel count per layer-step |
|---|---|---|
| Prefill/verify store | stock `set_kv_buffer` elementwise e4m3 cast (+ per-tensor scale) | 1 small cast kernel |
| Decode attention | ONE fused FA v2 varlen kernel: paged walk (page_size = block_size), fp8 load, counter-operand convert, dot, online softmax, split-combine via in-kernel scheduler metadata | 1 kernel |
| MTP verify (q_len=k+1) | the SAME single kernel — extra rows are just more cu_seqlens entries; causal masking is positional | 1 kernel |
| Host logic | none per layer beyond the varlen call | 0 |

`get_scheduler_metadata(..., qkv_dtype=fp8_dtype, num_splits=max_num_splits)` — the
split-KV combine that MX implements as a separate second kernel is, in FA, scheduled
INSIDE the single launch (persistent split work distribution + in-kernel reduce).

## 3. Why it is fast on hardware without native FP8

Xe2/B70 has no native FP8 dot ([[xe2-xmx-fp8-nativeity-verdict]]: counter-operand
conversion only). The fp8 lane does not need one:

1. **Decode is bandwidth-bound.** The win is bytes: 256 B/slot/head-D256-rank vs 512 B
   fp16 — the ALU cost of converting each loaded fp8 to fp16 for the XMAD dot is
   hidden behind the memory pipe.
2. **Conversion is fused into the attention kernel.** No dequant pass, no extra
   tensors, no intermediate fp16 KV materialization.
3. **One launch per layer per step.** At agg8 × ~17 attn layers the ctrl path issues
   ~17 decode kernels per step; per-launch device dispatch (~µs each) is invisible.

## 4. Why it is always corruption-free

1. **Uniform layout.** An e4m3 slot is D bytes, no scale interleaving, no group
   deinterleave arithmetic to get wrong. The MX analogue ([K][Ks][V][Vs] with 1B
   E8M0 per 32 elements) adds two classes of addressable mistakes (scale-lane
   indexing, group-boundary handling) the fp8 lane simply does not have.
2. **Positional causal masking.** Flash's varlen causal handling derives each row's
   limit from cu_seqlens positions. The per-row-limit class of bug (our LIMIT LAW:
   MTP allocator-extent seq_lens leaking the next draft's K/V) is structurally
   impossible in the flash shape.
3. **Uniform per-layer geometry.** One dtype cast, one head size per spec — the
   hybrid two-head-size pools (gemma: 50×256-D + 10×512-D per rank) exercise nothing
   MX-specific in the store path.
4. **Upstream maturity.** The kernel and the fp8 KV plumbing are mainline vllm code
   paths with fleet-scale coverage; our MX lane is 100% first-party code.

## 5. The MX gap table (pre-fix → post-fix)

| Property | fp8 ctrl | MX pre-cert25y | MX post-batched-verify |
|---|---|---|---|
| Decode kernels per layer-step | 1 | ~1 (kernel math fine) | 1 |
| Verify-batch kernels per layer-step | 1 | ~16/req (per-row mx_decode + per-req inversion cascade) | 1 batched (+~8 once per step, layer-cached) |
| Per-step host kernels at agg8 ×17 layers | ~17 | ~2,000+ | ~80 |
| Store per layer-step | 1 cast kernel | 1 mx_store (quantize+pack) — comparable | unchanged |
| Split-combine | in-kernel | separate tiny combine kernel | unchanged (watch in A10) |
| Per-row limit derivation | structural (positional) | slot→position inversion law | same law, batched once per step |
| Bytes per slot (D=256) | 256 (fp8) | 528 (mxfp8) / 272 (mxfp4) / 400 (mxk8v4) | unchanged — the format law (VERDICT.md §1) |

**Reading:** the offline kernel batteries (A4-A8) showed the v3 kernel itself at or
below flash-class timings at equal shape; the e2e deficit was the engine structure
around it — overwhelmingly the MTP verify dispatch. `_split_verify_decode`
(engine.py, cert25y) collapses the per-step kernel count to the fp8 lane's order.
The remaining structural deltas (separate combine kernel, separate store
quantize) are the A10 telemetry's next attribution targets.

## 6. Corruption-freedom obligations for the MX lane

From §4, the MX lane's equivalent guarantees are: the LIMIT LAW per-row limits
(slot-derived, never allocator-extent seq_lens), the KV-store separate-strides law
([[kv-store-separate-strides-law]]), the two-head-size pool audit (gemma KV_2LTD —
per-group block tables, page 64 vs 128), and the end-to-end corruption battery on
every dtype × both models (r161 L1-L6). The m8g disposition (native-g0 longprobe,
r164) applies §4.3/§4.4 as the control.
