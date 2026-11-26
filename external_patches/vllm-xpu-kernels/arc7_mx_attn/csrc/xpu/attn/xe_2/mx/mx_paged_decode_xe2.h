// Entry point for the arc7 MX-native paged-decode kernel (wheel-side).
// Mirrors paged_decode_xe2.h's role for the stock kernel: one function
// the op-registration layer (attn_interface patch) calls with the
// current SYCL queue. The MX deltas: a single uint8 TQ-AoS slot cache
// in place of separate K/V caches, a format id in place of the dtype
// pair, and decode-fixed flags (paged, varlen, no causal — the vendor
// decode law). Sliding-window bounds flow through since P-C2
// (2026-11-25); -1/-1 keeps the historical windowless decode.
#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

void cutlass_mx_paged_decode_xe2(
    sycl::queue& queue,
    const at::Tensor& query,     // [total_q, num_heads_q, D] bf16
    const at::Tensor& mx_cache,  // [num_blocks, block_size, Hk, slot_bytes] u8
    int64_t fmt_id,              // 0=mxfp8 1=mxfp4 2=mxk8v4
    at::Tensor& out,
    at::Tensor& temp_out,  // [total_q, num_heads_q * num_kv_splits, D]
    at::Tensor& exp_sums,  // [total_q, num_heads_q, num_kv_splits] f32
    at::Tensor& max_logits,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,  // per-seq used KV lengths (seqused_k
                                     // rides this slot in the decode flow)
    int max_seqlen_q,
    int max_seqlen_k,
    double sm_scale,
    // P-C2 (2026-11-25): sliding-window bounds, vendor convention
    // (stock cutlass_paged_decode_impl order; -1/-1 = no local mask).
    // W-token window = (W-1, 0) — the last W tokens per row.
    int window_size_left,
    int window_size_right,
    int num_kv_splits,
    std::optional<at::Tensor>& splits_per_seq,
    std::optional<at::Tensor>& work_list);
