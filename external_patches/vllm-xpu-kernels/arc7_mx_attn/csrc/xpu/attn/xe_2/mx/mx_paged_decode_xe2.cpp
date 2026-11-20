// arc7 MX-native paged decode: runtime entry (fork of
// xe_2/paged_decode_xe2.cpp + the decode-buffer allocation and
// split-default rules from flash_attn/flash_api.cpp).
//
// The cache is ONE uint8 TQ-AoS slot pool [num_blocks, block_size, Hk,
// slot_bytes] (vllm_gaudi/xpu/patches/mx_kv/mx_layout.py). The four byte
// surfaces are slot sub-ranges at the format's offsets; the pool's
// (seq, head, page) byte strides are passed verbatim as the surfaces'
// element strides — zero re-layout.
#include "xpu/attn/xe_2/mx/mx_paged_decode_xe2.h"

#include "xpu/attn/paged_kv_utils.h"
#include "xpu/attn/xe_2/fmha_utils.hpp"
#include "xpu/attn/xe_2/mx/mx_paged_decode.hpp"
#include "xpu/attn/xe_2/mx/mx_types.hpp"

using namespace cute;

namespace {

// Split-count default — the vendor rule from flash_attn/flash_api.cpp
// get_num_splits, mirrored (device WG-slot saturation, the ~12-tiles-
// per-split bandwidth knee, the reduction-cost cap). num_splits <= 0
// from python means "use this default".
int mx_get_num_splits(
    const sycl::queue& queue,
    const int& batch_size,
    const int& num_heads_q,
    const int& num_heads_kv,
    const int& max_seqlen_k,
    const int& block_size) {
  auto device = queue.get_device();
  int num_xe_cores =
      device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
      device
          .get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();

  int kv_tile;
  int sg_per_wg;
  int policy_split_cap;
  if (block_size == 16) {
    kv_tile = 16;
    sg_per_wg = 1;
    // r207: 16 -> 64. The old cap equaled the kernel's (sanity-only)
    // max_num_kv_splits and pinned auto at 16 splits; the splits sweep
    // shows the walk still scales there (r207 bank). Reduce cap and
    // splits_bw still bound the request for short/mixed batches.
    policy_split_cap = 64;
  } else if (block_size == 32) {
    kv_tile = 32;
    sg_per_wg = 2;
    policy_split_cap = 32;
  } else {
    kv_tile = 64;
    sg_per_wg = 4;
    policy_split_cap = 64;
  }

  int kv_tiles = (max_seqlen_k + kv_tile - 1) / kv_tile;

  if (kv_tiles < 16) return 1;

  int num_wg_slots = num_xe_cores * 4 / sg_per_wg;
  int wgs_per_split = batch_size * num_heads_kv;

  if (wgs_per_split >= num_wg_slots && kv_tiles < 64) return 1;

  int splits_par =
      std::max(1, (4 * num_wg_slots + wgs_per_split - 1) / wgs_per_split);
  int splits_bw = std::max(1, kv_tiles / 12);
  int splits = std::max(splits_par, splits_bw);

  // Reduction-cost cap: ReduceSplitK output volume scales with
  // splits * num_heads_q; keep it bounded (stock 4-slots/head knee,
  // RESTORED in r210: the r209 8x lift let 32K run 64 splits = +0-13%
  // worse on the probe A/B; 40 slots/head is the measured optimum on
  // BOTH probe (B=32) and telemetry (B=1) geometries, and the r207/r208
  // bench sweep that favored 64 was a 2-seq slot-fill artifact — 4 WGs
  // per split-set wants deep splits, engine batches do not).
  int splits_reduce_cap =
      std::max(1, 4 * num_wg_slots / std::max(1, num_heads_q));
  splits = std::min(splits, splits_reduce_cap);

  // r210 workload bound: splits_par over-requests at small batch +
  // short ctx (B=1 1K asked 40 splits for 64 tiles = 1.6 tiles/split
  // and lost 12% to the 5-split posture, r209 telemetry chain). Cap at
  // the SAME divisor as the bandwidth estimate (kv_tiles/12) — never
  // denser than the bw knee. Measured: 1K optimum s=5 = 64/12 (r210 1K
  // forced sweep: s=1 is latency-bound 1.5x, plateau 0.43-0.53, best
  // at 5); at 8K/32K the bound is loose (42/170) and the 4x reduce cap
  // binds at 40 — the r209 probe A/B showed 32 there costs +58-65%.
  splits = std::min(splits, std::max(1, kv_tiles / 12));

  return std::min(splits, policy_split_cap);
}

// Slot geometry per format — MUST match mx_types.hpp MxTraits (which
// mirrors vllm_gaudi/xpu/patches/mx_kv/mx_layout.py). Offsets in bytes
// from the head-slot base; widths are the surface column extents.
struct MxSlotLayout {
  int kb_off;
  int k_width;
  int ks_off;
  int ks_width;
  int vb_off;
  int v_width;
  int vs_off;
  int vs_width;
  int slot_bytes;
};

MxSlotLayout mx_slot_layout(int64_t fmt_id, int d) {
  const int n = d / cutlass::fmha::collective::mx::kMxGroupSize;
  switch (fmt_id) {
    case 0:  // mxfp8: K e4m3, V e4m3
      return {0, d, d, n, d + n, d, 2 * d + n, n, 2 * d + 2 * n};
    case 1:  // mxfp4: K e2m1, V e2m1
      return {
          0,
          d / 2,
          d / 2,
          n,
          d / 2 + n,
          d / 2,
          d + n,
          n,
          d + 2 * n};
    case 2: {  // mxk8v4: K e4m3, V e2m1
      return {
          0,
          d,
          d,
          n,
          d + n,
          d / 2,
          3 * d / 2 + n,
          n,
          3 * d / 2 + 2 * n};
    }
    default:
      TORCH_CHECK(false, "mx_paged_decode: unknown MX format id ", fmt_id);
  }
}

}  // namespace

void cutlass_mx_paged_decode_xe2(
    sycl::queue& queue,
    const at::Tensor& query,     // [total_q, num_heads_q, D] bf16
    const at::Tensor& mx_cache,  // [num_blocks, block, Hk, slot_bytes] u8
    int64_t fmt_id,
    at::Tensor& out,
    at::Tensor& temp_out,
    at::Tensor& exp_sums,
    at::Tensor& max_logits,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    double sm_scale,
    int num_kv_splits,
    std::optional<at::Tensor>& splits_per_seq,
    std::optional<at::Tensor>& work_list) {
  TORCH_CHECK(
      mx_cache.scalar_type() == at::ScalarType::Byte && mx_cache.dim() == 4,
      "mx_paged_decode: cache must be the uint8 MX slot pool "
      "[num_blocks, block_size, num_kv_heads, slot_bytes], got ",
      mx_cache.dim(),
      "-D ",
      mx_cache.scalar_type());
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::BFloat16 && query.dim() == 3,
      "mx_paged_decode: query must be [total_q, heads, D] bf16");

  const int batch_size = int(cu_seqlens_q.numel() - 1);
  const int total_seqlen_q = int(query.size(0));
  const int num_heads_q = int(query.size(1));
  const int head_size = int(query.size(2));
  const int num_heads_kv = int(mx_cache.size(2));
  const int num_blocks = int(mx_cache.size(0));
  const int block_size = int(mx_cache.size(1));
  const int max_blocks_per_seq = int(block_table.size(1));

  const MxSlotLayout slot = mx_slot_layout(fmt_id, head_size);
  TORCH_CHECK(
      int(mx_cache.size(3)) >= slot.slot_bytes,
      "mx_paged_decode: slot width ",
      mx_cache.size(3),
      " < ",
      slot.slot_bytes,
      " required for fmt ",
      fmt_id,
      " at D=",
      head_size);
  TORCH_CHECK(
      head_size == 128,
      "mx_paged_decode: head_size ",
      head_size,
      " not yet compiled (first-cut arc7 instantiations cover D=128; "
      "widen mx_paged_decode.hpp's dispatch set to add buckets)");
  TORCH_CHECK(
      block_size == 16,
      "mx_paged_decode: block_size ",
      block_size,
      " not yet compiled (first-cut arc7 covers page 16)");
  TORCH_CHECK(
      query.stride(-1) == 1 && mx_cache.stride(-1) == 1,
      "mx_paged_decode: query and cache must be contiguous in the last "
      "dimension");

  int total_seqlen_k = num_blocks * block_size;
  // Non-contiguous pools (cross-layer gaps): enlarge the load-surface
  // height so 2D-block loads at higher physical addresses stay valid
  // (stock law, paged_decode_xe2.cpp).
  const int64_t effective_total =
      get_paged_kv_cache_effective_total_seqlen(mx_cache);
  if (effective_total > total_seqlen_k) {
    total_seqlen_k = int(effective_total);
  }

  if (num_kv_splits <= 0) {
    num_kv_splits = mx_get_num_splits(
        queue,
        batch_size,
        num_heads_q,
        num_heads_kv,
        max_seqlen_k,
        block_size);
  }

  // Decode buffers (vendor shapes, flash_api.cpp decode branch).
  if (!out.defined()) {
    out = at::empty(
        {total_seqlen_q, num_heads_q, head_size},
        query.options().device(query.device()));
  }
  if (num_kv_splits == 1) {
    temp_out = out;
  } else if (!temp_out.defined()) {
    temp_out = at::empty(
        {total_seqlen_q, num_heads_q * num_kv_splits, head_size},
        query.options().device(query.device()));
  }
  if (!exp_sums.defined()) {
    exp_sums = at::empty(
        {total_seqlen_q, num_heads_q, num_kv_splits},
        query.options().dtype(at::kFloat).device(query.device()));
  }
  if (!max_logits.defined()) {
    max_logits = at::empty(
        {total_seqlen_q, num_heads_q, num_kv_splits},
        query.options().dtype(at::kFloat).device(query.device()));
  }

  // Four byte surfaces at the slot offsets; the pool's byte strides
  // are the surfaces' element strides verbatim (zero re-layout).
  const uint8_t* base = mx_cache.data_ptr<uint8_t>();
  mx_paged_decode_args_t args = {
      query.data_ptr(),
      const_cast<uint8_t*>(base + slot.kb_off),
      const_cast<uint8_t*>(base + slot.ks_off),
      const_cast<uint8_t*>(base + slot.vb_off),
      const_cast<uint8_t*>(base + slot.vs_off),
      out.data_ptr(),
      temp_out.data_ptr(),
      exp_sums.data_ptr(),
      max_logits.data_ptr(),
      block_table.data_ptr(),
      cu_seqlens_q.data_ptr(),
      cu_seqlens_k.data_ptr(),
      max_seqlen_q,
      max_seqlen_k,
      total_seqlen_q,
      total_seqlen_k,
      static_cast<float>(sm_scale),
      nullptr,  // sm_sink (no sink on this lane)
      batch_size,
      num_heads_q,
      num_heads_kv,
      head_size,
      head_size,  // v_head_size == D on this lane
      max_blocks_per_seq,
      block_size,
      -1,  // window_size_left (no local mask)
      -1,  // window_size_right
      false,  // is_causal: always false for decode (vendor law)
      false,  // is_local
      false,  // is_sink
      num_kv_splits,
      nullptr,  // splits_per_seq, filled below
      nullptr,  // work_list, filled below
      0,        // total_wgs, filled below
      mx_cache.stride(0),  // page stride (bytes == surface elements)
      mx_cache.stride(1),  // seq stride within a page
      mx_cache.stride(2),  // head stride
      mx_cache.stride(0),
      mx_cache.stride(1),
      mx_cache.stride(2),
      nullptr,  // is_prefill (decode-only lane)
      query.stride(0),
      query.stride(1),
      int64_t{0},  // batch stride unused for varlen
      static_cast<int>(mx_cache.stride(0) / mx_cache.stride(1))};

  if (splits_per_seq.has_value() && splits_per_seq->numel() > 0) {
    args.splits_per_seq = splits_per_seq->data_ptr<int>();
  }
  if (work_list.has_value() && work_list->numel() > 0) {
    args.work_list = work_list->data_ptr();
    args.total_wgs = int(work_list->size(0));  // [total_wgs, 4]
  }

  using cutlass::fmha::collective::mx::MxFormat;

  // qgroup bucket: the decode kernel tiles the GQA head-group across
  // the grid's Q dimension (stock law, paged_decode_xe2.cpp).
  const int num_q_group_size = num_heads_q / num_heads_kv;
  auto dispatch_fmt = [&](auto qgroup) {
    using policy_t = decode_policy_qpacked_head<decltype(qgroup), _128, _16>;
    switch (fmt_id) {
      case 0:
        mx_decode_policy_dispatch<policy_t, MxFormat::mxfp8>(queue, args);
        break;
      case 1:
        mx_decode_policy_dispatch<policy_t, MxFormat::mxfp4>(queue, args);
        break;
      default:
        mx_decode_policy_dispatch<policy_t, MxFormat::mxk8v4>(queue, args);
        break;
    }
  };
  if (num_q_group_size <= 8) {
    dispatch_fmt(_8{});
  } else {
    dispatch_fmt(_16{});
  }
}
