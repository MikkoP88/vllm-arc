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
    // r219b: 64 -> 128 = the validated optimum (r217/r217b A/A;
    // r219 tail cells show 192/256 regress). Not the engine branch
    // (page 64 -> else), kept consistent for page-16 deployments.
    policy_split_cap = 128;
  } else if (block_size == 32) {
    kv_tile = 32;
    sg_per_wg = 2;
    policy_split_cap = 32;
  } else {
    kv_tile = 64;
    sg_per_wg = 4;
    // r219b: 64 -> 128. This ELSE branch is the ENGINE branch (page
    // 64): r219 e2e showed auto pinned at exactly 64 by this cap
    // while the 13x reduce cap allowed 130; ns128 is the validated
    // optimum (r217/r217b/r218), and ns192/ns256 REGRESS (+7-13 pct,
    // r219 tail cells) so the cap sits AT the optimum, not above it.
    policy_split_cap = 128;
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
  // splits * num_heads_q. r219: 4 -> 13 slots/head, re-priced at
  // ENGINE geometry — the r210 4-slots/head knee (40 splits at the
  // c32768 bucket: W=120, H_q=12) was probe-priced and INVERTED on the
  // e2e faces (third sign-inversion-class instance): 64 = -11.45 pct,
  // 128 = -8.79 pct further, both A/A-confirmed (r216b/r217b); auto
  // now lands ~128-130 at the bucket (min with the policy cap). The
  // r207/r208 bench sweep that favored shallow splits was a 2-seq
  // slot-fill artifact. Short ctx stays workload-bounded (kv_tiles/12).
  int splits_reduce_cap =
      std::max(1, 13 * num_wg_slots / std::max(1, num_heads_q));
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
    // P-C2 (2026-11-25): sliding-window bounds, the stock
    // cutlass_paged_decode_impl arg order. W-token window = (W-1, 0)
    // — the last W tokens per row (right bound 0 keeps the diagonal;
    // the vendor mainloop masks col > pos + w_right). -1/-1 = the
    // historical windowless decode.
    int window_size_left,
    int window_size_right,
    int num_kv_splits,
    std::optional<at::Tensor>& splits_per_seq,
    std::optional<at::Tensor>& work_list) {
  const bool is_local =
      window_size_left >= 0 || window_size_right >= 0;
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
      head_size == 128 || head_size == 256 || head_size == 512,
      "mx_paged_decode: head_size ",
      head_size,
      " not yet compiled (arc7 instantiations cover D=128/256/512)");
  TORCH_CHECK(
      block_size == 16 || block_size == 64 || block_size == 128,
      "mx_paged_decode: block_size ",
      block_size,
      " not yet compiled (arc7 covers page 16/64/128)");
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
    // r212c2 law: the auto-split heuristic prices WG parallelism by
    // TILE count; every admitted geometry walks 16-row tiles (the _16
    // policy — page-64 pools ride it via the SG law below), so pricing
    // by the POOL's block size asked for 4x fewer tiles/splits at page
    // 64 and halved the walk's parallelism (r212c2: native 1147-1236
    // vs 598-644 us at 1K, 5040 vs 2891-adjacent at 32K). Widening the
    // dispatch to a true 64-row tile policy must revisit this together
    // with the SG-law route.
    num_kv_splits = mx_get_num_splits(
        queue,
        batch_size,
        num_heads_q,
        num_heads_kv,
        max_seqlen_k,
        16);
  }

  // Decode buffers (vendor shapes, flash_api.cpp decode branch).
  if (!out.defined()) {
    out = at::empty(
        {total_seqlen_q, num_heads_q, head_size},
        query.options().device(query.device()));
  }
  // r225 delta (4): partials ALWAYS fp32 in a dedicated temp buffer.
  // The old alias (num_kv_splits == 1 -> temp_out = out) and the bf16
  // partial allocation quantized each split's unnormalized
  // accumulator to 2^-9 relative BEFORE the combine divided by the
  // softmax sum -- hundreds-thousands of output ulp in the flat
  // deep-layer decode regime (see mx_paged_decode.hpp delta (4)).
  // Single-split runs keep the fp32 buffer too; the reduce kernel is
  // the one bf16 quantization point (sentinel pass-through).
  if (!temp_out.defined()) {
    temp_out = at::empty(
        {total_seqlen_q, num_heads_q * num_kv_splits, head_size},
        query.options().dtype(at::kFloat).device(query.device()));
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
      window_size_left,  // P-C2: -1/-1 keeps the windowless decode
      window_size_right,
      false,  // is_causal: always false for decode (vendor law)
      is_local,
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
  // r212b geometry widening (fail2 law, 2026-11-21): the e2e target
  // qwen3.8-27b-fp8 carries head_dim 256 at engine pool page 64 (TP2:
  // HQ 12 / HK 2 / G 6 -> the _8 qgroup). The first-cut set (D=128
  // page 16) was calibrated off the config's GDN linear head dim, not
  // the full-attention head_dim. Policies and kernels are shape-generic
  // (mx_paged_decode.hpp) — only the instantiation set widens:
  //   128/16  certified offline-calibration cell (r208-r212 ladder)
  //   256/16  head-width isolation at the harness pool's page
  //   256/64  the e2e engine pool page — served by the _16 policy's
  //           16-row sub-tiles (r212c SG law: the MX staging is
  //           single-SG; see the dispatch branch below)
  //   512/64  the gemma-4-31b full_attention head class (P-C1) —
  //           same _16 sub-tile frame; V tiles across grid.x per
  //           the vendor accumulator budget (see the branch below)
  // P-C2: `local` rides a compile-time tag (std::true_type/false_type)
  // — a runtime bool would instantiate BOTH masked/unmasked twins for
  // every geometry this lambda touches (the vendor
  // decode_policy_dispatch_func bool-resolution law). The masked twin
  // exists ONLY where a sliding-window model reaches: 256/16 +
  // 256/64-via-_16 at the qgroup<=8 bucket (gemma-4-31b sliding
  // class); the geometry gates TORCH_CHECK-refuse everything else.
  auto dispatch_fmt = [&](auto qgroup, auto head, auto page, auto local) {
    using policy_t = decode_policy_qpacked_head<
        decltype(qgroup),
        decltype(head),
        decltype(page)>;
    constexpr bool kLocal = decltype(local)::value;
    switch (fmt_id) {
      case 0:
        mx_decode_policy_dispatch<policy_t, MxFormat::mxfp8, kLocal>(
            queue, args);
        break;
      case 1:
        mx_decode_policy_dispatch<policy_t, MxFormat::mxfp4, kLocal>(
            queue, args);
        break;
      default:
        mx_decode_policy_dispatch<policy_t, MxFormat::mxk8v4, kLocal>(
            queue, args);
        break;
    }
  };
  // allow_local rides a compile-time tag so dispatch_geom<_16> never
  // instantiates a masked twin (the qgroup>8 + is_local posture is
  // refused above the call); only _8 x {256/16, 256/64-via-_16} carries
  // the masked twins — the instantiation discipline (P-C2).
  auto dispatch_geom = [&](auto qgroup, auto allow_local) {
    constexpr bool kAllowLocal = decltype(allow_local)::value;
    if (head_size == 128 && block_size == 16) {
      TORCH_CHECK(
          !is_local,
          "mx_paged_decode: window masking not compiled for D=",
          head_size,
          " page=",
          block_size,
          " (masked twins compiled: 256/16 + 256/64 at the qgroup<=8 "
          "bucket)");
      dispatch_fmt(qgroup, _128{}, _16{}, std::false_type{});
    } else if (head_size == 256 && block_size == 16) {
      if constexpr (kAllowLocal) {
        if (is_local) {
          dispatch_fmt(qgroup, _256{}, _16{}, std::true_type{});
        } else {
          dispatch_fmt(qgroup, _256{}, _16{}, std::false_type{});
        }
      } else {
        dispatch_fmt(qgroup, _256{}, _16{}, std::false_type{});
      }
    } else if (head_size == 256 && block_size == 64) {
      // r212c law (2026-11-21): page-64 pools ride the _16 policy — the
      // vendor _64 policy's SubgroupLayoutQK (1,4,1) puts FOUR subgroups
      // in the workgroup, but the MX staging scatter and its tv-decoder
      // are single-SG laws ("One subgroup per workgroup: no per-SG
      // offsets", mx_paged_decode_mainloop.hpp r200b/r203): multi-SG
      // thr_ids decode out-of-frame coordinates and corrupt ~0.64
      // relative (r212c eng256p64 c1024, all three formats, auto+es).
      // 16-row sub-tiles (4 per 64-page via the stock page-table
      // indirection) keep the certified frame; the true multi-SG p64
      // tile (1.5-2x lower fixed term: 306-404 vs ~600 us at 1K) is the
      // r212d deep-dive.
      //
      // r220a (2026-11-22): the multi-SG decode laws landed (band-baked
      // coordinates in the mainloop/dequant tv sites); this env route
      // probes the TRUE _64 policy offline before any shipping flip.
      // Probe instrument only -- the default stays the certified _16
      // sub-tile frame until p204 oracle + canary + e2e A/A pass.
      if (std::getenv("ARC_MXKV_P64NATIVE") != nullptr) {
        TORCH_CHECK(
            !is_local,
            "mx_paged_decode: window masking not compiled on the "
            "ARC_MXKV_P64NATIVE probe policy (256/64-via-_16 carries "
            "the masked twin)");
        dispatch_fmt(qgroup, _256{}, _64{}, std::false_type{});
      } else if constexpr (kAllowLocal) {
        if (is_local) {
          dispatch_fmt(qgroup, _256{}, _16{}, std::true_type{});
        } else {
          dispatch_fmt(qgroup, _256{}, _16{}, std::false_type{});
        }
      } else {
        dispatch_fmt(qgroup, _256{}, _16{}, std::false_type{});
      }
    } else if (head_size == 512 && block_size == 64) {
      // P-C1 (2026-11-25): gemma-4-31b full_attention layers — the
      // hybrid's D=512 head class (10/60 layers; sliding layers ride
      // 256/64 above). The FIRST r240b decode attempt refused here
      // ("head_size 512 not yet compiled") and no MX KV lane on this
      // image ever served gemma. Same certified _16 sub-tile frame as
      // 256/64 (r212c single-SG law — the _64 policy's (1,4,1) SG
      // layout breaks the MX staging); the policy carries _512 so the
      // vendor accumulator-budget V-cap tiles V across grid.x
      // (q_packed _8: ShapeOut V 256, two slices; gemma GQA G=2/8 both
      // land the _8 bucket). K walks D in the same 64-elem chunks — 8
      // per QK pass instead of 4, loop-count-generic (size<4> of the
      // runtime-strided partition). No new kernel surface: the stock
      // fp16 paged-decode lane already serves this head class with
      // the same policy machinery on this image (r240b gf8 cells).
      TORCH_CHECK(
          !is_local,
          "mx_paged_decode: window masking not compiled for the D=512 "
          "full-attention class (gemma-4-31b full_attention layers carry "
          "sliding_window=None; a windowed D=512 model needs the masked "
          "twin instantiated first)");
      dispatch_fmt(qgroup, _512{}, _16{}, std::false_type{});
    } else if (head_size == 512 && block_size == 128) {
      // r245a (2026-11-26): gemma-4-31b's hybrid KV manager gives the
      // D=512 full_attention group a PAGE-128 pool (the sliding D=256
      // group rides page 64 — the same boot banks both; r245 e2e died
      // at the first real request with "block_size 128 not yet
      // compiled" while the sliding layers' capture-time native calls
      // with window=1024 had already succeeded). Same _512/_16
      // sub-tile frame: the mainloop's page indirection is
      // runtime-generic (tiles_per_page = page_size/16 -> 8 sub-tiles
      // per 128-page vs 4 at page 64; page_local_idx = tile*16/
      // page_size), so no new kernel surface — only the admission.
      // Windowless class (see the 512/64 branch's refusal note).
      TORCH_CHECK(
          !is_local,
          "mx_paged_decode: window masking not compiled for the D=512 "
          "full-attention class (gemma-4-31b full_attention layers carry "
          "sliding_window=None; a windowed D=512 model needs the masked "
          "twin instantiated first)");
      dispatch_fmt(qgroup, _512{}, _16{}, std::false_type{});
    } else {
      TORCH_CHECK(
          false,
          "mx_paged_decode: no instantiation for D=",
          head_size,
          " page=",
          block_size,
          " (compiled: 128/16, 256/16, 256/64-via-_16, 512/64, "
          "512/128-via-_16)");
    }
  };
  TORCH_CHECK(
      !is_local || num_q_group_size <= 8,
      "mx_paged_decode: window masking compiled at the qgroup<=8 bucket "
      "only (this call carries G=",
      num_q_group_size,
      "; instantiate the _16 masked twin on a real model's evidence — "
      "no unevidenced surface)");
  if (num_q_group_size <= 8) {
    dispatch_geom(_8{}, std::true_type{});
  } else {
    dispatch_geom(_16{}, std::false_type{});
  }
}
