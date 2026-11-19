// MX-native split-KV decode kernel (arc7) — the stock
// XeFMHAFwdSplitKVKernel (kernel/paged_decode_kernel.hpp) with the
// K/V tensor construction extended to the four MX byte surfaces.
// ReduceSplitK is REUSED verbatim from the stock header (it only sees
// O accumulators + LSE — format-agnostic), as is the epilogue.
//
// Deltas from stock, all in tensor plumbing:
//   - KernelArguments carries the four surface base pointers (K bytes,
//     K scales, V bytes, V scales) instead of p.K/p.V;
//   - shape_K/shape_V use the format's byte widths (D or D/2 for the
//     payload, D/32 for the scales) as the column extent;
//   - the mainloop call passes the five tensors (Q + four surfaces).
// The paging walk, split logic, scheduler, softmax accumulation and
// epilogue call are byte-identical in structure.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/util/type_traits.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp"
#include "xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class MxXeFMHAFwdSplitKVKernel {
 public:
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len =
      cutlass::fmha::collective::is_variable_length_v<
          typename ProblemShape::SeqLenType>;
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::ElementQ;
  using ElementK = typename CollectiveMainloop::ElementK;  // uint8

  using Mx = typename CollectiveMainloop::Mx;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  // The payload/scale surfaces share the cache's (seq, head, page)
  // strides; the byte widths differ per surface.
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideKS = decltype(stride(typename CollectiveMainloop::TensorKS{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));
  using StrideVS = decltype(stride(typename CollectiveMainloop::TensorVS{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  static constexpr int max_num_kv_splits = SGPerWG::value * intel::sg_size;
  static constexpr int dpas_max_repeat_count = 8;
  static constexpr bool Sink = CollectiveEpilogue::Sink;
  using ElementSink = typename CollectiveEpilogue::ElementSink;

  static constexpr int kXeMaxSurfaceRows = 1 << 24;

  // Device side arguments. head_size_qk stays the LOGICAL element count
  // (problem-shape semantics); the byte widths derive from it via Mx.
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* Kb;    // K payload bytes
    const ElementK* Ks;    // K e8m0 scale bytes
    StrideK dK;            // (seq, _1, head, page) of the cache
    const ElementK* Vb;    // V payload bytes
    const ElementK* Vs;    // V e8m0 scale bytes
    StrideV dV;
    ElementO* Oaccum;
    StrideO dOaccum;
    ElementLSE* exp_sums;
    StrideO dExp_sums;
    ElementLSE* max_logits;
    StrideO dMax_logits;

    const ElementSink* sm_sink;

    const bool* is_prefill;
    const int* splits_per_seq = nullptr;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;
  };

  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    if (!is_var_len && args.kernel.shape.seq_len_qo != 1) {
      return false;
    }
    if (args.num_kv_splits > max_num_kv_splits) {
      return false;
    }
    return CollectiveMainloop::can_implement(args.mainloop) &&
           CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(
        params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(
      ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
      auto q_len = cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength>{problem_shape.seq_len_qo}, batch);
      return Shape<int, int>{
          get<0>(q_len), problem_shape.seq_len_kv.cumulative_length[batch]};
    } else {
      return Shape<int, int>{
          problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(
        sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits_;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto
          [blk_q,
           blk_v,
           head,
           idx_b,
           idx_kv_split,
           wl_tile_start,
           wl_tile_count] =
              tile_scheduler.get_block_coord();

      if (p.is_prefill != nullptr && p.is_prefill[idx_b]) continue;

      auto blk_qv = make_coord(blk_q, blk_v);
      int head_q_start = head * head_group_q;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;
      if (blk_q * get<0>(TileShapeQK{}) >= head_group_q) continue;

      auto offset = cute::min(seq_len_qo, seq_len_kv);
      auto discard_seq_coord = seq_len_qo - offset;
      auto full_tile_offset = seq_len_kv - offset;
      int seq_coord =
          cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

      if (CollectiveMainloop::CausalMask && seq_coord < discard_seq_coord)
        continue;
      const int seq_len = seq_len_kv;
      const int k_block0 =
          CollectiveMainloop::LocalMask
              ? cute::max(seq_len - 1 - params.mainloop.window_size_left, 0) /
                    get<1>(TileShapeQK{})
              : 0;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));
      const int windowed_k_blocks = k_blocks - k_block0;

      int offset_q = 0, offset_o = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;
        offset_q = get<0>(p.dQ) * qo_cumulative[idx_b];
        offset_o = s.num_heads_q * s.head_size_vo * num_kv_splits *
                   qo_cumulative[idx_b];
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits =
            s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        seq_len_qo = 1;
      }

      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q =
          make_shape(head_group_q, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto total_seqlen_kv = params.mainloop.total_seqlen_kv;

      // MX byte-surface shapes: payload widths D or D/2 per format
      // leg, scale widths D/32. head_size_qk is the LOGICAL element
      // count (== D); the widths below must match Mx (mx_types.hpp).
      const int d = s.head_size_qk;
      const int k_bw = Mx::k_is_e4m3 ? d : d / 2;
      const int v_bw = Mx::v_is_e4m3 ? d : d / 2;
      const int n_groups = d / collective::mx::kMxGroupSize;

      auto shape_K = make_shape(total_seqlen_kv, k_bw, s.num_heads_kv, batch_dim);
      auto shape_KS =
          make_shape(total_seqlen_kv, n_groups, s.num_heads_kv, batch_dim);
      // V surfaces are (width, total) — column-extent = seq positions.
      auto shape_V = make_shape(v_bw, total_seqlen_kv, s.num_heads_kv, batch_dim);
      auto shape_VS =
          make_shape(n_groups, total_seqlen_kv, s.num_heads_kv, batch_dim);

      auto shape_O = make_shape(
          head_group_q,
          s.head_size_vo,
          s.num_heads_kv,
          num_kv_splits,
          batch_dim);
      auto shape_exp_sums =
          make_shape(head_group_q, num_kv_splits, s.num_heads_kv, batch_dim);
      auto shape_max_logits =
          make_shape(head_group_q, num_kv_splits, s.num_heads_kv, batch_dim);
      auto shape_sink = make_shape(s.num_heads_kv, head_group_q);

      int kv_split_offset;
      int num_effective_kv_blocks;
      int seq_num_kv_splits;
      bool is_single_split;

      if (wl_tile_start >= 0) {
        kv_split_offset = k_block0 + wl_tile_start;
        num_effective_kv_blocks = wl_tile_count;
        seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                ? p.splits_per_seq[idx_b]
                                : num_kv_splits;
        is_single_split = (seq_num_kv_splits <= 1);
      } else {
        seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                ? p.splits_per_seq[idx_b]
                                : num_kv_splits;

        if (idx_kv_split >= seq_num_kv_splits) {
          continue;
        }

        int num_blocks_per_split =
            cute::ceil_div(windowed_k_blocks, seq_num_kv_splits);

        constexpr int tile_n = get<1>(TileShapeQK{});
        constexpr int kMinBlocksForSplit = (tile_n <= 64) ? 32 : 128;
        is_single_split =
            (seq_num_kv_splits > 1) && (windowed_k_blocks < kMinBlocksForSplit);

        if (is_single_split) {
          if (idx_kv_split > 0) {
            continue;
          }
          kv_split_offset = k_block0;
          num_effective_kv_blocks = windowed_k_blocks;
        } else {
          kv_split_offset = k_block0 + idx_kv_split * num_blocks_per_split;
          num_effective_kv_blocks = cute::min(
              windowed_k_blocks - idx_kv_split * num_blocks_per_split,
              num_blocks_per_split);
        }
      }

      if (num_effective_kv_blocks <= 0) {
        continue;
      }

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcKb = const_cast<ElementK*>(p.Kb);
      auto dcKs = const_cast<ElementK*>(p.Ks);
      auto dcVb = const_cast<ElementK*>(p.Vb);
      auto dcVs = const_cast<ElementK*>(p.Vs);
      auto ptrO = p.Oaccum + offset_o;
      auto ptrExp_sums = p.exp_sums + offset_exp_sums;
      auto ptrMax_logits = p.max_logits + offset_max_logits;

      auto layout_q = make_layout(
          shape_Q,
          make_stride(
              get<2>(p.dQ),
              _1{},
              static_cast<int>(head_group_q) * get<2>(p.dQ),
              get<3>(p.dQ)));
      auto layout_k = make_layout(
          shape_K, make_stride(get<0>(p.dK), _1{}, get<2>(p.dK), get<3>(p.dK)));
      auto layout_ks = make_layout(
          shape_KS,
          make_stride(get<0>(p.dK), _1{}, get<2>(p.dK), get<3>(p.dK)));
      auto layout_v = make_layout(
          shape_V, make_stride(_1{}, get<1>(p.dV), get<2>(p.dV), get<3>(p.dV)));
      auto layout_vs = make_layout(
          shape_VS,
          make_stride(_1{}, get<1>(p.dV), get<2>(p.dV), get<3>(p.dV)));

      auto layout_o = make_ordered_layout(shape_O, Step<_1, _0, _2, _3, _4>{});
      auto layout_exp_sums =
          make_ordered_layout(shape_exp_sums, Step<_1, _0, _2, _3>{});
      auto layout_max_logits =
          make_ordered_layout(shape_max_logits, Step<_1, _0, _2, _3>{});
      auto layout_sink = make_ordered_layout(shape_sink, Step<_1, _0>{});

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), layout_q);
      Tensor Kb = make_tensor(make_gmem_ptr(dcKb), layout_k);
      Tensor Ks = make_tensor(make_gmem_ptr(dcKs), layout_ks);
      Tensor Vb = make_tensor(make_gmem_ptr(dcVb), layout_v);
      Tensor Vs = make_tensor(make_gmem_ptr(dcVs), layout_vs);
      Tensor O = make_tensor(make_gmem_ptr(ptrO), layout_o);
      Tensor exp_sums =
          make_tensor(make_gmem_ptr(ptrExp_sums), layout_exp_sums);
      Tensor max_logits =
          make_tensor(make_gmem_ptr(ptrMax_logits), layout_max_logits);
      Tensor sinks = make_tensor(
          make_gmem_ptr(const_cast<ElementSink*>(p.sm_sink)), layout_sink);

      FragA tArA;
      FragARow tA_max, tA_sum;

      int l_coord = is_var_len ? 0 : idx_b;

      int start_blk = kv_split_offset;
      int end_blk = kv_split_offset + num_effective_kv_blocks;

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      bool has_large_surface = kXeMaxSurfaceRows < total_seqlen_kv;
      if (has_large_surface) {
        mainloop.template operator()<true>(
            Q(_, _, head, l_coord),
            Kb(_, _, head, l_coord),
            Ks(_, _, head, l_coord),
            Vb(_, _, head, l_coord),
            Vs(_, _, head, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            idx_b,
            start_blk,
            end_blk,
            k_blocks,
            thr_id,
            seq_len,
            full_tile_offset,
            discard_seq_coord);
      } else {
        mainloop.template operator()<false>(
            Q(_, _, head, l_coord),
            Kb(_, _, head, l_coord),
            Ks(_, _, head, l_coord),
            Vb(_, _, head, l_coord),
            Vs(_, _, head, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            idx_b,
            start_blk,
            end_blk,
            k_blocks,
            thr_id,
            seq_len,
            full_tile_offset,
            discard_seq_coord);
      }

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      if constexpr (Sink) {
        auto sinks_per_kv = sinks(head, _);
        epilogue(
            O(_, _, head, idx_kv_split, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            thr_id,
            exp_sums(_, _, head, l_coord),
            max_logits(_, _, head, l_coord),
            idx_kv_split,
            head_group_q,
            sinks_per_kv,
            num_kv_splits,
            is_single_split);
      } else {
        epilogue(
            O(_, _, head, idx_kv_split, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            thr_id,
            exp_sums(_, _, head, l_coord),
            max_logits(_, _, head, l_coord),
            idx_kv_split,
            head_group_q,
            sinks,
            num_kv_splits,
            is_single_split);
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel
