// mx_reduce_split.hpp -- arc7 MX-native split-KV reduce (r208).
//
// Fork-owned copy of the stock ReduceSplitK
// (csrc/xpu/attn/xe_2/kernel/paged_decode_kernel.hpp, class ReduceSplitK)
// with THREE deltas: two functional ones required once num_kv_splits
// exceeds
// the workgroup thread count (16 at page 16; raised to 64 in r207):
//
// 1. Stock step-1 loader was `thr_id < effective_splits` -- ONE THREAD
//    PER SPLIT. Past 16 threads, splits 16+ were never loaded into SLM
//    (the global-max reduce missed them) while the combine loop read
//    UNWRITTEN SLM for those indices -> garbage-rescaled partials, error
//    growing with S (r207 sweep canary: 2.4e-4 at s<=16, 3.5e-2 at 24,
//    8.5e-2 at 64). Here the loader is thread-strided (same walk shape
//    as the combine's D loop) and zero-fills guarded-out slots so the
//    combine's exp_sums<=0 skip stays the only emptiness signal.
//
// 2. Stock combine loop broke at `i * num_blocks_per_split >=
//    windowed_k_blocks` -- the UNIFORM-ceil partition invariance. The
//    host work-list plan partitions per seq with divmod (first rem
//    splits get base+1 tiles), so a tail split can own tiles where the
//    uniform guard says none (wkb=1366, S=64: split 63 owns 21 tiles,
//    63*22 >= 1366 breaks anyway) -- its contribution is silently
//    dropped. With splits_per_seq present the plan contract guarantees
//    every split in [0, effective_splits) owns >= 1 tile, so the break
//    is only applied on the legacy (no-plan) path where the uniform
//    ceil partition is the truth.
//
// Everything else is byte-equivalent to the stock class (provenance:
// stock file lines 531-855 at the r207 anchor; bitwise faces at s<=16
// must not move -- the strided loader degenerates to the stock
// one-thread-per-split mapping there).
#pragma once

#include "xpu/attn/xe_2/fmha_utils.hpp"
#include "xpu/attn/xe_2/kernel/paged_decode_kernel.hpp"  // scheduler, ProblemShape

namespace cutlass::fmha::kernel {

template <class ProblemShape_, class TileScheduler_, class FMHAKernel_>
class MxReduceSplitK {
 public:
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len =
      cutlass::fmha::collective::is_variable_length_v<
          typename ProblemShape::SeqLenType>;
  using TileScheduler = TileScheduler_;
  static_assert(
      is_same_v<
          TileScheduler,
          cutlass::fmha::kernel::XeReduceSplitKTileScheduler>,
      "MxReduceSplitK kernel requires XeReduceSplitKTileScheduler");
  using TileSchedulerParams = typename TileScheduler::Params;

  using ElementO = typename FMHAKernel_::ElementO;
  // r225 delta (4): the O WRITE is the final bf16 out (the decode
  // kernel's ElementO became the fp32 partial type -- see
  // mx_paged_decode.hpp delta (4)); Oaccum follows it as fp32.
  using ElementOFinal = bfloat16_t;
  using StrideO = typename FMHAKernel_::StrideO;
  using TileShapeO = typename FMHAKernel_::TileShapeO;
  using TileShapeQK = typename FMHAKernel_::TileShapeQK;

  using ElementLSE = typename FMHAKernel_::ElementLSE;

  using SGPerWG = typename FMHAKernel_::SGPerWG;

  // num values (head_dim) processed by each thread
  constexpr static int num_vals_per_thread =
      int(get<1>(TileShapeO{}) / (SGPerWG::value * intel::sg_size));

  //
  // Types
  //

  struct KernelArguments {
    ProblemShape shape;
    // outputs:
    ElementOFinal* O;
    StrideO dO;
    // below are inputs
    // TODO(stock): whether same dtype as output or accum?
    const ElementO* Oaccum;
    StrideO dOaccum;
    const ElementLSE* exp_sums;
    StrideO dExp_sums;
    const ElementLSE* max_logits;
    StrideO dMax_logits;
    int window_size_left = -1;

    // per-batch mask: true = prefill, false = decode; nullptr = process all
    const bool* is_prefill;
    const int* splits_per_seq = nullptr;  // per-seq split counts
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;  // no split by default
  };

  /// Params structure
  struct Params {
    KernelParams kernel;
    TileSchedulerParams scheduler;
  };

  struct SharedStorage {
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits>
        max_logits_slm_array;
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits>
        exp_sums_slm_array;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

 public:
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    // only support decode
    if (!is_var_len && args.kernel.shape.seq_len_qo > 1) {
      return false;
    }

    if (args.num_kv_splits > FMHAKernel_::max_num_kv_splits) {
      return false;
    }
    return true;
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

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int tid_in_sg = thr_id % intel::sg_size;

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits;

    auto batch_dim = is_var_len ? 1 : s.batch;
    auto num_heads_q = s.num_heads_q;
    auto head_size_vo = s.head_size_vo;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [seq_idx, head_q, idx_b] = tile_scheduler.get_block_coord();

      // Skip prefill batches when is_prefill mask is provided
      if (p.is_prefill != nullptr && p.is_prefill[idx_b]) continue;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;

      // when varlen enabled, use largest seq_len_qo to decide work group num
      if (seq_idx >= seq_len_qo) continue;

      const int k_blocks = cute::ceil_div(seq_len_kv, get<1>(TileShapeQK{}));
      // Sliding window: skip blocks before the window
      constexpr bool LocalMask = FMHAKernel_::CollectiveMainloop::LocalMask;
      const int k_block0 =
          LocalMask ? cute::max(seq_len_kv - 1 - p.window_size_left, 0) /
                          get<1>(TileShapeQK{})
                    : 0;
      const int windowed_k_blocks = k_blocks - k_block0;
      // Per-sequence adaptive split count
      // r231d MIRROR (seqused-tightened splits): must derive the SAME
      // clamped count the FMHA kernel derived for this seq -- the
      // legacy path clamps on the windowed block count (>=4 k-tiles
      // per split) so shallow-ctx decode stops over-provisioning
      // splits it then has to memset/reduce; a count mismatch here
      // would iterate past the last written Oaccum slot.
      int seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                  ? p.splits_per_seq[idx_b]
                                  : cute::min(num_kv_splits,
                                              cute::max(1,
                                                        windowed_k_blocks /
                                                            4));

      int num_blocks_per_split =
          cute::ceil_div(windowed_k_blocks, seq_num_kv_splits);

      // is_single_split is a heuristic owned by the FMHA kernel; when the
      // host provides splits_per_seq, the host has already applied the same
      // policy (see build_decode_split_plan in flash_attn_interface.py), so
      // trust it directly. Otherwise (legacy path), mirror the FMHA kernel.
      int effective_splits;
      if (p.splits_per_seq != nullptr) {
        effective_splits = seq_num_kv_splits;
      } else {
        constexpr int tile_n = get<1>(typename FMHAKernel_::TileShapeQK{});
        constexpr int kMinBlocksForSplit = (tile_n <= 64) ? 32 : 128;
        bool is_single_split =
            (seq_num_kv_splits > 1) && (windowed_k_blocks < kMinBlocksForSplit);
        effective_splits = is_single_split ? 1 : seq_num_kv_splits;
      }

      int offset_o = 0, offset_o_accum = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;

      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;

        offset_o_accum = s.num_heads_q * s.head_size_vo * num_kv_splits *
                         qo_cumulative[idx_b];
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits =
            s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];

        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
      }

      auto shape_O =
          make_shape(seq_len_qo, head_size_vo, num_heads_q, batch_dim);
      auto shape_Oaccum = is_var_len ? make_shape(
                                           seq_len_qo,
                                           head_size_vo,
                                           num_heads_q * num_kv_splits,
                                           batch_dim)
                                     : make_shape(
                                           seq_len_qo,
                                           head_size_vo,
                                           num_heads_q * num_kv_splits,
                                           batch_dim);

      auto shape_exp_sums =
          make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);
      auto shape_max_logits =
          make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);

      auto dcOaccum = const_cast<ElementO*>(p.Oaccum + offset_o_accum);
      auto ptrExp_sums = const_cast<ElementLSE*>(p.exp_sums + offset_exp_sums);
      auto ptrMax_logits =
          const_cast<ElementLSE*>(p.max_logits + offset_max_logits);
      auto ptrO = p.O + offset_o;

      auto stride_o = is_var_len
                          ? cutlass::make_cute_packed_stride(StrideO{}, shape_O)
                          : p.dO;
      auto stride_o_accum =
          is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_Oaccum)
                     : p.dOaccum;
      auto stride_exp_sums = is_var_len ? cutlass::make_cute_packed_stride(
                                              StrideO{}, shape_exp_sums)
                                        : p.dExp_sums;
      auto stride_max_logits = is_var_len ? cutlass::make_cute_packed_stride(
                                                StrideO{}, shape_max_logits)
                                          : p.dMax_logits;

      Tensor Oaccum = make_tensor(
          make_gmem_ptr(dcOaccum), make_layout(shape_Oaccum, stride_o_accum));
      Tensor O =
          make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      Tensor exp_sums = make_tensor(
          make_gmem_ptr(ptrExp_sums),
          make_layout(shape_exp_sums, stride_exp_sums));
      Tensor max_logits = make_tensor(
          make_gmem_ptr(ptrMax_logits),
          make_layout(shape_max_logits, stride_max_logits));

      int l_coord = is_var_len ? 0 : idx_b;

      // Step 1: reduce max logits across different partitions
      // store into SLM for later use
      //
      // r208 fork delta (1): thread-strided loader (see file header).
      // At effective_splits <= threads this degenerates to the stock
      // one-thread-per-split mapping (bitwise-parity at s <= 16).
      ElementLSE global_max_logits{
          cutlass::platform::numeric_limits<ElementLSE>::lowest()};
      ElementLSE global_exp_sums{0};
      for (int i = thr_id; i < effective_splits;
           i += SGPerWG::value * intel::sg_size) {
        if (i * num_blocks_per_split < windowed_k_blocks ||
            p.splits_per_seq != nullptr) {
          ElementLSE cur_max_logit = max_logits(seq_idx, i, head_q, l_coord);
          global_max_logits = sycl::max(global_max_logits, cur_max_logit);
          shared_storage.max_logits_slm_array[i] = cur_max_logit;

          ElementLSE cur_exp_sum = exp_sums(seq_idx, i, head_q, l_coord);
          shared_storage.exp_sums_slm_array[i] = cur_exp_sum;
        } else {
          // zero-fill guarded-out slots so the combine's exp_sums<=0
          // skip stays the only emptiness signal (legacy path only).
          shared_storage.max_logits_slm_array[i] =
              cutlass::platform::numeric_limits<ElementLSE>::lowest();
          shared_storage.exp_sums_slm_array[i] = ElementLSE(0);
        }
      }

      // barrier for SLM writes finished
      sycl::group_barrier(get_work_group<3>());

      // reduce across wg
      global_max_logits = reduce_over_group(
          get_work_group<1>(), global_max_logits, sycl::maximum<>());

      // broadcast to all other threads
      global_max_logits =
          sycl::group_broadcast(get_work_group<1>(), global_max_logits, 0);

      for (int idx = thr_id; idx < s.head_size_vo;
           idx += SGPerWG::value * intel::sg_size) {
        ElementLSE acc = 0;
        global_exp_sums = 0;
        for (int i = 0; i < effective_splits; ++i) {
          // r208 fork delta (2): uniform-ceil break only on the legacy
          // path; the plan guarantees every split below
          // effective_splits owns tiles (see file header).
          if (p.splits_per_seq == nullptr &&
              i * num_blocks_per_split >= windowed_k_blocks) {
            break;
          }
          ElementLSE local_max_logit = shared_storage.max_logits_slm_array[i];
          ElementLSE local_exp_sum = shared_storage.exp_sums_slm_array[i];

          // Skip splits with no valid data (short sequences treated as
          // single-split have exp_sums=0 / max_logits=-inf for unused splits).
          if (local_exp_sum <= ElementLSE(0)) continue;

          ElementLSE rescale =
              // r225 delta (3), numerics: precise exp2. Stock (and the
              // r208 fork copy) rescaled partials with native::exp2 --
              // ~1e-3 relative (the instruction r198 evicted from the
              // mainloop softmax). Invisible in a peaked softmax (the
              // dominant split rescales by exp2(0) == 1 exactly, the
              // rest by ~0), but a FLAT decode softmax (deep-layer Ks
              // codes 119-122, r225k) makes every per-split delta
              // nontrivial and the approximation error lands on the
              // output -- the r225 residual (err/|O| p50 ~ 2^-9.4,
              // 15-26% of elements off-grid; r225l: nsplit1 exact,
              // auto/nsplit2/nsplit8 wrong, a flat synthetic with zero
              // real bytes reproduces it).
              sycl::exp2(local_max_logit - global_max_logits);

          // Partial outputs are unnormalized (not divided by exp_sum in the
          // epilogue), so combine them directly with the rescale factor.
          ElementLSE o_accum_val = static_cast<ElementLSE>(
              Oaccum(seq_idx, idx, i * num_heads_q + head_q, l_coord));
          acc += o_accum_val * rescale;

          // update global exp sum
          global_exp_sums += local_exp_sum * rescale;
        }

        ElementLSE inv_global_exp_sums = 1. / global_exp_sums;

        acc *= inv_global_exp_sums;
        O(seq_idx, idx, head_q, l_coord) =
            static_cast<ElementOFinal>(acc);
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel
