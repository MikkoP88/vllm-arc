// MX-native paged decode: config, launcher and policy dispatch (arc7).
// Fork of xe_2/paged_decode.hpp with the MX surface deltas:
//
//   - the cache is ONE uint8 TQ-AoS slot pool [num_blocks, block, Hk,
//     slot_bytes]; the four byte surfaces (K payload, K e8m0 scales, V
//     payload, V e8m0 scales) are slot sub-ranges at the mx_types.hpp
//     offsets, sharing the pool's (seq, head, page) byte strides;
//   - ElementK/V = uint8 for every surface; the MMA stays bf16 (the
//     dequant fills float fragments that reorder() casts down, the
//     stock fp8 division of labor);
//   - explicit KV strides are REQUIRED (the packed-stride fallback
//     would compute column-contiguous strides that the slot pool
//     does not have);
//   - the instantiation set is self-contained (no .conf gating): the
//     three MX formats x qgroup {_8, _16} x head _128 x page _16 x
//     causal {false, true}. The stock ReduceSplitK, DecodeTileScheduler
//     and DecodeFwdEpilogue are reused verbatim.
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"  // workspace allocation (stock include)
#include <cute/tensor.hpp>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "xpu/attn/xe_2/collective/chunk_prefill_scheduler.hpp"
#include "xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp"
#include "xpu/attn/xe_2/kernel/paged_decode_kernel.hpp"  // DecodeProblemShape, ReduceSplitK

#include "xpu/attn/xe_2/fmha_utils.hpp"
#include "xpu/attn/xe_2/mx/mx_types.hpp"
#include "xpu/attn/xe_2/mx/mx_paged_decode_kernel.hpp"

using namespace cute;

struct mx_paged_decode_args_t {
  void* query;    // [total_q, num_heads_q, D] bf16
  void* k_bytes;  // uint8 slot pool sub-ranges (entry computes offsets)
  void* k_scales;
  void* v_bytes;
  void* v_scales;
  void* out;
  void* tem_out;
  void* exp_sums;
  void* max_logits;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  float sm_scale;
  void* sm_sink;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;  // LOGICAL element count (== D); byte widths are
                  // compile-time via MxTraits
  int v_head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
  int num_kv_splits = 1;
  const int* splits_per_seq = nullptr;
  const void* work_list = nullptr;
  int total_wgs = 0;
  // Cache byte strides (shared by all four surfaces): the pool's
  // [num_blocks, block, Hk, slot_bytes] strides (1), (2), (0).
  int64_t k_stride_page = 0;
  int64_t k_stride_seq = 0;
  int64_t k_stride_heads = 0;
  int64_t v_stride_page = 0;
  int64_t v_stride_seq = 0;
  int64_t v_stride_heads = 0;
  void* is_prefill = nullptr;
  int64_t q_stride_seq = 0;
  int64_t q_stride_heads = 0;
  int64_t q_stride_batch = 0;
  int page_stride_elements = 0;
};

template <class FMHAKernel, class ReductionSplitKernel>
struct MxDecodeKernelLauncher {
  // Decode is always varlen + paged on this path (mirrors the stock
  // decode config's hardcoded VarLen/Paged).
  static constexpr bool isVarLen = true;

  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementO = typename FMHAKernel::ElementO;
  using ElementLSE = typename FMHAKernel::ElementLSE;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::DecodeProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::DecodeProblemShape<false>;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideO stride_Oaccum;
  StrideO stride_exp_sums;
  StrideO stride_max_logits;

  int num_kv_splits;

  ProblemShapeType initialize(const mx_paged_decode_args_t& args) {
    ProblemShapeType shape;
    ProblemShapeTypeInit shape_init;
    auto batch = shape.batch = shape_init.batch = args.batch_size;
    auto num_heads_q = shape.num_heads_q = shape_init.num_heads_q =
        args.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv = shape_init.num_heads_kv =
        args.num_heads_k;
    auto head_size_qk = shape.head_size_qk = shape_init.head_size_qk =
        args.head_size;
    auto head_size_vo = shape.head_size_vo = shape_init.head_size_vo =
        args.v_head_size;

    batch = shape_init.batch = 1;
    shape_init.seq_len_qo = args.total_seqlen_q;
    shape_init.seq_len_kv = args.total_seqlen_k;

    shape.seq_len_qo =
        cutlass::fmha::collective::VariableLength{args.max_queries};
    shape.seq_len_qo.cumulative_length =
        reinterpret_cast<int*>(args.cu_seqlens_q);
    shape.seq_len_kv =
        cutlass::fmha::collective::VariableLength{args.max_keys};
    shape.seq_len_kv.cumulative_length =
        reinterpret_cast<int*>(args.cu_seqlens_k);

    auto seq_len_qo = shape_init.seq_len_qo;
    auto seq_len_kv = shape_init.seq_len_kv;

    num_kv_splits = args.num_kv_splits;

    TORCH_CHECK(
        args.q_stride_seq > 0 && args.q_stride_heads > 0,
        "mx_paged_decode: explicit Q strides required");
    constexpr int64_t kIntMax =
        static_cast<int64_t>(std::numeric_limits<int>::max());
    TORCH_CHECK(
        args.q_stride_seq <= kIntMax && args.q_stride_heads <= kIntMax &&
            args.q_stride_batch <= kIntMax,
        "Q stride exceeds int32 max (",
        kIntMax,
        "): q_stride_seq=",
        args.q_stride_seq,
        " q_stride_heads=",
        args.q_stride_heads,
        " q_stride_batch=",
        args.q_stride_batch);
    stride_Q = StrideQ{
        static_cast<int>(args.q_stride_seq),
        _1{},
        static_cast<int>(args.q_stride_heads),
        static_cast<int>(args.q_stride_batch)};

    // The slot pool is never column-contiguous at page granularity:
    // require the pool's byte strides verbatim (they are shared by the
    // payload and scale surfaces — one stride triple for K, one for V).
    TORCH_CHECK(
        args.k_stride_seq > 0,
        "mx_paged_decode: explicit KV cache strides required "
        "(TQ-AoS slot pool)");
    TORCH_CHECK(
        args.k_stride_seq <= kIntMax && args.k_stride_heads <= kIntMax &&
            args.k_stride_page <= kIntMax && args.v_stride_seq <= kIntMax &&
            args.v_stride_heads <= kIntMax && args.v_stride_page <= kIntMax,
        "KV cache stride exceeds int32 max (",
        kIntMax,
        "): k_stride_seq=",
        args.k_stride_seq,
        " k_stride_heads=",
        args.k_stride_heads,
        " k_stride_page=",
        args.k_stride_page,
        " v_stride_seq=",
        args.v_stride_seq,
        " v_stride_heads=",
        args.v_stride_heads,
        " v_stride_page=",
        args.v_stride_page);
    stride_K = StrideK{
        static_cast<int>(args.k_stride_seq),
        _1{},
        static_cast<int>(args.k_stride_heads),
        static_cast<int>(args.k_stride_page)};
    stride_V = StrideV{
        _1{},
        static_cast<int>(args.v_stride_seq),
        static_cast<int>(args.v_stride_heads),
        static_cast<int>(args.v_stride_page)};

    stride_O = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch));
    stride_Oaccum = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(
            seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch));

    stride_exp_sums = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    stride_max_logits = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    return shape;
  }

  cutlass::Status
  run(sycl::queue& queue,
      const mx_paged_decode_args_t& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(args);

    typename FMHAKernel::Arguments arguments{
        {
            shape,
            reinterpret_cast<ElementQ*>(args.query),
            stride_Q,
            reinterpret_cast<ElementK*>(args.k_bytes),
            reinterpret_cast<ElementK*>(args.k_scales),
            stride_K,
            reinterpret_cast<ElementK*>(args.v_bytes),
            reinterpret_cast<ElementK*>(args.v_scales),
            stride_V,
            reinterpret_cast<ElementO*>(args.tem_out),
            stride_Oaccum,
            reinterpret_cast<ElementLSE*>(args.exp_sums),
            stride_exp_sums,
            reinterpret_cast<ElementLSE*>(args.max_logits),
            stride_max_logits,
            reinterpret_cast<ElementQ*>(args.sm_sink),
            static_cast<const bool*>(args.is_prefill),
            args.splits_per_seq,
        },
        {args.sm_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         args.window_size_left,
         args.window_size_right,
         args.page_stride_elements},
        {},
        hw_info,
        args.num_kv_splits};

    typename ReductionSplitKernel::Arguments reduce_arg{
        {shape,
         reinterpret_cast<ElementO*>(args.out),
         stride_O,
         reinterpret_cast<ElementO*>(args.tem_out),
         stride_Oaccum,
         reinterpret_cast<ElementLSE*>(args.exp_sums),
         stride_exp_sums,
         reinterpret_cast<ElementLSE*>(args.max_logits),
         stride_max_logits,
         args.window_size_left,
         static_cast<const bool*>(args.is_prefill),
         args.splits_per_seq},
        hw_info,
        args.num_kv_splits};

    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    size_t reduce_workspace_size =
        ReductionSplitKernel::get_workspace_size(reduce_arg);
    cutlass::device_memory::allocation<uint8_t> workspace(
        workspace_size + reduce_workspace_size);

    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << args.batch_size << 'x'
                << args.num_heads_q << 'x' << args.max_queries << 'x'
                << args.max_keys << 'x' << args.head_size << 'x'
                << args.v_head_size << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    FMHAKernel::initialize_workspace(arguments, workspace.get());

    auto params =
        FMHAKernel::to_underlying_arguments(arguments, workspace.get());
    auto reduce_params = ReductionSplitKernel::to_underlying_arguments(
        reduce_arg, workspace.get() + workspace_size);

    // Compact grid: patch scheduler params with the host split plan.
    if (args.work_list != nullptr && args.total_wgs > 0) {
      params.scheduler.work_list =
          reinterpret_cast<const cutlass::fmha::kernel::DecodeWorkItem*>(
              args.work_list);
      params.scheduler.total_wgs = args.total_wgs;
      params.scheduler.grid.z = args.total_wgs * args.num_heads_k;
    }

    ReductionSplitKernel::initialize_workspace(
        reduce_arg, workspace.get() + workspace_size);
    run(queue, params, reduce_params, args.num_kv_splits > 1);

    return cutlass::Status::kSuccess;
  }

  static void
  run(sycl::queue& queue,
      typename FMHAKernel::Params params,
      typename ReductionSplitKernel::Params reduce_params,
      bool need_reduce) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
        policy, queue, params);

    if (need_reduce) {
      dim3 const reduce_grid =
          ReductionSplitKernel::get_grid_shape(reduce_params);
      int reduce_smem_size = ReductionSplitKernel::SharedStorageSize;
      const auto reduce_sycl_block = compat::dim3(block.x, block.y, block.z);
      const auto reduce_sycl_grid =
          compat::dim3(reduce_grid.x, reduce_grid.y, reduce_grid.z);
      compat::experimental::launch_properties launch_props_reduce{
          syclex::work_group_scratch_size(reduce_smem_size),
      };
      compat::experimental::launch_policy reduce_policy{
          reduce_sycl_grid,
          reduce_sycl_block,
          launch_props_reduce,
          kernel_props};

      compat::experimental::launch<
          cutlass::device_kernel<ReductionSplitKernel>>(
          reduce_policy, queue, reduce_params);
    }
  }
};

template <
    typename MxTraits,
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK>
struct MxPagedDecodeConfig {
  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  // The MMA is typed on Q (bf16 MACs — the dequant feeds converted
  // fragments, exactly as the stock fp8 lane types on ElementQ).
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, bfloat16_t>;
  using SubgroupLayoutPV =
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using ElementQ = bfloat16_t;
  using ElementBytes = uint8_t;
  using ElementO = bfloat16_t;  // the out dtype; accumulators stay float

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;

  template <class Scheduler>
  static void run(sycl::queue& queue, const mx_paged_decode_args_t& args) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::DecodeProblemShape<true>;

    using TiledMMAQK = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapeQK>,
        SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapePV>,
        SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(
          make_gmem_ptr(&val),
          make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementBytes{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementBytes{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
    using TensorLSE = decltype(make_dummy_tensor(float{}, StrideO{}));

    // Mainloop: the scale surfaces share the K/V stride types.
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<1>;
    using CollectiveMainloop = cutlass::fmha::collective::MxDecodeFwdMainloop<
        MainloopDispatchPolicy,
        true,   // Paged
        false,  // Causal: always false for decode (vendor law —
                // seqused_k already bounds KV to valid past tokens;
                // is_causal=true adds +q_sg_tile KV positions and
                // pollutes the output with invalid cache entries)
        MxTraits,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorK,  // TensorKS: same type as TensorK (u8, K strides)
        TensorV,
        TensorV,  // TensorVS: same type as TensorV (u8, V strides)
        false>;   // Local

    // Epilogue (stock, format-agnostic)
    using CollectiveEpilogue = cutlass::fmha::collective::DecodeFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        TensorLSE,
        void,
        false>;  // Sink

    using FMHAKernel = cutlass::fmha::kernel::MxXeFMHAFwdSplitKVKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    using ReduceSplitKernel = cutlass::fmha::kernel::ReduceSplitK<
        ProblemShapeType,
        cutlass::fmha::kernel::XeReduceSplitKTileScheduler,
        FMHAKernel>;

    MxDecodeKernelLauncher<FMHAKernel, ReduceSplitKernel> launcher;

    launcher.run(queue, args, hw_info);
  }

  static void
  kernel_dispatch(sycl::queue& queue, const mx_paged_decode_args_t& args) {
    return run<cutlass::fmha::kernel::DecodeTileScheduler>(queue, args);
  }
};

// Policy dispatch: format x qgroup, head _128, page _16 (the served
// target's cell). Head/page coverage widens by adding instantiations
// here — the policies and kernels themselves are shape-generic.
template <typename decode_policy, cutlass::fmha::collective::mx::MxFormat Fmt>
void mx_decode_policy_dispatch(
    sycl::queue& queue,
    const mx_paged_decode_args_t& args) {
  using MxTraits =
      cutlass::fmha::collective::mx::MxTraits<Fmt>;
  MxPagedDecodeConfig<
      MxTraits,
      typename decode_policy::ShapeQK,
      typename decode_policy::ShapePV,
      typename decode_policy::ShapeOut,
      typename decode_policy::SubgroupLayoutQK>::kernel_dispatch(queue, args);
}
