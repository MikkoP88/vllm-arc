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

#include <cstdlib>  // getenv/atoi (r200c probe knob)
#include <cstdio>   // FILE/fwrite (r200c debug dump)

#include "xpu/attn/xe_2/collective/chunk_prefill_scheduler.hpp"
#include "xpu/attn/xe_2/collective/chunk_prefill_epilogue.hpp"
#include "xpu/attn/xe_2/kernel/paged_decode_kernel.hpp"  // DecodeProblemShape, ReduceSplitK
#include "xpu/attn/xe_2/mx/mx_reduce_split.hpp"  // r208: fork-owned reduce (S>threads)

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
  // r200c V-lane probe (env ARC_MXKV_VPROBE, injected by the config's
  // run(); ablation telemetry ONLY — never set on serving paths):
  //   0 full lane | 1 skip staging (gather reads stale SLM; timing
  //   only) | 2 skip gather (tArV const-fill) | 3 skip both (floor) |
  //   4 staging without the barrier | 5 gather without scale reads.
  // Diffs against the full run attribute the V lane stage-by-stage.
  int vlane_probe = 0;
  // r200c debug lane (env ARC_MXKV_DEBUG allocates a static device
  // buffer in the config's run() and dumps it to /tmp/mxkv_debug.bin
  // after each call; SINGLE-WORKGROUP problem shapes only — multiple
  // WGs race on the buffer). Layout (bytes): A @0, 4 k-tiles x 16
  // rows x 128 (padded row pitch; live bytes = kVStagePitch per row);
  // B @8192, 8 int64 = per-tile (v_krow_base, v_byte_base); C @8256,
  // 4 tiles x VTiles x 16 WI x 32 floats = dequant_vslm outputs.
  // Null = off, zero cost.
  void* debug_buf = nullptr;
  // r212m direct-bf16 decode switch (env ARC_MXKV_DIRECTBF16; the
  // fast path folds the e8m0 code into the bf16 exponent — bit-
  // identical in the normal band, see mx_dequant.hpp). Default on;
  // 0 forces the float path (A/B oracle).
  int direct_bf16 = 1;
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
    // PITCH LAW (r200d correlator, both payload dtypes): the 2D-block
    // byte copies encode the cache row pitch in 16-byte units, so a row
    // pitch of 8 mod 16 (odd head count x 264/200-byte MX slots) rounds
    // up +8 bytes/row — row 0 lands exact while row k reads
    // cache[v_off + (slot+8)*k]: silent corruption. Refuse it loudly;
    // even head counts (production qgroup geometry) and every stock
    // dtype slot (256/128 bytes) keep the pitch aligned.
    TORCH_CHECK(
        args.k_stride_seq % 16 == 0 && args.v_stride_seq % 16 == 0,
        "mx_paged_decode: KV row pitch must be 16-byte aligned for the "
        "2D-block byte copies (pitch law): k_stride_seq=",
        args.k_stride_seq,
        " v_stride_seq=",
        args.v_stride_seq,
        " (odd head count x MX slot bytes drifts +8/row)");
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
         args.page_stride_elements,
         args.vlane_probe,
         args.debug_buf,
         args.direct_bf16},
        {},
        hw_info,
        args.num_kv_splits};

    typename ReductionSplitKernel::Arguments reduce_arg{
        {shape,
         reinterpret_cast<bfloat16_t*>(args.out),
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
    // r225 delta (4): the combine ALWAYS runs -- it is the single
    // bf16 quantization point now (fp32 partials in, final cast out;
    // one-split sequences ride the epilogue's sentinel pass-through).
    run(queue, params, reduce_params, /*need_reduce=*/true);

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
    typename SubgroupLayoutQK,
    bool LocalMask = false>
struct MxPagedDecodeConfig {
  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  // Both MMAs ride bf16 DPAS (full rate; the dequant feeds converted
  // fragments exactly as the stock fp8 lane types on ElementQ). P
  // enters the PV gemm as a SPLIT bf16 TRIPLE (mainloop): p = bf16(p)
  // + bf16(r1) + bf16(r2), r1 = p - bf16(p), r2 = r1 - bf16(r1),
  // holds p to 2^-27 relative -- below fp32's own accumulation noise.
  // The plain bf16 cast deviates up to ~8e-3 from the certified v2
  // envelope where rounding flips the output's last bit (r198
  // element-ledger: adjacent-grid-step flips dominate; tf32 only
  // shrank the walk), a PAIR's 2^-18 floor still showed 1-2% of |O|
  // at deep-cancellation elements (r198 mini-ledger), and no
  // fp32-operand DPAS exists on Xe2 (cute/arch/mma_xe.hpp set).
  using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, bfloat16_t>;
  using SubgroupLayoutPV =
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{}));

  using ElementQ = bfloat16_t;
  using ElementBytes = uint8_t;
  // r225 delta (4): the O element roles split. FINAL out stays bf16,
  // but the decode kernel's O write -- the KV-split partials, or the
  // normalized value at one split -- lands in the temp buffer, which
  // the binding now allocates fp32: bf16 partials quantized every
  // split to 2^-9 relative BEFORE the combine divides by the softmax
  // sum, which a FLAT decode softmax (deep-layer Ks codes 119-122;
  // r225l) amplifies to hundreds-thousands of output ulp (sum ~ 5e3
  // shrinks |O| ~5000x under the same absolute partial error). A
  // PEAKED softmax (sum ~ 1) masked it sub-ulp -- why every synthetic
  // gate was clean while real deep-layer payloads carried 15-26%
  // off-grid elements. The reduce kernel is now the ONE bf16
  // quantization point.
  using ElementO = bfloat16_t;  // the FINAL out dtype (reduce write)
  using ElementOAccum = float;  // decode O write / split partials

  using StrideQ = Stride<int, _1, int, int>;
  using StrideK = Stride<int, _1, int, int>;
  using StrideV = Stride<_1, int, int, int>;
  using StrideO = Stride<int, _1, int, int>;

  template <class Scheduler>
  static void run(sycl::queue& queue, const mx_paged_decode_args_t& args) {
    // r200c: the probe knob rides an env var so the ablation harness
    // needs no API changes (read once per call; unset = full lane).
    mx_paged_decode_args_t args_p = args;
    if (const char* vprobe = std::getenv("ARC_MXKV_VPROBE")) {
      args_p.vlane_probe = std::atoi(vprobe);
    }
    // r212m: direct-bf16 decode switch (default on; 0 = float path).
    if (const char* dbf = std::getenv("ARC_MXKV_DIRECTBF16")) {
      args_p.direct_bf16 = std::atoi(dbf);
    }
    // r200c debug lane: env-gated static device buffer + post-launch
    // D2H dump (see args.debug_buf layout comment). Leak-once on
    // purpose — ablation diagnosis only, never on serving paths.
    static void* dbg_dev = nullptr;
    static uint8_t* dbg_host = nullptr;
    constexpr size_t kDbgBufBytes = 8256 + 4 * 4 * 16 * 32 * 4;
    if (std::getenv("ARC_MXKV_DEBUG")) {
      if (dbg_dev == nullptr) {
        dbg_dev = sycl::malloc_device(kDbgBufBytes, queue);
        dbg_host = new uint8_t[kDbgBufBytes];
      }
      queue.memset(dbg_dev, 0xAA, kDbgBufBytes);
      args_p.debug_buf = dbg_dev;
    }
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
    using TensorO =
        decltype(make_dummy_tensor(ElementOAccum{}, StrideO{}));
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
        // P-C2 (2026-11-25): LocalMask rides the config template — the
        // vendor masking machinery (mainloop ~L1270, kernel k_block0
        // walk-bound) is compiled in only where a sliding-window model
        // can reach (256/16 + 256/64 at qgroup _8, the gemma-4-31b
        // sliding class; see the binding's dispatch_geom). is_local at
        // runtime pairs window_size_left = W-1 / right = 0 = the last
        // W tokens per row (wheel-law: triton_unified
        // sliding_window_val = 1 + window_size[0]).
        LocalMask>;

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

    // r208: fork-owned reduce -- the stock ReduceSplitK loads one split
    // per THREAD (breaks past 16 at page 16) and its combine-loop break
    // assumes the uniform-ceil partition (drops plan tail splits).
    using ReduceSplitKernel = cutlass::fmha::kernel::MxReduceSplitK<
        ProblemShapeType,
        cutlass::fmha::kernel::XeReduceSplitKTileScheduler,
        FMHAKernel>;

    MxDecodeKernelLauncher<FMHAKernel, ReduceSplitKernel> launcher;

    // r217: can_implement failures (e.g. num_kv_splits > the kernel
    // ceiling) previously returned kErrorInvalidProblem SILENTLY --
    // the engine streamed uninitialized attention (the ns96
    // fast-garbage cell, r216b). Fail loudly instead.
    TORCH_CHECK(
        launcher.run(queue, args_p, hw_info) ==
            cutlass::Status::kSuccess,
        "mx_paged_decode: kernel cannot implement the request "
        "(check num_kv_splits <= max_num_kv_splits)");

    if (args_p.debug_buf != nullptr) {
      constexpr size_t kDbgBufBytes = 8256 + 4 * 4 * 16 * 32 * 4;
      queue.memcpy(dbg_host, dbg_dev, kDbgBufBytes).wait();
      if (std::FILE* f = std::fopen("/tmp/mxkv_debug.bin", "wb")) {
        std::fwrite(dbg_host, 1, kDbgBufBytes, f);
        std::fclose(f);
      }
    }
  }

  static void
  kernel_dispatch(sycl::queue& queue, const mx_paged_decode_args_t& args) {
    return run<cutlass::fmha::kernel::DecodeTileScheduler>(queue, args);
  }
};

// Policy dispatch: format x qgroup, head _128, page _16 (the served
// target's cell). Head/page coverage widens by adding instantiations
// here — the policies and kernels themselves are shape-generic.
// P-C2: the trailing bool routes is_local into the compile-time
// LocalMask instantiation (masked/unmasked kernels are separate TUs;
// the binding refuses geometries whose masked twin is not compiled).
template <
    typename decode_policy,
    cutlass::fmha::collective::mx::MxFormat Fmt,
    bool LocalMask = false>
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
      typename decode_policy::SubgroupLayoutQK,
      LocalMask>::kernel_dispatch(queue, args);
}
