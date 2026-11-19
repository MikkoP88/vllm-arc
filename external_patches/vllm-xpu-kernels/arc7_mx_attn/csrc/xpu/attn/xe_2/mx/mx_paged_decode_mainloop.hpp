// MX-native decode mainloop (arc7, P6.47 Phase 8) — the stock
// DecodeFwdMainloop (collective/chunk_prefill_mainloop.hpp) with the
// K/V load path replaced for MX KV caches. Everything downstream of
// the MMA fragments is byte-identical in structure and semantics:
// paged tile indirection, 2^24-row surface split, register-resident
// online softmax, split-KV accumulation. The stock fp8 path casts
// fp8->bf16 inside reorder(); the MX path dequants uint8 payload +
// e8m0 group codes to bf16 in the copy-fragment layout
// (mx::dequant_dtile) and then rides the SAME stock reorder() for the
// layout change.
//
// Cache geometry (mx_layout.py law; no re-layout ever): one uint8 pool
// [num_blocks, block, Hk, slot_bytes]; per slot
// [K bytes][Ks e8m0 x D/32][V bytes][Vs e8m0 x D/32]. The kernel
// receives FOUR base pointers (K bytes, K scales, V bytes, V scales)
// sharing the cache's own (seq, head, page) strides; each is a 2D
// surface of (total_kv x width) uint8 with width from mx::MxTraits.
//
// D-tiling: the MMA consumes 64-element d-tiles (TileShapeQK[2]).
// e4m3 payloads are 1 byte/element (64-byte byte-tiles); e2m1 are 2
// elements/byte (32-byte byte-tiles). Each d-tile spans exactly
// 64 / kMxGroupSize = 2 scale groups for both formats. The V tile per
// VV step is 32 v-elements = exactly 1 scale group per row.
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"

#include "xpu/attn/xe_2/mx/mx_types.hpp"
#include "xpu/attn/xe_2/mx/mx_dequant.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template <
    class DispatchPolicy_,
    bool PagedKV_,
    bool CausalMask_,
    class MxTraits_,          // mx::MxTraits<format> instantiation
    class TiledMMAQK_,        // Tiling for Q*K GEMM
    class TiledMMAPV_,        // Tiling for P*V GEMM
    int VTiles_,              // # of tiles in V dimension
    class TensorQ_,           // Global tensors: Q (bf16), K/V/S scales (u8)
    class TensorK_,
    class TensorKS_,
    class TensorV_,
    class TensorVS_,
    bool LocalMask_ = false>
struct MxDecodeFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

template <
    int Stages,
    bool PagedKV_,
    bool CausalMask_,
    class MxTraits_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorKS_,
    class TensorV_,
    class TensorVS_,
    bool LocalMask_>
struct MxDecodeFwdMainloop<
    XeDefault<Stages>,
    PagedKV_,
    CausalMask_,
    MxTraits_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorKS_,
    TensorV_,
    TensorVS_,
    LocalMask_> {
  //
  // Type Aliases
  //
  using Mx = MxTraits_;
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{})));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;    // (k, k_bytes) u8 surface per head
  using TensorKS = TensorKS_;  // (k, k_groups) u8 surface per head
  using TensorV = TensorV_;    // (v_bytes, k) u8 surface per head
  using TensorVS = TensorVS_;  // (v_groups, k) u8 surface per head

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;  // uint8

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorKS2D =
      decltype(TensorKS_{}(append<rank_v<TensorKS_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));
  using TensorVS2D =
      decltype(TensorVS_{}(append<rank_v<TensorVS_>>(make_coord(_, _), 0)));

  // K/V payload byte counts per (64-element) d-tile, and groups per
  // tile. Uniform across formats: 2 groups per d-tile; V tiles of 32
  // elements carry 1 group per row.
  static constexpr int kDTileElems = 64;
  static constexpr int kElemsPerByteK = Mx::k_is_e4m3 ? 1 : 2;
  static constexpr int kElemsPerByteV = Mx::v_is_e4m3 ? 1 : 2;
  static constexpr int kBytesPerDTileK = kDTileElems / kElemsPerByteK;
  static constexpr int kBytesPerDTileV = kDTileElems / kElemsPerByteV;
  static constexpr int kGroupsPerDTile = kDTileElems / mx::kMxGroupSize;

  //
  // Accumulator types (identical to stock)
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk())));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool LocalMask = LocalMask_;
  // MX KV always: per-group scales cannot fold into the softmax scale
  // (they vary within a row); dequant-before-MMA is the math.
  static constexpr bool Fp8KV = false;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    // Paged KV Cache (shared geometry for all four byte surfaces)
    int const* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int window_size_left;
    int window_size_right;
    // Physical stride between paged blocks in seq-position units
    int page_stride_elements;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //
  MxDecodeFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.window_size_left,
        args.window_size_right,
        args.page_stride_elements};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,    // (q,d)
      TensorK2D const& K_2D,    // (k, k_bytes) u8
      TensorKS2D const& KS_2D,  // (k, k_groups) u8
      TensorV2D const& V_2D,    // (v_bytes, k) u8
      TensorVS2D const& VS_2D,  // (v_groups, k) u8
      FragA& tArA,              // Output accumulator (q,v)
      FragARow& tA_max,         // Softmax row-wise max accumulator
      FragARow& tA_sum,         // Softmax row-wise sum accumulator
      QVCoord blk_qv,           // WG tile indices: (Q,V)
      int const& idx_b,         // WG tile indices: (B)
      int blk_k0,               // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord) {
    using namespace sycl::ext::oneapi::this_work_item;

    constexpr int tile_k = get<1>(TileShapeQK{});

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());                // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());                // (k,kb)
    Tensor cKS = make_identity_tensor(KS_2D.shape());              // (k,kg)
    Tensor cV = make_identity_tensor(V_2D.shape());                // (vb,k)
    Tensor cVS = make_identity_tensor(VS_2D.shape());              // (vg,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));   // (q,k)

    auto K_2D_0 = K_2D;
    auto KS_2D_0 = KS_2D;
    auto V_2D_0 = V_2D;
    auto VS_2D_0 = VS_2D;

    // --- Xe 2D-block surface-height split (stock; the split math is
    // width-agnostic — it tiles rows/pages only). See the stock
    // mainloop for the full rationale (24-bit LSC surface height).
    [[maybe_unused]] int tiles_per_surface = 0x3fffffff;
    [[maybe_unused]] int rows_per_surface = 0;
    [[maybe_unused]] int k_rows_total = get<0>(K_2D.shape());
    [[maybe_unused]] int v_cols_total = get<1>(V_2D.shape());
    [[maybe_unused]] int64_t k_surf_shift = 0;
    [[maybe_unused]] int64_t ks_surf_shift = 0;
    [[maybe_unused]] int64_t v_surf_shift = 0;
    [[maybe_unused]] int64_t vs_surf_shift = 0;
    [[maybe_unused]] int k_pre_surf = -1;

    [[maybe_unused]] auto make_k_surface = [&](int surf) {
      int64_t rem = int64_t(k_rows_total) - int64_t(surf) * rows_per_surface;
      int rows = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (rows < 1) {
        rows = 1;
      }
      return make_tensor(
          K_2D.data() + int64_t(surf) * k_surf_shift,
          make_layout(make_shape(rows, get<1>(K_2D.shape())), K_2D.stride()));
    };
    [[maybe_unused]] auto make_ks_surface = [&](int surf) {
      int64_t rem = int64_t(k_rows_total) - int64_t(surf) * rows_per_surface;
      int rows = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (rows < 1) {
        rows = 1;
      }
      return make_tensor(
          KS_2D.data() + int64_t(surf) * ks_surf_shift,
          make_layout(make_shape(rows, get<1>(KS_2D.shape())), KS_2D.stride()));
    };
    [[maybe_unused]] auto make_v_surface = [&](int surf) {
      int64_t rem = int64_t(v_cols_total) - int64_t(surf) * rows_per_surface;
      int cols = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (cols < 1) {
        cols = 1;
      }
      return make_tensor(
          V_2D.data() + int64_t(surf) * v_surf_shift,
          make_layout(make_shape(get<0>(V_2D.shape()), cols), V_2D.stride()));
    };
    [[maybe_unused]] auto make_vs_surface = [&](int surf) {
      int64_t rem = int64_t(v_cols_total) - int64_t(surf) * rows_per_surface;
      int cols = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (cols < 1) {
        cols = 1;
      }
      return make_tensor(
          VS_2D.data() + int64_t(surf) * vs_surf_shift,
          make_layout(make_shape(get<0>(VS_2D.shape()), cols), VS_2D.stride()));
    };

    if constexpr (has_large_surface) {
      constexpr int kXeMaxSurfaceRows = 1 << 24;
      if constexpr (PagedKV) {
        int page_stride_tiles = params.page_stride_elements / tile_k;
        int blocks_per_surface =
            kXeMaxSurfaceRows / params.page_stride_elements;
        if (blocks_per_surface < 1) {
          blocks_per_surface = 1;
        }
        tiles_per_surface = blocks_per_surface * page_stride_tiles;
        rows_per_surface = tiles_per_surface * tile_k;
      }

      int k_rows0, v_cols0;
      if constexpr (PagedKV) {
        k_rows0 =
            k_rows_total < rows_per_surface ? k_rows_total : rows_per_surface;
        k_surf_shift = int64_t(rows_per_surface) * get<0>(K_2D.stride());
        ks_surf_shift = int64_t(rows_per_surface) * get<0>(KS_2D.stride());
        v_cols0 =
            v_cols_total < rows_per_surface ? v_cols_total : rows_per_surface;
        v_surf_shift = int64_t(rows_per_surface) * get<1>(V_2D.stride());
        vs_surf_shift = int64_t(rows_per_surface) * get<1>(VS_2D.stride());
      } else {
        k_rows0 = k_rows_total;
        v_cols0 = v_cols_total;
      }

      K_2D_0 = make_tensor(
          K_2D.data(),
          make_layout(
              make_shape(k_rows0, get<1>(K_2D.shape())), K_2D.stride()));
      KS_2D_0 = make_tensor(
          KS_2D.data(),
          make_layout(
              make_shape(k_rows0, get<1>(KS_2D.shape())), KS_2D.stride()));
      V_2D_0 = make_tensor(
          V_2D.data(),
          make_layout(
              make_shape(get<0>(V_2D.shape()), v_cols0), V_2D.stride()));
      VS_2D_0 = make_tensor(
          VS_2D.data(),
          make_layout(
              make_shape(get<0>(VS_2D.shape()), v_cols0), VS_2D.stride()));

      cK = make_identity_tensor(make_shape(
          PagedKV ? rows_per_surface : k_rows_total,
          get<1>(K_2D.shape())));  // (k,kb)
      cKS = make_identity_tensor(make_shape(
          PagedKV ? rows_per_surface : k_rows_total,
          get<1>(KS_2D.shape())));  // (k,kg)
      cV = make_identity_tensor(make_shape(
          get<0>(V_2D.shape()),
          PagedKV ? rows_per_surface : v_cols_total));  // (vb,k)
      cVS = make_identity_tensor(make_shape(
          get<0>(VS_2D.shape()),
          PagedKV ? rows_per_surface : v_cols_total));  // (vg,k)
    }

    /* Partition global tensors into workgroup tiles. The byte tiles
    (tile_k x kBytesPerDTileK) and scale tiles (tile_k x groups) are
    tiled over their own column extents; the K tile-index mode (over
    the paged pool) is shared. */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    using DTileBytes = Shape<Int<kBytesPerDTileK>>;
    using STileGroups = Shape<Int<kGroupsPerDTile>>;
    Tensor gK = local_tile(
        cK,
        make_shape(Int<get<1>(TileShapeQK{})>{}, DTileBytes{}),
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,kb,K,DB)
    Tensor gKS = local_tile(
        cKS,
        make_shape(Int<get<1>(TileShapeQK{})>{}, STileGroups{}),
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,kg,K,SG)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (vb,k,K)
    Tensor gV_split = local_tile(
        gV,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (vb,k,VV,K)

    /* Create global -> register copies. The Q copy is the stock B/A
    copy for the MMA; the K/V byte and scale copies are built over the
    u8 surfaces via the generic block-2D copy of the matching tile
    shapes (the stock make_block_2d_copy_B derives from the MMA tile
    and element type — u8 payloads tile by BYTES, so the copy is built
    from the tile shape directly). */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_kb{K_2D_0};
    TiledCopyKS copy_ks{KS_2D_0};
    TiledCopyV copy_vb{V_2D_0};
    TiledCopyVS copy_vs{VS_2D_0};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_kb = copy_kb.get_slice(thr_id);
    auto thr_copy_ks = copy_ks.get_slice(thr_id);
    auto thr_copy_vb = copy_vb.get_slice(thr_id);
    auto thr_copy_vs = copy_vs.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKBgK = thr_copy_kb.partition_S(gK);      // (atom_val,k',kb',K,DB)
    auto tKSgKS = thr_copy_ks.partition_S(gKS);    // (atom_val,k',sg',K,SG)
    auto tVbgV = thr_copy_vb.partition_S(gV_split);  // (atom_val,vb',k',VV,K)
    auto tVSgVS = thr_copy_vs.partition_S(
        local_tile(cVS, TileShapeVS{}, make_coord(get<1>(blk_qv), _),
                   Step<X, _1, _1>{}));  // (atom_val,vg',k',K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    auto tKrU8 = thr_copy_kb.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tKsU8 = thr_copy_ks.partition_sg_fragment_D(gKS(_, _, 0, 0));
    auto tKrDeq = make_fragment_like<ElementQ>(tKrU8);  // copy-layout dequant out
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK_deq_tile());
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tVrU8 = thr_copy_vb.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tVsU8 = thr_copy_vs.partition_sg_fragment_D(
        local_tile(cVS, TileShapeVS{}, make_coord(get<1>(blk_qv), _),
                   Step<X, _1, _1>{})(_, _, 0));
    auto tVrDeq = make_fragment_like<ElementQ>(tVrU8);
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    // NOTE [frag sizing]: tKrDeq holds kDTileElems-per-row values
    // while tKrU8 holds kBytesPerDTileK — make_fragment_like matches
    // SIZE, so for e2m1 the dequant destination is sized by hand:
    // see dequant fragment plumbing below. tSrK/tArV sizes follow the
    // MMA fragments (stock).

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_kb = make_block_2d_prefetch(copy_kb);
    auto prefetch_vb =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_0);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKBgK = prefetch_kb.get_slice(thr_id).partition_S(gK);
    auto pVbgV = prefetch_vb.get_slice(thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    // PagedKV (stock tile indirection; scale surfaces ride the same
    // physical tile index since they share the page/seq geometry)
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int page_stride_tiles = params.page_stride_elements / get<1>(TileShapeQK{});
    int tile_idx = blk_k0;
    int b_offset = idx_b * params.max_pages_per_seq;
    if constexpr (PagedKV) {
      int page_local_idx = tile_idx * get<1>(TileShapeQK{}) / params.page_size;
      int block_idx = params.ptr_page_table[b_offset + page_local_idx];
      tile_idx = block_idx * page_stride_tiles + tile_idx % tiles_per_page;
    }

    /* Q prefetch */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    [[maybe_unused]] auto update_surface_info = [&](int& next_page_idx) {
      int k_surf = PagedKV ? next_page_idx / tiles_per_surface : 0;
      next_page_idx = next_page_idx - k_surf * tiles_per_surface;

      if (k_pre_surf != k_surf) {
        k_pre_surf = k_surf;

        auto K_2D_n = make_k_surface(k_surf);
        auto KS_2D_n = make_ks_surface(k_surf);
        auto V_2D_n = make_v_surface(k_surf);
        auto VS_2D_n = make_vs_surface(k_surf);

        copy_kb = TiledCopyK{K_2D_n};
        copy_ks = TiledCopyKS{KS_2D_n};
        copy_vb = TiledCopyV{V_2D_n};
        copy_vs = TiledCopyVS{VS_2D_n};

        auto thr_copy_kb_n = copy_kb.get_slice(thr_id);
        auto thr_copy_vb_n = copy_vb.get_slice(thr_id);

        prefetch_kb = make_block_2d_prefetch(copy_kb);
        prefetch_vb =
            make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_n);

        Tensor cK_n = make_identity_tensor(K_2D_n.shape());   // (k,kb)
        Tensor cKS_n = make_identity_tensor(KS_2D_n.shape()); // (k,kg)
        Tensor cV_n = make_identity_tensor(V_2D_n.shape());   // (vb,k)
        Tensor cVS_n = make_identity_tensor(VS_2D_n.shape()); // (vg,k)

        Tensor gK_n = local_tile(
            cK_n,
            make_shape(Int<get<1>(TileShapeQK{})>{}, DTileBytes{}),
            make_coord(_, _, _),
            Step<X, _1, _1>{});
        Tensor gKS_n = local_tile(
            cKS_n,
            make_shape(Int<get<1>(TileShapeQK{})>{}, STileGroups{}),
            make_coord(_, _, _),
            Step<X, _1, _1>{});
        Tensor gV_n = local_tile(
            cV_n, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (vb,k,K)
        Tensor gV_split_n = local_tile(
            gV_n,
            TileShapePV{},
            make_coord(_, _, 0),
            Step<X, _1, _1>{});

        tKBgK = thr_copy_kb_n.partition_S(gK_n);
        tVbgV = thr_copy_vb_n.partition_S(gV_split_n);

        pKBgK = prefetch_kb.get_slice(thr_id).partition_S(gK_n);
        pVbgV = prefetch_vb.get_slice(thr_id).partition_S(gV_n);
      }
    };

    if constexpr (has_large_surface) {
      update_surface_info(tile_idx);
    }

    for (int D = 0; D < size<4>(pKBgK); D++) {
      prefetch(prefetch_kb, pKBgK(_, _, _, tile_idx, D));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    /* Softmax scale: MX has no per-tensor fold (per-group scales are
    applied element-wise during dequant) — the sm_scale is the whole
    effective scale. */
    ElementS effective_scale = params.scale;

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      auto tKBgK_cache =
          PagedKV ? tKBgK(_, _, _, tile_idx, _) : tKBgK(_, _, _, K, _);
      auto tKSgKS_cache =
          PagedKV ? tKSgKS(_, _, _, tile_idx, _) : tKSgKS(_, _, _, K, _);
      auto tVbgV_cache =
          PagedKV ? tVbgV(_, _, _, _, tile_idx) : tVbgV(_, _, _, _, K);

      /* GEMM 1: S = K * Q */
      clear(tSrS);
      for (int D = 0; D < size<4>(tKBgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_kb, tKBgK_cache(_, _, _, D), tKrU8);
        copy(copy_ks, tKSgKS_cache(_, _, _, D), tKsU8);
        reorder(tQrQ, tSrQ);

        // MX dequant: uint8 payload + e8m0 codes -> bf16 in the copy
        // layout, then the stock reorder performs the layout change.
        mx::dequant_dtile<
            Mx::k_is_e4m3,
            kDTileElems,
            get<1>(TileShapeQK{})>(
            tKrU8,
            thr_copy_kb.partition_S(
                identity_dtile_partition<D, get<1>(TileShapeQK{})>()),
            tKsU8,
            thr_copy_ks.partition_S(
                identity_stile_partition<D, get<1>(TileShapeQK{})>()),
            tKrDeq,
            thr_copy_kb_deq.partition_S(
                identity_deq_partition<D, get<1>(TileShapeQK{})>()));
        reorder(tKrDeq, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_vb, pVbgV(_, _, _, tile_idx));

      /* Local/sliding window masking (stock) */
      if constexpr (LocalMask) {
        int decode_row = seq_len - 1 - full_tile_offset;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(
            cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask = col_idx < decode_row - params.window_size_left;
          bool right_mask = col_idx > decode_row + params.window_size_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for remainder tiles (stock) */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKBgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* Apply softmax and scaling */
      softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_vb, tVbgV_cache(_, _, _, VV), tVrU8);
        copy(copy_vs, tVSgVS_paged(tile_idx), tVsU8);
        mx::dequant_dtile<
            Mx::v_is_e4m3,
            get<1>(TileShapePV{}),
            get<2>(TileShapePV{})>(
            tVrU8,
            thr_copy_vb.partition_S(identity_vbytes_partition<VV>()),
            tVsU8,
            thr_copy_vs.partition_S(identity_vscale_partition<VV>()),
            tVrDeq,
            thr_copy_vb_deq.partition_S(identity_vdeq_partition<VV>()));
        reorder(tVrDeq, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      barrier();

      // next tile_idx (stock)
      int next_tile_idx = K + 1;
      if constexpr (PagedKV) {
        int next_page_local_idx =
            next_tile_idx * get<1>(TileShapeQK{}) / params.page_size;
        if (next_page_local_idx < params.max_pages_per_seq) {
          int next_block_idx =
              params.ptr_page_table[b_offset + next_page_local_idx];
          next_tile_idx = next_block_idx * page_stride_tiles +
                          next_tile_idx % tiles_per_page;
        } else {
          next_tile_idx = params.max_pages_per_seq * tiles_per_page - 1;
        }
      }
      tile_idx = next_tile_idx;

      if constexpr (has_large_surface) {
        update_surface_info(tile_idx);
      }

      /* K prefetch */
      for (int D = 0; D < size<4>(pKBgK); D++) {
        prefetch(prefetch_kb, pKBgK(_, _, _, tile_idx, D));
      }
    }
  }

  // Single step of blocked softmax (stock, verbatim).
  CUTLASS_DEVICE
  void
  softmax(
      ElementS scale,    // Effective softmax scale
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum,  // Softmax row-wise sum accumulator
      FragA& tA) {       // O accumulator (for rescaling)

    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), scale * tS_bmax(i));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(scale * tS(i) - broadcast<0>(tS_max, tS, i));

    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

}  // namespace cutlass::fmha::collective
