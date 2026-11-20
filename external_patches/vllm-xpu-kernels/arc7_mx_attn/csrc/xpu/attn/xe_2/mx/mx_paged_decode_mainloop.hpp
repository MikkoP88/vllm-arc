// MX-native decode mainloop (arc7, P6.47 Phase 8) — the stock
// DecodeFwdMainloop (collective/chunk_prefill_mainloop.hpp) with the
// K/V load path replaced for MX KV caches. Everything downstream of
// the MMA fragments is byte-identical in structure and semantics:
// paged tile indirection, 2^24-row surface split, register-resident
// online softmax, split-KV accumulation. The stock fp8 path casts
// fp8->bf16 inside reorder(); the MX path dequants uint8 payload +
// e8m0 group codes to bf16 in a copy-fragment layout and then rides
// the SAME stock reorder() for the layout change.
//
// r199b LOAD PATH (after the r199a V1 const-fill ablation attributed
// ~90-97% of device time at 32K ctx to the dequant's scattered
// per-element scalar reads): e4m3 lanes load payload tiles with the
// REAL vectorized 2D-block byte copies (the same descriptors that
// previously only prefetched) and dequant register-to-register into
// ElementQ mirrors sharing the tv layout (mx::dequant_reg); e2m1
// lanes keep the coordinate walk but fetch bytes through staged u32
// window loads (mx::dequant_dtile). Scale codes are still read from
// the scale surfaces by COORDINATE (L1-cached, 4-entry register
// hoist) — no scale copies, no scale fragments, no cross-lane traffic
// (an earlier subgroup-broadcast design would have spent
// ~tile_k*groups shuffles per d-tile, ~8x the MMA itself).
//
// r200a DENSE V LANE (after r199b's cross-format triangulation showed
// the V lane dominant in BOTH forms — V-scalar ~5.2 ms and V-copy
// ~4.4 ms of ~5.5 ms @32K — and form (ii) (8 rows x stride-4
// bytes/WI) defeating the 2D-block copy as badly as the scalar walk):
// V is treated EXACTLY like K through the transposed vb_surf view
// (k, v_bytes) — bytes contiguous, rows strided by slot pitch, the K
// surface's structure. The V tile rides a form-(i) bwidth byte copy
// (copy_vdb -> tVrDB) in that frame. The attempt to finish the lane
// IN REGISTERS (dequant in the K orientation + a 4-arg reorder with a
// basis-swapped slayout bridging into tArV) FAILED: 33/33 faces on
// all formats (full scramble — the tv-codomain flat-basis bridge
// derivation was wrong) and classify_xe_reorder demoted the crossed
// permutation to Generic (scalar shuffles, +0.36 ms on the e2m1
// lanes). r199b's e2m1 register path for V never existed and the
// (vb, k)-frame V copies are gone entirely.
//
// r200b SLM-STAGED V LANE (replaces the failed register bridge): the
// dense form-(i) copy loads the k-tile's V bytes (all VV steps), each
// WI scatter-writes its bytes to SLM at tv-decoded LOCAL (k, vb)
// coordinates, one barrier() orders write->read (the loop-end barrier
// orders use->write across tiles), and mx::dequant_vslm walks tArV's
// form-(ii) slots reading SLM + scales and writing tArV directly. NO
// reorder, NO cross-WI register traffic, NO dtype branch (VIsE4m3 is
// a template flag: nibble select vs whole byte). The layout change
// happens at the scatter's coordinate decode — exact by construction,
// the same decoder that passed r199b's bitwise faces. Anchor law
// (dequant_reg): identity slot 0 minus its own tv position gives the
// dense-frame (k-row, v-byte) tile base. SGPerWG=1: SharedStorage is
// exclusive to the one subgroup; v_stage is 16 rows x VTiles*
// kVBytesPerStep bytes (2 KiB e4m3 / 1 KiB e2m1).
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

// ---------------------------------------------------------------------------
// Custom-extent B-copies for the u8 byte surfaces. The stock
// make_block_2d_copy_B (cute/atom/copy_traits_xe_2d.hpp) bakes the
// MMA's (N,K) ELEMENT tile into the copy's coordinate space and
// sv-layout; the payload surfaces tile the same rows at a different
// extent (K: d-tile bytes per row; V: v-tile bytes per column). These
// builders repeat the stock construction verbatim with one tile extent
// replaced, so the load keeps the MMA's B-operand value distribution
// over the byte columns. For e4m3 payloads (1 byte/element) the
// replaced extent equals the stock value and the copy degenerates to
// the stock derivation exactly.
// ---------------------------------------------------------------------------
template <
    class ValType,
    int Width,
    class TiledMMA,
    class CopyOp,
    class... Strides,
    class XMode,
    class YMode>
CUTE_HOST_DEVICE auto
make_block_2d_copy_bwidth(
    CopyOp const& op,
    TiledMMA const& mma,
    Stride<Strides...> const& gstride,
    XMode const& x_mode,
    YMode const& y_mode) {
  // (N, Width) coordinate space: the K-extent of the B tile in BYTES.
  // (mma.tile_mnk() is a member call — its static-typed values pass
  // through un-wrapped, exactly like the stock select<1,2>(...) pair.)
  auto tile_nk = make_shape(
      get<0>(select<1, 2>(mma.tile_mnk())), Int<Width>{});

  auto thr_vmnk = mma.get_thr_layout_vmnk();
  auto shape_vmnk = shape(thr_vmnk);
  auto drop_m = make_layout(
      shape_vmnk,
      make_stride(
          _1{},
          _0{},
          get<0>(shape_vmnk),
          get<0>(shape_vmnk) * get<2>(shape_vmnk)));
  auto thr_to_vnk = composition(drop_m, right_inverse(thr_vmnk));
  auto sg_to_vnk = composition(
      thr_to_vnk,
      make_layout(product(take<1, 4>(shape_vmnk)), get<0>(shape_vmnk)));

  auto svB = composition(
      mma.thrfrg_B(make_layout(tile_nk)), make_tile(sg_to_vnk, _));

  return make_block_2d_copy_X<ValType>(
      op, gstride, x_mode, y_mode, tile_nk, svB);
}

template <class ValType, int Width, class TiledMMA, class... Strides>
CUTE_HOST_DEVICE auto
make_block_2d_copy_bwidth(
    TiledMMA const& mma, Stride<Strides...> const& gstride) {
  auto cB = make_identity_tensor(make_shape(
      get<0>(select<1, 2>(mma.tile_mnk())), Int<Width>{}));
  using MMAType = typename TiledMMA::ValTypeB;
  auto op = block_2d_selector<ValType, MMAType>(
      mma.get_slice(0).atom_partition_B(cB).layout(), gstride);
  return make_block_2d_copy_bwidth<ValType, Width>(
      op, mma, gstride, find_x_mode(gstride), find_y_mode(gstride));
}

template <
    class ValType,
    int Height,
    class TiledMMA,
    class CopyOp,
    class... Strides,
    class XMode,
    class YMode>
CUTE_HOST_DEVICE auto
make_block_2d_copy_bheight(
    CopyOp const& op,
    TiledMMA const& mma,
    Stride<Strides...> const& gstride,
    XMode const& x_mode,
    YMode const& y_mode) {
  // (Height, K) coordinate space: the N-extent of the B tile in BYTES.
  // (mma.tile_mnk() is a member call — its static-typed values pass
  // through un-wrapped, exactly like the stock select<1,2>(...) pair.)
  auto tile_nk = make_shape(
      Int<Height>{}, get<1>(select<1, 2>(mma.tile_mnk())));

  auto thr_vmnk = mma.get_thr_layout_vmnk();
  auto shape_vmnk = shape(thr_vmnk);
  auto drop_m = make_layout(
      shape_vmnk,
      make_stride(
          _1{},
          _0{},
          get<0>(shape_vmnk),
          get<0>(shape_vmnk) * get<2>(shape_vmnk)));
  auto thr_to_vnk = composition(drop_m, right_inverse(thr_vmnk));
  auto sg_to_vnk = composition(
      thr_to_vnk,
      make_layout(product(take<1, 4>(shape_vmnk)), get<0>(shape_vmnk)));

  auto svB = composition(
      mma.thrfrg_B(make_layout(tile_nk)), make_tile(sg_to_vnk, _));

  return make_block_2d_copy_X<ValType>(
      op, gstride, x_mode, y_mode, tile_nk, svB);
}

template <class ValType, int Height, class TiledMMA, class... Strides>
CUTE_HOST_DEVICE auto
make_block_2d_copy_bheight(
    TiledMMA const& mma, Stride<Strides...> const& gstride) {
  auto cB = make_identity_tensor(make_shape(
      Int<Height>{}, get<1>(select<1, 2>(mma.tile_mnk()))));
  using MMAType = typename TiledMMA::ValTypeB;
  auto op = block_2d_selector<ValType, MMAType>(
      mma.get_slice(0).atom_partition_B(cB).layout(), gstride);
  return make_block_2d_copy_bheight<ValType, Height>(
      op, mma, gstride, find_x_mode(gstride), find_y_mode(gstride));
}

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
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

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
  static_assert(kGroupsPerDTile >= 1, "d-tile must span at least one MX group");

  // v-extent (elements) of one VV step of the PV MMA; its byte width.
  static constexpr int kVStepElems = get<1>(TileShapePV{});
  static constexpr int kVBytesPerStep = kVStepElems / kElemsPerByteV;
  // r200b SLM V staging: one k-tile of V payload bytes in the dense
  // (k, byte) frame — full v-byte width as the row pitch.
  static constexpr int kVStagePitch = VTiles * kVBytesPerStep;

  // Tiled copies. Q is the stock A-copy. The K payload copy is a
  // B-copy over the u8 byte surface (byte-extent builder above; for
  // e4m3 the extents equal the stock element tiles and the stock
  // derivation applies verbatim). The dequant-destination copy
  // (copy_kd) is a full stock-width B-copy typed at the dequant
  // output: it never issues loads (mx::dequant_dtile fills its
  // fragments by coordinate) — it exists so the destination
  // fragments and their matching coordinate views carry the stock
  // MMA-facing layout that reorder() converts into the MMA operands.
  using TiledCopyQ =
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{}));
  using TiledCopyK = decltype(make_block_2d_copy_bwidth<uint8_t, kBytesPerDTileK>(
      TiledMMAQK{}, decltype(stride(TensorK2D{})){}));
  using TiledCopyKD = decltype(make_block_2d_copy_B<ElementQ>(
      TiledMMAQK{}, decltype(stride(TensorK2D{})){}));
  // r200a dense V copies: the (vb, k) payload surface re-viewed
  // row-major (k, v_bytes) has the K surface's exact structure (rows
  // strided by slot pitch, bytes contiguous), so the bwidth builder
  // over TiledMMAQK hands the V tile the SAME form-(i) per-WI
  // distribution the K lane rides — 2 rows x contiguous-phase bytes —
  // instead of the PV B-operand's form (ii) (8 rows x stride-4 bytes)
  // that defeats both the 2D-block copy and window compression.
  //
  // r200d PITCH LAW (supersedes the retracted r200c width diagnosis):
  // the vendored 2D-block copy op encodes the row pitch in 16-byte
  // units, so the (k, byte) payload surface's row pitch — the cache's
  // seq stride, i.e. num_kv_heads * slot_bytes — must be a multiple of
  // 16. Odd head counts (1 x 264, 1 x 200: pitch 8 mod 16) round up to
  // +8 bytes/row: row 0 lands byte-exact while row k reads
  // cache[v_off + (slot+8)*k] (r200d correlator, exact window match on
  // both payload dtypes). The r200c "width 16 loads row 0 only" reading
  // was this same drift at HK=1 — width-independent, since the r200c
  // and r200d A-images were byte-identical. Production qgroup geometry
  // (even head counts) and every stock dtype slot (256/128 bytes) are
  // always aligned; the host entry refuses misaligned pitches outright.
  // Width 32 is the shipped copy geometry regardless: e4m3 stages one
  // VV step per copy (kVBytesPerStep verbatim), e2m1 stages VV pairs
  // (2 x 16) — proven byte-exact at HK=2 (r200f synth-v2 A-stage 0/0
  // and 42/42 faces on the r200d wheel).
  static constexpr int kVCopyWidth =
      Mx::v_is_e4m3 ? kVBytesPerStep : 2 * kVBytesPerStep;
  static constexpr int kVCopySteps =
      Mx::v_is_e4m3 ? VTiles : VTiles / 2;
  static_assert(
      kVCopyWidth >= 32,
      "bwidth V byte copies need width >= 32 (2D-block copy row window "
      "law: width 16 does not cover a staged VV pair)");
  static_assert(
      kVCopyWidth * kVCopySteps == VTiles * kVBytesPerStep,
      "V copy steps must cover the tile's full staged width");
  using TensorVDnsStride = decltype(make_stride(
      get<1>(stride(TensorV2D{})),
      get<0>(stride(TensorV2D{}))));  // (k_row_stride, byte_stride)
  using TiledCopyVDB = decltype(make_block_2d_copy_bwidth<
      uint8_t,
      kVCopyWidth>(TiledMMAQK{}, TensorVDnsStride{}));

  //
  // Accumulator types (identical to stock)
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

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
    // r200c V-lane probe (ARC_MXKV_VPROBE via the config's run();
    // ablation telemetry only — never set on serving paths):
    //   0 full lane | 1 skip staging (gather reads stale SLM) |
    //   2 skip gather (tArV const-fill) | 3 skip both (floor) |
    //   4 staging without the barrier | 5 gather without scale reads.
    int vlane_probe = 0;
    // r200c debug lane (args.debug_buf layout documented at the args
    // struct; SINGLE-workgroup shapes only — WGs race on the buffer).
    void* debug_buf = nullptr;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {
    // r200b: the k-tile's V payload bytes staged dense (k-major, full
    // v-byte row width) — written by the form-(i) copy + tv-decoded
    // scatter, read by the form-(ii) dequant straight into tArV. One
    // subgroup per workgroup (SGPerWG=1), so no per-SG offsets. Full
    // overwrite each k-tile; the loop's barriers order write->read
    // within a tile and use->write across tiles.
    alignas(16) uint8_t v_stage[get<1>(TileShapeQK{}) * kVStagePitch];
  };

  Params params;
  SharedStorage& smem;

  //
  // Methods
  //
  MxDecodeFwdMainloop(Params const& params_, SharedStorage& smem_)
      : params(params_), smem(smem_) {}

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
        args.page_stride_elements,
        args.vlane_probe,
        args.debug_buf};
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
    // byte-height V tile (payload bytes per v extent) for the u8 surface
    auto tile_shape_vb = make_shape(
        Int<(get<1>(TileShapePV{}) * VTiles) / kElemsPerByteV>{},
        get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors. cK/cV span the u8 byte
    surfaces; cKD/cVD span the ELEMENT space the dequant destinations
    and MMA operands live in (the role of stock's cK/cV for a bf16
    surface — the byte/element extents differ exactly by the packing
    factor). */
    Tensor cQ = make_identity_tensor(Q_2D.shape());                // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());                // (k,kb)
    Tensor cKD = make_identity_tensor(make_shape(
        get<0>(K_2D.shape()),
        get<1>(K_2D.shape()) * Int<kElemsPerByteK>{}));            // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());                // (vb,k)
    Tensor cVD = make_identity_tensor(make_shape(
        get<0>(V_2D.shape()) * Int<kElemsPerByteV>{},
        get<1>(V_2D.shape())));                                    // (v,k)
    // r200a dense V frame identity (vb_surf's frame): payload
    // re-viewed (k, v_bytes), byte-granular (e4m3 copy coordinates).
    [[maybe_unused]] Tensor cVdns = make_identity_tensor(make_shape(
        get<1>(V_2D.shape()),
        get<0>(V_2D.shape())));                                     // (k,vb)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));   // (q,k)

    // Dequant-facing byte + scale surfaces for the CURRENT
    // (sub-)surface: payload bytes as (row, byte-col) — K bytes are
    // already (k, k_bytes), V bytes re-viewed to (k, v_bytes) — and
    // scales as (k, groups) either way. dequant's contract is
    // (dst-row, byte/scale-col) with coordinates read directly from
    // gmem (L1-resident from the prefetches; first-pass byte-fragment
    // lookup removed — the custom-extent and stock copies distribute
    // tiles on incompatible work-item grids). The 2^24 surface split
    // below/lambda updates these alongside the payload surfaces.
    [[maybe_unused]] auto kb_surf = K_2D;
    [[maybe_unused]] auto vb_surf = make_tensor(
        V_2D.data(),
        make_layout(
            make_shape(get<1>(V_2D.shape()), get<0>(V_2D.shape())),
            make_stride(get<1>(V_2D.stride()), get<0>(V_2D.stride()))));
    [[maybe_unused]] auto ks_surf = KS_2D;
    [[maybe_unused]] auto vs_surf = make_tensor(
        VS_2D.data(),
        make_layout(
            make_shape(get<1>(VS_2D.shape()), get<0>(VS_2D.shape())),
            make_stride(get<1>(VS_2D.stride()), get<0>(VS_2D.stride()))));

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
      cKD = make_identity_tensor(make_shape(
          PagedKV ? rows_per_surface : k_rows_total,
          get<1>(K_2D.shape()) * Int<kElemsPerByteK>{}));  // (k,d)
      cV = make_identity_tensor(make_shape(
          get<0>(V_2D.shape()),
          PagedKV ? rows_per_surface : v_cols_total));  // (vb,k)
      cVD = make_identity_tensor(make_shape(
          get<0>(V_2D.shape()) * Int<kElemsPerByteV>{},
          PagedKV ? rows_per_surface : v_cols_total));  // (v,k)
      cVdns = make_identity_tensor(make_shape(
          PagedKV ? rows_per_surface : v_cols_total,
          get<0>(V_2D.shape())));  // (k,vb)

      kb_surf = K_2D_0;
      vb_surf = make_tensor(
          V_2D_0.data(),
          make_layout(
              make_shape(get<1>(V_2D_0.shape()), get<0>(V_2D_0.shape())),
              make_stride(
                  get<1>(V_2D_0.stride()), get<0>(V_2D_0.stride()))));
      ks_surf = KS_2D_0;
      vs_surf = make_tensor(
          VS_2D_0.data(),
          make_layout(
              make_shape(get<1>(VS_2D_0.shape()), get<0>(VS_2D_0.shape())),
              make_stride(
                  get<1>(VS_2D_0.stride()), get<0>(VS_2D_0.stride()))));
    }

    /* Partition global tensors into workgroup tiles. Byte tiles cover
    the u8 payload surfaces; the element tiles (gKD/gVD) cover the
    dequant-destination + MMA-operand coordinate space (stock's gK/gV
    role). DB (byte tiles per row) == DD (element tiles per row) for
    every format: total_bytes/bytes_per_dtile == D/64. */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK,
        make_shape(
            get<0>(TileShapeQK{}),
            Int<tile_k>{},
            Int<kBytesPerDTileK>{}),
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,kb,K,DB)
    Tensor gKD = local_tile(
        cKD,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,K,DD)
    Tensor gV =
        local_tile(cV, tile_shape_vb, make_coord(get<1>(blk_qv), _));  // (vb,k,K)
    Tensor gVD =
        local_tile(cVD, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gVD_split = local_tile(
        gVD,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,K)
    // r200a dense V tile view in the (k, v_bytes) frame — the same
    // tiling law as gK (mode 0 kept, modes 1-2 tiled), so the bwidth
    // copy distributes it form-(i) like the K tiles it mirrors.
    // r200c: byte-tile width = kVCopyWidth (VV pairs on e2m1 — the
    // width-16 A-stage verdict above); mode 3 walks copy steps.
    [[maybe_unused]] Tensor gVdns = local_tile(
        cVdns,
        make_shape(
            get<0>(TileShapeQK{}),
            Int<tile_k>{},
            Int<kVCopyWidth>{}),
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,vb,K,VP)

    /* Create global -> register copies. copy_kd is built
    over an element-typed pointer proxy (reinterpret cast: static_cast
    cannot cross unsigned char* -> ElementQ*; the ctor consumes only
    value_type + the stride pattern — shape unused by
    make_block_2d_copy_B(mma, gmem) — and the copy never issues
    loads, dequant fills its fragments by coordinate). r200a: copy_vdb
    is the REAL V byte load (form-(i) dense-frame bwidth distribution);
    r200b stages its fragment into SLM for dequant_vslm. */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_kb{K_2D_0};
    TiledCopyKD copy_kd{make_tensor(
        make_gmem_ptr(reinterpret_cast<ElementQ*>(raw_pointer_cast(K_2D_0.data()))),
        K_2D_0.layout())};
    TiledCopyVDB copy_vdb{vb_surf};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_kb = copy_kb.get_slice(thr_id);
    auto thr_copy_kd = copy_kd.get_slice(thr_id);
    auto thr_copy_vdb = copy_vdb.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy. On e4m3 lanes the byte
    tiles are the REAL load path (r199b): tKBgK hands copy() the same
    vectorized 2D-block descriptors the prefetches warm — r200a gives
    the V lane the same via the dense (k, v_bytes) frame. On e2m1
    lanes dequant reads payload bytes through staged u32 windows from
    the gmem surfaces (kb/vb/ks/vs_surf above) and only the prefetch
    partitions matter. */
    auto tQgQ = thr_copy_q.partition_S(gQ);          // (atom_val,q',d',D)
    auto tKBgK = thr_copy_kb.partition_S(gK);        // (atom_val,k',kb',K,DB)
    auto tKDgKD = thr_copy_kd.partition_S(gKD);      // (atom_val,k',d',K,DD)
    auto tVDBgV = thr_copy_vdb.partition_S(gVdns);   // (atom_val,k',vb',K,VV)

    /* Create register fragments for MMA and copies. tKrDeq comes
    from the ELEMENT-dim tiles, so its size is the MMA-facing element
    count regardless of payload format (e2m1's 2x expansion is
    absorbed by the byte->element coordinate mapping in dequant). */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    [[maybe_unused]] auto tKrDeq =
        thr_copy_kd.partition_sg_fragment_D(gKD(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gKD(_, _, 0, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    // Split-P residual operands: the pair/triple decomposition of the
    // softmax probabilities (see the softmax/reorder site) -- tArP2
    // carries bf16(p - bf16(p)), tArP3 the second residual.
    auto tArP2 = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tArP3 = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gVD_split(_, _, 0, 0));

    // r199b e4m3 K load-path fragments: tKrB receives the REAL byte
    // copy; tKrBq is an ElementQ mirror on the SAME tv layout
    // (make_subgroup_tensor preserves it — src and dst registers map
    // 1:1), so mx::dequant_reg fills it register-to-register and the
    // stock reorder() performs only the layout change. e2m1 lanes leave
    // them unused (the staged dequant keeps the tKrDeq walk).
    [[maybe_unused]] auto tKrB =
        thr_copy_kb.partition_sg_fragment_D(gK(_, _, 0, 0));
    [[maybe_unused]] auto tKrBq = make_subgroup_tensor(
        make_tensor<ElementQ>(tKrB.layout()), tKrB.tv_layout());

    // r200a/r200b dense V fragment: tVrDB receives the dense byte copy
    // (all formats — e2m1 included, the dense frame makes the copy
    // format-blind); r200b scatter-stages its bytes into SLM at
    // tv-decoded (k, vb) local coordinates and dequant_vslm walks
    // tArV's form-(ii) slots straight out of SLM. No register
    // mirror, no reorder, no dtype branch.
    [[maybe_unused]] auto tVrDB =
        thr_copy_vdb.partition_sg_fragment_D(gVdns(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_kb = make_block_2d_prefetch(copy_kb);
    auto prefetch_vb =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_vb, V_2D_0);

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
        auto vb_surf_n = make_tensor(
            V_2D_n.data(),
            make_layout(
                make_shape(get<1>(V_2D_n.shape()), get<0>(V_2D_n.shape())),
                make_stride(
                    get<1>(V_2D_n.stride()), get<0>(V_2D_n.stride()))));
        copy_vdb = TiledCopyVDB{vb_surf_n};

        prefetch_kb = make_block_2d_prefetch(copy_kb);
        prefetch_vb =
            make_block_2d_prefetch<SGPerWG::value>(tile_shape_vb, V_2D_n);

        Tensor cK_n = make_identity_tensor(K_2D_n.shape());  // (k,kb)
        Tensor cKD_n = make_identity_tensor(make_shape(
            get<0>(K_2D_n.shape()),
            get<1>(K_2D_n.shape()) * Int<kElemsPerByteK>{}));  // (k,d)
        Tensor cV_n = make_identity_tensor(V_2D_n.shape());  // (vb,k)
        Tensor cVdns_n = make_identity_tensor(make_shape(
            get<1>(V_2D_n.shape()), get<0>(V_2D_n.shape())));  // (k,vb)

        Tensor gK_n = local_tile(
            cK_n,
            make_shape(
                get<0>(TileShapeQK{}),
                Int<tile_k>{},
                Int<kBytesPerDTileK>{}),
            make_coord(_, _, _),
            Step<X, _1, _1>{});  // (k,kb,K,DB)
        Tensor gKD_n = local_tile(
            cKD_n,
            TileShapeQK{},
            make_coord(_, _, _),
            Step<X, _1, _1>{});  // (k,d,K,DD)
        Tensor gV_n = local_tile(
            cV_n, tile_shape_vb, make_coord(get<1>(blk_qv), _));  // (vb,k,K)
        Tensor gVdns_n = local_tile(
            cVdns_n,
            make_shape(
                get<0>(TileShapeQK{}),
                Int<tile_k>{},
                Int<kVCopyWidth>{}),
            make_coord(_, _, _),
            Step<X, _1, _1>{});  // (k,vb,K,VP)

        // copy_kd is stride-shaped and never issues loads, so its
        // sub-surface state stays valid — only its coordinate
        // tensors need the refresh.
        tKDgKD = thr_copy_kd.partition_S(gKD_n);

        pKBgK = prefetch_kb.get_slice(thr_id).partition_S(gK_n);
        pVbgV = prefetch_vb.get_slice(thr_id).partition_S(gV_n);

        // r199b/r200a: the byte-LOAD partitions follow the rebuilt
        // copies (same state-refresh law as the prefetch partitions).
        tKBgK = copy_kb.get_slice(thr_id).partition_S(gK_n);
        tVDBgV = copy_vdb.get_slice(thr_id).partition_S(gVdns_n);

        kb_surf = K_2D_n;
        vb_surf = vb_surf_n;
        ks_surf = KS_2D_n;
        vs_surf = make_tensor(
            VS_2D_n.data(),
            make_layout(
                make_shape(get<1>(VS_2D_n.shape()), get<0>(VS_2D_n.shape())),
                make_stride(
                    get<1>(VS_2D_n.stride()), get<0>(VS_2D_n.stride()))));
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
      [[maybe_unused]] auto tKDgKD_cache =
          PagedKV ? tKDgKD(_, _, _, tile_idx, _) : tKDgKD(_, _, _, K, _);
      [[maybe_unused]] auto tKBgK_cache =
          PagedKV ? tKBgK(_, _, _, tile_idx, _) : tKBgK(_, _, _, K, _);
      // r200a/r200b dense V partition caches the K mode (tile index),
      // VV stays live for the stage + split-P loops.
      [[maybe_unused]] auto tVDBgV_cache =
          PagedKV ? tVDBgV(_, _, _, tile_idx, _) : tVDBgV(_, _, _, K, _);

      /* GEMM 1: S = K * Q */
      clear(tSrS);
      for (int D = 0; D < size<4>(tKDgKD); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        reorder(tQrQ, tSrQ);

        if constexpr (Mx::k_is_e4m3) {
          // r199b e4m3 load path: the REAL vectorized 2D-block byte
          // copy fills tKrB, dequant runs register-to-register into
          // its ElementQ mirror (group scales are the only gmem
          // reads), and the stock reorder performs the layout change.
          copy(copy_kb, tKBgK_cache(_, _, _, D), tKrB);
          mx::dequant_reg<true, true>(
              ks_surf, tKrB, tKrBq, tKBgK_cache(_, _, _, D));
          reorder(tKrBq, tSrK);
        } else {
          // MX dequant (e2m1): staged u32 byte windows -> bf16 in the
          // dequant copy's fragment layout, then the stock reorder
          // performs only the layout change into the MMA operand.
          // Positions come from the fragment's tv_layout (per-WI;
          // partition_S identity views are WI-canonical in this tree)
          // anchored to the identity slot-0 tuple — see mx_dequant.hpp
          // for the coordinate + staging laws. The QK B-tile frame is
          // (k, d): tv codomain mode 0 IS the row.
          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tKrDeq,
              tKDgKD_cache(_, _, _, D));
          reorder(tKrDeq, tSrK);
        }

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_vb, pVbgV(_, _, _, tile_idx));

      /* r200b: stage the k-tile's V payload into SLM (dense (k, v_bytes)
      frame, ALL VV steps at once). The form-(i) copy fills tVrDB with
      this WI's bytes; each WI scatter-writes them at tv-decoded LOCAL
      (k, vb) coordinates — the layout change happens in the coordinate
      decode, exact by construction (same decoder that passed r199b
      bitwise). One subgroup per workgroup (SGPerWG=1): no per-SG
      offsets. The barrier orders write->read within the tile; the
      loop-end barrier orders use->write across tiles. Stale bytes on
      remainder k-tiles are finite (previous tile's payload) and land
      under p=0 rows.
      r200c: copies run at kVCopyWidth (VV PAIRS on e2m1 — the width-16
      law above); a copy step VP's bytes scatter to VP*kVCopyWidth + vb,
      which equals the per-VV layout (VP*2+s)*kVBytesPerStep + vb.
      Probe gates: 1/3 skip the staging entirely (timing runs — the
      gather then reads stale SLM), 4 stages WITHOUT the barrier
      (isolates the barrier's serialization cost). */
      uint8_t* vst = reinterpret_cast<uint8_t*>(&smem.v_stage);
      bool stage_on =
          params.vlane_probe != 1 && params.vlane_probe != 3;
      if (stage_on) {
        CUTLASS_PRAGMA_UNROLL
        for (int VP = 0; VP < kVCopySteps; VP++) {
          copy(copy_vdb, tVDBgV_cache(_, _, _, VP), tVrDB);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tVrDB.size(); ++i) {
            auto pc = mx::detail::tv_coord(tVrDB.tv_layout(), thr_id, i);
            vst[int(cute::get<0>(pc)) * kVStagePitch +
                VP * kVCopyWidth + int(cute::get<1>(pc))] =
                uint8_t(tVrDB(i));
          }
        }
        if (params.vlane_probe != 4) {
          barrier();
        }
      }

      // Tile bases, the dequant_reg anchor law: identity slot 0
      // (WI-canonical) minus its own tv position, both in the dense
      // (k, vb) frame. v_byte_base is in surface BYTES; the element
      // base for the scales' group index is derived at the call site.
      auto pv00 = mx::detail::tv_coord(tVrDB.tv_layout(), 0, 0);
      auto cv0 = tVDBgV_cache(_, _, _, 0)(0);
      int v_krow_base = int(cute::get<0>(cv0)) - int(cute::get<0>(pv00));
      int v_byte_base = int(cute::get<1>(cv0)) - int(cute::get<1>(pv00));

      // r200c debug lane (single-WG shapes only): A = the staged SLM
      // image (this k-tile, padded 128-byte row pitch), B = anchors,
      // C = per-VV gathered fragments (written in the VV loop).
      uint8_t* dbg = reinterpret_cast<uint8_t*>(params.debug_buf);
      int dbg_tile = K - blk_k0;
      bool dbg_on = dbg != nullptr && dbg_tile < 4;
      if (dbg_on) {
        if (thr_id == 0) {
          int64_t* dbgB = reinterpret_cast<int64_t*>(dbg + 8192);
          dbgB[2 * dbg_tile + 0] = v_krow_base;
          dbgB[2 * dbg_tile + 1] = v_byte_base;
        }
        uint8_t* dbgA = dbg + dbg_tile * 2048;
        for (int j = thr_id;
             j < get<1>(TileShapeQK{}) * kVStagePitch;
             j += 16) {
          dbgA[(j / kVStagePitch) * 128 + (j % kVStagePitch)] = vst[j];
        }
      }

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
        int k = get<0>(tKDgKD(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
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
      /* Split-P TRIPLE (deliberate deviation, r198 element-ledger +
       * mini-ledger law): a plain bf16 P operand leaves a 2^-9
       * relative error whose random walk flips the output's last bit
       * at rounding-midpoint magnitudes -- adjacent-grid-step
       * disagreements vs the certified v2 envelope (tf32 operands
       * only shrank the walk). A PAIR holds p to 2^-18 relative --
       * enough for every normal output -- but at deep-cancellation
       * elements (|O| ~ 2^-13 of the p*V term scale) that floor is
       * still 1-2% of |O| (mini-ledger: single elements at 2 and 38
       * grid steps while C == F32 exactly). The TRIPLE
       * bf16(p) + bf16(r1) + bf16(r2), r1 = p - bf16(p),
       * r2 = r1 - bf16(r1), holds p to 2^-27 relative -- below fp32's
       * own accumulation noise, i.e. v2-grade -- and every gemm rides
       * the full-rate bf16 atom (register operands; the PV gemms are
       * a small share of this memory-bound walk -- priced at r199). */
      FragS tSrS2 = tSrS;
      FragS tSrS3;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tSrS2(i) = tSrS(i) - static_cast<float>(bfloat16_t(tSrS(i)));
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tSrS3(i) = tSrS2(i) - static_cast<float>(bfloat16_t(tSrS2(i)));
      }
      reorder(tSrS, tArP);
      reorder(tSrS2, tArP2);
      reorder(tSrS3, tArP3);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        // r200b SLM V lane: dequant_vslm walks tArV's form-(ii) slots
        // straight out of the staged bytes (per-WI tv decode; group
        // scales through the 4-entry cache) — no register bridge, no
        // reorder, no dtype branch. The r200a cross-frame reorder
        // failed faces (33/33, full scramble — the tv-codomain
        // flat-basis bridge derivation was wrong) AND classified
        // Generic in reorder_atom_xe (scalar shuffles, +0.36 ms on the
        // e2m1 lanes); staging deletes both problems at once.
        if (params.vlane_probe == 2 || params.vlane_probe == 3) {
          // r200c probe 2/3: const-fill (timing only — severs the
          // gather chain; a structural floor, not a reachable target).
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArV.size(); ++i) {
            tArV(i) = typename decltype(tArV)::value_type(1.f);
          }
        } else {
          mx::dequant_vslm<Mx::v_is_e4m3>(
              vst + VV * kVBytesPerStep,
              kVStagePitch,
              vs_surf,
              tArV,
              v_krow_base,
              v_byte_base * kElemsPerByteV + VV * kVStepElems,
              params.vlane_probe == 5);
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
        cute::gemm(mma_pv, tArP2, tArV, tArA(_, _, _, VV));
        cute::gemm(mma_pv, tArP3, tArV, tArA(_, _, _, VV));
        if (dbg_on) {
          float* dbgC =
              reinterpret_cast<float*>(dbg + 8256) +
              ((dbg_tile * VTiles + VV) * 16 + thr_id) * 32;
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArV.size(); ++i) {
            dbgC[i] = float(tArV(i));
          }
        }
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

  // Single step of blocked softmax (stock structure; exp2 precision is
  // the ONE deliberate deviation: stock's sycl::native::exp2 carries
  // ~1e-3 relative error that showed as the beyond-cast tail in the
  // r198 softmax-probe (n(dp > 1.5*cast) = 13-17/120); the certified v2
  // envelope (d <= 2e-3 vs fp64 oracle, matched by v2 at 3e-7) needs
  // the precise sycl::exp2. Perf-priced at r199.)
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
      tS(i) = sycl::exp2(scale * tS(i) - broadcast<0>(tS_max, tS, i));

    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::exp2(tS_prev_max(i) - tS_max(i));
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
