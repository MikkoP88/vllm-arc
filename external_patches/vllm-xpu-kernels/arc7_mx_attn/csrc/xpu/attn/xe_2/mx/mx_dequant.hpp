// In-register MX dequant for the arc7 MX-native paged-decode kernel.
// Converts the packed payload + e8m0 codes up to ElementQ in the
// dequant-copy fragment layout, leaving the layout change to the stock
// subgroup reorder() — the same division of labor the stock fp8 path
// uses (cast in registers, reorder to the MMA fragment).
//
// COORDINATE LAW (learned across two r198 debug iterations):
//
// 1. Copy partition_S of an IDENTITY tensor returns WI-CANONICAL
//    coordinates in this tree: the SG-scoped copy atom exposes the
//    subgroup (not the work-item) as its thread mode, so every WI of
//    the subgroup reads the same slot-dv tuple. The vendor's own
//    remainder mask compensates by hand (`k = get<0>(tKgK(0,0,0,K,0))
//    + get_local_id()[0]`) — reading those tuples as per-WI positions
//    makes dequant correct only where the WI offset is zero (observed:
//    exactly the elements at d % 8 == 0, rest garbage).
//
// 2. The per-WI truth lives in the DESTINATION fragment's tv_layout:
//    partition_sg_fragment_D builds it over the tidfrg_D (thr, val)
//    mapping, and its codomain is the tile's 2D coordinate space
//    itself (ScaledBasis strides — instantiated evidence, K lane:
//    row = t1(wi) + 8*v0(dv) spanning 16, elem = t0(wi) + 2*v1(dv) +
//    16*v2(dv) spanning 64, injective over the tile). Evaluating the
//    tv at (this WI, register dv) yields the tile-local (row, element)
//    directly — no frame decode arithmetic.
//
// The tile origin in surface coordinates is recovered from the
// identity view's slot-0 tuple minus its own tv position — the two
// describe the same (WI-0, register-0) element, so any canonical
// slot-0 offset cancels and only the tile base remains.
//
// Payload bytes and scales are then read coordinate-driven direct
// from gmem (L1-resident from the copy prefetches; a 4-entry register
// cache hoists repeat scale reads out of the unrolled loop). Which
// codomain mode is the surface ROW differs per operand: the QK B-tile
// frame is (k, d) (mode 0 = row), the PV B-tile is (v, k) (mode 0 =
// element) — kFrame0IsRow selects. The IDENTITY tuple orientation
// also differs: K carries (k, d) (row = mode 0), V carries (v, k)
// (row = mode 1) — RowIsMode0 selects.
#pragma once

#include <cstdint>

#include <cutlass/float8.h>
#include <cutlass/numeric_types.h>
#include <cute/config.hpp>
#include <cute/tensor.hpp>
#include <sycl/sycl.hpp>

#include "xpu/attn/xe_2/mx/mx_types.hpp"

namespace cutlass::fmha::collective::mx {

namespace detail {

CUTE_HOST_DEVICE
inline float
e4m3_to_float(uint8_t b) {
  cutlass::float_e4m3_t e;
  e.storage = b;
  return static_cast<float>(e);
}

// Evaluate a subgroup tv_layout at flat (work-item, register) indices.
// partition_sg_fragment_D produces the rank-3 nesting (Thr, Val, Rest)
// (instantiated evidence, K lane: ((2,8), ((2,8),4), (1,1))): idx2crd
// expands the flat indices into the hierarchical coordinates the
// layout consumes; the Rest modes (extent 1 for our tile slices) park
// at zero. The return value is the tile-local coordinate pair.
template <class TVLayout>
CUTE_HOST_DEVICE auto
tv_coord(TVLayout const& tv, int wi, int dv) {
  static_assert(
      cute::rank_v<TVLayout> == 3,
      "unexpected tv_layout nesting (expected (Thr, Val, Rest))");
  auto thr_crd = cute::idx2crd(wi, cute::shape<0>(tv));
  auto val_crd = cute::idx2crd(dv, cute::shape<1>(tv));
  auto rest_crd = cute::idx2crd(0, cute::shape<2>(tv));
  return tv(cute::make_coord(thr_crd, val_crd, rest_crd));
}

}  // namespace detail

// dequant_dtile — one d-tile (K) or v-step (V) of one row-block.
//
//   bytes        : gmem u8 view (row, byte-col), FULL surface width;
//                 column = global element / kElemsPerByte
//   scales       : gmem u8 view (row, group), full width; indexed
//                  directly by coordinate
//   dst          : per-WI ElementQ fragment in the dequant copy's
//                  layout (the fragment reorder() converts into the
//                  MMA operand); carries the tv_layout used for the
//                  per-WI position decoding
//   dst_coords   : WI-canonical identity view of the same tile; only
//                  slot 0 is read (tile-origin anchor)
//   kFrame0IsRow : tv codomain mode 0 is the surface row (K lane) vs
//                  the element (V lane)
//
// Every loop runs over static fragment sizes and unrolls fully.
template <
    bool KIsE4M3,
    bool RowIsMode0,
    bool kFrame0IsRow,
    class ByteView,
    class ScaleTensor,
    class DstFrag,
    class CoordView>
CUTE_HOST_DEVICE
void
dequant_dtile(
    ByteView const& bytes,
    ScaleTensor const& scales,
    DstFrag& dst,
    CoordView const& dst_coords) {
  constexpr int kElemsPerByte = KIsE4M3 ? 1 : 2;
  // The mainloop resolves get_sub_group via its function-scope
  // using-directive (vendor pattern); this header sits outside it.
  using namespace sycl::ext::oneapi::this_work_item;
  auto wi = get_sub_group().get_local_id()[0];

  // Tile origin: identity slot 0 (WI-canonical) minus its own tv
  // position — both describe (WI 0, register 0), so the difference is
  // the tile base in surface coordinates.
  auto p00 = detail::tv_coord(dst.tv_layout(), 0, 0);
  auto c0 = dst_coords(0);
  int org_row =
      (RowIsMode0 ? int(cute::get<0>(c0)) : int(cute::get<1>(c0))) -
      (kFrame0IsRow ? int(cute::get<0>(p00)) : int(cute::get<1>(p00)));
  int org_elem =
      (RowIsMode0 ? int(cute::get<1>(c0)) : int(cute::get<0>(c0))) -
      (kFrame0IsRow ? int(cute::get<1>(p00)) : int(cute::get<0>(p00)));

  // Per-WI scale cache: the destination fragment of one tile spans few
  // (row, group) pairs; anything beyond 4 entries still reads correctly
  // through L1.
  int cr[4] = {-1, -1, -1, -1};
  int cg[4] = {-1, -1, -1, -1};
  float cv[4];

  // SubgroupTensor::size() is a non-static member in this tree (unlike
  // cute::Tensor's static size()) — call it on the object.
  CUTE_UNROLL
  for (int dv = 0; dv < dst.size(); ++dv) {
    // This work-item's register dv: tile-local (row, element) from the
    // fragment's own tv_layout, plus the tile base.
    auto pc = detail::tv_coord(dst.tv_layout(), wi, dv);
    int row = org_row +
        (kFrame0IsRow ? int(cute::get<0>(pc)) : int(cute::get<1>(pc)));
    int elem = org_elem +
        (kFrame0IsRow ? int(cute::get<1>(pc)) : int(cute::get<0>(pc)));
    int grp = elem / kMxGroupSize;

    float scale = 0.f;
    bool hit = false;
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
      if (cr[i] == row && cg[i] == grp) {
        scale = cv[i];
        hit = true;
      }
    }
    if (!hit) {
      scale = e8m0_to_scale(scales(row, grp));
      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        if (cr[i] < 0) {
          cr[i] = row;
          cg[i] = grp;
          cv[i] = scale;
          break;
        }
      }
    }

    // Payload byte for this element: direct gmem read by coordinate
    // (L1-resident from the tile prefetch; element 2j = low nibble,
    // 2j+1 = high nibble of byte j — the mx_layout quantizer law).
    uint8_t b = bytes(row, elem / kElemsPerByte);

    float val;
    if constexpr (KIsE4M3) {
      val = detail::e4m3_to_float(b);
    } else {
      val = e2m1_to_float(uint8_t(elem & 1 ? (b >> 4) : (b & 0xF)));
    }
    // bfloat16_t's float ctor is explicit (no implicit conversion on
    // assignment); build the element type directly.
    dst(dv) = typename DstFrag::value_type(val * scale);
  }
}

}  // namespace cutlass::fmha::collective::mx
