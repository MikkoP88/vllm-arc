// In-register MX dequant for the arc7 MX-native paged-decode kernel.
// Consumes the packed payload + e8m0 codes already in cache and
// converts up, in the copy-fragment layout, leaving the layout change
// to the stock subgroup reorder() — the same division of labor the
// stock fp8 path uses (cast in registers, reorder to the MMA fragment).
//
// BOTH the payload bytes and the scales are read COORDINATE-DRIVEN
// DIRECT from gmem: the caller passes (row, column) views of the byte
// and scale surfaces plus the partitioned coordinate view of the
// destination fragment. Each destination value computes its (row,
// byte, group) from its coordinate and reads the u8 codes directly;
// the tile's cache lines are already L2/L1-resident from the
// prefetches, so these are L1 hits, and a 4-entry register cache hoists
// the scale reads out of the unrolled loop.
//
// This replaces the FIRST-PASS byte-fragment lookup, which was broken
// by construction: the custom-extent byte copies and the stock
// dequant-destination copies distribute the SAME tile across
// work-items on DIFFERENT sub-grids (different tv_layouts), so a
// work-item-local coordinate match only succeeded where the grids
// coincided — every 8th element, with the rest reading b=0. Direct
// gmem reads are layout-coupling-free by construction.
//
// Row/element orientation differs per operand: K fragments carry
// (k, d) coordinates (row = mode 0); the V surface family carries
// (v, k) (row = mode 1). RowIsMode0 selects the reading.
//
// Coordinate access follows the in-tree convention (the masking paths
// in chunk_prefill_mainloop.hpp): the caller passes identity-tensor
// coordinate views partitioned by the same thread slices that produced
// the fragments.
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

}  // namespace detail

// dequant_dtile — one d-tile (K) or v-step (V) of one row-block.
//
//   bytes        : gmem u8 view (row, byte-col), FULL surface width;
//                 column = global element / kElemsPerByte
//   scales       : gmem u8 view (row, group), full width; indexed
//                  directly by coordinate
//   dst          : per-thread ElementQ fragment in the dequant copy's
//                  layout (the fragment reorder() converts into the
//                  MMA operand)
//   dst_coords   : partitioned coordinate view -> (x, row) of each
//                  destination value (x = element column/row); global
//                  within the surface, matching the byte/scale views
//
// Every loop runs over static fragment sizes and unrolls fully.
template <
    bool KIsE4M3,
    bool RowIsMode0,
    class ByteView,
    class ScaleTensor,
    class DstFrag,
    class DstCrd>
CUTE_HOST_DEVICE
void
dequant_dtile(
    ByteView const& bytes,
    ScaleTensor const& scales,
    DstFrag& dst,
    DstCrd const& dst_coords) {
  constexpr int kElemsPerByte = KIsE4M3 ? 1 : 2;

  // Per-WI scale cache: the destination fragment of one 64-wide tile
  // spans few (row, group) pairs; anything beyond 4 entries still
  // reads correctly through L1.
  int cr[4] = {-1, -1, -1, -1};
  int cg[4] = {-1, -1, -1, -1};
  float cv[4];

  // SubgroupTensor::size() is a non-static member in this tree (unlike
  // cute::Tensor's static size()) — call it on the object.
  CUTE_UNROLL
  for (int dv = 0; dv < dst.size(); ++dv) {
    auto dc = dst_coords(dv);
    int row = RowIsMode0 ? int(cute::get<0>(dc)) : int(cute::get<1>(dc));
    int elem = RowIsMode0 ? int(cute::get<1>(dc)) : int(cute::get<0>(dc));
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
