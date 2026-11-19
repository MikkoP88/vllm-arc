// In-register MX dequant for the arc7 MX-native paged-decode kernel.
// Consumes the packed payload + e8m0 codes already in cache and
// converts up, in the copy-fragment layout, leaving the layout change
// to the stock subgroup reorder() — the same division of labor the
// stock fp8 path uses (cast in registers, reorder to the MMA fragment).
//
// Scale access is COORDINATE-DRIVEN DIRECT READ: the caller passes the
// gmem scale surface as a (dst-row, group) view plus the partitioned
// coordinate views of the byte and destination fragments. Each
// destination value computes its (row, group) from its coordinate and
// reads the e8m0 code from the surface; repeated (row, group) reads
// hit L1 (the scale tile is a few hundred bytes and the K/V payload
// prefetch has already touched those cache lines' neighborhood), and a
// 4-entry register cache hoists the reads out of the unrolled loop.
// This replaces an earlier subgroup-broadcast design: replicating the
// full tile's scales per work-item costs ~tile_k*groups SG shuffles
// per d-tile, which would have dominated the MMA by ~8x.
//
// Row/element orientation differs per operand: K fragments carry
// (k, d) coordinates (row = mode 0); the V surface family carries
// (v, k) (row = mode 1). RowIsMode0 selects the reading.
//
// Coordinate access follows the in-tree convention (the masking paths
// in chunk_prefill_mainloop.hpp): the caller passes identity-tensor
// coordinate views partitioned by the same thread slices that produced
// the fragments. The byte-lookup match folds at compile time for
// static fragment layouts.
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
//   bytes        : per-thread u8 fragment, payload bytes of the tile
//                 (e4m3: 1 byte/element; e2m1: 2 elements/byte,
//                 element 2j = low nibble, 2j+1 = high)
//   byte_coords  : partitioned coordinate view -> (x, row) of each
//                  byte value (x = byte column for K, byte row for V)
//   scales       : gmem u8 surface view (row, group) for the CURRENT
//                  (sub-)surface; indexed directly by coordinate
//   dst          : per-thread ElementQ fragment in the dequant copy's
//                  layout (the fragment reorder() converts into the
//                  MMA operand)
//   dst_coords   : partitioned coordinate view -> (x, row) of each
//                  destination value (x = element column/row)
//
// Every loop runs over static fragment sizes and unrolls fully.
template <
    bool KIsE4M3,
    bool RowIsMode0,
    class ByteFrag,
    class ByteCrd,
    class ScaleTensor,
    class DstFrag,
    class DstCrd>
CUTE_HOST_DEVICE
void
dequant_dtile(
    ByteFrag const& bytes,
    ByteCrd const& byte_coords,
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

  CUTE_UNROLL
  for (int dv = 0; dv < DstFrag::size(); ++dv) {
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

    // Payload byte for this element: coordinate match against this
    // thread's byte fragment (static size, unrolled; folds away when
    // the copy layouts make the match thread-local).
    uint8_t b = 0;
    CUTE_UNROLL
    for (int bv = 0; bv < ByteFrag::size(); ++bv) {
      auto bc = byte_coords(bv);
      int brow = RowIsMode0 ? int(cute::get<0>(bc)) : int(cute::get<1>(bc));
      int bx = RowIsMode0 ? int(cute::get<1>(bc)) : int(cute::get<0>(bc));
      if (brow == row && bx == elem / kElemsPerByte) {
        b = bytes(bv);
      }
    }

    float val;
    if constexpr (KIsE4M3) {
      val = detail::e4m3_to_float(b);
    } else {
      val = e2m1_to_float(uint8_t(elem & 1 ? (b >> 4) : (b & 0xF)));
    }
    dst(dv) = val * scale;
  }
}

}  // namespace cutlass::fmha::collective::mx
