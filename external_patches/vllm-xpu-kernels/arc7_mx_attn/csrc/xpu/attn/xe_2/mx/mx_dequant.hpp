// In-register MX dequant for the arc7 MX-native paged-decode kernel:
// the structural mirror of cutlass-sycl's cute::quantize_block_wise
// (include/cute/quantization/quantize.hpp), inverted. Where quantize
// reduces abs-max, emits e8m0 codes and converts down, we consume the
// packed payload + e8m0 codes already in cache and convert up, in the
// copy-fragment layout, leaving the layout change to the stock
// subgroup reorder() — the same division of labor the stock fp8 path
// uses (cast in registers, reorder to the MMA fragment).
//
// The two fragments arrive per-thread from 2D-block copies of the SAME
// K (or V) tile: `bytes` holds the packed payload rows, `scales` the
// e8m0 group codes for those rows. Row/column ownership between the
// data copy and the scale copy is not assumed to match — each scale
// value is replicated across the subgroup with group_broadcast keyed
// by the SENDING lane's coordinates, so correctness never depends on
// copy-atom ownership details (32 bytes per (16-row x 64-element)
// d-tile; amortized over the tile's 1k MACs).
//
// Coordinate access follows the in-tree convention (the masking paths
// in chunk_prefill_mainloop.hpp): the caller passes identity-tensor
// coordinate views partitioned by the same thread slices that produced
// the fragments, so every fragment value's logical (row, col) lookup
// folds at compile time.
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

// dequant_dtile — one d-tile of one K (or V) row-block.
//
//   bytes        : per-thread u8 fragment, payload bytes of the tile
//                 (e4m3: 1 byte/element; e2m1: 2 elements/byte,
//                 element 2j = low nibble, 2j+1 = high)
//   byte_coords  : partitioned coordinate view -> (row, byte_col)
//   scales       : per-thread u8 fragment, e8m0 codes of the tile's
//                 groups (DPerTile / kMxGroupSize per row)
//   scale_coords : partitioned coordinate view -> (row, group)
//   dst          : per-thread float fragment, DPerTile elements per
//                  row in the copy-layout family, filled by logical
//                  coordinate via dst_coords ((row, elem))
//
// Every loop runs over static fragment sizes and unrolls fully; the
// group-broadcast phase is the only cross-lane traffic.
template <
    bool KIsE4M3,
    int DPerTile,
    int Rows,
    class ByteFrag,
    class ByteCrd,
    class ScaleFrag,
    class ScaleCrd,
    class DstFrag,
    class DstCrd>
CUTE_HOST_DEVICE
void
dequant_dtile(
    ByteFrag const& bytes,
    ByteCrd const& byte_coords,
    ScaleFrag const& scales,
    ScaleCrd const& scale_coords,
    DstFrag& dst,
    DstCrd const& dst_coords) {
  constexpr int kGroupsPerTile = DPerTile / kMxGroupSize;
  constexpr int kElemsPerByte = KIsE4M3 ? 1 : 2;

  auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
  constexpr int kSG = cute::intel::sg_size;

  // Phase 1 — replicate the tile's scale factors subgroup-wide.
  // Every lane broadcasts each of its scale values together with the
  // (row, group) coordinates that value lives at; each broadcast is
  // claimed by exactly one (lane, sv), so after the loop every thread
  // holds scales_full[row * kGroupsPerTile + group] = 2^(code-127).
  float scales_full[Rows * kGroupsPerTile];
  CUTE_UNROLL
  for (int sv = 0; sv < ScaleFrag::size(); ++sv) {
    float f = e8m0_to_scale(scales(sv));
    auto crd = scale_coords(sv);
    int row = int(cute::get<0>(crd));
    int grp = int(cute::get<1>(crd));
    CUTE_UNROLL
    for (int lane = 0; lane < kSG; ++lane) {
      float v = sycl::group_broadcast(sg, f, lane);
      int r = sycl::group_broadcast(sg, row, lane);
      int g = sycl::group_broadcast(sg, grp, lane);
      scales_full[r * kGroupsPerTile + g] = v;
    }
  }

  // Phase 2 — dequant this thread's payload bytes into its dst values.
  // A dst value (row, elem) reads byte (row, elem / kElemsPerByte) and
  // scale group (elem / kMxGroupSize). The byte lookup is a coordinate
  // match against this thread's byte fragment (static size, unrolled,
  // folds away); fragments from the same tiled-copy family make the
  // match thread-local.
  CUTE_UNROLL
  for (int dv = 0; dv < DstFrag::size(); ++dv) {
    auto crd = dst_coords(dv);
    int row = int(cute::get<0>(crd));
    int elem = int(cute::get<1>(crd));  // 0..DPerTile-1
    float scale = scales_full[row * kGroupsPerTile + elem / kMxGroupSize];
    uint8_t b = 0;
    CUTE_UNROLL
    for (int bv = 0; bv < ByteFrag::size(); ++bv) {
      auto bc = byte_coords(bv);
      if (int(cute::get<0>(bc)) == row &&
          int(cute::get<1>(bc)) == elem / kElemsPerByte) {
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
