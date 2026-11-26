// In-register MX dequant for the arc7 MX-native paged-decode kernel.
// Converts the packed payload + e8m0 codes up to ElementQ in a
// copy-fragment layout, leaving the layout change to the stock
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
// r199b LOAD-PATH LAW (after the r199a V1 const-fill ablation
// attributed ~90-97% of device time to the dequant's read path at
// 32K ctx — the loop structure itself already ran 3x faster than the
// certified v2 Triton decoder): the per-element scattered scalar gmem
// reads are replaced wholesale.
//
//   e4m3 lanes (mxfp8 K/V, mxk8v4 K): dequant_reg — the mainloop
//   loads the payload tile with the REAL vectorized 2D-block byte
//   copy (the same copy_kb/copy_vb machinery that until now only
//   prefetched), and dequant reads each byte from the source
//   FRAGMENT register (src and dst share one tv layout, so index i
//   maps 1:1). No per-element memory traffic at all; only the group
//   scales still touch gmem (4 reads/WI/tile, register-cached).
//
//   e2m1 lanes (mxfp4 K/V, mxk8v4 V): the element tile carries 2
//   elements per byte, so a byte-copy fragment cannot be reordered
//   into the MMA operand (half the values). Instead the same
//   dequant_dtile coordinate walk stages each (row, byte-window) ONCE
//   as a pair of u32 loads: the val form ((2,8),N) puts a WI's
//   elements at elem = t0 + 2*v1 + 16*v2, so dv = v0 + 2*v1 + 16*v2
//   addresses byte windows [8*v2, 8*v2+8) of row(v0) — 16 u32 loads
//   per WI per K d-tile (8 per V step) replace 64 scattered u8 reads.
//   u32 alignment is guaranteed by construction: slot pitches are
//   136/200/264 bytes, payload offsets 0/68/132 — all multiples of 4.
//   The tv form is pinned by static_assert; the r198 faces (bitwise
//   at seq_len==1) re-verify any distribution drift loudly.
//
// Which codomain mode is the surface ROW differs per operand: the QK B-tile
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
  // r203 LEAN decode (in-cache law): the TQ-AoS quantizer saturates
  // (torch float8_e4m3fn semantics), so the e4m3 NaN band 0x7F/0xFF
  // never lands in cache and the cutlass generic convert (NaN check +
  // normal/subnormal branch + subnormal normalize while-loop, ~20 ops
  // per byte on the shared K/V decode slice) reduces to predicated
  // bit assembly. Normals: bits = s | (e+120)<<23 | m<<20 (bias 7 ->
  // 127). Subnormals (e==0): value = m * 2^-9, exact as a float
  // multiply; the sign select keeps -0.0 bitwise (the faces gate is
  // bit-exact). A stray NaN-band byte decodes as 57344 — any writer
  // violating the saturation law fails the faces loudly.
  uint32_t s = uint32_t(b >> 7);
  uint32_t e = (uint32_t(b) >> 3) & 0xF;
  uint32_t m = uint32_t(b) & 7;
  uint32_t bits = (s << 31) | ((e + 120) << 23) | (m << 20);
  float fn;
  __builtin_memcpy(&fn, &bits, 4);
  float sub = float(m) * 0x1p-9f;
  sub = s ? -sub : sub;
  return (e == 0) ? sub : fn;
}

// r212m DIRECT-BF16 DECODE LAW (the probe-9 measurement: the decode
// tail -- bit assembly + scale multiply + f32->bf16 convert -- is
// 56-72% of the 32K-context walk): assemble the bf16 BITS of
// payload * 2^k directly. Every e2m1/e4m3 value times a power of
// two is exactly representable in bf16 (<= 4 significant bits vs 8),
// so whenever the biased exponent lands in the normal band [1, 254]
// the direct bits are IDENTICAL to the float-assembly + multiply +
// convert bits -- no rounding anywhere. base_e is the biased bf16
// exponent of the group's smallest normal payload value scaled:
// e2m1 base_e = 126 + k = code - 1 (normals; the subnormal m=1 is
// exactly 0.5 = 2^(base_e-127) with mantissa ZERO -- r225: the
// assembly must DROP the mantissa bit for ee==0, subnormals carry no
// implicit leading 1; ee=0 m=0 is +-0); e4m3
// base_e = 120 + k = code - 7 (normals only -- payload subnormals
// ee==0 take the float path). Out-of-band codes (including code 0,
// whose bit-assembly float semantics the legacy path preserves)
// route slow via the band checks. The r198 bitwise faces gate the
// whole claim exhaustively at seq_len==1.
CUTE_HOST_DEVICE
inline uint16_t
e2m1_scaled_bf16(uint32_t nib, int base_e) {
  uint32_t s = (nib >> 3) & 1u;
  uint32_t ee = (nib >> 1) & 3u;
  uint32_t m = nib & 1u;
  // r225: subnormals (ee==0) carry NO implicit bit — value = m * 0.5,
  // so m=1 scaled is exactly 2^(base_e-127): mantissa ZERO, not m<<6.
  // The old form applied the implicit 1 to the subnormal band and
  // decoded every +-0.5 element 1.5x (the mxk8v4 NATIVE_WRONG root
  // cause — 23-25% of deep-layer V nibbles are that class; r225a
  // factor matrix + r225c nibble simulation, P6.47).
  uint32_t man = ee ? (m << 6) : 0u;
  uint16_t nz = uint16_t(
      (s << 15) | (uint32_t(base_e + int(ee)) << 7) | man);
  return (m | ee) ? nz : uint16_t(s << 15);
}

// r212q U32-NATIVE VARIANT: same field math, no intermediate 16-bit
// truncation -- the u8/u16 form pays GRF emulation in the hot loop
// (r212p: the tail machinery pipelines free; the bit-assembly codegen
// is the 361 us). In-band base_e+ee <= 254 so the assembled value
// always fits 16 bits; the caller memcpy's the low 2 bytes.
CUTE_HOST_DEVICE
inline uint32_t
e2m1_scaled_bf16_u32(uint32_t nib, uint32_t base_e) {
  uint32_t s = (nib >> 3) & 1u;
  uint32_t ee = (nib >> 1) & 3u;
  uint32_t m = nib & 1u;
  // r225: same subnormal fix as the u16 variant (see above).
  uint32_t man = ee ? (m << 6) : 0u;
  uint32_t bits = (s << 15) | ((base_e + ee) << 7) | man;
  return (m | ee) ? bits : (s << 15);
}

CUTE_HOST_DEVICE
inline uint16_t
e4m3_scaled_bf16(uint32_t b, int base_e) {
  uint32_t s = (b >> 7) & 1u;
  uint32_t ee = (b >> 3) & 0xFu;
  uint32_t m = b & 7u;
  return uint16_t(
      (s << 15) | (uint32_t(base_e + int(ee)) << 7) | (m << 4));
}

// Group-band checks: every payload value of a group whose code passes
// stays in the bf16 normal band after scaling, so the fast path covers
// the whole group (one predictable branch per group; e2m1 needs no
// per-element predication at all).
CUTE_HOST_DEVICE
inline bool
e2m1_band_ok(uint8_t code) {
  int k = int(code) - 127;
  return k >= -125 && k <= 125;  // E' = 126+ee+k in [1, 254]
}

CUTE_HOST_DEVICE
inline bool
e4m3_band_ok(uint8_t code) {
  int k = int(code) - 127;
  return k >= -120 && k <= 119;  // E' = 120+ee+k in [1, 254], ee >= 1
}

// Evaluate a subgroup tv_layout at flat (work-item, register) indices.
// partition_sg_fragment_D produces the rank-3 nesting (Thr, Val, Rest)
// (instantiated evidence, K lane: ((2,8), ((2,8),4), (1,1))): idx2crd
// expands the flat indices into the hierarchical coordinates the
// layout consumes; the Rest modes (extent 1 for our tile slices) park
// at zero. MMA B-fragments (partition_sg_fragment_B, r200b tArV)
// carry the rank-2 nesting (Thr, Val) with no Rest mode — same
// decode, one fewer coordinate. The return value is the tile-local
// coordinate pair.
template <class TVLayout>
CUTE_HOST_DEVICE auto
tv_coord(TVLayout const& tv, int wi, int dv) {
  static_assert(
      cute::rank_v<TVLayout> == 2 || cute::rank_v<TVLayout> == 3,
      "unexpected tv_layout nesting (expected (Thr, Val[, Rest]))");
  auto thr_crd = cute::idx2crd(wi, cute::shape<0>(tv));
  auto val_crd = cute::idx2crd(dv, cute::shape<1>(tv));
  if constexpr (cute::rank_v<TVLayout> == 3) {
    auto rest_crd = cute::idx2crd(0, cute::shape<2>(tv));
    return tv(cute::make_coord(thr_crd, val_crd, rest_crd));
  } else {
    return tv(cute::make_coord(thr_crd, val_crd));
  }
}

// Pin the staged-read addressing law to the tv form it was derived
// from: thr ((2,8)) and val ((2,8),N) — i.e. register dv decodes as
// dv = v0 + 2*v1 + 16*v2 and the tile coordinate advance is
// elem = t0(wi) + 2*v1 + 16*v2, row = t1(wi) + 8*v0 (r198 instantiated
// evidence on both lanes; N = 4 for the 64-element K d-tile, 2 for the
// 32-element V step). A distribution change fails HERE at compile time
// instead of corrupting silently downstream.
template <class TVLayout>
CUTE_HOST_DEVICE void
assert_staged_tv_form(TVLayout const& tv) {
  static_assert(cute::is_static_v<TVLayout>, "tv_layout must be static");
  static_assert(
      cute::rank_v<TVLayout> == 3,
      "unexpected tv_layout nesting (expected (Thr, Val, Rest))");
  constexpr auto thr_sh = cute::shape<0>(TVLayout{});
  static_assert(
      cute::size(thr_sh) == 16 && cute::size(cute::shape<0>(thr_sh)) == 2,
      "thr mode must be ((2,8)) for the staged byte-window law");
  constexpr auto val_sh = cute::shape<1>(TVLayout{});
  static_assert(
      cute::size(cute::get<0>(val_sh)) == 16 &&
          cute::size(cute::shape<0>(cute::get<0>(val_sh))) == 2 &&
          cute::size(val_sh) % 16 == 0,
      "val mode must be ((2,8),N) for the staged byte-window law");
  (void) tv;
}

}  // namespace detail

// dequant_reg — one d-tile (K) or v-step (V) of one row-block, from a
// LOADED byte fragment. The mainloop issues the real 2D-block byte
// copy into src (u8 fragment carrying its own tv_layout); dst is the
// mainloop-built ElementQ fragment with the SAME tv layout (register
// i maps 1:1). e4m3 only: 1 byte = 1 element.
//
//   scales      : gmem u8 view (row, group), full width; indexed
//                 directly by coordinate
//   src         : per-WI u8 byte fragment (the copy destination)
//   dst         : per-WI ElementQ fragment, same tv layout as src
//   src_coords  : WI-canonical identity view of the same byte tile;
//                 only slot 0 is read (tile-origin anchor)
template <
    bool RowIsMode0,
    bool kFrame0IsRow,
    class ScaleTensor,
    class SrcFrag,
    class DstFrag,
    class CoordView>
CUTE_HOST_DEVICE
void
dequant_reg(
    ScaleTensor const& scales,
    SrcFrag const& src,
    DstFrag& dst,
    CoordView const& src_coords,
    bool probe_val = false,
    bool direct_bf16 = true) {
  // The mainloop resolves get_sub_group via its function-scope
  // using-directive (vendor pattern); this header sits outside it.
  using namespace sycl::ext::oneapi::this_work_item;
  auto wi = get_sub_group().get_local_id()[0];

  // Tile origin: identity slot 0 (WI-canonical) minus its own tv
  // position — both describe (WI 0, register 0), so the difference is
  // the tile base in surface BYTE coordinates (for e4m3 the byte
  // column IS the element column).
  // r220a3 NO BAKE: the bwidth TiledCopy composes sg_to_vnk into svB
  // (WG-spanning value distribution — same construction as the kd
  // copy), so this SG's partition coords land IN its own k-band: the
  // slot-0 anchor already carries the band and the scale rows
  // self-resolve. The r220a bake double-counted (foreign scale rows =
  // the r220c uniform deviation on _64; _16 with one SG/WG was never
  // affected). Mirrors the r220a2 dtile verdict.
  auto p00 = detail::tv_coord(src.tv_layout(), 0, 0);
  auto c0 = src_coords(0);
  int org_row =
      (RowIsMode0 ? int(cute::get<0>(c0)) : int(cute::get<1>(c0))) -
      (kFrame0IsRow ? int(cute::get<0>(p00)) : int(cute::get<1>(p00)));
  int org_col =
      (RowIsMode0 ? int(cute::get<1>(c0)) : int(cute::get<0>(c0))) -
      (kFrame0IsRow ? int(cute::get<1>(p00)) : int(cute::get<0>(p00)));

  // r201b K-lane scale VECTOR (r200g's V-lane law, LEAN form): the old
  // 4-entry scanned cache reset per call, and the QK B-form spans more
  // (row, group) pairs than 4 slots — most elements missed, and every
  // miss was an exposed scalar gmem load (r200h: the K load chain is
  // 58-66% of the walk; probe 5 only ablates V scales, so this slice
  // hid inside it). The first r201 attempt used a generic dedup
  // (16-slot int tables + scans) and REGRESSED 2.5x — the tables
  // spilled and the unrolled dedup walks blew up the hot loop. The
  // lean law instead derives the index from the fragment's own tv
  // form. The QK B-tile is NOT the PV B-tile's form-(ii) (v1 advances
  // the row there); its bitwise-validated e2m1 staging proves the row
  // is CONSTANT in v1 and varies over (v0, v2) — bank v0's window v2
  // stages row(v0 + 16*v2) — i.e. EIGHT rows per WI at
  //   slot = v0 + 2*v2 = (i & 1) + 2*(i >> 4)
  // (the mirror of the V law with the val roles swapped; registers
  // 0..7 are the v1=0 representatives), with the d-tile's columns
  // covering at most two scale groups. kr[8] reads the rows through
  // tv_coord at registers 0..7; the scales ride TWO flat vectors
  // sc0/sc1[8] (r200g's flat-index law — a 2-D sc[slot][goff] with
  // runtime goff would demote the array to scratch and replay the r201
  // spill); the walk selects between them on the runtime bit goff.
  // A form that breaks the law trips `direct` and falls back to
  // per-element reads: correct, merely slow.
  //
  // r204 NOTE (reverted tight walk): the ((2,8),((2,8),N)) tight law
  // belongs to the DEQUANT DST fragment (dequant_dtile's assert pins
  // it) — the COPY fragment riding in src carries its own tv_layout
  // whose thr first sub-mode is NOT extent 2 (preflight static_assert
  // fired at instantiation), so the t0/t1 law arithmetic cannot be
  // validated against this object yet; dump the copy form before
  // trying again.
  int kr[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    auto pc0 = detail::tv_coord(src.tv_layout(), wi, r);
    kr[r] = org_row +
        (kFrame0IsRow ? int(cute::get<0>(pc0)) : int(cute::get<1>(pc0)));
  }
  int grp0 = org_col / kMxGroupSize;
  // r212m3 E4M3 REVERT: the e4m3 lanes keep the r212l float path —
  // the direct-bf16 gain (+17.4% same-wheel) did not cover the
  // restructured tail's cost (float path 1359 -> 1862 monatone across
  // r212m/r212m2; net fast = +13% over the old float path). e2m1
  // lanes keep direct-bf16 (see dequant_vslm / dequant_dtile).
  float sc0[8], sc1[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    sc0[r] = e8m0_to_scale(scales(kr[r], grp0));
    sc1[r] = e8m0_to_scale(scales(kr[r], grp0 + 1));
  }
  bool direct = true;

  // SubgroupTensor::size() is a non-static member in this tree (unlike
  // cute::Tensor's static size()) — call it on the object.
  CUTE_UNROLL
  for (int i = 0; i < src.size(); ++i) {
    // This work-item's register i: tile-local (row, byte column) from
    // the fragment's own tv_layout, plus the tile base.
    auto pc = detail::tv_coord(src.tv_layout(), wi, i);
    int row = org_row +
        (kFrame0IsRow ? int(cute::get<0>(pc)) : int(cute::get<1>(pc)));
    int col = org_col +
        (kFrame0IsRow ? int(cute::get<1>(pc)) : int(cute::get<0>(pc)));
    int grp = col / kMxGroupSize;

    float scale = 0.f;
    if (direct) {
      // Direct index: slot = (i&1) + 2*(i>>4) selects the row slot
      // (compile-time under the unroll; the kr guard itself folds to
      // a compile-time compare of the two tv offsets), the column's
      // group offset goff = grp - grp0 selects between the two flat
      // vectors — the only runtime index is this 0/1 select.
      int slot = (i & 1) + 2 * (i >> 4);
      int goff = grp - grp0;
      if (kr[slot] != row || goff < 0 || goff > 1) {
        direct = false;
        scale = e8m0_to_scale(scales(row, grp));
      } else {
        scale = goff ? sc1[slot] : sc0[slot];
      }
    } else {
      scale = e8m0_to_scale(scales(row, grp));
    }

    // Byte from the loaded fragment register — no per-element memory.
    // r212m probe 9: sever ONLY the decode tail (assembly + scale mul);
    // loads/extracts/selects/writes stay.
    float val = probe_val
        ? float(uint8_t(src(i)))
        : detail::e4m3_to_float(uint8_t(src(i)));
    // bfloat16_t's float ctor is explicit (no implicit conversion on
    // assignment); build the element type directly.
    dst(i) = typename DstFrag::value_type(probe_val ? val : val * scale);
  }
}

// dequant_vslm — one v-step of the r200b SLM-staged V lane. The
// k-tile's V payload bytes were cooperatively staged into SLM in the
// dense (k, byte) frame by the mainloop's form-(i) copy + tv-decoded
// scatter; this walk fills the PV B-operand fragment DIRECTLY (form
// (ii), (v, k) tv) — no reorder, no cross-WI register exchange, and
// no tv-basis games (coordinates are read as TUPLES from the dst's
// own tv, the proven r198 orientation). Byte and nibble laws are
// identical to the r198 walk; only the byte source moved to SLM.
// One code path for both payload dtypes (VIsE4m3 picks the byte law).
//
// r212l SPLIT: the form-(ii) law makes this WI's EIGHT rows and their
// scale lines (this workgroup's whole VV-step group span, two u32;
// slot pitches and scale offsets are multiples of 4 by the mx_types
// law, so u32 line loads are aligned by construction) VV-INVARIANT.
// dequant_vslm_prep derives them once per k-tile; this walk consumes
// the registers (was: 8 tv evals + 8 scalar gmem scale loads PER VV
// step = 64 of each per k-tile).
//
// P-C1 r242b SLICE WINDOW: head_size_vo > kDecodeMaxShapeOutV (256)
// splits V across grid.x, so a workgroup's VV steps span only the
// blk_v SLICE's groups — at most ShapeOutV/32 <= 8. The line is
// therefore SLICE-LOCAL: prep gathers the row's groups
// [grp_base, grp_base + slice_groups) (grp_base a multiple of 4 by
// the ShapeOut law — the u32 alignment holds) and the walk's grp0 is
// within-slice by construction (v_base below). D=512 rows span 16
// bytes (4 u32) — windowing keeps the r212l 2-u32 footprint instead
// of caching all D/32 groups. D<=256 is blk_v=0 with grp_base=0:
// byte-identical to the r212l law.
//
//   slm         : this VV step's byte window base (rows at slm_pitch)
//   dst         : per-WI ElementQ PV B-operand fragment (tArV)
//   kr          : the WI's 8 k-rows (tile-local), from prep
//   line        : the 8 rows' scale lines (u32 x2 each), from prep
//   v_base      : this VV step's first element, within the slice
//   skip_scales : r200c probe 5 — true = val only, scale 1 (timing)
template <class ScaleTensor>
CUTE_HOST_DEVICE
void
dequant_vslm_prep(
    ScaleTensor const& scales,
    int k_row_base,
    int const (&kr)[8],
    uint32_t (&line)[8][2],
    int grp_base,
    int slice_groups,
    bool skip_scales) {
  // The second u32 is loaded ONLY when the slice actually spans 8
  // groups — at D=128 (or a 128-wide budget slice) the window is 4
  // bytes and an unpredicated pair read would walk into the next row
  // (past the pool on the last page). grp0 < 4 there, so line[.][1]
  // is never selected by the walk.
  if (!skip_scales) {
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      const uint32_t* p = reinterpret_cast<const uint32_t*>(
          &scales(k_row_base + kr[r], grp_base));
      line[r][0] = p[0];
      line[r][1] = (slice_groups >= 8) ? p[1] : 0u;
    }
  } else {
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      line[r][0] = 0x7F7F7F7Fu;  // every group byte = code 127 = 1.0
      line[r][1] = 0x7F7F7F7Fu;
    }
  }
}

template <bool VIsE4m3, class DstFrag>
CUTE_HOST_DEVICE
void
dequant_vslm(
    uint8_t const* slm,
    int slm_pitch,
    DstFrag& dst,
    int const (&kr)[8],
    uint32_t const (&line)[8][2],
    int v_base,
    bool skip_scales = false,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false,
    int probe_mode = 0) {
  // The mainloop resolves get_sub_group via its function-scope
  // using-directive (vendor pattern); this header sits outside it.
  using namespace sycl::ext::oneapi::this_work_item;
  auto wi = get_sub_group().get_local_id()[0];

  // Pin the dst tv to the form-(ii) B-fragment law this walk decodes
  // (instantiated evidence, r200b tArV: thr ((2,8)), val ((2,8),(2,1))
  // — v = t1 + 8*v0 + 16*v2, k = t0 + 2*v1, mode 0 = v; CONFIRMED
  // exhaustively by the r200f synth-v2 validator: C-stage 0/2048
  // mismatches on both payload dtypes at HK=2). A distribution change
  // fails HERE at compile time instead of corrupting silently
  // downstream (the assert_staged_tv_form law).
  static_assert(
      cute::rank_v<decltype(dst.tv_layout())> == 2,
      "dequant_vslm expects the B-fragment (Thr, Val) tv nesting");
  constexpr auto vslm_thr_sh = cute::shape<0>(decltype(dst.tv_layout()){});
  static_assert(
      cute::size(vslm_thr_sh) == 16 &&
          cute::size(cute::shape<0>(vslm_thr_sh)) == 2,
      "thr mode must be ((2,8)) for the form-(ii) V walk");
  constexpr auto vslm_val_sh = cute::shape<1>(decltype(dst.tv_layout()){});
  static_assert(
      cute::size(cute::get<0>(vslm_val_sh)) == 16 &&
          cute::size(cute::shape<0>(cute::get<0>(vslm_val_sh))) == 2 &&
          cute::size(vslm_val_sh) == 32,
      "val mode must be ((2,8),2) for the form-(ii) V walk");

  // r212l: kr and the full scale lines arrive from prep (per-k-tile
  // invariants). Per VV step only the group byte extracts: grp0 is
  // constant within the step (v spans [0,32)); byte = line[r] >>
  // (8*grp0) selects the group from the row's staged line.
  int grp0 = v_base / kMxGroupSize;
  uint32_t gsh = 8 * (grp0 & 3);
  const uint32_t* gl = (grp0 < 4) ? &line[0][0] : &line[0][1];
  // r212m3 DTYPE SPLIT: e2m1 (mxfp4 V, mxk8v4 V) keeps the trimmed
  // direct-bf16 machinery — kc[r] (two u32 registers) is the only
  // added live state, base_e derives inline (int(code) - 1), the rare
  // out-of-band element converts its own code on demand. e4m3
  // (mxfp8 V) restores the r212l float-vector prep and tail — the
  // direct-bf16 gain did not cover the restructured tail's cost
  // (see dequant_reg's r212m3 note).
  uint8_t kc[8];
  bool fast_band = false;
  float sc[8] = {};
  // r212p LIGHT PROBE (probe 11) severs the prep too — the light
  // tail needs no scale codes, no band, no float vector.
  if (!probe_light) {
    if constexpr (VIsE4m3) {
      CUTE_UNROLL
      for (int r = 0; r < 8; ++r) {
        sc[r] = skip_scales ? 1.f : e8m0_to_scale(
            uint8_t((gl[2 * r] >> gsh) & 0xFF));
      }
    } else {
      bool fb = direct_bf16;
      CUTE_UNROLL
      for (int r = 0; r < 8; ++r) {
        kc[r] = skip_scales ? uint8_t(127)
                            : uint8_t((gl[2 * r] >> gsh) & 0xFF);
        fb = fb && detail::e2m1_band_ok(kc[r]);
      }
      fast_band = fb;
      // r212o: the e2m1 sc vector was dead (the slow tail derives
      // its scale inline from kc[r]) — removed; only the e4m3 arm
      // reads sc.
    }
  }

  // r203 PERMUTED-FRAME READ: the staging scatter lands bytes by the
  // transpose law (see the mainloop's staging block), so this WI's
  // four payload bytes of each k-row are ONE aligned u32 — 8 u32
  // loads per VV step replace the 32 scattered byte transactions that
  // r202 isolated as the ~0.9 ms V-dequant slice. Register i's byte
  // within the u32 is j = (i&1) + 2*(i>>4) (= v>>3, compile-time under
  // the unroll); the e2m1 nibble phase is t1&1 (all of a WI's v share
  // t1's parity). k_loc = kr[(i>>1)&7] and v_loc = t1 + 8*(i&1) +
  // 16*(i>>4) are the r202 form-law arithmetic (t1 = (wi>>1)&7).
  int t1 = (wi >> 1) & 7;
  int qbase = VIsE4m3 ? 4 * t1 : 4 * (t1 >> 1);
  int nib_phase = t1 & 1;
  uint32_t u[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    u[r] = *reinterpret_cast<const uint32_t*>(
        slm + kr[r] * slm_pitch + qbase);
  }
  // r212p LIGHT PROBE TAIL (probe 11): extracts + float convert +
  // store ONLY — severs more than probe 9 (whose tail kept the nibble
  // decode arithmetic and scale re-derivation). Prices the
  // extracts/store floor vs the decode machinery (r212o's puzzle).
  if (probe_light) {
    CUTE_UNROLL
    for (int i = 0; i < dst.size(); ++i) {
      int r = (i >> 1) & 7;
      int j = (i & 1) + 2 * (i >> 4);
      uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
      // r212s/r212t PROBE LADDER: 11 = light tail; 12/13 = +4/+8
      // dependent scalar int ops (2 shifts + xor + and, chained so
      // nothing folds); 14 = + the REAL helper call (full assembly
      // incl. the ternary select, base 120) with the convert-shaped
      // store; 15 = trivial bits with the memcpy-shaped store. The
      // rungs split the m0-m11 pool into assembly/select vs store
      // shape (r212s: int ops alone are FREE — the ladder is flat).
      uint32_t nib = VIsE4m3
          ? uint32_t(b)
          : uint32_t(nib_phase ? (b >> 4) : (b & 0xF));
      int kops = probe_mode == 12 ? 4 : (probe_mode == 13 ? 8 : 0);
      for (int k = 0; k + 3 <= kops; k += 3) {
        uint32_t t = nib;
        nib = ((t << 1) ^ (t >> 1)) & 0xFu;
      }
      if (probe_mode == 15) {
        typename DstFrag::value_type o;
        uint32_t bits = nib;
        __builtin_memcpy(&o, &bits, 2);
        dst(i) = o;
      } else {
        if (probe_mode == 14) {
          uint32_t bits = detail::e2m1_scaled_bf16_u32(nib, 120u);
          nib = bits & 0xFu;
        }
        dst(i) = typename DstFrag::value_type(float(uint8_t(nib)));
      }
    }
    return;
  }
  // r212o LOOP SPLIT: the per-element runtime branch
  // (!probe_val && fast_band) evaluated dst.size() times per step is
  // a measured cost class (r212n: e2m1 skeleton floor +104 us across
  // the m-series; the assembly ALU itself counts ~2-3 us of prep).
  // The branch hoists to per-step: a branch-free direct-assembly
  // loop for the in-band common case, the float loop for slow bands
  // and every probe mode. Identical math per element.
  if constexpr (!VIsE4m3) {
    if (!probe_val && fast_band) {
      // r212r NO-UNROLL PROBE: the identical r212q body under a
      // runtime loop — prices the 32x-unrolled code size (icache /
      // cross-element interleave) against the m11-class light loops.
      for (int i = 0; i < dst.size(); ++i) {
        int r = (i >> 1) & 7;
        int j = (i & 1) + 2 * (i >> 4);
        // r212q U32-DOMAIN ASSEMBLY: one fused native shift for the
        // nibble (byte extract folded away), u32 field math, single
        // 2-byte touch at the store.
        uint32_t nib = (u[r] >> (8 * j + 4 * nib_phase)) & 0xFu;
        uint32_t bits = detail::e2m1_scaled_bf16_u32(
            nib, uint32_t(int(kc[r]) - 1));
        typename DstFrag::value_type o;
        __builtin_memcpy(&o, &bits, 2);
        dst(i) = o;
      }
      return;
    }
  }
  CUTE_UNROLL
  for (int i = 0; i < dst.size(); ++i) {
    int r = (i >> 1) & 7;
    int j = (i & 1) + 2 * (i >> 4);
    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    // r212m probe 9: sever ONLY the decode tail (assembly + scale
    // mul); the SLM reads, extracts and fragment writes stay.
    float val = probe_val
        ? float(b)
        : (VIsE4m3 ? detail::e4m3_to_float(b)
                   : e2m1_to_float(uint8_t(
                         nib_phase ? (b >> 4) : (b & 0xF))));

    if constexpr (VIsE4m3) {
      dst(i) = typename DstFrag::value_type(
          probe_val ? val : val * sc[r]);
    } else {
      dst(i) = typename DstFrag::value_type(
          probe_val ? val
                    : val * (skip_scales ? 1.f : e8m0_to_scale(kc[r])));
    }
  }
}

// dequant_dtile — one d-tile (K) or v-step (V) of one row-block,
// e2m1 lanes: staged wide reads. Same contract as before r199b
// (destination fragment in the dequant copy's layout; reorder()
// converts it into the MMA operand) but the payload bytes reach
// registers through paired u32 window loads instead of 64 scattered
// scalar reads — see the r199b LOAD-PATH LAW above.
//
//   bytes        : gmem u8 view (row, byte-col), FULL surface width;
//                 column = global element / 2
//   scales       : gmem u8 view (row, group), full width; indexed
//                  directly by coordinate
//   dst          : per-WI ElementQ fragment in the dequant copy's
//                  layout (the fragment reorder() converts into the
//                  MMA operand); carries the tv_layout used for the
//                  per-WI position decoding
//   dst_coords   : WI-canonical identity view of the same tile; only
//                  slot 0 is read (tile-origin anchor)
template <
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
    CoordView const& dst_coords,
    bool probe_val = false,
    bool direct_bf16 = true,
    bool probe_light = false,
    int probe_mode = 0) {
  // The mainloop resolves get_sub_group via its function-scope
  // using-directive (vendor pattern); this header sits outside it.
  using namespace sycl::ext::oneapi::this_work_item;
  auto wi = get_sub_group().get_local_id()[0];
  auto tv = dst.tv_layout();

  // Tile origin: identity slot 0 (WI-canonical) minus its own tv
  // position — both describe (WI 0, register 0), so the difference is
  // the tile base in surface coordinates.
  // r220a2 NO BAKE: TiledCopyKD is the make_block_2d_copy_B family
  // whose svB composition bakes sg_to_vnk into the value layout —
  // this anchor is SLICE-canonical (carries the SG's k-band) under
  // multi-SG, so the walk adds NO band. Evidence: r220a offline —
  // mxfp8/mxk8v4 (bwidth-copy lanes: reg + V) clean with the band
  // baked at decode; mxfp4 (e2m1 K = this path) NaN with the bake.
  auto p00 = detail::tv_coord(tv, 0, 0);
  auto c0 = dst_coords(0);
  int org_row =
      (RowIsMode0 ? int(cute::get<0>(c0)) : int(cute::get<1>(c0))) -
      (kFrame0IsRow ? int(cute::get<0>(p00)) : int(cute::get<1>(p00)));
  int org_elem =
      (RowIsMode0 ? int(cute::get<1>(c0)) : int(cute::get<0>(c0))) -
      (kFrame0IsRow ? int(cute::get<1>(p00)) : int(cute::get<0>(p00)));

  // FORM GATE (compile-time; uniform across the subgroup): both MMAs
  // ride the SAME DPAS atom, but the N/K extents swap between the QK
  // B-tile (N = 16 rows, K = 64 elems — val mode ((2,8),N), the
  // r198-evidenced K form) and the PV B-tile (N = 32 v, K = 16 rows —
  // val mode (2,8,2): v1 advances the ROW). Contiguous 8-byte windows
  // per (row, v2) exist only in the first case; the second (e2m1 V)
  // has 8 rows x stride-4 byte positions with no compressible window
  // and falls back to the r198 PASS scalar walk until its structural
  // fix (SLM staging / repacked lane). The val-mode nesting IS the
  // form discriminator at compile time; the staged branch then pins
  // the full law (thr form + first-16 val form) with
  // assert_staged_tv_form, which only instantiates there.
  constexpr bool kStagedValForm =
      cute::size(cute::get<0>(cute::shape<1>(decltype(tv){}))) == 16;

  // Per-WI scale cache (unchanged law).
  int cr[4] = {-1, -1, -1, -1};
  int cg[4] = {-1, -1, -1, -1};
  float cv[4];

  if constexpr (kStagedValForm) {
    // Staged byte windows: the val form dv = v0 + 2*v1 + 16*v2
    // (asserted) makes register dv = v0 + 16*v2 the representative of
    // window (row(v0), bytes [8*v2, 8*v2+8)). K d-tile: 2x4 windows
    // (64 elements, 2 u32 each); a same-form V step would be 2x2.
    // r212x NESTED-FORM GUARD: dst may be the MMA B operand (tSrK --
    // the K reorder deletion). Its tv is rank-2 with the rest mode
    // folded into val as a stride-0 degenerate: ((2,8),((2,8),(1,4)))
    // -- the SAME walk law as the pinned rank-3 form (dv = v0 + 2*v1
    // + 16*v2; row = t_b + 8*v0, elem = t_a + 2*v1 + 16*v2), so the
    // staged body below applies verbatim. Only the form pin differs.
    if constexpr (cute::rank_v<decltype(tv)> == 3) {
      detail::assert_staged_tv_form(tv);
    } else {
      constexpr auto vsh = cute::shape<1>(decltype(tv){});
      static_assert(
          cute::rank_v<decltype(vsh)> == 2 &&
              cute::size(cute::get<0>(vsh)) == 16 &&
              cute::size(vsh) % 16 == 0,
          "r212x: unexpected rank-2 tv form (expected the nested "
          "((2,8),((2,8),(1,4))) MMA B operand)");
    }
    using TV = decltype(tv);
    constexpr int kV2Max = int(cute::size(cute::shape<1>(TV{}))) / 16;
    static_assert(kV2Max >= 1, "val count must be a multiple of 16");
    uint32_t st[2][2 * kV2Max];

    // r212l STATIC WINDOW LAW: the pinned form (assert_staged_tv_form)
    // gives register r + 16*g the element t0 + 16*g of row t1 + 8*r, so
    // the window address is pure arithmetic — the per-window tv_coord
    // eval (16 layout evaluations per d-tile) retires. t0 floors away
    // in the byte window ((even + t0) >> 1 == even >> 1) and org_elem
    // is 64-aligned (d-tile base), so w = org_elem/2 + 8*g exactly.
    {
      int t1w = (wi >> 1) & 7;
      CUTE_UNROLL
      for (int r = 0; r < 2; ++r) {
        int row = org_row + t1w + 8 * r;
        CUTE_UNROLL
        for (int g = 0; g < kV2Max; ++g) {
          int w = (org_elem >> 1) + 8 * g;
          const uint32_t* p =
              reinterpret_cast<const uint32_t*>(&bytes(row, w));
          st[r][2 * g + 0] = p[0];
          st[r][2 * g + 1] = p[1];
        }
      }
    }

    // r204 LAW-ARITHMETIC WALK (same law as dequant_reg; the form is
    // pinned by assert_staged_tv_form above, so the r201b runtime
    // guard retires): the tight law is TWO rows per WI — row =
    // t1(wi) + 8*v0 (the staged banks r = 0/1 ARE those rows), elem
    // = t0 + 2*v1 + 16*v2, nibble phase = t0 (byte origins are
    // even). FOUR scale loads issue upfront (2 rows x 2 groups);
    // goff = (off + org_elem%32) >> 5 is exact for any tile origin.
    int t0 = wi & 1;
    int t1 = (wi >> 1) & 7;
    int grp0 = org_elem / kMxGroupSize;
    int org_emod = org_elem & (kMxGroupSize - 1);
    int row0 = org_row + t1;
    // r212m CODE-DOMAIN SCALES: base_e = code - 1 (= 126 + k); the
    // slow path derives its float on demand. Identical gmem traffic
    // (same 4 byte reads).
    // r212p LIGHT PROBE (probe 11): prep severed — the light tail
    // needs no scale codes and no band.
    uint8_t c00 = 127, c01 = 127, c10 = 127, c11 = 127;
    bool fast_band = false;
    int be00 = 0, be01 = 0, be10 = 0, be11 = 0;
    if (!probe_light) {
      c00 = uint8_t(scales(row0, grp0));
      c01 = uint8_t(scales(row0, grp0 + 1));
      c10 = uint8_t(scales(row0 + 8, grp0));
      c11 = uint8_t(scales(row0 + 8, grp0 + 1));
      fast_band = direct_bf16 && detail::e2m1_band_ok(c00) &&
                  detail::e2m1_band_ok(c01) &&
                  detail::e2m1_band_ok(c10) &&
                  detail::e2m1_band_ok(c11);
      be00 = int(c00) - 1;
      be01 = int(c01) - 1;
      be10 = int(c10) - 1;
      be11 = int(c11) - 1;
    }

    // SubgroupTensor::size() is a non-static member in this tree
    // (unlike cute::Tensor's static size()) — call it on the object.
    // r212p LIGHT PROBE TAIL (probe 11): extracts + float convert +
    // store ONLY (see dequant_vslm). The st[] window loads above stay
    // — they ARE the walk's load path.
    if (probe_light) {
      CUTE_UNROLL
      for (int dv = 0; dv < dst.size(); ++dv) {
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;
        // r212s/r212t PROBE LADDER (see dequant_vslm).
        uint32_t nib = t0 ? (b >> 4) : (b & 0xF);
        int kops = probe_mode == 12 ? 4 : (probe_mode == 13 ? 8 : 0);
        for (int k = 0; k + 3 <= kops; k += 3) {
          uint32_t t = nib;
          nib = ((t << 1) ^ (t >> 1)) & 0xFu;
        }
        if (probe_mode == 15) {
          typename DstFrag::value_type o;
          uint32_t bits = nib;
          __builtin_memcpy(&o, &bits, 2);
          dst(dv) = o;
        } else {
          if (probe_mode == 14) {
            uint32_t bits = detail::e2m1_scaled_bf16_u32(nib, 120u);
            nib = bits & 0xFu;
          }
          dst(dv) = typename DstFrag::value_type(float(uint8_t(nib)));
        }
      }
      return;
    }
    // r212o LOOP SPLIT (see dequant_vslm): the per-element runtime
    // branch hoists to per-d-tile; the fast loop is branch-free and
    // drops the jb decode comments (same arithmetic verbatim).
    if (!probe_val && fast_band) {
      // r212r NO-UNROLL PROBE (see dequant_vslm).
      for (int dv = 0; dv < dst.size(); ++dv) {
        int off = t0 + 2 * ((dv >> 1) & 7) + 16 * (dv >> 4);
        int goff = (off + org_emod) >> 5;
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        // r212q U32-DOMAIN ASSEMBLY (see dequant_vslm).
        uint32_t nib =
            (st[v0][jb >> 2] >> ((jb & 3) * 8 + 4 * t0)) & 0xFu;
        uint32_t bits = detail::e2m1_scaled_bf16_u32(
            nib,
            uint32_t((dv & 1) ? (goff ? be11 : be10)
                              : (goff ? be01 : be00)));
        typename DstFrag::value_type o;
        __builtin_memcpy(&o, &bits, 2);
        dst(dv) = o;
      }
      return;
    }
    CUTE_UNROLL
    for (int dv = 0; dv < dst.size(); ++dv) {
      int off = t0 + 2 * ((dv >> 1) & 7) + 16 * (dv >> 4);
      int goff = (off + org_emod) >> 5;

      // Staged byte fetch. Address the staging array by the DECODED
      // val coordinates (compile-time per dv after unrolling — a
      // runtime index here would push st[] to scratch):
      // byte-within-row jb = v1 + 8*v2 holds for BOTH nibble phases,
      // since (t0 + 2*v1 + 16*v2) / 2 = t0/2 + v1 + 8*v2 and
      // t0/2 == 0. Element 2j = low nibble, 2j+1 = high nibble of
      // byte j — the mx_layout quantizer law; phase is t0.
      int v0 = dv & 1;
      int v1 = (dv >> 1) & 7;
      int v2 = dv >> 4;
      int jb = v1 + 8 * v2;
      uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;

      // Slow path: derive the float scale on demand.
      float scale = e8m0_to_scale(
          goff ? (dv & 1 ? c11 : c01) : (dv & 1 ? c10 : c00));
      // r212m probe 9: sever ONLY the decode tail (assembly + scale
      // mul); the staged window reads and decode arithmetic on the
      // addresses stay.
      float val = probe_val
          ? float(uint8_t(b))
          : e2m1_to_float(uint8_t(t0 ? (b >> 4) : (b & 0xF)));
      // bfloat16_t's float ctor is explicit (no implicit conversion on
      // assignment); build the element type directly.
      dst(dv) = typename DstFrag::value_type(
          probe_val ? val : val * scale);
    }
    return;
  }

  // FALLBACK (e2m1 V form): the r198 PASS scalar walk, verbatim.
  CUTE_UNROLL
  for (int dv = 0; dv < dst.size(); ++dv) {
    auto pc = detail::tv_coord(tv, wi, dv);
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

    uint8_t b = bytes(row, elem / 2);
    float val = e2m1_to_float(uint8_t(elem & 1 ? (b >> 4) : (b & 0xF)));
    dst(dv) = typename DstFrag::value_type(val * scale);
  }
}

}  // namespace cutlass::fmha::collective::mx
