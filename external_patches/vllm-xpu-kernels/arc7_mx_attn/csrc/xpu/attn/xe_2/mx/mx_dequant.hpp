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
    CoordView const& src_coords) {
  // The mainloop resolves get_sub_group via its function-scope
  // using-directive (vendor pattern); this header sits outside it.
  using namespace sycl::ext::oneapi::this_work_item;
  auto wi = get_sub_group().get_local_id()[0];

  // Tile origin: identity slot 0 (WI-canonical) minus its own tv
  // position — both describe (WI 0, register 0), so the difference is
  // the tile base in surface BYTE coordinates (for e4m3 the byte
  // column IS the element column).
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
  int kr[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    auto pc0 = detail::tv_coord(src.tv_layout(), wi, r);
    kr[r] = org_row +
        (kFrame0IsRow ? int(cute::get<0>(pc0)) : int(cute::get<1>(pc0)));
  }
  int grp0 = org_col / kMxGroupSize;
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
    float val = detail::e4m3_to_float(uint8_t(src(i)));
    // bfloat16_t's float ctor is explicit (no implicit conversion on
    // assignment); build the element type directly.
    dst(i) = typename DstFrag::value_type(val * scale);
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
//   slm         : this VV step's byte window base (rows at slm_pitch)
//   scales      : gmem u8 view (k row, group), full width; indexed
//                 directly by coordinate
//   dst         : per-WI ElementQ PV B-operand fragment (tArV)
//   k_row_base  : global (surface-local) k row of the tile, from the
//                 identity-slot-0 anchor law
//   v_base      : global v element of this VV step's first element
//   skip_scales : r200c probe 5 — true = val only, scale 1 (timing)
template <
    bool VIsE4m3,
    class ScaleTensor,
    class DstFrag>
CUTE_HOST_DEVICE
void
dequant_vslm(
    uint8_t const* slm,
    int slm_pitch,
    ScaleTensor const& scales,
    DstFrag& dst,
    int k_row_base,
    int v_base,
    bool skip_scales = false) {
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

  // Per-VV scale VECTOR, direct-indexed: the pinned form-(ii) law
  // gives register i the row k_loc = t0 + 2*v1 with v1 = (i>>1)&7 —
  // exactly 8 rows per WI — and the group is constant within a VV
  // step (v spans [0,32)). So scales ride sc[v1] with zero scan and
  // zero eviction (r200f probe 5: the 4-entry scanned cache thrashed
  // at ~50% miss and cost 1.8-1.9 ms of the 32K walk; rows are read
  // through tv_coord at even registers — kr[r] == k_loc(2r) — so the
  // vector stays derived from the layout, not a parallel law).
  int kr[8];
  float sc[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    auto pc0 = detail::tv_coord(dst.tv_layout(), wi, 2 * r);
    kr[r] = int(cute::get<1>(pc0));
    sc[r] = 1.f;
  }
  if (!skip_scales) {
    int grp0 = v_base / kMxGroupSize;
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      sc[r] = e8m0_to_scale(scales(k_row_base + kr[r], grp0));
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
  CUTE_UNROLL
  for (int i = 0; i < dst.size(); ++i) {
    int r = (i >> 1) & 7;
    int j = (i & 1) + 2 * (i >> 4);
    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    float val = VIsE4m3 ? detail::e4m3_to_float(b)
                        : e2m1_to_float(uint8_t(
                              nib_phase ? (b >> 4) : (b & 0xF)));

    dst(i) = typename DstFrag::value_type(val * sc[r]);
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
    CoordView const& dst_coords) {
  // The mainloop resolves get_sub_group via its function-scope
  // using-directive (vendor pattern); this header sits outside it.
  using namespace sycl::ext::oneapi::this_work_item;
  auto wi = get_sub_group().get_local_id()[0];
  auto tv = dst.tv_layout();

  // Tile origin: identity slot 0 (WI-canonical) minus its own tv
  // position — both describe (WI 0, register 0), so the difference is
  // the tile base in surface coordinates.
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
    detail::assert_staged_tv_form(tv);
    using TV = decltype(tv);
    constexpr int kV2Max = int(cute::size(cute::shape<1>(TV{}))) / 16;
    static_assert(kV2Max >= 1, "val count must be a multiple of 16");
    uint32_t st[2][2 * kV2Max];

    CUTE_UNROLL
    for (int r = 0; r < 2; ++r) {
      CUTE_UNROLL
      for (int g = 0; g < kV2Max; ++g) {
        auto pg = detail::tv_coord(tv, wi, r + 16 * g);
        int row = org_row +
            (kFrame0IsRow ? int(cute::get<0>(pg)) : int(cute::get<1>(pg)));
        int elem = org_elem +
            (kFrame0IsRow ? int(cute::get<1>(pg)) : int(cute::get<0>(pg)));
        // 8-byte window holding elements [16*g, 16*g+16) of this row:
        // bytes (elem>>1) & ~7 .. +8. Both nibble phases (t0 = 0/1)
        // share the window — phases only select the nibble.
        int w = (elem >> 1) & ~7;
        const uint32_t* p =
            reinterpret_cast<const uint32_t*>(&bytes(row, w));
        st[r][2 * g + 0] = p[0];
        st[r][2 * g + 1] = p[1];
      }
    }

    // r201b K-lane scale VECTOR, lean form (see dequant_reg for the
    // full law and the r201 generic-dedup regression that motivated
    // it): the staged byte windows above already issue together; the
    // 4-entry scanned scale cache was the last dependent-load chain.
    // The staging loop above reads the row PER WINDOW (v0, v2) — so
    // the row is constant in v1 but varies over BOTH v0 and v2: EIGHT
    // rows per WI at slot = v0 + 2*v2 = (dv & 1) + 2*(dv >> 4), at
    // most two groups per d-tile — kr[8] + two flat vectors sc0/sc1
    // (r200g's flat-index law; a 2-D sc[slot][goff] with runtime goff
    // would demote to scratch and replay the r201 spill).
    int kr[8];
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      auto pc0 = detail::tv_coord(tv, wi, r);
      kr[r] = org_row +
          (kFrame0IsRow ? int(cute::get<0>(pc0)) : int(cute::get<1>(pc0)));
    }
    int grp0 = org_elem / kMxGroupSize;
    float sc0[8], sc1[8];
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      sc0[r] = e8m0_to_scale(scales(kr[r], grp0));
      sc1[r] = e8m0_to_scale(scales(kr[r], grp0 + 1));
    }
    bool direct = true;

    // SubgroupTensor::size() is a non-static member in this tree
    // (unlike cute::Tensor's static size()) — call it on the object.
    CUTE_UNROLL
    for (int dv = 0; dv < dst.size(); ++dv) {
      // Same coordinate law as the scalar path (cheap ALU; only the
      // byte fetch below changed).
      auto pc = detail::tv_coord(tv, wi, dv);
      int row = org_row +
          (kFrame0IsRow ? int(cute::get<0>(pc)) : int(cute::get<1>(pc)));
      int elem = org_elem +
          (kFrame0IsRow ? int(cute::get<1>(pc)) : int(cute::get<0>(pc)));
      int grp = elem / kMxGroupSize;

      float scale = 0.f;
      if (direct) {
        int slot = (dv & 1) + 2 * (dv >> 4);
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

      // Staged byte fetch. Address the staging array by the DECODED
      // val coordinates (compile-time per dv after unrolling — a
      // runtime index here would push st[] to scratch):
      // byte-within-row jb = v1 + 8*v2 holds for BOTH nibble phases,
      // since (t0 + 2*v1 + 16*v2) / 2 = t0/2 + v1 + 8*v2 and
      // t0/2 == 0. Element 2j = low nibble, 2j+1 = high nibble of
      // byte j — the mx_layout quantizer law; phase is elem & 1.
      int v0 = dv & 1;
      int v1 = (dv >> 1) & 7;
      int v2 = dv >> 4;
      int jb = v1 + 8 * v2;
      uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;

      float val = e2m1_to_float(uint8_t(elem & 1 ? (b >> 4) : (b & 0xF)));
      // bfloat16_t's float ctor is explicit (no implicit conversion on
      // assignment); build the element type directly.
      dst(dv) = typename DstFrag::value_type(val * scale);
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
