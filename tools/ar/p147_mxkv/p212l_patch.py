#!/usr/bin/env python3
"""p212l_patch.py -- r212l kernel iteration: hoist the per-tile invariants.

Evidence (r212k decomposition, build21 wheel, 32K mxfp4): V_gather 392 us
+ K_chain 357 us dominate the walk; the mxfp8-vs-mxfp4 cross-format
triangulation (dense-copy K at 2x bytes ~= scattered-window K) proves
both chains are per-element instruction-bound, NOT load-bound. The
redundant instructions targeted here:

  (1) dequant_vslm re-derives its 8 rows (tv_coord x8) and re-reads the
      same 64 scale bytes as 8 scalar gmem loads EVERY VV step (x8 =
      64 tv evals + 64 loads per k-tile). The rows and the FULL scale
      lines (16 u32; slot pitches/offsets are multiples of 4 by the
      mx_types law) are VV-invariant -> hoist once per k-tile.
  (2) dequant_dtile's staged branch pays 16 tv_coord layout evals per
      d-tile for window addresses that are pure arithmetic under the
      form assert_staged_tv_form already pins (row = t1 + 8*r,
      w = org_elem/2 + 8*g -- t0 floors away in the byte window).

Run on the rig against /root/build/vxk-arc-src. Anchored replacements;
aborts (rc 1) on any anchor miss so a drifted tree fails loudly.
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # ---- (2) static window law in dequant_dtile ----
        (
            """    CUTE_UNROLL
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
""",
            """    // r212l STATIC WINDOW LAW: the pinned form (assert_staged_tv_form)
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
""",
        ),
        # ---- (1a) dequant_vslm split: prep + walk ----
        (
            """// dequant_vslm — one v-step of the r200b SLM-staged V lane. The
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
""",
            """// dequant_vslm — one v-step of the r200b SLM-staged V lane. The
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
// FULL scale lines (all VV steps' groups — D/32 <= 8 bytes per row,
// two u32; slot pitches and scale offsets are multiples of 4 by the
// mx_types law, so u32 line loads are aligned by construction) VV-
// INVARIANT. dequant_vslm_prep derives them once per k-tile; this
// walk consumes the registers (was: 8 tv evals + 8 scalar gmem scale
// loads PER VV step = 64 of each per k-tile).
//
//   slm         : this VV step's byte window base (rows at slm_pitch)
//   dst         : per-WI ElementQ PV B-operand fragment (tArV)
//   kr          : the WI's 8 k-rows (tile-local), from prep
//   line        : the 8 rows' scale lines (u32 x2 each), from prep
//   v_base      : global v element of this VV step's first element
//   skip_scales : r200c probe 5 — true = val only, scale 1 (timing)
template <class ScaleTensor>
CUTE_HOST_DEVICE
void
dequant_vslm_prep(
    ScaleTensor const& scales,
    int k_row_base,
    int const (&kr)[8],
    uint32_t (&line)[8][2],
    bool skip_scales) {
  // groups per scale row (D/32): the second u32 is loaded ONLY when
  // the row actually spans 8 groups — at D=128 the row is 4 bytes and
  // an unpredicated pair read would walk 4 bytes into the next row
  // (past the pool on the last page). grp0 < 4 there, so line[.][1]
  // is never selected by the walk.
  int groups = int(cute::get<1>(scales.shape()));
  if (!skip_scales) {
    CUTE_UNROLL
    for (int r = 0; r < 8; ++r) {
      const uint32_t* p = reinterpret_cast<const uint32_t*>(
          &scales(k_row_base + kr[r], 0));
      line[r][0] = p[0];
      line[r][1] = (groups >= 8) ? p[1] : 0u;
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
    bool skip_scales = false) {
""",
        ),
        # ---- (1b) dequant_vslm body: kr/sc derive from registers ----
        (
            """  // Per-VV scale VECTOR, direct-indexed: the pinned form-(ii) law
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
""",
            """  // r212l: kr and the full scale lines arrive from prep (per-k-tile
  // invariants). Per VV step only the group byte extracts: grp0 is
  // constant within the step (v spans [0,32)); byte = line[r] >>
  // (8*grp0) selects the group from the row's staged line.
  int grp0 = v_base / kMxGroupSize;
  uint32_t gsh = 8 * (grp0 & 3);
  const uint32_t* gl = (grp0 < 4) ? &line[0][0] : &line[0][1];
  float sc[8];
  CUTE_UNROLL
  for (int r = 0; r < 8; ++r) {
    sc[r] = skip_scales ? 1.f : e8m0_to_scale(
        uint8_t((gl[2 * r] >> gsh) & 0xFF));
  }
""",
        ),
    ],
    "mx_paged_decode_mainloop.hpp": [
        # ---- (1c) mainloop: prep before the VV loop + new call ----
        (
            """      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {""",
            """      /* r212l per-tile invariants: the PV B-form's 8 rows and their
      FULL scale lines (2 u32 per row; every VV step's group byte lives
      in them) are VV-invariant — derived once here, consumed by every
      dequant_vslm walk below (was 8 tv evals + 8 scalar gmem scale
      loads PER VV step). */
      int vslm_kr[8];
      uint32_t vslm_line[8][2];
      {
        CUTE_UNROLL
        for (int r = 0; r < 8; ++r) {
          auto pc0 = mx::detail::tv_coord(
              tArV.tv_layout(), thr_id, 2 * r);
          vslm_kr[r] = int(cute::get<1>(pc0));
        }
        mx::dequant_vslm_prep(
            vs_surf,
            v_krow_base,
            vslm_kr,
            vslm_line,
            params.vlane_probe == 5);
      }

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {""",
        ),
        (
            """          mx::dequant_vslm<Mx::v_is_e4m3>(
              vst + VV * kVBytesPerStep,
              kVStagePitch,
              vs_surf,
              tArV,
              v_krow_base,
              v_byte_base * kElemsPerByteV + VV * kVStepElems,
              params.vlane_probe == 5);""",
            """          mx::dequant_vslm<Mx::v_is_e4m3>(
              vst + VV * kVBytesPerStep,
              kVStagePitch,
              tArV,
              vslm_kr,
              vslm_line,
              v_byte_base * kElemsPerByteV + VV * kVStepElems,
              params.vlane_probe == 5);""",
        ),
    ],
}


def main():
    for fname, pairs in EDITS.items():
        path = SRC / fname
        text = path.read_text()
        for old, new in pairs:
            if text.count(old) != 1:
                print(f"PATCH_ANCHOR_MISS {fname}: count={text.count(old)}")
                print(f"  anchor head: {old[:72]!r}")
                sys.exit(1)
            text = text.replace(old, new)
        path.write_text(text)
        print(f"PATCHED {fname} ({len(pairs)} edits)")
    print("P212L_PATCH_OK")


if __name__ == "__main__":
    main()
