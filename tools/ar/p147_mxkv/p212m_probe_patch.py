#!/usr/bin/env python3
"""p212m_probe_patch.py -- r212m pre-flight: the decode-math probe.

r212k priced the lanes (mxfp4 32K: K_chain 366 + V_gather 376); r212l
proved loop-invariant bookkeeping is NOT on the critical path. The
remaining ALU theory: the per-element decode tail (e2m1/e4m3 bit
assembly + scale multiply + f32->bf16 convert, ~10 ops x 512 elements
per WI per k-tile) binds the walk. This probe severs EXACTLY that tail
and nothing else -- loads, extracts, selects, fragment writes all stay:
ARC_MXKV_VPROBE=9 -> val = float(raw payload byte) (no decode, no
scale mul), threaded to the three dequant sites via a bool arg.

Timing-only (wrong values by design, the const-fill ablation law):
delta(0 -> 9) prices the decode-math slice. >= ~100 us -> the direct-
bf16 assembly iteration is worth it; ~0 -> decode ALU is already
hidden under load latency and the levers are load-structure only
(V-stage u32 stores, tile width).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # dequant_reg: signature + probe tail
        (
            """template <
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
    CoordView const& src_coords) {""",
            """template <
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
    bool probe_val = false) {""",
        ),
        (
            """    // Byte from the loaded fragment register — no per-element memory.
    float val = detail::e4m3_to_float(uint8_t(src(i)));
    // bfloat16_t's float ctor is explicit (no implicit conversion on
    // assignment); build the element type directly.
    dst(i) = typename DstFrag::value_type(val * scale);""",
            """    // Byte from the loaded fragment register — no per-element memory.
    // r212m probe 9: sever ONLY the decode tail (assembly + scale mul);
    // loads/extracts/selects/writes stay.
    float val = probe_val
        ? float(uint8_t(src(i)))
        : detail::e4m3_to_float(uint8_t(src(i)));
    // bfloat16_t's float ctor is explicit (no implicit conversion on
    // assignment); build the element type directly.
    dst(i) = typename DstFrag::value_type(probe_val ? val : val * scale);""",
        ),
        # dequant_vslm: signature + walk tail
        (
            """template <bool VIsE4m3, class DstFrag>
CUTE_HOST_DEVICE
void
dequant_vslm(
    uint8_t const* slm,
    int slm_pitch,
    DstFrag& dst,
    int const (&kr)[8],
    uint32_t const (&line)[8][2],
    int v_base,
    bool skip_scales = false) {""",
            """template <bool VIsE4m3, class DstFrag>
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
    bool probe_val = false) {""",
        ),
        (
            """    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    float val = VIsE4m3 ? detail::e4m3_to_float(b)
                        : e2m1_to_float(uint8_t(
                              nib_phase ? (b >> 4) : (b & 0xF)));

    dst(i) = typename DstFrag::value_type(val * sc[r]);""",
            """    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    // r212m probe 9: sever ONLY the decode tail (assembly + scale
    // mul); the SLM reads, extracts and fragment writes stay.
    float val = probe_val
        ? float(b)
        : (VIsE4m3 ? detail::e4m3_to_float(b)
                   : e2m1_to_float(uint8_t(
                         nib_phase ? (b >> 4) : (b & 0xF))));

    dst(i) = typename DstFrag::value_type(
        probe_val ? val : val * sc[r]);""",
        ),
        # dequant_dtile: signature + staged-walk tail
        (
            """template <
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
    CoordView const& dst_coords) {""",
            """template <
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
    bool probe_val = false) {""",
        ),
        (
            """      uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;

      float val = e2m1_to_float(uint8_t(t0 ? (b >> 4) : (b & 0xF)));
      // bfloat16_t's float ctor is explicit (no implicit conversion on
      // assignment); build the element type directly.
      dst(dv) = typename DstFrag::value_type(val * scale);""",
            """      uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;

      // r212m probe 9: sever ONLY the decode tail (assembly + scale
      // mul); the staged window reads and decode arithmetic on the
      // addresses stay.
      float val = probe_val
          ? float(uint8_t(b))
          : e2m1_to_float(uint8_t(t0 ? (b >> 4) : (b & 0xF)));
      // bfloat16_t's float ctor is explicit (no implicit conversion on
      // assignment); build the element type directly.
      dst(dv) = typename DstFrag::value_type(
          probe_val ? val : val * scale);""",
        ),
    ],
    "mx_paged_decode_mainloop.hpp": [
        (
            """          copy(copy_kb, tKBgK_cache(_, _, _, D), tKrB);
          mx::dequant_reg<true, true>(
              ks_surf, tKrB, tKrBq, tKBgK_cache(_, _, _, D));
          reorder(tKrBq, tSrK);""",
            """          copy(copy_kb, tKBgK_cache(_, _, _, D), tKrB);
          mx::dequant_reg<true, true>(
              ks_surf,
              tKrB,
              tKrBq,
              tKBgK_cache(_, _, _, D),
              params.vlane_probe == 9);
          reorder(tKrBq, tSrK);""",
        ),
        (
            """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tKrDeq,
              tKDgKD_cache(_, _, _, D));
          reorder(tKrDeq, tSrK);""",
            """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tKrDeq,
              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9);
          reorder(tKrDeq, tSrK);""",
        ),
        (
            """          mx::dequant_vslm<Mx::v_is_e4m3>(
              vst + VV * kVBytesPerStep,
              kVStagePitch,
              tArV,
              vslm_kr,
              vslm_line,
              v_byte_base * kElemsPerByteV + VV * kVStepElems,
              params.vlane_probe == 5);""",
            """          mx::dequant_vslm<Mx::v_is_e4m3>(
              vst + VV * kVBytesPerStep,
              kVStagePitch,
              tArV,
              vslm_kr,
              vslm_line,
              v_byte_base * kElemsPerByteV + VV * kVStepElems,
              params.vlane_probe == 5,
              params.vlane_probe == 9);""",
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
    print("P212M_PROBE_PATCH_OK")


if __name__ == "__main__":
    main()
