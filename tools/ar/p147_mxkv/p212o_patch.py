#!/usr/bin/env python3
"""p212o_patch.py -- r212o: hoist the per-element runtime branches out
of the two e2m1 decode walks (loop split) + drop the dead e2m1 sc prep.

Evidence chain (r212n): the post-direct-bf16 decode residual is still
>=370.5 us (49% of the m4 32K call) while the ALU count of the assembly
itself is ~36-140 us, and the e2m1 skeleton floor grew +104 us across
the m-series (143 -> 246.6) while m8's reproduced r212k's 171.0 exactly.
The m-series threaded probe_val/direct_bf16 as RUNTIME bools into the
element loops: every one of ~50M elements/call evaluates
`!probe_val && fast_band` (2 tests + and), and the fast/slow tails
carry dual codegen. The prep ALU alone is ~2-3 us (too small) -- the
cost is the per-element branch machinery and the register/code shape
of the merged loop.

r212o splits each walk at the per-step granularity (the branch was
already uniform within a step for probe_val; fast_band is per-WI but
loop-invariant): a BRANCH-FREE direct-assembly loop for the in-band
common case, the float loop for slow bands and every probe mode. Math
per element is identical -- bit-exact by construction. The e4m3 paths
(dequant_reg, dequant_vslm's e4m3 arm) are byte-identical source: the
m8 lane must hold its 1319.7-class timing (the dbf0 oracle checks it).
Also removes the e2m1 sc prep -- dead since r212m3 (the e2m1 slow tail
derives its scale inline from kc[r]; only the e4m3 arm reads sc).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # (1) dequant_vslm prep: drop the dead e2m1 sc vector
        (
            """    fast_band = fb;
    if (!fast_band || probe_val) {
      CUTE_UNROLL
      for (int r = 0; r < 8; ++r) {
        sc[r] = skip_scales ? 1.f : e8m0_to_scale(kc[r]);
      }
    }
  }""",
            """    fast_band = fb;
    // r212o: the e2m1 sc vector was dead (the slow tail derives its
    // scale inline from kc[r]) — removed; only the e4m3 arm reads sc.
  }""",
        ),
        # (2) dequant_vslm walk: split fast/slow loops
        (
            """  CUTE_UNROLL
  for (int i = 0; i < dst.size(); ++i) {
    int r = (i >> 1) & 7;
    int j = (i & 1) + 2 * (i >> 4);
    uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
    if constexpr (!VIsE4m3) {
      // r212m DIRECT-BF16 fast path (e2m1 only): every in-band nibble
      // is covered (zero nibbles return +-0); out-of-band groups fall
      // back to the float path below.
      if (!probe_val && fast_band) {
        uint16_t bits = detail::e2m1_scaled_bf16(
            uint8_t(nib_phase ? (b >> 4) : (b & 0xF)), int(kc[r]) - 1);
        typename DstFrag::value_type o;
        __builtin_memcpy(&o, &bits, 2);
        dst(i) = o;
        continue;
      }
    }
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
}""",
            """  // r212o LOOP SPLIT: the per-element runtime branch
  // (!probe_val && fast_band) evaluated dst.size() times per step is
  // a measured cost class (r212n: e2m1 skeleton floor +104 us across
  // the m-series; the assembly ALU itself counts ~2-3 us of prep).
  // The branch hoists to per-step: a branch-free direct-assembly
  // loop for the in-band common case, the float loop for slow bands
  // and every probe mode. Identical math per element.
  if constexpr (!VIsE4m3) {
    if (!probe_val && fast_band) {
      CUTE_UNROLL
      for (int i = 0; i < dst.size(); ++i) {
        int r = (i >> 1) & 7;
        int j = (i & 1) + 2 * (i >> 4);
        uint8_t b = uint8_t((u[r] >> (8 * j)) & 0xFF);
        uint16_t bits = detail::e2m1_scaled_bf16(
            uint8_t(nib_phase ? (b >> 4) : (b & 0xF)), int(kc[r]) - 1);
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
}""",
        ),
        # (3) dequant_dtile walk: split fast/slow loops
        (
            """    // SubgroupTensor::size() is a non-static member in this tree
    // (unlike cute::Tensor's static size()) — call it on the object.
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

      // r212m DIRECT-BF16 fast path: every in-band nibble is covered
      // (zero nibbles return +-0); out-of-band groups fall through.
      if (!probe_val && fast_band) {
        uint16_t bits = detail::e2m1_scaled_bf16(
            uint8_t(t0 ? (b >> 4) : (b & 0xF)),
            (dv & 1) ? (goff ? be11 : be10) : (goff ? be01 : be00));
        typename DstFrag::value_type o;
        __builtin_memcpy(&o, &bits, 2);
        dst(dv) = o;
        continue;
      }
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
    return;""",
            """    // SubgroupTensor::size() is a non-static member in this tree
    // (unlike cute::Tensor's static size()) — call it on the object.
    // r212o LOOP SPLIT (see dequant_vslm): the per-element runtime
    // branch hoists to per-d-tile; the fast loop is branch-free and
    // drops the jb decode comments (same arithmetic verbatim).
    if (!probe_val && fast_band) {
      CUTE_UNROLL
      for (int dv = 0; dv < dst.size(); ++dv) {
        int off = t0 + 2 * ((dv >> 1) & 7) + 16 * (dv >> 4);
        int goff = (off + org_emod) >> 5;
        int v0 = dv & 1;
        int jb = ((dv >> 1) & 7) + 8 * (dv >> 4);
        uint32_t b = (st[v0][jb >> 2] >> ((jb & 3) * 8)) & 0xFF;
        uint16_t bits = detail::e2m1_scaled_bf16(
            uint8_t(t0 ? (b >> 4) : (b & 0xF)),
            (dv & 1) ? (goff ? be11 : be10) : (goff ? be01 : be00));
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
    return;""",
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
    print("P212O_PATCH_OK")


if __name__ == "__main__":
    main()
