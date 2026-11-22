#!/usr/bin/env python3
"""p212q_patch.py -- r212q: U32-DOMAIN fast-path assembly.

Evidence chain (r212p): the light tail (extracts + one convert +
store) and the heavy tail (extra decode/scale/mul/converts) time
IDENTICALLY (380.8 vs 380.6) -- tail arithmetic pipelines free --
while the fast bit-assembly path pays +361 us over that base doing
FEWER ops. The wall is the u8/u16 codegen (16-bit GRF op emulation /
masked inserts / mid-chain truncation), not work.

r212q rewrites the two e2m1 fast loops in the u32 domain end to end:
the byte+nibble extracts fuse into ONE native u32 shift
(u >> (8*j + 4*phase)) & 0xF), the field assembly runs as u32
shifts/ORs with no intermediate truncation, and the only sub-32-bit
touch is the 2-byte memcpy at the fragment store. A u32-native
e2m1_scaled_bf16_u32 helper joins the detail namespace (the u16
helper stays for the history; it is no longer referenced by the fast
paths). Bit-identical outputs by construction -- same field math.

Scope: the two e2m1 FAST loops only (slow/probe loops and every e4m3
path untouched -- m8 byte-identity discipline).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

EDITS = {
    "mx_dequant.hpp": [
        # (1) the u32-native helper after the u16 one
        (
            """e2m1_scaled_bf16(uint32_t nib, int base_e) {
  uint32_t s = (nib >> 3) & 1u;
  uint32_t ee = (nib >> 1) & 3u;
  uint32_t m = nib & 1u;
  uint16_t nz = uint16_t(
      (s << 15) | (uint32_t(base_e + int(ee)) << 7) | (m << 6));
  return (m | ee) ? nz : uint16_t(s << 15);""",
            """e2m1_scaled_bf16(uint32_t nib, int base_e) {
  uint32_t s = (nib >> 3) & 1u;
  uint32_t ee = (nib >> 1) & 3u;
  uint32_t m = nib & 1u;
  uint16_t nz = uint16_t(
      (s << 15) | (uint32_t(base_e + int(ee)) << 7) | (m << 6));
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
  uint32_t bits = (s << 15) | ((base_e + ee) << 7) | (m << 6);
  return (m | ee) ? bits : (s << 15);""",
        ),
        # (2) vslm fast loop: fused u32 extract + u32 assembly
        (
            """    if (!probe_val && fast_band) {
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
    }""",
            """    if (!probe_val && fast_band) {
      CUTE_UNROLL
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
    }""",
        ),
        # (3) dtile fast loop: same u32 rewrite
        (
            """    if (!probe_val && fast_band) {
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
    }""",
            """    if (!probe_val && fast_band) {
      CUTE_UNROLL
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
    }""",
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
    print("P212Q_PATCH_OK")


if __name__ == "__main__":
    main()
