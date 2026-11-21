#!/usr/bin/env python3
"""p212z_patch.py -- r212z: DOUBLE-BUFFERED K DECODE (mainloop only).

r212x proved the K lane's ~600 us is tSrK LIVENESS -- the per-(K,D)
chain loads -> staging -> decode -> operand -> DPAS serializes
(~73 ns/step x 8192 steps at 32K; the reorder was NOT the cost,
deletion moved nothing). r212y showed the attractor basin is stable,
so the 595-606 band is the comparison baseline. r212z installs a
depth-1 software pipeline for the e2m1 lane: decode (K,D+1) issues
inside iteration D, BEFORE gemm(D), so its chain overlaps the DPAS
pipe. Two B-fragment buffers (tSrK, tSrK1) alternate by D parity
(DD=4 at d=256); a prologue decodes (K,0) after the caches settle.
Probe sever cells fill the CURRENT buffer; the e4m3 lane (m8 oracle)
and the dead-18 block stay byte-identical sequential. Register
pressure is the risk (two 32-reg K buffers + accumulators -- r212t
scratch-wall precedent); m17 vs the band decides.
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

ANCHOR_HEAD = """      /* GEMM 1: S = K * Q */
      clear(tSrS);
      for (int D = 0; D < size<4>(tKDgKD); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        reorder(tQrQ, tSrQ);

        if (kp_sever) {"""

HEAD = """      /* GEMM 1: S = K * Q */
      clear(tSrS);
      // r212z DOUBLE-BUFFERED K DECODE (e2m1 lane): the K lane's
      // ~600 us (m17-m11, stable across 4 attractor wheels) is the
      // tSrK LIVENESS chain -- loads -> staging -> decode -> operand
      // -> DPAS serializes per (K,D) step (r212x: the reorder was
      // NOT the cost; deletion moved nothing). Depth-1 software
      // pipeline: decode (K,D+1) issues inside iteration D, before
      // gemm(D), so the chain overlaps the DPAS pipe. Two B-fragment
      // buffers alternate by D parity; the prologue decodes (K,0).
      // Probe cells and the e4m3 lane (m8 oracle) stay sequential.
      auto tSrK1 = thr_mma_qk.partition_sg_fragment_B(gKD(_, _, 0, 0));
      if constexpr (!Mx::k_is_e4m3) {
        if (!kp_sever) {
          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tSrK,
              tKDgKD_cache(_, _, _, 0),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              (params.vlane_probe >= 11 && params.vlane_probe <= 16) ||
                  params.vlane_probe == 20,
              params.vlane_probe);
        }
      }
      for (int D = 0; D < size<4>(tKDgKD); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        reorder(tQrQ, tSrQ);

        auto& tKrCur = (D & 1) ? tSrK1 : tSrK;
        if (kp_sever) {"""

ANCHOR_FILL = """          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrK.size(); ++i) {
            tSrK(i) = typename decltype(tSrK)::value_type(1.f);
          }
        } else if constexpr (Mx::k_is_e4m3) {"""

FILL = """          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tKrCur.size(); ++i) {
            tKrCur(i) = typename decltype(tSrK)::value_type(1.f);
          }
        } else if constexpr (Mx::k_is_e4m3) {"""

ANCHOR_E4M3 = """          reorder(tKrBq, tSrK);
        } else {"""

E4M3 = """          reorder(tKrBq, tKrCur);
        } else {"""

ANCHOR_CALL = """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tSrK,
              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              (params.vlane_probe >= 11 && params.vlane_probe <= 16) ||
                  params.vlane_probe == 20,
              params.vlane_probe);"""

CALL = """          // r212z: (K,D) was decoded by the prologue (D==0) or the
          // previous iteration's lookahead; issue the lookahead for
          // (K,D+1) NOW so its chain overlaps this gemm.
          if (D + 1 < size<4>(tKDgKD)) {
            mx::dequant_dtile<true, true>(
                kb_surf,
                ks_surf,
                (D & 1) ? tSrK : tSrK1,
                tKDgKD_cache(_, _, _, D + 1),
                params.vlane_probe == 9,
                params.direct_bf16 != 0,
                (params.vlane_probe >= 11 && params.vlane_probe <= 16) ||
                    params.vlane_probe == 20,
                params.vlane_probe);
          }"""

ANCHOR_GEMM = """        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);"""

GEMM = """        cute::gemm(mma_qk, tSrQ, tKrCur, tSrS);"""


def main():
    path = SRC / "mx_paged_decode_mainloop.hpp"
    text = path.read_text()
    if text.count("r212z DOUBLE-BUFFERED K DECODE") != 0:
        print("PATCH_ALREADY_APPLIED (r212z DOUBLE-BUFFERED K DECODE)")
        sys.exit(1)
    for name, anchor in (("HEAD", ANCHOR_HEAD), ("FILL", ANCHOR_FILL),
                         ("E4M3", ANCHOR_E4M3), ("CALL", ANCHOR_CALL),
                         ("GEMM", ANCHOR_GEMM)):
        n = text.count(anchor)
        if n != 1:
            print(f"PATCH_ANCHOR_MISS {name}: count={n}")
            sys.exit(1)
    text = text.replace(ANCHOR_HEAD, HEAD)
    text = text.replace(ANCHOR_FILL, FILL)
    text = text.replace(ANCHOR_E4M3, E4M3)
    text = text.replace(ANCHOR_CALL, CALL)
    text = text.replace(ANCHOR_GEMM, GEMM)
    path.write_text(text)
    print("PATCHED mx_paged_decode_mainloop.hpp (5 edits)")
    print("P212Z_PATCH_OK")


if __name__ == "__main__":
    main()
