#!/usr/bin/env python3
"""p212ab_patch.py -- r212ab: COMPILE-TIME-PARITY PEELED K-DECODE
PIPELINE in the SHIPPING e2m1 body only.

r212z proved ~490 us of the K lane (tSrK-liveness chain) is
overlap-coverable by decoding (K,D+1) one step ahead of gemm(D), but
its runtime-parity ternary (auto& tKrCur = (D&1) ? tSrK1 : tSrK)
demoted the MMA operand off static registers -- whole-region collapse
(m11 414->1107, m8 +109%). The peel removes the selection entirely:
the D-loop runs unrolled-by-2 with STATIC buffer names per textual
half (even: cur tSrK, lookahead tSrK1; odd: cur tSrK1, lookahead
tSrK), a prologue decodes (K,0), and DD (size<4> of the static
layout; 4 at d=256) folds every guard at compile time. Odd-DD tails
take a constexpr epilogue (the last lookahead wrote tSrK).

The instrumented body (kInstr) and the e4m3 lane (m8 new best 1193.7
on r212aa) keep the original loop textually untouched -- wrapped in
if constexpr (kInstr || Mx::k_is_e4m3), which folds to no branch for
them. The patch slices the existing loop out by markers (no comment
retyping) and re-emits it verbatim inside the wrapper.

Register pressure (2x32-reg B fragments + staging) is the declared
risk (r212t scratch-wall class -- though that wall was RUNTIME
indexing, which this form has none of). m0 + canary gate the bytes
and the net; m11 gates the instrumented lane's stability; m8 gates
the e4m3 lane against the 1193.7 baseline (1-sample, +8 pct band).
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

START = "      /* GEMM 1: S = K * Q */\n      clear(tSrS);\n"
LOOP_HEAD = "      for (int D = 0; D < size<4>(tKDgKD); D++) {"
GEMM_LINE = "        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);\n"
MARK = "r212ab PEELED-BY-2 K DECODE PIPELINE"

WRAP_OPEN = """      // r212ab: the SHIPPING e2m1 lane takes the compile-time-parity
      // pipelined D-loop below (r212z proved ~490 us of the K lane
      // overlaps decode-ahead, but its runtime-parity ternary demoted
      // the MMA operand off static registers -- whole-region
      // collapse). The peeled form keeps every buffer name STATIC per
      // textual half: no ternary, no demotion. The instrumented body
      // and the e4m3 lane (m8 new best 1193.7 on r212aa) keep the
      // original loop textually untouched.
      if constexpr (kInstr || Mx::k_is_e4m3) {
"""

PIPE = """      } else {
        // r212ab PEELED-BY-2 K DECODE PIPELINE (shipping e2m1 only):
        // prologue decodes (K,0) into tSrK; the loop processes D in
        // pairs -- the EVEN half's lookahead decodes (K,D+1) into
        // tSrK1 BEFORE gemm(D) consumes tSrK; the ODD half's
        // lookahead decodes (K,D+2) into tSrK BEFORE gemm(D+1)
        // consumes tSrK1. Each decode issues one half-step ahead of
        // its consumer and overlaps the DPAS pipe. DD is a
        // compile-time constant (size<4> of the static layout; 4 at
        // d=256), so the unrolled loop folds every guard and every
        // buffer name is statically bound. Odd-DD tails take the
        // constexpr epilogue (the last lookahead wrote tSrK).
        auto tSrK1 = thr_mma_qk.partition_sg_fragment_B(gKD(_, _, 0, 0));
        mx::dequant_dtile<true, true>(
            kb_surf,
            ks_surf,
            tSrK,
            tKDgKD_cache(_, _, _, 0),
            false,
            params.direct_bf16 != 0,
            false,
            0);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D + 1 < size<4>(tKDgKD); D += 2) {
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          reorder(tQrQ, tSrQ);
          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tSrK1,
              tKDgKD_cache(_, _, _, D + 1),
              false,
              params.direct_bf16 != 0,
              false,
              0);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
          copy(copy_q, tQgQ(_, _, _, D + 1), tQrQ);
          reorder(tQrQ, tSrQ);
          if (D + 2 < size<4>(tKDgKD)) {
            mx::dequant_dtile<true, true>(
                kb_surf,
                ks_surf,
                tSrK,
                tKDgKD_cache(_, _, _, D + 2),
                false,
                params.direct_bf16 != 0,
                false,
                0);
          }
          cute::gemm(mma_qk, tSrQ, tSrK1, tSrS);
        }
        // size<4>(tKDgKD) is RUNTIME here (the gmem K tensor carries
        // int extents, not static C<>), so the tail guard is a plain
        // if -- the static-buffer property of the peel does not need
        // constexpr bounds, only statically named halves.
        if ((size<4>(tKDgKD) % 2) != 0) {
          int D = int(size<4>(tKDgKD)) - 1;
          copy(copy_q, tQgQ(_, _, _, D), tQrQ);
          reorder(tQrQ, tSrQ);
          cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
        }
      }
"""


def main():
    path = SRC / "mx_paged_decode_mainloop.hpp"
    text = path.read_text(encoding="utf-8")
    if text.count(MARK) != 0:
        print("PATCH_ALREADY_APPLIED (r212ab marker present)")
        sys.exit(1)
    if text.count("r212aa TEMPLATE-SPLIT") < 1:
        print("PATCH_BASE_MISS (r212aa split not applied)")
        sys.exit(1)
    for name, m in (("START", START), ("LOOP_HEAD", LOOP_HEAD),
                    ("GEMM_LINE", GEMM_LINE)):
        n = text.count(m)
        if n != 1:
            print(f"PATCH_ANCHOR_MISS {name}: count={n}")
            sys.exit(1)
    # Slice the original loop verbatim: LOOP_HEAD through the first
    # "\n      }\n" after GEMM_LINE (the loop's closing brace).
    s = text.find(LOOP_HEAD)
    g = text.find(GEMM_LINE)
    assert s != -1 and g != -1 and s < g
    e = text.find("\n      }\n", g)
    if e == -1:
        print("PATCH_ANCHOR_MISS LOOP_CLOSE")
        sys.exit(1)
    e += len("\n      }\n")
    loop = text[s:e]
    if loop.count("cute::gemm(mma_qk") != 1 or not loop.endswith("}\n"):
        print("PATCH_SLICE_SANITY_FAIL")
        sys.exit(1)
    new = text[:s] + WRAP_OPEN + loop + PIPE + text[e:]
    path.write_text(new, encoding="utf-8", newline="\n")
    print("PATCHED mx_paged_decode_mainloop.hpp (GEMM-1 loop wrapped +"
          " peeled pipeline installed)")
    print("P212AB_PATCH_OK")


if __name__ == "__main__":
    main()
