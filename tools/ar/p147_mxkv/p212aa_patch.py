#!/usr/bin/env python3
"""p212aa_patch.py -- r212aa: TEMPLATE-SPLIT the probe apparatus out of
the shipping instantiation (the principled attractor escape).

r212y proved the +34% GEMM-1 codegen attractor is a stable basin (4
wheels), and r212z proved the K-overlap mechanics but collapsed on
runtime-parity operand demotion. The basin taxes are the biggest known
recoverable chunks (~290 us mxfp4 m0, ~450 us mxfp8 m8 oracle vs the
r212q-era region) and every future cross-wheel absolute needs a clean
shipping instantiation. Split:

  operator()  -> thin dispatcher: vlane_probe==0 -> body<false> (the
                 SHIPPING path -- every probe read folds to a
                 constant: sever gates false, probe args neutral,
                 barrier always, softmax always), else body<true>
                 (the r200c/r212 apparatus, runtime selection as
                 before -- m11/m17/... cells unchanged).

body<false> contains ZERO probe branches -- structurally cleaner than
even the r212u pre-attractor state (probes were present-but-untried
there). m0/m8 ride body<false> BY CONSTRUCTION (the oracle now
measures the shipping path). Cells 0/11/17 + m8 oracle:
  m8 in band (<= +-6% of 1320.5)  -> attractor ESCAPED; absolutes
         regain trust (compare m0 vs r212u clean 718.0)
  m8 still +34                     -> basin survived the split;
         escalate structure (peel D-loop compile-time parity next)
"""
import pathlib
import sys

SRC = pathlib.Path("/root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx")

ANCHOR_SIG = """  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,    // (q,d)
      TensorK2D const& K_2D,    // (k, k_bytes) u8
      TensorKS2D const& KS_2D,  // (k, k_groups) u8
      TensorV2D const& V_2D,    // (v_bytes, k) u8
      TensorVS2D const& VS_2D,  // (v_groups, k) u8
      FragA& tArA,              // Output accumulator (q,v)
      FragARow& tA_max,         // Softmax row-wise max accumulator
      FragARow& tA_sum,         // Softmax row-wise sum accumulator
      QVCoord blk_qv,           // WG tile indices: (Q,V)
      int const& idx_b,         // WG tile indices: (B)
      int blk_k0,               // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord) {
    using namespace sycl::ext::oneapi::this_work_item;
"""

SIG = """  // r212aa TEMPLATE-SPLIT dispatch: vlane_probe==0 runs the
  // SHIPPING body (kInstr=false -- every probe read folds to a
  // constant at compile time; zero probe branches survive in the
  // GEMM regions). Any probe value runs the instrumented body
  // (kInstr=true -- the r200c/r212 apparatus, runtime cell
  // selection as before). m0 and the m8 mxfp8 oracle ride the
  // shipping body BY CONSTRUCTION.
  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,    // (q,d)
      TensorK2D const& K_2D,    // (k, k_bytes) u8
      TensorKS2D const& KS_2D,  // (k, k_groups) u8
      TensorV2D const& V_2D,    // (v_bytes, k) u8
      TensorVS2D const& VS_2D,  // (v_groups, k) u8
      FragA& tArA,              // Output accumulator (q,v)
      FragARow& tA_max,         // Softmax row-wise max accumulator
      FragARow& tA_sum,         // Softmax row-wise sum accumulator
      QVCoord blk_qv,           // WG tile indices: (Q,V)
      int const& idx_b,         // WG tile indices: (B)
      int blk_k0,               // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord) {
    if (params.vlane_probe == 0) {
      this->template body<has_large_surface, false, QVCoord>(
          Q_2D, K_2D, KS_2D, V_2D, VS_2D, tArA, tA_max, tA_sum, blk_qv,
          idx_b, blk_k0, blk_k1, total_blk, thr_id, seq_len,
          full_tile_offset, discard_seq_coord);
    } else {
      this->template body<has_large_surface, true, QVCoord>(
          Q_2D, K_2D, KS_2D, V_2D, VS_2D, tArA, tA_max, tA_sum, blk_qv,
          idx_b, blk_k0, blk_k1, total_blk, thr_id, seq_len,
          full_tile_offset, discard_seq_coord);
    }
  }

  template <bool has_large_surface, bool kInstr, typename QVCoord>
  CUTLASS_DEVICE void body(
      TensorQ2D const& Q_2D,    // (q,d)
      TensorK2D const& K_2D,    // (k, k_bytes) u8
      TensorKS2D const& KS_2D,  // (k, k_groups) u8
      TensorV2D const& V_2D,    // (v_bytes, k) u8
      TensorVS2D const& VS_2D,  // (v_groups, k) u8
      FragA& tArA,              // Output accumulator (q,v)
      FragARow& tA_max,         // Softmax row-wise max accumulator
      FragARow& tA_sum,         // Softmax row-wise sum accumulator
      QVCoord blk_qv,           // WG tile indices: (Q,V)
      int const& idx_b,         // WG tile indices: (B)
      int blk_k0,               // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord) {
    using namespace sycl::ext::oneapi::this_work_item;
"""

ANCHOR_BOOLS = """    // r212y ATTRACTOR RE-ROLL: the probe gates' comparison chains in
    // the GEMM regions pinned a bad codegen layout for three
    // consecutive wheels (m8 +33.8/+34.1/+34.5 pct after r212v's
    // null gate edit; k_lane 595.5/600.1 vs the clean r212u pool
    // 338). Hoist the sever conditions to precomputed bools --
    // semantically identical, fresh layout dice for both regions.
    const int vprobe = params.vlane_probe;
    const bool kp_sever = vprobe >= 6 && vprobe != 17 && vprobe != 20;
    const bool vp_sever =
        vprobe >= 2 && vprobe != 4 && vprobe != 5 && vprobe != 6 &&
        vprobe != 17;
"""

BOOLS = """    // r212y hoist (kept): sever conditions precomputed before the K
    // loop. r212aa TEMPLATE-SPLIT: under kInstr=false the kInstr
    // conjunct folds both gates constant-false -- the severed arms
    // die at compile time and the shipping GEMM regions carry zero
    // probe branches.
    const int vprobe = params.vlane_probe;
    const bool kp_sever =
        kInstr && vprobe >= 6 && vprobe != 17 && vprobe != 20;
    const bool vp_sever =
        kInstr && vprobe >= 2 && vprobe != 4 && vprobe != 5 &&
        vprobe != 6 && vprobe != 17;
"""

ANCHOR_REG = """          mx::dequant_reg<true, true>(
              ks_surf,
              tKrB,
              tKrBq,
              tKBgK_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0);"""

REG = """          mx::dequant_reg<true, true>(
              ks_surf,
              tKrB,
              tKrBq,
              tKBgK_cache(_, _, _, D),
              kInstr && params.vlane_probe == 9,
              params.direct_bf16 != 0);"""

ANCHOR_DTILE = """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tSrK,
              tKDgKD_cache(_, _, _, D),
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              (params.vlane_probe >= 11 && params.vlane_probe <= 16) ||
                  params.vlane_probe == 20,
              params.vlane_probe);"""

DTILE = """          mx::dequant_dtile<true, true>(
              kb_surf,
              ks_surf,
              tSrK,
              tKDgKD_cache(_, _, _, D),
              kInstr && params.vlane_probe == 9,
              params.direct_bf16 != 0,
              kInstr &&
                  ((params.vlane_probe >= 11 &&
                    params.vlane_probe <= 16) ||
                   params.vlane_probe == 20),
              kInstr ? params.vlane_probe : 0);"""

ANCHOR_STAGE = """      bool stage_on = params.vlane_probe != 1 && params.vlane_probe != 3 &&
          params.vlane_probe != 7 && params.vlane_probe != 8;"""

STAGE = """      // r212aa: under kInstr=false stage_on folds constant-true.
      bool stage_on =
          !kInstr || (params.vlane_probe != 1 && params.vlane_probe != 3 &&
                      params.vlane_probe != 7 && params.vlane_probe != 8);"""

ANCHOR_BARRIER = """        if (params.vlane_probe != 4) {
          barrier();
        }"""

BARRIER = """        if constexpr (kInstr) {
          if (params.vlane_probe != 4) {
            barrier();
          }
        } else {
          barrier();
        }"""

ANCHOR_DBG = """      bool dbg_on = dbg != nullptr && dbg_tile < 4;"""

DBG = """      // r212aa: debug lane folds away in the shipping body.
      bool dbg_on = kInstr && dbg != nullptr && dbg_tile < 4;"""

ANCHOR_P = """      if (params.vlane_probe == 8) {
        // r200g probe 8: const-P — skip the softmax math entirely
        // (max/sum stay neutral, epilogue LSE garbage). Timing floor
        // only: severed chains understate.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          tSrS(i) = ElementS(1.f);
        }
      } else {
        softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      }"""

P = """      if constexpr (kInstr) {
        if (params.vlane_probe == 8) {
          // r200g probe 8: const-P — skip the softmax math entirely
          // (max/sum stay neutral, epilogue LSE garbage). Timing floor
          // only: severed chains understate.
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            tSrS(i) = ElementS(1.f);
          }
        } else {
          softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum,
                  tArA);
        }
      } else {
        softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      }"""

ANCHOR_PREP = """        mx::dequant_vslm_prep(
            vs_surf,
            v_krow_base,
            vslm_kr,
            vslm_line,
            params.vlane_probe == 5);"""

PREP = """        mx::dequant_vslm_prep(
            vs_surf,
            v_krow_base,
            vslm_kr,
            vslm_line,
            kInstr && params.vlane_probe == 5);"""

ANCHOR_VSLM = """              v_byte_base * kElemsPerByteV + VV * kVStepElems,
              params.vlane_probe == 5,
              params.vlane_probe == 9,
              params.direct_bf16 != 0,
              (params.vlane_probe >= 11 && params.vlane_probe <= 16) ||
                  params.vlane_probe == 17,
              params.vlane_probe);"""

VSLM = """              v_byte_base * kElemsPerByteV + VV * kVStepElems,
              kInstr && params.vlane_probe == 5,
              kInstr && params.vlane_probe == 9,
              params.direct_bf16 != 0,
              kInstr &&
                  ((params.vlane_probe >= 11 &&
                    params.vlane_probe <= 16) ||
                   params.vlane_probe == 17),
              kInstr ? params.vlane_probe : 0);"""


def main():
    path = SRC / "mx_paged_decode_mainloop.hpp"
    text = path.read_text(encoding="utf-8")
    if text.count("r212aa TEMPLATE-SPLIT") != 0:
        print("PATCH_ALREADY_APPLIED (r212aa TEMPLATE-SPLIT)")
        sys.exit(1)
    edits = (
        ("SIG", ANCHOR_SIG, SIG),
        ("BOOLS", ANCHOR_BOOLS, BOOLS),
        ("REG", ANCHOR_REG, REG),
        ("DTILE", ANCHOR_DTILE, DTILE),
        ("STAGE", ANCHOR_STAGE, STAGE),
        ("BARRIER", ANCHOR_BARRIER, BARRIER),
        ("DBG", ANCHOR_DBG, DBG),
        ("P", ANCHOR_P, P),
        ("PREP", ANCHOR_PREP, PREP),
        ("VSLM", ANCHOR_VSLM, VSLM),
    )
    for name, anchor, _ in edits:
        n = text.count(anchor)
        if n != 1:
            print(f"PATCH_ANCHOR_MISS {name}: count={n}")
            sys.exit(1)
    for _, anchor, new in edits:
        text = text.replace(anchor, new)
    path.write_text(text, encoding="utf-8", newline="\n")
    print("PATCHED mx_paged_decode_mainloop.hpp (10 edits)")
    print("P212AA_PATCH_OK")


if __name__ == "__main__":
    main()
