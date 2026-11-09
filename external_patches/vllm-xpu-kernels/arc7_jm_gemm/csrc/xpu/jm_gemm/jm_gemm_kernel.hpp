#pragma once

// JM (joint-matrix / DPAS tensor-core) skinny-M weight-only GEMM kernels
// (P6.34, arc7 wheel).
//
// The arc6 scalar kernel (skinny_gemm_kernel.hpp, P6.33) proved the
// bytes-optimal traffic layout but is ALU-bound at ~300 GB/s: every weight
// byte pays a ~10-15 op f32 decode plus two f32 FMAs on the vector pipe.
// The r128a bar table put the stock oneDNN in-graph routes at 490-530 GB/s
// (f8 big shapes) and 780-885 GB/s on the L2-resident small shapes, with
// the f4 lane at only ~300-380 GB/s of ACTUAL bytes (oneDNN weight
// decomposition overhead -- the softest target). Device envelope (r128a2):
// DRAM read 589 GB/s, L2-resident read 992 GB/s, D2D copy 535 GB/s r+w.
// The scalar family cannot reach those bars; the tensor-core family is the
// P6.34 directive (all-around superiority on every shape x lane x M cell).
//
// This kernel keeps the arc6 traffic law -- one thread streams ONE weight
// column K-contiguously, the M x K activation panel stays L2-resident,
// fixed-order split-K with raw-fp32 partials [S][8][N], epilogue scale
// applied exactly once -- and moves the math onto the Xe2 matrix engine
// through the OFFICIAL sycl::ext::oneapi::experimental::matrix::
// joint_matrix (SPIR-V CooperativeMatrixKHR) path, M8xN16xK16
// half x half -> f32:
//
//   * Fragment layouts are the MEASURED r128e2 law (B70, Intel's own
//     lowering; matches the corrected cutlass mma_traits_xe.hpp algebra):
//     a 16-thread subgroup owns output tile [m0, m0+8) x [n0, n0+16) and
//     thread t (lane) holds
//       A: 8 halves -- slot m = row, at k = k0 + t (lane = k)
//       B: 16 halves -- slot k, of column n0 + t (lane = column)
//       C: 8 f32 -- slot m = row, of column n0 + t (lane = column)
//     The cut-4 raw inline-asm route is retired: cutlass's .decl text
//     with "rw" operands does NOT reproduce this placement on this
//     compiler (the r128d sweep2 product law is lane-local with
//     parity-coupled k rotations -- unmatchable by any repack; the
//     defect is register materialization of raw asm operands, not the
//     instruction). Cuts 1-3 were wrong-B layout derivations, cut 4 the
//     broken decode + this binding (GATE-1 cuts 1-4, 2026-11-09).
//   * The B layout makes the weight repack a near-identity reshape of
//     [N, K]: lane t's fragment = column n0+t's 16 CONSECUTIVE k codes
//     -- one uint4 (f8, [N/16, K/16, 256] tiles, HALF the cut-2/4
//     VNNI-pad bytes) or one uint2 (f4, [N/16, K/16, 128] tiles)
//     contiguous load, decoded with the proven pair decoders. A: each
//     lane does 8 k-strided 2B gathers down the rows (per-row 32B
//     contiguous across the subgroup, panel L2-resident); C stores one
//     2B element per row per lane (per-row 32B contiguous per subgroup).
//   * Packed pair decoders: f8/f4 bytes decode to f16 pairs with
//     SIMD-in-register 32-bit integer ALU (~7-9 ops per PAIR), all
//     lane-isolated (no cross-lane carries; algebra in the decoder
//     comments). e4m3/e5m2/e2m1 values are exact in f16, products exact
//     into the f32 accumulator -- the same numeric class as oneDNN's own
//     f16 DPAS joint-dtype route (gate tolerance 2e-3 rel vs f64).
//   * MX group scales by accumulator flush: a 16-k mad step always sits
//     inside ONE 32-k e8m0 group (k-steps are 16-aligned, slices are
//     group-whole), so scales apply per-group as
//       d_tot += 2^(e8m0) * d_grp
//     in f32 -- exact over the FULL e8m0 range (an f16 fold would
//     overflow at 2^120 probe scales), and cheaper than per-element muls.
//     Checkpoint per-tensor/per-channel scales stay epilogue folds
//     (constant along k, applied once after the fixed-order sum).
//   * Fragment op LAW (r128e2/r128e3, both runtimes -- the ONLY sequence
//     that survives JIT on this stack): per 16-k step, stage the decoded
//     halves to GLOBAL scratch, subgroup barrier, joint_matrix_load A/B
//     (row-major 8x16 / col-major 16x16, stride 16), barrier, apply-zero
//     BOTH accumulators, ONE mad, apply-read into a per-lane f32 side
//     total. Broken alternatives (banked cuts, 2026-11-09): apply-WRITES
//     on A/B fragments and loads from LOCAL memory silently produce dead
//     kernels; a second mad consuming un-reloaded fragments hangs the
//     device; D == C aliasing yields zeros (r128e). The staging traffic
//     (48 B/lane/step, L1-resident) is the price of the only working
//     matrix-engine path.
//
// Launch: formula-derived for any shape (all-model-types clause): tiles =
// ceil(N/16) subgroups x splits, splits driven by a subgroup floor with
// the arc6 group-divisor + kc>=1024 laws, golden-ratio tile swizzle.
// Probe knobs ARC_JM_SGF / ARC_JM_SWZ / ARC_JM_VAR (env, read per call)
// sweep the floor, swizzle, and kernel variant; winners get baked as
// constants in the certified wheel (the arc6 v2.4 -> v2.5 pattern: no env
// reads on the serving path). r128k2 (2026-11-09): a fragment load whose
// ADDRESS varies across loop iterations reads zeros (swap-based
// ping-pong mis-lowering) and standalone-harness numerics are a codegen
// lottery -- so V14 (fixed-pointer unrolled ping-pong) ships beside the
// proven V5 as a per-call template choice, and the wheel gate alone
// certifies.
//
// r128 windows measure JM vs stock vs scalar per shape x lane x M with
// the 50-call amortized XPUGraph replay probe (P6.33 measurement law:
// eager loops are host-rate illusions). Superiority bar: JM < stock on
// EVERY cell; ceiling-bound cells close as tie-at-ceiling with the math
// banked (plan P6.34).

#include <torch/all.h>
#include <sycl/sycl.hpp>
#include <cstdlib>

#include <cstdint>

namespace vllm::xpu::jm {

// ---------------------------------------------------------------------------
// Tensor-core fragments: the OFFICIAL SPIR-V CooperativeMatrixKHR path
// (sycl::ext::oneapi::experimental::matrix::joint_matrix), M8xN16xK16
// half x half -> f32 per 16-thread subgroup. Fragment element placement
// is the r128e2 measured law (B70, oneAPI 2026.0.0, both runtimes):
// A lane = k / slot = row, B lane = column / slot = k, C lane = column /
// slot = row. Only the load-from-global-memory + apply(accumulator) +
// single-mad-per-load sequence works (see the fragment op LAW below);
// the cutlass mma_xe.hpp raw-asm text (cut 4) does not bind "rw"
// operands to these registers and is retired.
// ---------------------------------------------------------------------------
namespace jmx = sycl::ext::oneapi::experimental::matrix;

using JmA =
    jmx::joint_matrix<sycl::sub_group, sycl::half, jmx::use::a, 8, 16,
                      jmx::layout::row_major>;
using JmB =
    jmx::joint_matrix<sycl::sub_group, sycl::half, jmx::use::b, 16, 16,
                      jmx::layout::col_major>;
using JmC = jmx::joint_matrix<sycl::sub_group, float, jmx::use::accumulator,
                              8, 16, jmx::layout::dynamic>;

// ---------------------------------------------------------------------------
// Packed pair decoders: one u32 holding two source elements (lanes [7:0]
// and [23:16]) -> one u32 holding two f16 (same lane positions). All ops
// are 32-bit integer, branchless, and LANE-ISOLATED: every intermediate
// stays below its 16-bit lane boundary so no carry crosses lanes.
// ---------------------------------------------------------------------------

// Lane-pair predicate broadcast: p holds 0/1 predicate bits at [0] and
// [16]; (p << 16) - p in wrapping u32 is the EXACT per-lane 0xFFFF mask
// (p=1 -> 0x0000FFFF, p=0x10000 -> 0xFFFF0000, p=0x10001 -> 0xFFFFFFFF,
// 0 -> 0). A multiply broadcast (p * 0xFFFF) is NOT lane-isolated
// (0x00010001 * 0xFFFF == 0xFFFEFFFF) -- that cross-lane contamination
// was the first-cut GATE-1 failure (2026-11-09).
inline std::uint32_t jm_lane_mask(std::uint32_t p) {
  return (p << 16) - p;
}

// e4m3fn byte pair -> f16x2. Per lane: byte = s|eeee|mmm.
//   lane  = byte & 0x7F (strip sign); s moves to f16 bit 15 via << 8.
//   e4m3 bias 7, f16 bias 15: normal bits = ((e + 8) << 10) | (m << 7)
//     == ((lane << 7) + 0x2000) exactly, since lane<<7 = e<<10 | m<<7 and
//     e + 8 <= 23 keeps the per-lane sum below 0x8000 (no lane carry).
//   e == 0 (subnormal m * 2^-9, m in 1..7): these live ABOVE f16's max
//     subnormal (1023 * 2^-24) -- they are NORMAL f16 values with a
//     normalized mantissa, so no linear bit pattern exists; the fixup is
//     a 3-way per-lane select (m >= 4: 0x2000 | (m&3) << 8; m in 2..3:
//     0x1C00 | (m&1) << 9; m == 1: 0x1800; m == 0 -> +0 via the value
//     mask). f16 subnormal scaling is 2^-24 (mantissa/1024 * 2^-14) --
//     the (m << 5) * 2^-14 form of cuts 1-2 landed at 2^-19, the
//     third-cut GATE-1 failure.
//   e4m3fn's only non-finite code (e=15, m=7 -> 448) maps finite, the
//   arc6 decoder stance (quantized weights never carry it).
// Exhaustively checked over all 256 codes on host (r128b gate).
inline std::uint32_t e4m3_pair_f16x2(std::uint32_t x) {
  const std::uint32_t lane = x & 0x007F007Fu;
  const std::uint32_t sg = (x & 0x00800080u) << 8;
  const std::uint32_t naive = (lane << 7) + 0x20002000u;
  // e >= 1 <=> lane >= 0x08; val != 0 <=> lane >= 0x01 (sums stay < 0x100)
  const std::uint32_t pe1 =
      ((lane + 0x00780078u) >> 7) & 0x00010001u;
  const std::uint32_t pv1 =
      ((lane + 0x007F007Fu) >> 7) & 0x00010001u;
  const std::uint32_t zall = jm_lane_mask(pe1 ^ 0x00010001u);  // e == 0
  const std::uint32_t fall = jm_lane_mask(pv1);                // val != 0
  const std::uint32_t m = lane & 0x00070007u;
  // m-range predicates (per-lane, isolated; see the comment above). All
  // ADD-based: a paired subtract (t - 2) would let a low-lane borrow
  // corrupt the high lane (t_lo = 0 eats t_hi's bits).
  const std::uint32_t p4 = ((m + 0x00040004u) >> 3) & 0x00010001u;
  const std::uint32_t t = m & 0x00060006u;            // e4m3 e bits
  const std::uint32_t pr = ((t + 0x00020002u) >> 2) & 0x00010001u;
  const std::uint32_t t4 = (t >> 2) & 0x00010001u;
  const std::uint32_t p23 = pr & (t4 ^ 0x00010001u);  // t == 2 exactly
  const std::uint32_t p1 = ((m + 0x000E000Eu) >> 3) & 0x00010001u;
  const std::uint32_t hi = 0x20002000u | ((m & 0x00030003u) << 8);
  const std::uint32_t mid = 0x1C001C00u | ((m & 0x00010001u) << 9);
  const std::uint32_t fx =
      ((hi & jm_lane_mask(p4)) | (mid & jm_lane_mask(p23)) |
       (0x18001800u & jm_lane_mask(p1))) & fall;
  return (naive ^ ((naive ^ fx) & zall)) | sg;
}

// e5m2 byte pair -> f16x2. Per lane: byte = s|eeeee|mm.
//   e5m2 bias 15 == f16 bias 15: normal bits = e<<10 | m<<8 == lane << 8
//   EXACTLY (no rebias add). e == 0 (subnormal m * 2^-16 = (m << 8) *
//   2^-14): true f16 bits are the raw SUBNORMAL mantissa m << 8, except
//   m == 0 -> +0.
//   Inf/NaN codes (e == 31) map finite -- weights never carry them.
inline std::uint32_t e5m2_pair_f16x2(std::uint32_t x) {
  const std::uint32_t lane = x & 0x007F007Fu;
  const std::uint32_t sg = (x & 0x00800080u) << 8;
  const std::uint32_t naive = lane << 8;
  // e >= 1 <=> lane >= 0x04; val != 0 <=> lane >= 0x01
  const std::uint32_t pe1 =
      ((lane + 0x007C007Cu) >> 7) & 0x00010001u;
  const std::uint32_t pv1 =
      ((lane + 0x007F007Fu) >> 7) & 0x00010001u;
  const std::uint32_t zall = jm_lane_mask(pe1 ^ 0x00010001u);
  const std::uint32_t fx = (lane << 8) & jm_lane_mask(pv1);
  return (naive ^ ((naive ^ fx) & zall)) | sg;
}

// e2m1 nibble pair -> f16x2. Per lane: nibble = s|ee|m (input lanes hold
// nibbles at [3:0]/[19:16]).
//   e2m1 bias 1, f16 bias 15: normal bits = ((e + 14) << 10) | (m << 9)
//     == (lane << 9) + 0x3800 (lane<<9 = e<<10 | m<<9; e <= 3 keeps the
//     per-lane sum below 0x8000).
//   e == 0: m == 0 -> +0; m == 1 -> 0.5 = 0x3800 (fixup masked like f8).
//   e >= 1 <=> lane >= 2 and val != 0 <=> lane >= 1: test via bit 3 of
//   lane + 6 / lane + 7 (7 + 7 = 14 < 16, no nibble carry).
inline std::uint32_t e2m1_pair_f16x2(std::uint32_t y) {
  const std::uint32_t lane = y & 0x00070007u;
  const std::uint32_t sg = (y & 0x00080008u) << 12;
  const std::uint32_t naive = (lane << 9) + 0x38003800u;
  const std::uint32_t pe1 = ((lane + 0x00060006u) >> 3) & 0x00010001u;
  const std::uint32_t pv1 = ((lane + 0x00070007u) >> 3) & 0x00010001u;
  const std::uint32_t zall = jm_lane_mask(pe1 ^ 0x00010001u);
  const std::uint32_t fx = 0x38003800u & jm_lane_mask(pv1);
  return (naive ^ ((naive ^ fx) & zall)) | sg;
}

// e8m0 -> 2^(b - 127): group scales; one exp2 per column per group.
inline float e8m0_to_float(std::uint32_t b) {
  return sycl::exp2(static_cast<float>(static_cast<int>(b) - 127));
}

// bf16 -> f16 bits (exact for f16-normal-range values; the activation
// panel class. Extreme-exponent bf16 flushes the way the f16 cvt does).
inline std::uint16_t bf16_to_f16_bits(std::uint16_t b) {
  const float f = sycl::bit_cast<float>(
      static_cast<std::uint32_t>(b) << 16);
  return sycl::bit_cast<std::uint16_t>(static_cast<sycl::half>(f));
}

// ---------------------------------------------------------------------------
// Stage 1: joint-matrix GEMM over one K-slice per 16-thread subgroup.
// ---------------------------------------------------------------------------
enum class WeightFormat : int { kF8E4m3 = 0, kF8E5m2 = 1, kF4E2m1 = 2 };

// Weight layout: column-contiguous tiles (python repack helper runs ONCE
// at weight load; the repack zero-pads N up to a 16 multiple and pads the
// group-scale columns to match). The r128e2 B layout makes each tile a
// near-identity reshape of the 16x16 [N, K] block:
//   f8: [N/16, K/16, 256] u8 -- tile byte (c, k) at 16c + k; lane t's B
//       fragment = the 16 CONSECUTIVE bytes at tile + 16t (column n0+t's
//       k-run; one uint4 load, HALF the cut-2/4 VNNI-pad bytes).
//   f4: [N/16, K/16, 128] u8 -- nibble (c, k) at byte 8c + (k>>1),
//       nibble slot = k&1 (lo = even k); lane t's 8-byte block at
//       tile + 8t is its whole fragment (one uint2 load).
//   scales (GROUP32): byte (n, g) at bsc_[n * sc_ns + g * sc_gs], N padded.
//
// Scaling: GROUP32 lanes flush the e8m0 group scales into the accumulator
// per 32-k group (ONE output column per lane -- one scale byte);
// checkpoint lanes fold pc/pt at the epilogue exactly once (direct: stage
// 1; split: stage 2 after the fixed-order sum).
// PP (ARC_JM_VAR): true = the SLM form (r128k7 V24, 2026-11-09): A loads
// direct from the launcher-built k-blocked f16 activation buffer a16_
// (never kernel-written -- immune to the scratch visibility lottery), B
// stages per-step into per-subgroup SLM slots; false = the gate-proven V5
// same-step global-scratch sequence (lottery control, build6 wheel
// byte-exact). Both variants ship in the wheel; only the wheel gate
// certifies (r128k2 codegen-lottery law).
template <WeightFormat WF, bool GROUP32, bool A_BF16, bool DIRECT, bool PP>
class JmGemmStage1Kernel {
 public:
  JmGemmStage1Kernel(
      const void* a,             // [M, K] contiguous activations
      const sycl::half* a16,     // [K/16, 8, 16] k-blocked f16 (PP: A-direct)
      const std::uint8_t* b,     // column-contiguous weight tiles (above)
      const std::uint8_t* bsc,   // e8m0 group scales (GROUP32)
      sycl::half* scratch,  // [n_subgroups][768B] (V5 arm staging)
      sycl::local_accessor<sycl::half, 1> slm,  // [1024] per-WG B slots (PP)
      float* partials,           // [S][8][N] fp32 (not DIRECT)
      void* out,                 // [M, N] activation dtype (DIRECT)
      const float* bias,         // [N] (DIRECT, may be nullptr)
      const float* pc_scale,     // [N] per-channel or nullptr
      float pt_scale,            // per-tensor scalar (1.f if none)
      int m_rows, int n_cols, int k_elems, int splits, int k_per_split,
      int64_t sc_n_stride, int64_t sc_g_stride, bool swizzle,
      int64_t pc_stride)         // checkpoint-scale N stride (0 = scalar)
      : a_(static_cast<const std::uint8_t*>(a)),
        a16_(a16),
        b_(b),
        bsc_(bsc),
        scratch_(scratch),
        slm_(std::move(slm)),
        partials_(partials),
        out_(static_cast<std::uint8_t*>(out)),
        bias_(bias),
        pc_(pc_scale),
        pt_(pt_scale),
        sc_ns_(sc_n_stride),
        sc_gs_(sc_g_stride),
        m_(m_rows),
        n_(n_cols),
        k_(k_elems),
        s_(splits),
        kc_(k_per_split),
        swzl_(swizzle),
        pcs_(pc_stride) {}

  void operator()(sycl::nd_item<1> it) const {
    // Subgroup geometry: local size is a multiple of 16, so global id maps
    // cleanly to (subgroup, lane). The 16-thread subgroup IS the matrix
    // engine unit: fragment ops (apply / load / mad) are
    // subgroup-collective.
    auto sgc = it.get_sub_group();
    const int gid = static_cast<int>(it.get_global_linear_id());
    const int lane = static_cast<int>(sgc.get_local_linear_id());
    const int sg = gid >> 4;
    const int tiles = n_ >> 4;   // padded N is a 16 multiple (repack)
    // Golden-ratio tile swizzle (arc6 v2.5 law): (t * A) % tiles is a
    // bijection (A = 0x9E3779B1 prime, tiles < A); every tile owned once,
    // deterministic.
    const std::uint32_t tt = static_cast<std::uint32_t>(sg % tiles);
    const int ctile =
        swzl_ ? static_cast<int>((tt * 2654435761ull) % tiles)
              : static_cast<int>(tt);
    const int n0 = ctile * 16;
    const int slice = sg / tiles;
    if (slice >= s_) return;  // grid padded to the local-size multiple

    const int k_lo = slice * kc_;
    const int k_hi = sycl::min(k_lo + kc_, k_);
    // k_lo/kc_/k_ are group-aligned by the launcher (K % 32 == 0 checked
    // at the interface; splits divide the group count), so every 16-k mad
    // step lies inside ONE 32-k e8m0 group: (k0 >> 5) is its group id.
    // One column-contiguous tile per (n-tile, k-step): stride 256 B (f8) /
    // 128 B (f4); lane t's codes sit at +16t / +8t inside it.
    const std::uint8_t* tbase =
        b_ + static_cast<std::size_t>(ctile) * (k_ >> 4) *
                 ((WF == WeightFormat::kF4E2m1) ? 128 : 256);

    // Dual variant, selected per call by ARC_JM_VAR (launcher; default 14,
    // 5 = the gate-proven V5 sequence). r128k2 (2026-11-09): the swap-based
    // ping-pong reads ZERO fragments -- on this stack a fragment load whose
    // address varies across loop iterations is mis-lowered, and
    // barriers/zero-applies do not rescue it; standalone-harness numerics
    // are a codegen lottery, so BOTH variants ship in the wheel and only
    // the wheel gate certifies. PP unrolls by 2 so each load SITE is a
    // lexically constant pointer; staging runs one step ahead with OOB
    // guards; every half-step keeps the V5-exact fragment op sequence.
    JmA ja;
    JmB jb;
    JmC c0, c1;
    if constexpr (PP) {
      // The SLM form (r128k7 V24, 2026-11-09 evening). The global-scratch
      // store->barrier->load round trip is the whole V5 wall (r128k6
      // decomposition: A-staging ~3.3 us/step, B-staging ~1.55 us/step of
      // the 5.4 us/step total, while V22 all-direct runs 585 ns/step).
      // Here: A loads DIRECT from a16_ -- the launcher-built k-blocked f16
      // activation buffer x16[(k>>4)*128 + m*16 + t], never kernel-written
      // (r128k2 Law-1-immunity class; the ladder proved direct loads from
      // such memory exact at any varying address). B keeps the same-step
      // discipline but stages into PER-SUBGROUP SLM slots (4 sg x 256
      // halves per work group): r128k7 refuted r128e3's "SLM fragment
      // loads are dead" -- V23/V24 are EXACT (8.7e-7) on every shape,
      // hardware-ordered with one barrier, and cost only ~130 ns/step over
      // the all-direct floor. No post-load barrier (V23 vs V24: identical
      // numerics AND perf). The swap-based ping-pong family (V6..V16,
      // incl. the fixed-pointer V14 this arm used to carry) reads exact
      // ZEROS in every binary -- cross-iteration scratch dependencies are
      // dropped by the lowering; V14 is retired.
      const sycl::half* ap =
          a16_ + (static_cast<std::size_t>(k_lo) >> 4) * 128;
      const int sgl = static_cast<int>(sgc.get_group_linear_id());
      sycl::half* sbs =
          slm_.get_pointer().get() + static_cast<std::size_t>(sgl) * 256;
      float tot[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
        stage_b(sbs, tbase, k0, lane);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_load(
            sgc, ja,
            sycl::multi_ptr<
                sycl::half,
                sycl::access::address_space::global_space>(
                const_cast<sycl::half*>(ap +
                                        ((k0 - k_lo) >> 4) * 128)),
            16);
        jmx::joint_matrix_load(
            sgc, jb,
            sycl::multi_ptr<
                sycl::half,
                sycl::access::address_space::local_space>(sbs),
            16);
        jmx::joint_matrix_apply(sgc, c0, [](float &e) { e = 0.0f; });
        jmx::joint_matrix_apply(sgc, c1, [](float &e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        // GROUP32 scales each step by its 32-k group's e8m0 byte
        // (k0 >> 5); checkpoint lanes scale by 1 and fold pc/pt at the
        // epilogue (store_out).
        const float s =
            GROUP32 ? e8m0_to_float(
                          bsc_[static_cast<std::size_t>(n0 + lane) * sc_ns_ +
                               static_cast<std::size_t>(k0 >> 5) * sc_gs_])
                    : 1.0f;
        int i = 0;
        jmx::joint_matrix_apply(sgc, c1, [&](float &e) { tot[i++] += s * e; });
      }
      epilogue_arr(tot, slice, n0, lane);
    } else {
      // PP=false: the gate-proven V5 sequence, BYTE-EXACT (build6 wheel,
      // r128b cut 7: GATE-1 40/40, deterministic, poison-immune). The
      // lottery control -- if THIS arm fails a gate on a wheel that
      // carries it unchanged, the failure is codegen, not source.
      // Global staging scratch, per GLOBAL subgroup (concurrent work
      // groups must not share slots): A 8x16 halves + B 16x16 halves
      // = 768 B.
      sycl::half* sa = scratch_ + static_cast<std::size_t>(sg) * 384;
      sycl::half* sb = sa + 128;
      // Per 16-k step, the r128e2/r128e3-proven fragment sequence EXACTLY:
      // stage halves to global memory, barrier, load A/B, barrier, zero
      // both accumulators, ONE mad, apply-read into the side total. On this
      // stack nothing else survives JIT: apply-WRITES on A/B fragments
      // silently produce dead kernels, a second mad consuming stale
      // un-reloaded fragments hangs the device, and D == C aliasing yields
      // zeros (r128e/r128e3 cuts 1-3, 2026-11-09). r128e3's "any load from
      // local memory is dead" claim was REFUTED by r128k7 (V23/V24 exact)
      // -- the SLM path works; this V5 form stays as the lottery control.
      // The staging traffic (48 B/lane/step, L1-resident) is the measured
      // ~4.9 us/step wall (r128k6); weights still stream K-contiguously
      // per column (traffic law intact).
      float tot[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
        stage_a(sa, k0, lane);
        stage_b(sb, tbase, k0, lane);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_load(
            sgc, ja,
            sycl::multi_ptr<sycl::half,
                            sycl::access::address_space::global_space>(sa),
            16);
        jmx::joint_matrix_load(
            sgc, jb,
            sycl::multi_ptr<sycl::half,
                            sycl::access::address_space::global_space>(sb),
            16);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_apply(sgc, c0, [](float &e) { e = 0.0f; });
        jmx::joint_matrix_apply(sgc, c1, [](float &e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        const float s =
            GROUP32 ? e8m0_to_float(
                          bsc_[static_cast<std::size_t>(n0 + lane) * sc_ns_ +
                               static_cast<std::size_t>(k0 >> 5) * sc_gs_])
                    : 1.0f;
        int i = 0;
        jmx::joint_matrix_apply(sgc, c1, [&](float &e) { tot[i++] += s * e; });
      }
      epilogue_arr(tot, slice, n0, lane);
    }
  }

 private:
  // One activation half at (row, k): f16 bits direct or bf16-converted.
  std::uint16_t ld_act(int row, int k) const {
    const std::uint16_t h = *reinterpret_cast<const std::uint16_t*>(
        a_ + (static_cast<std::size_t>(row) * k_ + k) * 2);
    return A_BF16 ? bf16_to_f16_bits(h) : h;
  }

  // A staging (row-major 8x16 halves, stride 16; the load consumes it):
  // lane t writes column t -- sa[m*16 + t] = x[m][k0 + t], rows >= M
  // zeros (pad compute is free, the kernel is bytes-bound). Per row the
  // subgroup's 16 lanes write 32 contiguous bytes.
  void stage_a(sycl::half* sa, int k0, int lane) const {
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      const std::uint16_t bits =
          m < m_ ? ld_act(m, k0 + lane) : 0;
      sa[m * 16 + lane] = sycl::bit_cast<sycl::half>(bits);
    }
  }

  // B staging (col-major 16x16 halves, stride 16): lane t DECODES its
  // column's 16 contiguous codes -- one uint4 (f8) / uint2 (f4) load at
  // tbase + (k0>>4)*256/128 + lane*16/8 -- to f16 pairs (decoders above;
  // pair input contract lo code at [7:0], hi at [23:16]) and writes its
  // 32-byte block sb[t*16 .. t*16+15].
  void stage_b(sycl::half* sb, const std::uint8_t* tbase, int k0,
               int lane) const {
    if constexpr (WF == WeightFormat::kF4E2m1) {
      const sycl::uint2 rw = *reinterpret_cast<const sycl::uint2*>(
          tbase + (static_cast<std::size_t>(k0 >> 4) << 7) +
          static_cast<std::size_t>(lane) * 8);
      const std::uint32_t wds[2] = {rw.x(), rw.y()};
#pragma unroll
      for (int v1 = 0; v1 < 2; ++v1) {
        const std::uint32_t w = wds[v1];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          // byte p = nibble pair = codes k = 8*v1+2p (lo nibble, even k)
          // and +1 (hi) of the lane's 8-byte block.
          const std::uint32_t y =
              ((w >> (8 * p)) & 0x0000000Fu) |
              (((w >> (8 * p + 4)) & 0x0000000Fu) << 16);
          const std::uint32_t pr = e2m1_pair_f16x2(y);
          sb[lane * 16 + v1 * 8 + 2 * p] =
              sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(pr));
          sb[lane * 16 + v1 * 8 + 2 * p + 1] =
              sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(pr >> 16));
        }
      }
    } else {
      const sycl::uint4 rw = *reinterpret_cast<const sycl::uint4*>(
          tbase + (static_cast<std::size_t>(k0 >> 4) << 8) +
          static_cast<std::size_t>(lane) * 16);
      const std::uint32_t wds[4] = {rw.x(), rw.y(), rw.z(), rw.w()};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        // u32 j = codes k = 4j..4j+3 (four consecutive bytes, no pad):
        // pairs (4j, 4j+1) and (4j+2, 4j+3) of THIS lane's column. PAIR
        // CONTRACT: the decoders read their lanes at [7:0]/[23:16], so the
        // hi code must LAND at [23:16] -- (w & 0xFF000000u) leaves it at
        // [31:24], the decoder reads zeros, and every k==4j+3 weight
        // silently decodes +0.0 (the cut-5 GATE-1 defect; r128f pinpoint
        // + r128g scratch dump, 2026-11-09).
        const std::uint32_t w = wds[j];
        const std::uint32_t x1 =
            (w & 0x000000FFu) | ((w & 0x0000FF00u) << 8);
        const std::uint32_t x2 =
            ((w >> 16) & 0x000000FFu) | ((w >> 8) & 0x00FF0000u);
        const std::uint32_t p1 = (WF == WeightFormat::kF8E4m3)
                                     ? e4m3_pair_f16x2(x1)
                                     : e5m2_pair_f16x2(x1);
        const std::uint32_t p2 = (WF == WeightFormat::kF8E4m3)
                                     ? e4m3_pair_f16x2(x2)
                                     : e5m2_pair_f16x2(x2);
        sb[lane * 16 + 4 * j] =
            sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(p1));
        sb[lane * 16 + 4 * j + 1] =
            sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(p1 >> 16));
        sb[lane * 16 + 4 * j + 2] =
            sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(p2));
        sb[lane * 16 + 4 * j + 3] =
            sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(p2 >> 16));
      }
    }
  }

  // Epilogue off the per-slot side total (slot m = row m of this lane's
  // ONE column n0 + lane; r128e2 C layout). The padded N tail (repack
  // zero-fill) stores zeros x activations -- the caller slices the real
  // N columns.
  void epilogue_arr(const float (&tot)[8], int slice, int n0, int lane) const {
    const int nn = n0 + lane;
    for (int m = 0; m < m_ && m < 8; ++m) store_out(tot[m], m, nn, slice);
  }

  // Direct converted store (scaled + biased) or raw fp32 partials (scale
  // applied exactly once in stage 2). literal-8 pad stride: partials are
  // [S][8][N] (interface alloc and stage-2 read agree on 8).
  void store_out(float val, int m, int nn, int slice) const {
    if (DIRECT && !GROUP32) val *= pc_ ? pc_[nn * pcs_] : pt_;
    if (DIRECT) {
      if (bias_) val += bias_[nn];
      if (A_BF16) {
        // truncating convert (sub-ULP vs RTNE; gate rel-errs at 2e-3)
        const std::uint32_t bits =
            sycl::bit_cast<std::uint32_t>(val) >> 16;
        *reinterpret_cast<std::uint16_t*>(
            out_ + (static_cast<std::size_t>(m) * n_ + nn) * 2) =
            static_cast<std::uint16_t>(bits);
      } else {
        const sycl::half h = static_cast<sycl::half>(val);
        *reinterpret_cast<std::uint16_t*>(
            out_ + (static_cast<std::size_t>(m) * n_ + nn) * 2) =
            sycl::bit_cast<std::uint16_t>(h);
      }
    } else {
      partials_[((static_cast<std::size_t>(slice) * 8 + m) * n_) + nn] = val;
    }
  }

  const std::uint8_t* a_;
  const sycl::half* a16_;  // PP: k-blocked f16 activations [K/16, 8, 16]
  const std::uint8_t* b_;
  const std::uint8_t* bsc_;
  sycl::half* scratch_;
  sycl::local_accessor<sycl::half, 1> slm_;  // PP: per-WG B slots
  float* partials_;
  std::uint8_t* out_;
  const float* bias_;
  const float* pc_;
  float pt_;
  int64_t sc_ns_, sc_gs_;
  int m_, n_, k_, s_, kc_;
  bool swzl_;
  int64_t pcs_;
};

// ---------------------------------------------------------------------------
// Stage 2: fixed-order split-K reduce + epilogue (not-DIRECT path).
// Verbatim from arc6 SkinnyGemmReduceKernel (P6.33) -- same [S][8][N]
// contract, same scale/bias/convert epilogue.
// ---------------------------------------------------------------------------
class JmGemmReduceKernel {
 public:
  JmGemmReduceKernel(
      const float* partials, void* out, const float* bias,
      const float* pc_scale, float pt_scale, int m_rows, int n_cols,
      int splits, bool out_bf16, int64_t pc_stride)
      : p_(partials),
        out_(static_cast<std::uint8_t*>(out)),
        bias_(bias),
        pc_(pc_scale),
        pt_(pt_scale),
        m_(m_rows),
        n_(n_cols),
        s_(splits),
        bf16_(out_bf16),
        pcs_(pc_stride) {}

  void operator()(sycl::nd_item<1> it) const {
    const int n = static_cast<int>(it.get_global_linear_id());
    if (n >= n_) return;
    for (int mi = 0; mi < m_; ++mi) {
      float v = 0.0f;
      for (int slice = 0; slice < s_; ++slice) {  // fixed order: deterministic
        v += p_[(static_cast<std::size_t>(slice) * 8 + mi) * n_ + n];
      }
      v *= pc_ ? pc_[n * pcs_] : pt_;
      if (bias_) v += bias_[n];
      if (bf16_) {
        const std::uint32_t bits = sycl::bit_cast<std::uint32_t>(v) >> 16;
        *reinterpret_cast<std::uint16_t*>(out_ + (mi * n_ + n) * 2) =
            static_cast<std::uint16_t>(bits);
      } else {
        const sycl::half h = static_cast<sycl::half>(v);
        *reinterpret_cast<std::uint16_t*>(out_ + (mi * n_ + n) * 2) =
            sycl::bit_cast<std::uint16_t>(h);
      }
    }
  }

 private:
  const float* p_;
  std::uint8_t* out_;
  const float* bias_;
  const float* pc_;
  float pt_;
  int m_, n_, s_;
  bool bf16_;
  int64_t pcs_;
};

// ---------------------------------------------------------------------------
// Launcher: picks splits (formula for ANY shape -- all-model-types law),
// instantiates the format matrix, runs the stages.
// ---------------------------------------------------------------------------
struct JmLaunchCfg {
  int splits;
  int k_per_split;
  int direct;        // 1 = single-stage epilogue
  int n_subgroups;
};

// Probe knobs (cert wheel bakes winners as constants; arc6 v2.4 -> v2.5
// pattern). ARC_JM_SGF: subgroup floor (default 2048). ARC_JM_SWZ: tile
// swizzle off (default on).
inline int jm_env_int(const char* name, int dflt) {
  const char* v = std::getenv(name);
  return (v && *v) ? std::atoi(v) : dflt;
}

inline JmLaunchCfg jm_pick_cfg(int m, int n, int k) {
  (void)m;
  const int sg_floor = jm_env_int("ARC_JM_SGF", 2048);
  const int tiles = (n + 15) / 16;
  int splits = 1;
  if (tiles < sg_floor) {
    // Same laws as arc6 skinny_pick_cfg (windows 8c/9 evidence), scaled to
    // subgroups (16 columns each vs TN=2): floor-driven wants, capped at 8
    // and at k/1024, decremented to a group divisor so slices stay uniform
    // and group-whole (a slice boundary always coincides with a 32-k group
    // boundary -- the MX flush depends on it).
    int want = sg_floor / tiles;
    splits = want < 1 ? 1 : (want > 8 ? 8 : want);
    const int groups = k / 32;
    while (splits > 1 && (groups % splits) != 0) --splits;
    if (splits > groups) splits = groups < 1 ? 1 : groups;
    const int s_max = k / 1024;
    if (splits > s_max) {
      splits = s_max < 1 ? 1 : s_max;
      while (splits > 1 && ((k / 32) % splits) != 0) --splits;
    }
  }
  // 64-aligned slice length (arc6 law; every span a 64 multiple, last
  // slice clamped to k_ in-kernel).
  const int kc = ((k / splits) + 63) / 64 * 64;
  return JmLaunchCfg{splits, kc, splits == 1 ? 1 : 0, tiles * splits};
}

// wf: 0 = e4m3, 1 = e5m2, 2 = f4 (f4 requires group32). sc_ns/sc_gs are
// the group-scale byte strides; pc_stride is the checkpoint-scale N stride
// (0 = broadcast scalar). out/partials unused in the counterpart path.
inline void jm_launch(
    sycl::queue& q, WeightFormat wf, bool group32, bool a_bf16, const void* a,
    const std::uint8_t* b, const std::uint8_t* bsc, float* partials,
    void* out, const float* bias, const float* pc_scale, float pt_scale,
    bool out_bf16, int m, int n, int k, int64_t sc_ns = 0,
    int64_t sc_gs = 1, int64_t pc_stride = 1) {
  TORCH_CHECK(k % 32 == 0, "jm gemm requires K % 32 == 0, got K=", k);
  TORCH_CHECK(m >= 1 && m <= 8, "jm gemm requires 1 <= M <= 8, got M=", m);
  if (wf == WeightFormat::kF4E2m1) {
    TORCH_CHECK(group32, "f4 jm gemm requires group-32 e8m0 scales");
  }
  const auto cfg = jm_pick_cfg(m, n, k);
  const bool swizzle = jm_env_int("ARC_JM_SWZ", 1) != 0;
  // ARC_JM_VAR: 5 = the gate-proven V5 sequence (lottery control), any
  // other value (default 14) = the SLM form (A-direct + B-in-SLM,
  // r128k7). Read PER CALL so one wheel runs both variants through the
  // gate.
  const bool pp = jm_env_int("ARC_JM_VAR", 14) != 5;
  // 4 subgroups per work group (local size 64): subgroup boundaries stay
  // aligned (16 | 64) and the scheduler gets group-level granularity.
  const int lsz = 64;
  const int subs = cfg.n_subgroups;
  const int wgs = ((subs * 16 + lsz - 1) / lsz) * lsz;
  // V5-arm staging scratch, 768 B/sg: decoded A/B halves round-trip
  // through GLOBAL memory per k-step (the measured ~4.9 us/step wall).
  // Torch caching allocator, same stream-ordered lifetime class as the
  // split-K partials.
  auto scratch = at::empty(
      {static_cast<int64_t>(subs) * 768},
      at::TensorOptions().dtype(at::kByte).device(at::kXPU));
  sycl::half* scr_p = reinterpret_cast<sycl::half*>(scratch.data_ptr());
  // PP arm's A-direct source: x -> zero-padded 8-row f16 -> k-blocked
  // [K/16, 8, 16] (x16[s*128 + m*16 + t], the A fragment's native tile
  // layout, perfectly coalesced). bf16->f16 is exact for activations;
  // ~80 KB max of traffic per call (sub-us under graph replay). Built
  // in-op so the torch schema and the python route stay untouched.
  // Stream-ordered lifetime class, same as scratch.
  const sycl::half* a16p = nullptr;
  if (pp) {
    auto a_t = at::from_blob(
        const_cast<void*>(a), {m, static_cast<int64_t>(k)},
        at::TensorOptions().dtype(a_bf16 ? at::kBFloat16 : at::kHalf)
            .device(at::kXPU));
    auto a16 = at::zeros(
        {8, static_cast<int64_t>(k)},
        at::TensorOptions().dtype(at::kHalf).device(at::kXPU));
    a16.narrow(0, 0, m).copy_(a_t);  // convert + zero-pad rows >= M
    auto a16v = a16.view({8, static_cast<int64_t>(k) / 16, 16})
                    .permute({1, 0, 2})
                    .contiguous();
    a16p = reinterpret_cast<const sycl::half*>(a16v.data_ptr());
  }

#define JM_LAUNCH(WF_C, G32_C, ABF_C, DIR_C, PP_C)                          \
  q.submit([&](sycl::handler &cgh) {                                        \
    sycl::local_accessor<sycl::half, 1> slm(1024, cgh);                     \
    cgh.parallel_for<JmGemmStage1Kernel<WF_C, G32_C, ABF_C, DIR_C, PP_C>>(  \
        sycl::nd_range<1>(wgs, lsz),                                        \
        JmGemmStage1Kernel<WF_C, G32_C, ABF_C, DIR_C, PP_C>(                \
            a, a16p, b, bsc, scr_p, std::move(slm), partials, out, bias,    \
            pc_scale, pt_scale, m, n, k, cfg.splits, cfg.k_per_split,       \
            sc_ns, sc_gs, swizzle, pc_stride));                             \
  })

  // No M_PAD dispatch: dpas always computes 8 rows; the A fragment pads
  // with zeros and the epilogue stores rows < M (runtime).
  // PP is a TEMPLATE bool -- the runtime ARC_JM_VAR read selects the
  // instantiation HERE, outside the template argument (the build8 arc8
  // attempt passed `pp` itself as a template arg: "non-type template
  // argument is not a constant expression", 2h into the build, 2026-11-09
  // 21:04 UTC; fixed in build9).
#define JM_LAUNCH_G(WF_C, G32_C, ABF_C)                    \
  do {                                                     \
    if (pp) {                                              \
      if (cfg.direct) {                                    \
        JM_LAUNCH(WF_C, G32_C, ABF_C, true, true);         \
      } else {                                             \
        JM_LAUNCH(WF_C, G32_C, ABF_C, false, true);        \
      }                                                    \
    } else {                                               \
      if (cfg.direct) {                                    \
        JM_LAUNCH(WF_C, G32_C, ABF_C, true, false);        \
      } else {                                             \
        JM_LAUNCH(WF_C, G32_C, ABF_C, false, false);       \
      }                                                    \
    }                                                      \
  } while (0)

#define JM_LAUNCH_A(WF_C, G32_C)          \
  do {                                    \
    if (a_bf16) {                         \
      JM_LAUNCH_G(WF_C, G32_C, true);     \
    } else {                              \
      JM_LAUNCH_G(WF_C, G32_C, false);    \
    }                                     \
  } while (0)

  if (wf == WeightFormat::kF8E4m3) {
    if (group32) {
      JM_LAUNCH_A(WeightFormat::kF8E4m3, true);
    } else {
      JM_LAUNCH_A(WeightFormat::kF8E4m3, false);
    }
  } else if (wf == WeightFormat::kF8E5m2) {
    if (group32) {
      JM_LAUNCH_A(WeightFormat::kF8E5m2, true);
    } else {
      JM_LAUNCH_A(WeightFormat::kF8E5m2, false);
    }
  } else {  // kF4E2m1 (group32 asserted above)
    JM_LAUNCH_A(WeightFormat::kF4E2m1, true);
  }
#undef JM_LAUNCH_A
#undef JM_LAUNCH_G
#undef JM_LAUNCH

  if (!cfg.direct) {
    // stage 2 wants one work-item per column (N)
    const int rgs = ((n + lsz - 1) / lsz) * lsz;
    q.parallel_for<JmGemmReduceKernel>(
        sycl::nd_range<1>(rgs, lsz),
        JmGemmReduceKernel(partials, out, bias, pc_scale, pt_scale, m, n,
                           cfg.splits, out_bf16, pc_stride));
  }
}

}  // namespace vllm::xpu::jm
