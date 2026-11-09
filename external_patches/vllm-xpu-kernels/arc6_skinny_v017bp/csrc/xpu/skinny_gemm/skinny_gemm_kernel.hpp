#pragma once

// Skinny-M weight-only GEMM kernels (P6.33, arc6 wheel).
//
// Decode-time linear layers run at M <= 8 (batch x spec tokens) where every
// stock oneDNN route measured in the r123 kernel matrix is latency-bound far
// above the pure-weights bytes floor (qwen.up K=5120 N=6912/rank, 608 GB/s:
// f8 35.4 MB -> 58 us floor; measured medians w16pt 179 us, w16pc 218 us,
// m816 327 us, g16 310 us). The Xe2 gemm tiles that oneDNN's joint-quant
// matmuls lower to are sized for large M; at M <= 8 the primitive pays full
// tile-fill traffic for an M dimension that is mostly padding.
//
// This kernel is the GEMV-regime replacement: one work-item owns TN output
// columns for one K-slice and streams its weight columns K-contiguously, so
// the only device traffic is the weights themselves (plus an M x K bf16
// activation panel that stays L2-resident). All dequantization happens in
// registers -- f8 e4m3/e5m2 via the f32 exponent-rebias bit identity,
// MXFP4 e2m1 via the same magnitude construction, e8m0 group scales via
// exp2 -- and math accumulates in fp32 with a fixed-order split-K reduction
// (deterministic across runs, unlike atomics).
//
// Weight-format coverage mirrors the three serving lanes:
//   kF8E4m3/kF8E5m2 + epilogue per-tensor/per-channel scale (checkpoint-fp8,
//   the w16pt/w16pc champion classes), the same f8 formats with in-kernel
//   group-32 e8m0 scales (online MXFP8 w8a16), and kF4E2m1 packed nibbles +
//   group-32 e8m0 (online MXFP4 w4a16).
//
// The launch is a two-stage split-K when N is too small to fill the device
// (fp32 partials, epilogue in stage 2) and a single direct-epilogue launch
// otherwise. Layouts are resolved by the caller (interface) from the same
// tensors the stock oneDNN ops receive -- [K, N] k-contiguous views for the
// f8 lanes, packed [N, K/2] row-major for the f4 lane -- so the python
// M-split patch swaps the call 1:1.
//
// r126 postmortem (the v1 kernel measured 7-30x SLOWER than stock at M<=8,
// ~20 GB/s effective on 600+ GB/s parts): the v1 inner loop streamed weights
// with 32-bit scalar loads while holding acc[TN][M_PAD]=32 fp32 registers
// live and unrolling M_PAD=8 even at runtime M=1 -- too few bytes in flight
// per thread and heavy register pressure serialized the load pipeline; time
// came out exactly proportional to weight bytes at 3% of bandwidth.
//
// v2 memory design (window4): 128-bit weight/activation loads (one uint4 =
// CHUNK elements), launch-time M_PAD dispatch {1,2,4,8}, TN=2, decode-and-
// consume transient w[CHUNK], [S][8][N] partials, 32768-workitem target
// with group-aligned split-K; v2.2 made the f8 decoders branchless.
//
// v2.3 (window6 targets, from the v2.2 GATE-2 residue):
//   * weight-panel restructure for every M_PAD: decode BOTH TN columns
//     first, then each row's activation chunk ONCE per k step. v2.2
//     re-decoded activations inside the j loop (per column), so M=8 paid
//     TN x M x CHUNK a-decodes where M x CHUNK suffice -- the M=8 lanes
//     lost to stock 1.0-2.35x on ALU that was half redundant.
//   * M=1 (the decode hot lane) runs a dual-chunk software pipeline: all
//     TN x 2 weight loads plus both activation loads issue BEFORE the
//     first decode -- twice the bytes in flight per thread without more
//     threads. v2.2's M=1 left bandwidth on the table on the short-K
//     shapes (qwen.o 142 GB/s effective vs gemma.o's 224 at the same
//     splits/stream length).
//   * branchless e2m1 (same biased-exponent f32 magnitude construction as
//     the f8 decoders) replaces the 8-entry LUT select chain that left the
//     f4 g16 lane 1.14-1.23x behind stock on the long-K gemma shapes.
//
// v2.4 (probe build, r126 sweep): the M>=2 generic body gains a one-chunk
// weight LOOKAHEAD (pv_next issues before the decode/FMA section; the v2.3
// panel starved next-iteration loads through the long M_PAD>=4 FMA
// stretch), and three env knobs (ARC_SKINNY_WI / ARC_SKINNY_SWIZZLE /
// ARC_SKINNY_KC_MIN, read per call, unset = v2.3 constants) let one kbench
// process sweep work-item target, a golden-ratio column swizzle (the
// power-of-two column-stride suspect behind qwen.o's 1.6x deficit vs
// gemma.o), and minimum split length. Winners get baked as constants in
// the certified wheel (no env reads on the serving path).
//
// v2.5 (baked; window8): swizzle always on; splits capped so each stream
// stays >= 1024 elements; kc rounded to 64 (kills the f4 single-chunk
// tail on every span); and the M=1 pair path decodes each half-chunk's
// activations transiently (a[CHUNK] in the h loop) instead of hoisting
// ah[2][CHUNK] -- the 32-register hoist sat at a scheduling cliff
// (combo-interleaved refine: p10/p90 spread 1.6x on this path vs 1.15x
// on the M>=2 lookahead path). No env reads anywhere: constants only.
//
// v2.5.1: per-tensor checkpoint scales pass as a device POINTER with
// N-stride 0 instead of a host .item<float>() -- that .item() was a
// device-to-host sync on every w16pt call (window8b round-interleaved
// p90s: 14-30 ms tails on the lane; the p50 carried the pipeline-stall
// tax that made w16pt look 1.16-3.03x behind stock when the kernel body
// was never the problem). Kernels read scale[n * pc_stride] (stride 0 =
// the same cached float for every column).
//
// v2.6 (window9): split selection lets the 32768 work-item floor drive
// splits up to 8 with a 640-element stream floor and a 64-aligned
// NON-UNIFORM last slice (kc rounds up, the last slice takes the remainder
// -- still a 64 multiple, so no f4 tail on any span and no
// group-divisibility constraint on the split count). Rationale: the 8b/8c
// windows read a ~180 GB/s ceiling on the 10.7-12.8k-work-item shapes
// while 21.5k+ shapes reached 270-296, and the refine's 8-split cells won
// qwen.up/qwen.o M=1. Split table: qwen.o 8x640, gemma.o 8x704+448,
// qwen.down 8x1088, gemma.down 8x1344 (all refine-confirmed); qwen.up
// 7x768+512 and gemma.up 6x896 interpolate the refine's 4-split winner
// and 8-split loser (its 8x672 evidence was tail-poisoned).
//
// v2.7 (window10): split law back to the v2.5 form (floor-capped wants,
// kc>=1024, group-div decrement, uniform slices) after window9 showed the
// v2.6 640-stream law regressing qwen.o 5-10% on all four lanes -- the
// refine's 8-split read there carried the w16pt .item() tax (the probe
// wheel predates the pointer-scale fix). Split table: qwen 5x1024, gemma
// 4x1344, downs 8x{1088,1344}; best measured on every shape.

#include <torch/all.h>
#include <sycl/sycl.hpp>

#include <cstdint>

namespace vllm::xpu::skinny {

// ---------------------------------------------------------------------------
// Format decode helpers (all bit-exact; no exp2 on the per-element paths).
// ---------------------------------------------------------------------------

// e4m3fn -> f32, branchless (v2.2: the branchy decode's e==0/e==15 guards
// serialized the SIMD lanes and capped the whole kernel at ~60-80 GB/s).
// Construct the biased-exponent f32 MAGNITUDE for every element:
// ((e + 120) << 23) | (m << 20) is exact for e > 0. For e == 0 it reads
// 2^-7 + m * 2^-10 (implicit leading 1, m at fraction m/8) while the true
// e4m3 subnormal is m * 2^-9: ((c - 2^-7) * 2) recovers it exactly
// (nearby-normal subtraction with an exactly-representable difference;
// m == 0 lands on 0) -- the fixup runs on the magnitude and the sign is
// applied after, so it cannot collide with the sign bit. e4m3fn's only
// non-finite code (e==15, m==7) maps to the finite 480.0f instead of NaN:
// quantized weights never carry it (max finite magnitude is 448) and the
// correctness gate feeds randn-quantized weights, so the branchy NaN->0
// guard was dead code. Exhaustively checked over all 256 codes.
inline float e4m3_to_float(std::uint32_t b) {
  const std::uint32_t e = (b >> 3) & 0xFu;
  const std::uint32_t m = b & 0x7u;
  const std::uint32_t bits = ((e + 120u) << 23) | (m << 20);
  const float c = sycl::bit_cast<float>(bits);
  const float v = e == 0u ? (c - 0x1p-7f) * 2.0f : c;
  return (b & 0x80u) ? -v : v;
}

// e5m2 -> f32, branchless, same magnitude construction: ((e + 112) << 23)
// | (m << 21) is exact for e > 0. For e == 0 it reads 2^-15 + m * 2^-17
// while the true e5m2 subnormal is m * 2^-16: ((c - 2^-15) * 2) recovers
// it exactly (fixup on the magnitude, sign applied after). Inf/NaN
// encodings (e == 31) map to finite values -- weights never carry them.
// Exhaustively checked over all 256 codes.
inline float e5m2_to_float(std::uint32_t b) {
  const std::uint32_t e = (b >> 2) & 0x1Fu;
  const std::uint32_t m = b & 0x3u;
  const std::uint32_t bits = ((e + 112u) << 23) | (m << 21);
  const float c = sycl::bit_cast<float>(bits);
  const float v = e == 0u ? (c - 0x1p-15f) * 2.0f : c;
  return (b & 0x80u) ? -v : v;
}

// e2m1 nibble (0..15) -> f32, branchless (v2.3): the same biased-exponent
// f32 magnitude construction as the f8 decoders -- ((e + 126) << 23) |
// (m << 22) is exact for e > 0 ({1, 1.5, 2, 3, 4, 6}; the mantissa bit
// sits at the f32 fraction MSB, bit 22 -- the host exhaustive check caught
// an m << 30 draft that landed on the exponent field). For e == 0 it reads
// 2^-1 + m * 2^-2 while the true e2m1 subnormal is m * 2^-1 ({0, 0.5}):
// ((c - 0.5) * 2) recovers it exactly (m == 0 lands on 0). Replaces the
// v2 8-entry LUT select chain. Exhaustively checked over all 16 codes.
inline float e2m1_to_float(std::uint32_t nib) {
  const std::uint32_t e = (nib >> 1) & 0x3u;
  const std::uint32_t m = nib & 0x1u;
  const std::uint32_t bits = ((e + 126u) << 23) | (m << 22);
  const float c = sycl::bit_cast<float>(bits);
  const float v = e == 0u ? (c - 0.5f) * 2.0f : c;
  return (nib & 0x8u) ? -v : v;
}

// e8m0 -> 2^(b - 127): group scales; the one exp2 on the hot path, K/32
// evaluations per column per slice.
inline float e8m0_to_float(std::uint32_t b) {
  return sycl::exp2(static_cast<float>(static_cast<int>(b) - 127));
}

inline float bf16_to_float(std::uint32_t h) {
  return sycl::bit_cast<float>(h << 16);
}

inline float f16_to_float(std::uint32_t h16) {
  return static_cast<float>(sycl::bit_cast<sycl::half>(
      static_cast<std::uint16_t>(h16)));
}

// ---------------------------------------------------------------------------
// Stage 1: partial (or direct) GEMM over one K-slice per work-item.
// ---------------------------------------------------------------------------

enum class WeightFormat : int { kF8E4m3 = 0, kF8E5m2 = 1, kF4E2m1 = 2 };

// Scaling rules: GROUP32 lanes (MX) multiply the decoded weight by its e8m0
// group scale inside the k-loop; checkpoint lanes apply the per-column (or
// per-tensor) scale once at the epilogue -- identical value, since the
// scale is constant along k for a fixed column.
//
// Layouts (resolved by the interface, asserted there):
//   f8:  element (k, n) at b_[k + n * k_]           -- [K, N], k contiguous
//   f4:  byte    (k, n) at b_[n * (k_/2) + (k>>1)], nibble (k & 1) low=even
//   scales (GROUP32): byte (n, g) at bsc_[n * sc_ns + g * sc_gs] -- either
//     [N, K/32] (sc_ns = K/32, sc_gs = 1) or [K/32, N] (sc_ns = 1,
//     sc_gs = N) order, whichever the layer hands the stock op.
template <WeightFormat WF, bool GROUP32, bool A_BF16, bool DIRECT,
          int M_PAD = 8, int TN = 2>
class SkinnyGemmStage1Kernel {
 public:
  SkinnyGemmStage1Kernel(
      const void* a,             // [M, K] contiguous activations
      const std::uint8_t* b,     // weights, k-contiguous (above)
      const std::uint8_t* bsc,   // e8m0 group scales [N, K/32] or nullptr
      float* partials,           // [S][8][N] fp32 (not DIRECT)
      void* out,                 // [M, N] activation dtype (DIRECT)
      const float* bias,         // [N] (DIRECT, may be nullptr)
      const float* pc_scale,     // [N] per-channel or nullptr
      float pt_scale,            // per-tensor scalar (1.f if none)
      int m_rows, int n_cols, int k_elems, int splits, int k_per_split,
      int64_t sc_n_stride,       // group-scale strides (GROUP32)
      int64_t sc_g_stride, bool swizzle,
      int64_t pc_stride)         // checkpoint-scale N stride (0 = scalar)
      : a_(static_cast<const std::uint8_t*>(a)),
        b_(b),
        bsc_(bsc),
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
    const int tid = static_cast<int>(it.get_global_linear_id());
    const int tiles = (n_ + TN - 1) / TN;
    // Column swizzle (v2.5, always on): natural tile order maps adjacent
    // work items to adjacent columns whose base addresses differ by
    // exactly K bytes -- power-of-two-ish K (qwen.o 5120 = 2^10 * 5)
    // aligns every concurrent stream to the same L2-set/channel pattern,
    // and qwen.o benched 1.6x below the near-identical gemma.o (5376)
    // across two windows. (t * A) % tiles is a bijection
    // (A = 0x9E3779B1 is prime and tiles < A), so every column stays
    // owned exactly once; deterministic.
    const std::uint32_t t = static_cast<std::uint32_t>(tid % tiles);
    const int ctile =
        swzl_ ? static_cast<int>((t * 2654435761ull) % tiles)
              : static_cast<int>(t);
    const int n0 = ctile * TN;
    const int slice = tid / tiles;
    if (n0 >= n_ || slice >= s_) return;

    constexpr int CHUNK = (WF == WeightFormat::kF4E2m1) ? 32 : 16;
    // f8: 16 elems = 1 uint4; f4: 32 nibbles = 1 uint4; bf16: 8 elems
    constexpr int AW = (CHUNK + 7) / 8;  // activation uint4 loads

    float acc[TN][M_PAD];
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int mi = 0; mi < M_PAD; ++mi) acc[j][mi] = 0.0f;

    const int k_lo = slice * kc_;
    const int k_hi = sycl::min(k_lo + kc_, k_);
    // k_lo, kc_ and CHUNK are group-aligned by the launcher/template, so
    // every chunk sits inside one e8m0 group: (k0 >> 5) is its group id.

    if constexpr (M_PAD == 1) {
      // v2.3 M=1 lane: dual-chunk software pipeline. All TN*2 weight loads
      // and both activation loads issue before the first decode, so each
      // thread keeps ~2x the bytes in flight of the single-chunk loop
      // without more threads (the 32768-workitem target already saturates
      // concurrency; v2.2's single-chunk M=1 left 40%+ bandwidth on the
      // table on the short-K shapes). Spans are multiples of 32 (kc, k
      // group-aligned), so the f8 pair loop covers every chunk and only
      // the f4 lane (CHUNK=32, pair=64) can carry a one-chunk tail.
      constexpr int C2 = CHUNK * 2;
      const int k_pair_end = k_lo + ((k_hi - k_lo) / C2) * C2;
      for (int k0 = k_lo; k0 < k_pair_end; k0 += C2) {
        sycl::uint4 pv[TN][2];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int n = n0 + j;
          if (n >= n_) continue;  // FMA loop skips j; pv[j] never read
          if constexpr (WF == WeightFormat::kF4E2m1) {
            pv[j][0] = *reinterpret_cast<const sycl::uint4*>(
                b_ + n * (k_ / 2) + (k0 >> 1));
            pv[j][1] = *reinterpret_cast<const sycl::uint4*>(
                b_ + n * (k_/ 2) + ((k0 + CHUNK) >> 1));
          } else {
            pv[j][0] = *reinterpret_cast<const sycl::uint4*>(
                b_ + n * k_ + k0);
            pv[j][1] = *reinterpret_cast<const sycl::uint4*>(
                b_ + n * k_ + k0 + CHUNK);
          }
        }
        // v2.5: decode each half-chunk's activations just before its FMA
        // pass (a[CHUNK] transient) instead of hoisting ah[2][CHUNK] --
        // the 32-register hoist sat at a scheduling cliff: combo-interleaved
        // refine measured p10/p90 spreads of 1.6x on this path vs 1.15x on
        // the M>=2 lookahead path. The four weight loads still issue first
        // and cover the activation load latency.
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float a[CHUNK];
          decode_a_chunk(k0 + h * CHUNK, 0, a);
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            const int n = n0 + j;
            if (n >= n_) continue;
            float w[CHUNK];
            decode_w_chunk(pv[j][h], group_scale(n, k0 + h * CHUNK), w);
#pragma unroll
            for (int c = 0; c < CHUNK; ++c) acc[j][0] += a[c] * w[c];
          }
        }
      }
      // f4 tail: span an odd multiple of 32 (e.g. kc=672) leaves one chunk
      for (int k0 = k_pair_end; k0 < k_hi; k0 += CHUNK) {
        float a[CHUNK];
        decode_a_chunk(k0, 0, a);
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int n = n0 + j;
          if (n >= n_) continue;
          float w[CHUNK];
          decode_w_chunk(load_w(n, k0), group_scale(n, k0), w);
#pragma unroll
          for (int c = 0; c < CHUNK; ++c) acc[j][0] += a[c] * w[c];
        }
      }
    } else {
      // v2.4 generic body (M_PAD >= 2): weight panel per k step with a
      // ONE-CHUNK LOOKAHEAD -- pv_next for k0+CHUNK issues before the
      // decode/FMA section consumes pv_cur, keeping the load pipeline fed
      // across k steps (v2.3's panel held wp live through the whole mi
      // section; at M_PAD >= 4 the long FMA stretch starved the next
      // iteration's loads and several M=4/8 cells regressed vs v2.2).
      // Activations decode once per (k step, row) and FMA against every
      // column (v2.2 re-decoded per column: TN x redundant a-decode ALU
      // that cost the M=8 lanes their GATE-2 parity, 1.0-2.35x behind).
      sycl::uint4 pv_cur[TN], pv_next[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int n = n0 + j;
        if (n < n_) pv_next[j] = load_w(n, k_lo);
      }
      for (int k0 = k_lo; k0 < k_hi; k0 += CHUNK) {
#pragma unroll
        for (int j = 0; j < TN; ++j) pv_cur[j] = pv_next[j];
        if (k0 + CHUNK < k_hi) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            const int n = n0 + j;
            if (n < n_) pv_next[j] = load_w(n, k0 + CHUNK);
          }
        }
        float wp[TN][CHUNK];
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          const int n = n0 + j;
          if (n >= n_) {
#pragma unroll
            for (int c = 0; c < CHUNK; ++c) wp[j][c] = 0.0f;
            continue;  // FMA against zeros; epilogue skips the store
          }
          decode_w_chunk(pv_cur[j], group_scale(n, k0), wp[j]);
        }
#pragma unroll
        for (int mi = 0; mi < M_PAD; ++mi) {
          if (mi >= m_) break;
          float a[CHUNK];
          decode_a_chunk(k0, mi, a);
#pragma unroll
          for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int c = 0; c < CHUNK; ++c) acc[j][mi] += a[c] * wp[j][c];
        }
      }
    }

    // Epilogue: direct converted store (scaled + biased) or fp32 partials.
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = n0 + j;
      if (n >= n_) continue;
#pragma unroll
      for (int mi = 0; mi < M_PAD; ++mi) {
        if (mi >= m_) break;
        float v = acc[j][mi];
        // DIRECT finishes here (scale + convert); !DIRECT partials stay
        // RAW fp32 -- stage 2 applies the epilogue scale exactly once
        // after the fixed-order sum (scaling here too would apply it
        // splits-times on the split-K path).
        if (DIRECT && !GROUP32) v *= pc_ ? pc_[n * pcs_] : pt_;
        if (DIRECT) {
          if (bias_) v += bias_[n];
          if (A_BF16) {
            // truncating convert (sub-ULP vs RTNE; the correctness gate
            // rel-errs at 2e-3 scale)
            const std::uint32_t bits = sycl::bit_cast<std::uint32_t>(v) >> 16;
            *reinterpret_cast<std::uint16_t*>(out_ + (mi * n_ + n) * 2) =
                static_cast<std::uint16_t>(bits);
          } else {
            const sycl::half h = static_cast<sycl::half>(v);
            *reinterpret_cast<std::uint16_t*>(out_ + (mi * n_ + n) * 2) =
                sycl::bit_cast<std::uint16_t>(h);
          }
        } else {
          // literal-8 pad stride: partials are [S][8][N] for every M_PAD
          // instantiation (interface alloc + stage-2 read agree on 8)
          partials_[((static_cast<std::size_t>(slice) * 8 + mi) * n_) + n] =
              v;
        }
      }
    }
  }

 private:
  // One 128-bit weight load for column n at k0 (format layout dispatched at
  // compile time).
  sycl::uint4 load_w(int n, int k0) const {
    if constexpr (WF == WeightFormat::kF4E2m1) {
      return *reinterpret_cast<const sycl::uint4*>(b_ + n * (k_ / 2) +
                                                   (k0 >> 1));
    } else {
      return *reinterpret_cast<const sycl::uint4*>(b_ + n * k_ + k0);
    }
  }

  // Group-32 e8m0 scale for column n / chunk k0 (1.f for checkpoint lanes).
  float group_scale(int n, int k0) const {
    if constexpr (GROUP32) {
      return e8m0_to_float(bsc_[n * sc_ns_ + (k0 >> 5) * sc_gs_]);
    } else {
      (void)n;
      (void)k0;
      return 1.0f;
    }
  }

  // One uint4 of weights -> CHUNK decoded values (transient registers),
  // multiplied by the (group) scale.
  static void decode_w_chunk(const sycl::uint4& pv, float gsc, float* w) {
    constexpr int CHUNK = (WF == WeightFormat::kF4E2m1) ? 32 : 16;
    const std::uint32_t lane[4] = {pv.x(), pv.y(), pv.z(), pv.w()};
    if constexpr (WF == WeightFormat::kF4E2m1) {
#pragma unroll
      for (int c = 0; c < CHUNK; ++c) {
        const std::uint32_t byte = lane[c >> 3] >> (8 * ((c >> 1) & 3));
        const std::uint32_t nib = (byte >> (4 * (c & 1))) & 0xFu;
        w[c] = e2m1_to_float(nib) * gsc;
      }
    } else {
#pragma unroll
      for (int c = 0; c < CHUNK; ++c) {
        const std::uint32_t bits = (lane[c >> 2] >> (8 * (c & 3))) & 0xFFu;
        const float v = (WF == WeightFormat::kF8E4m3)
                            ? e4m3_to_float(bits)
                            : e5m2_to_float(bits);
        w[c] = v * gsc;
      }
    }
  }

  // One CHUNK of row mi's activations at k0 -> decoded values. Each 32-bit
  // lane of the uint4 holds TWO halves: x=(0,1) y=(2,3) z=(4,5) w=(6,7) --
  // lane h>>1, half h&1 (r126 v2.1: the h<4 gate shifted lanes by >=32 and
  // garbled elements 2,3,6,7).
  void decode_a_chunk(int k0, int mi, float* out) const {
    constexpr int CHUNK = (WF == WeightFormat::kF4E2m1) ? 32 : 16;
    constexpr int AW = (CHUNK + 7) / 8;
    const sycl::uint4* av = reinterpret_cast<const sycl::uint4*>(
        a_ + (static_cast<std::size_t>(mi) * k_ + k0) * 2);
#pragma unroll
    for (int v = 0; v < AW; ++v) {
      const sycl::uint4 ap = av[v];
      const std::uint32_t alane[4] = {ap.x(), ap.y(), ap.z(), ap.w()};
#pragma unroll
      for (int h = 0; h < 8; ++h) {
        const std::uint32_t hw = (alane[h >> 1] >> (16 * (h & 1))) & 0xFFFFu;
        out[v * 8 + h] = A_BF16 ? bf16_to_float(hw) : f16_to_float(hw);
      }
    }
  }

  const std::uint8_t* a_;
  const std::uint8_t* b_;
  const std::uint8_t* bsc_;
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
// Partials are [S][8][N] regardless of the stage-1 M_PAD instantiation
// (literal-8 pad stride; the interface allocates the same shape).
// ---------------------------------------------------------------------------
class SkinnyGemmReduceKernel {
 public:
  SkinnyGemmReduceKernel(
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
// Launcher: picks splits, instantiates the format matrix, runs the stages.
// ---------------------------------------------------------------------------
struct SkinnyLaunchCfg {
  int splits;
  int k_per_split;
  int direct;        // 1 = single-stage epilogue
  int n_workitems;
};

inline SkinnyLaunchCfg skinny_pick_cfg(int m, int n, int k) {
  (void)m;
  // r126 v1 targeted 2048 work-items and never split our shapes; v2 (TN=2,
  // 128-bit loads) wants concurrent streams to cover HBM latency -- the v2
  // run showed the two DIRECT (unsplit) shapes were the two worst ratios
  // (3.2-3.4x). v2.7 law (windows 8c vs 9 head-to-head): splits want the
  // 32768 work-item floor, capped at 8 AND at k/1024, decremented to a
  // group-divisor so slices stay uniform and group-whole. The v2.6 floor-
  // only law (640-element streams, 8 splits on every K=5120/5376 shape)
  // won gemma.up 2% but REGRESSED qwen.o 5-10% on all four lanes -- the
  // refine's 8-split preference there was .item()-poisoned (the probe
  // wheel still had the per-call D2H sync on w16pt, distorting p50s).
  // Long streams win: kc floor 1024. Table: qwen 5x1024, gemma 4x1344,
  // downs 8x{1088,1344} -- best measured on every shape except gemma.up's
  // marginal 6x896 (kept 4x1344: one clean law, 2% on one cell).
  constexpr int kMinWorkitems = 32768;
  constexpr int kTileN = 2;
  constexpr int kMinSplitLen = 1024;
  const int tiles = (n + kTileN - 1) / kTileN;
  int splits = 1;
  if (tiles < kMinWorkitems) {
    const int want = kMinWorkitems / tiles;
    splits = want < 1 ? 1 : (want > 8 ? 8 : want);
    const int groups = k / 32;
    while (splits > 1 && (groups % splits) != 0) --splits;
    if (splits > groups) splits = groups < 1 ? 1 : groups;
    const int s_max = k / kMinSplitLen;
    if (splits > s_max) {
      splits = s_max < 1 ? 1 : s_max;
      while (splits > 1 && ((k / 32) % splits) != 0) --splits;
    }
  }
  // 64-aligned slice length: uniform group-divisible splits make kc exact
  // on our shapes; the round-up still guards arbitrary K, and the kernel
  // clamps each slice's k-hi to k. Every span stays a 64 multiple -- the
  // f4 pair loop never drops to the single-chunk tail (kc=672 on K=5376
  // paid it on every stream; the sweep measured gemma.up g16 -51%).
  const int kc = ((k / splits) + 63) / 64 * 64;
  return SkinnyLaunchCfg{splits, kc, splits == 1 ? 1 : 0, tiles * splits};
}

// wf: 0 = e4m3, 1 = e5m2, 2 = f4 (f4 requires group32). sc_ns/sc_gs are
// the group-scale byte strides (unused for non-group callers; defaults keep
// the canonical [N, K/32] order). pc_stride is the checkpoint-scale N
// stride (0 = broadcast scalar, 1+ = per-channel; no host sync either
// way). out/partials are unused in the direct/two-stage counterpart
// (pass nullptr).
inline void skinny_launch(
    sycl::queue& q, WeightFormat wf, bool group32, bool a_bf16, const void* a,
    const std::uint8_t* b, const std::uint8_t* bsc, float* partials,
    void* out, const float* bias, const float* pc_scale, float pt_scale,
    bool out_bf16, int m, int n, int k, int64_t sc_ns = 0,
    int64_t sc_gs = 1, int64_t pc_stride = 1) {
  TORCH_CHECK(k % 32 == 0, "skinny gemm requires K % 32 == 0, got K=", k);
  TORCH_CHECK(m >= 1 && m <= 8, "skinny gemm requires 1 <= M <= 8, got M=", m);
  if (wf == WeightFormat::kF4E2m1) {
    TORCH_CHECK(group32, "f4 skinny gemm requires group-32 e8m0 scales");
  }
  const auto cfg = skinny_pick_cfg(m, n, k);
  // column swizzle always on (refine: best mean at M=2/4/8 with tight
  // 1.15-1.25x p10/p90 spreads; composes with kc1024 for the best M=1
  // mean -- see the kernel's tile comment for the bijection proof)
  constexpr bool swizzle = true;
  const int lsz = 32;
  const int wgs = (cfg.n_workitems + lsz - 1) / lsz;

#define SKINNY_LAUNCH(WF_C, G32_C, ABF_C, DIR_C, MP_C)                     \
  q.parallel_for<SkinnyGemmStage1Kernel<WF_C, G32_C, ABF_C, DIR_C, MP_C>>( \
      sycl::nd_range<1>(wgs * lsz, lsz),                                   \
      SkinnyGemmStage1Kernel<WF_C, G32_C, ABF_C, DIR_C, MP_C>(             \
          a, b, bsc, partials, out, bias, pc_scale, pt_scale, m, n, k,     \
          cfg.splits, cfg.k_per_split, sc_ns, sc_gs, swizzle, pc_stride))

  // M_PAD dispatch: compile-time row bound over {1, 2, 4, 8} -- the decode
  // M=1 case drops the accumulator file and unrolled FMA work 8x (r126 v2
  // postmortem: v1's fixed M_PAD=8 was half the register-pressure story).
#define SKINNY_LAUNCH_M(WF_C, G32_C, ABF_C, DIR_C)                         \
  do {                                                                     \
    if (m <= 1) {                                                          \
      SKINNY_LAUNCH(WF_C, G32_C, ABF_C, DIR_C, 1);                         \
    } else if (m <= 2) {                                                   \
      SKINNY_LAUNCH(WF_C, G32_C, ABF_C, DIR_C, 2);                         \
    } else if (m <= 4) {                                                   \
      SKINNY_LAUNCH(WF_C, G32_C, ABF_C, DIR_C, 4);                         \
    } else {                                                               \
      SKINNY_LAUNCH(WF_C, G32_C, ABF_C, DIR_C, 8);                         \
    }                                                                      \
  } while (0)

  if (wf == WeightFormat::kF8E4m3) {
    if (!group32) {
      if (a_bf16 && cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E4m3, false, true, true);
      } else if (a_bf16 && !cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E4m3, false, true, false);
      } else if (!a_bf16 && cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E4m3, false, false, true);
      } else {
        SKINNY_LAUNCH_M(WeightFormat::kF8E4m3, false, false, false);
      }
    } else {
      if (a_bf16 && cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E4m3, true, true, true);
      } else if (a_bf16 && !cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E4m3, true, true, false);
      } else if (!a_bf16 && cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E4m3, true, false, true);
      } else {
        SKINNY_LAUNCH_M(WeightFormat::kF8E4m3, true, false, false);
      }
    }
  } else if (wf == WeightFormat::kF8E5m2) {
    if (!group32) {
      if (a_bf16 && cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E5m2, false, true, true);
      } else if (a_bf16 && !cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E5m2, false, true, false);
      } else if (!a_bf16 && cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E5m2, false, false, true);
      } else {
        SKINNY_LAUNCH_M(WeightFormat::kF8E5m2, false, false, false);
      }
    } else {
      if (a_bf16 && cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E5m2, true, true, true);
      } else if (a_bf16 && !cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E5m2, true, true, false);
      } else if (!a_bf16 && cfg.direct) {
        SKINNY_LAUNCH_M(WeightFormat::kF8E5m2, true, false, true);
      } else {
        SKINNY_LAUNCH_M(WeightFormat::kF8E5m2, true, false, false);
      }
    }
  } else {  // kF4E2m1 (group32 asserted above)
    if (a_bf16 && cfg.direct) {
      SKINNY_LAUNCH_M(WeightFormat::kF4E2m1, true, true, true);
    } else if (a_bf16 && !cfg.direct) {
      SKINNY_LAUNCH_M(WeightFormat::kF4E2m1, true, true, false);
    } else if (!a_bf16 && cfg.direct) {
      SKINNY_LAUNCH_M(WeightFormat::kF4E2m1, true, false, true);
    } else {
      SKINNY_LAUNCH_M(WeightFormat::kF4E2m1, true, false, false);
    }
  }
#undef SKINNY_LAUNCH_M
#undef SKINNY_LAUNCH

  if (!cfg.direct) {
    // stage 2 wants one work-item per column (N), NOT stage-1's
    // tiles x splits count (tiles = ceil(N / TN) < N)
    const int rgs = (n + lsz - 1) / lsz;
    q.parallel_for<SkinnyGemmReduceKernel>(
        sycl::nd_range<1>(rgs * lsz, lsz),
        SkinnyGemmReduceKernel(partials, out, bias, pc_scale, pt_scale, m, n,
                               cfg.splits, out_bf16, pc_stride));
  }
}

}  // namespace vllm::xpu::skinny
