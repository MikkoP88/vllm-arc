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
// MXFP4 e2m1 via the 8-entry magnitude set (sign in bit 3), e8m0 group
// scales via exp2 -- and
// math accumulates in fp32 with a fixed-order split-K reduction
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

// e2m1 nibble (0..15) -> f32: {0, .5, 1, 1.5, 2, 3, 4, 6} magnitude set
// (and doubles), sign in bit 3. An 8-entry table select is register-cheap
// and branch-free.
inline float e2m1_to_float(std::uint32_t nib) {
  static constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                    2.0f, 3.0f, 4.0f, 6.0f};
  const float v = kMag[nib & 0x7u];
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
//
// r126 postmortem (the v1 kernel measured 7-30x SLOWER than stock at M<=8,
// ~20 GB/s effective on 600+ GB/s parts): the v1 inner loop streamed weights
// with 32-bit scalar loads while holding acc[TN][M_PAD]=32 fp32 registers
// live and unrolling M_PAD=8 even at runtime M=1 -- too few bytes in flight
// per thread and heavy register pressure serialized the load pipeline; time
// came out exactly proportional to weight bytes at 3% of bandwidth.
//
// v2 memory design:
//   * weights stream in 128-bit loads: one sycl::uint4 covers CHUNK=16 f8
//     elements or CHUNK=32 f4 nibbles (f4 chunk = exactly one e8m0 group);
//     activations load as uint4 too (8 bf16 halves per load)
//   * M_PAD is dispatched at launch over {1, 2, 4, 8}: at decode M=1 the
//     accumulator file and FMA work drop 8x and the freed registers let the
//     compiler software-pipeline more weight loads per thread
//   * TN=2 (was 4) doubles the work-item count; skinny_pick_cfg targets
//     8192 work-items and group-aligned split-K reaches it on narrow N
//   * decode-and-consume: each column's decoded w[CHUNK] registers live
//     only inside its j iteration; for M_PAD <= 2 the activation chunk is
//     hoisted out of the j loop (loaded once per k step)
//   * partials keep the [S][8][N] layout (literal-8 stride) for every
//     M_PAD instantiation -- stage 2 and the interface alloc agree on 8
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
      int64_t sc_g_stride)
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
        kc_(k_per_split) {}

  void operator()(sycl::nd_item<1> it) const {
    const int tid = static_cast<int>(it.get_global_linear_id());
    const int tiles = (n_ + TN - 1) / TN;
    const int n0 = (tid % tiles) * TN;
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
    for (int k0 = k_lo; k0 < k_hi; k0 += CHUNK) {
      // Hoisted activation chunk for small M_PAD: decode a[mi][CHUNK] once
      // per k step instead of once per column (registers afford it at
      // M_PAD <= 2; larger M_PAD re-decodes inside the j loop).
      [[maybe_unused]] float ah[M_PAD <= 2 ? M_PAD : 1]
                           [M_PAD <= 2 ? CHUNK : 1];
      if constexpr (M_PAD <= 2) {
#pragma unroll
        for (int mi = 0; mi < M_PAD; ++mi) {
          if (mi >= m_) {
#pragma unroll
            for (int c = 0; c < CHUNK; ++c) ah[mi][c] = 0.0f;
            continue;
          }
          const sycl::uint4* av = reinterpret_cast<const sycl::uint4*>(
              a_ + (mi * k_ + k0) * 2);
#pragma unroll
          for (int v = 0; v < AW; ++v) {
            const sycl::uint4 pv = av[v];
            // each 32-bit lane holds TWO halves: x=(0,1) y=(2,3) z=(4,5)
            // w=(6,7) -- lane h>>1, half h&1 (r126 v2.1: the h<4 gate
            // shifted lanes by >=32 and garbled elements 2,3,6,7)
            const std::uint32_t alane[4] = {pv.x(), pv.y(), pv.z(), pv.w()};
#pragma unroll
            for (int h = 0; h < 8; ++h) {
              const std::uint32_t hw =
                  (alane[h >> 1] >> (16 * (h & 1))) & 0xFFFFu;
              ah[mi][v * 8 + h] =
                  A_BF16 ? bf16_to_float(hw) : f16_to_float(hw);
            }
          }
        }
      }

#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int n = n0 + j;
        if (n >= n_) continue;  // acc[j] stays 0; epilogue skips the store

        // One 128-bit weight load -> decode CHUNK values (transient regs).
        float w[CHUNK];
        if constexpr (WF == WeightFormat::kF4E2m1) {
          float gsc = 1.0f;
          if constexpr (GROUP32) {
            gsc = e8m0_to_float(bsc_[n * sc_ns_ + (k0 >> 5) * sc_gs_]);
          }
          const sycl::uint4 pv = *reinterpret_cast<const sycl::uint4*>(
              b_ + n * (k_ / 2) + (k0 >> 1));
          const std::uint32_t lane[4] = {pv.x(), pv.y(), pv.z(), pv.w()};
#pragma unroll
          for (int c = 0; c < CHUNK; ++c) {
            const std::uint32_t byte = lane[c >> 3] >> (8 * ((c >> 1) & 3));
            const std::uint32_t nib = (byte >> (4 * (c & 1))) & 0xFu;
            w[c] = e2m1_to_float(nib) * gsc;
          }
        } else {
          float gsc = 1.0f;
          if constexpr (GROUP32) {
            gsc = e8m0_to_float(bsc_[n * sc_ns_ + (k0 >> 5) * sc_gs_]);
          }
          const sycl::uint4 pv = *reinterpret_cast<const sycl::uint4*>(
              b_ + n * k_ + k0);
          const std::uint32_t lane[4] = {pv.x(), pv.y(), pv.z(), pv.w()};
#pragma unroll
          for (int c = 0; c < CHUNK; ++c) {
            const std::uint32_t bits = (lane[c >> 2] >> (8 * (c & 3))) & 0xFFu;
            const float v = (WF == WeightFormat::kF8E4m3)
                                ? e4m3_to_float(bits)
                                : e5m2_to_float(bits);
            w[c] = v * gsc;
          }
        }

        // Accumulate M rows against the decoded chunk.
        if constexpr (M_PAD <= 2) {
#pragma unroll
          for (int mi = 0; mi < M_PAD; ++mi) {
#pragma unroll
            for (int c = 0; c < CHUNK; ++c) acc[j][mi] += ah[mi][c] * w[c];
          }
        } else {
#pragma unroll
          for (int mi = 0; mi < M_PAD; ++mi) {
            if (mi >= m_) break;
            float a[CHUNK];
            const sycl::uint4* av = reinterpret_cast<const sycl::uint4*>(
                a_ + (mi * k_ + k0) * 2);
#pragma unroll
            for (int v = 0; v < AW; ++v) {
              const sycl::uint4 pv = av[v];
              // two halves per 32-bit lane (see the hoisted-path comment)
              const std::uint32_t alane[4] = {pv.x(), pv.y(), pv.z(),
                                              pv.w()};
#pragma unroll
              for (int h = 0; h < 8; ++h) {
                const std::uint32_t hw =
                    (alane[h >> 1] >> (16 * (h & 1))) & 0xFFFFu;
                a[v * 8 + h] =
                    A_BF16 ? bf16_to_float(hw) : f16_to_float(hw);
              }
            }
#pragma unroll
            for (int c = 0; c < CHUNK; ++c) acc[j][mi] += a[c] * w[c];
          }
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
        if (DIRECT && !GROUP32) v *= pc_ ? pc_[n] : pt_;
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
      int splits, bool out_bf16)
      : p_(partials),
        out_(static_cast<std::uint8_t*>(out)),
        bias_(bias),
        pc_(pc_scale),
        pt_(pt_scale),
        m_(m_rows),
        n_(n_cols),
        s_(splits),
        bf16_(out_bf16) {}

  void operator()(sycl::nd_item<1> it) const {
    const int n = static_cast<int>(it.get_global_linear_id());
    if (n >= n_) return;
    for (int mi = 0; mi < m_; ++mi) {
      float v = 0.0f;
      for (int slice = 0; slice < s_; ++slice) {  // fixed order: deterministic
        v += p_[(static_cast<std::size_t>(slice) * 8 + mi) * n_ + n];
      }
      v *= pc_ ? pc_[n] : pt_;
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
  // 128-bit loads) wants more concurrent streams to cover HBM latency.
  // The v2 run confirmed concurrency is a binding constraint: the two
  // DIRECT (unsplit) shapes were the two worst ratios (3.2-3.4x) while
  // split shapes landed 1.2-1.7x -- so v2.2 aims for 32768 work-items
  // (partial roundtrip is S*8*N*4 B, still negligible at these N) and the
  // branchless dequant removes the ~60-80 GB/s ALU ceiling observed
  // independent of thread count.
  constexpr int kMinWorkitems = 32768;
  constexpr int kTileN = 2;
  const int tiles = (n + kTileN - 1) / kTileN;
  int splits = 1;
  if (tiles < kMinWorkitems) {
    const int want = kMinWorkitems / tiles;
    splits = want < 1 ? 1 : (want > 8 ? 8 : want);
    const int groups = k / 32;
    while (splits > 1 && (groups % splits) != 0) --splits;
    if (splits > groups) splits = groups < 1 ? 1 : groups;
  }
  const int kc = ((k / splits) + 31) / 32 * 32;
  return SkinnyLaunchCfg{splits, kc, splits == 1 ? 1 : 0, tiles * splits};
}

// wf: 0 = e4m3, 1 = e5m2, 2 = f4 (f4 requires group32). sc_ns/sc_gs are
// the group-scale byte strides (unused for non-group callers; defaults keep
// the canonical [N, K/32] order). out/partials are unused in the
// direct/two-stage counterpart (pass nullptr).
inline void skinny_launch(
    sycl::queue& q, WeightFormat wf, bool group32, bool a_bf16, const void* a,
    const std::uint8_t* b, const std::uint8_t* bsc, float* partials,
    void* out, const float* bias, const float* pc_scale, float pt_scale,
    bool out_bf16, int m, int n, int k, int64_t sc_ns = 0,
    int64_t sc_gs = 1) {
  TORCH_CHECK(k % 32 == 0, "skinny gemm requires K % 32 == 0, got K=", k);
  TORCH_CHECK(m >= 1 && m <= 8, "skinny gemm requires 1 <= M <= 8, got M=", m);
  if (wf == WeightFormat::kF4E2m1) {
    TORCH_CHECK(group32, "f4 skinny gemm requires group-32 e8m0 scales");
  }
  const auto cfg = skinny_pick_cfg(m, n, k);
  const int lsz = 32;
  const int wgs = (cfg.n_workitems + lsz - 1) / lsz;

#define SKINNY_LAUNCH(WF_C, G32_C, ABF_C, DIR_C, MP_C)                     \
  q.parallel_for<SkinnyGemmStage1Kernel<WF_C, G32_C, ABF_C, DIR_C, MP_C>>( \
      sycl::nd_range<1>(wgs * lsz, lsz),                                   \
      SkinnyGemmStage1Kernel<WF_C, G32_C, ABF_C, DIR_C, MP_C>(             \
          a, b, bsc, partials, out, bias, pc_scale, pt_scale, m, n, k,     \
          cfg.splits, cfg.k_per_split, sc_ns, sc_gs))

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
                               cfg.splits, out_bf16));
  }
}

}  // namespace vllm::xpu::skinny
