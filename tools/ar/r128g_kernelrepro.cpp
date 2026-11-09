// r128g_kernelrepro.cpp -- P6.34 arc7 JM wheel-defect standalone repro.
//
// Law so far (r128f/r128h): through the wheel .so, fragment source
// halfwords at offsets == 3 (mod 4) of the staging scratch contribute
// exactly ZERO (F: 384 = 528-144 exact; C: x[7] invisible; D/E/H: k=3
// one-hots dead; H1/H2: B columns + epilogue stores at n=3 ALIVE).
// Poison over the recycled scratch block leaves out EXACTLY 24 ->
// the LOAD path zeroes those elements (store-drop + live-load ruled
// out; "stores also drop" still possible).
// Minimal probes (JIT + AOT + full device-link flags, shared/device
// scratch, 1/16 WGs) all PASS -> the trigger is the FULL kernel's code
// context or the merged .so device image. This probe compiles the
// ACTUAL JmGemmStage1Kernel class VERBATIM (torch-free) with knobs:
//   argv[1] scratch: 0 = zeroed, 1 = poisoned 0x3C (half 0x3C3C ~1.027)
//   argv[2] const_stage: 0 = real decode staging (verbatim), 1 = constant
//   argv[3] no_jm: 0 = full fragment chain, 1 = staging + barriers only
//   argv[4] m rows: 1..8 (default 1)
// Shape: the pinpoint cell M x 16 x 32, f8 e4m3, checkpoint lane
// (GROUP32=false), bf16 activations, DIRECT bf16 epilogue, splits=1.
// Run A: w all 1.0, x all 1.0 -> 32.0 correct / 24.0 defect /
//        24 + 8*1.027 poison-read.
// Run F: x[k] = k+1 -> 528.0 / 384.0 / 384 - 144 + 8.22.
// After run A the scratch is dumped (last k-step's staging): sa row 0/1,
// sb column 0/3 -- staged values at offsets 3 mod 4 => stores landed.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace vllm {
namespace xpu {
namespace jm {

// ------------------- verbatim from jm_gemm_kernel.hpp -------------------
namespace jmx = sycl::ext::oneapi::experimental::matrix;

using JmA =
    jmx::joint_matrix<sycl::sub_group, sycl::half, jmx::use::a, 8, 16,
                      jmx::layout::row_major>;
using JmB =
    jmx::joint_matrix<sycl::sub_group, sycl::half, jmx::use::b, 16, 16,
                      jmx::layout::col_major>;
using JmC = jmx::joint_matrix<sycl::sub_group, float, jmx::use::accumulator,
                              8, 16, jmx::layout::dynamic>;

inline std::uint32_t jm_lane_mask(std::uint32_t p) {
  return (p << 16) - p;
}

inline std::uint32_t e4m3_pair_f16x2(std::uint32_t x) {
  const std::uint32_t lane = x & 0x007F007Fu;
  const std::uint32_t sg = (x & 0x00800080u) << 8;
  const std::uint32_t naive = (lane << 7) + 0x20002000u;
  const std::uint32_t pe1 =
      ((lane + 0x00780078u) >> 7) & 0x00010001u;
  const std::uint32_t pv1 =
      ((lane + 0x007F007Fu) >> 7) & 0x00010001u;
  const std::uint32_t zall = jm_lane_mask(pe1 ^ 0x00010001u);
  const std::uint32_t fall = jm_lane_mask(pv1);
  const std::uint32_t m = lane & 0x00070007u;
  const std::uint32_t p4 = ((m + 0x00040004u) >> 3) & 0x00010001u;
  const std::uint32_t t = m & 0x00060006u;
  const std::uint32_t pr = ((t + 0x00020002u) >> 2) & 0x00010001u;
  const std::uint32_t t4 = (t >> 2) & 0x00010001u;
  const std::uint32_t p23 = pr & (t4 ^ 0x00010001u);
  const std::uint32_t p1 = ((m + 0x000E000Eu) >> 3) & 0x00010001u;
  const std::uint32_t hi = 0x20002000u | ((m & 0x00030003u) << 8);
  const std::uint32_t mid = 0x1C001C00u | ((m & 0x00010001u) << 9);
  const std::uint32_t fx =
      ((hi & jm_lane_mask(p4)) | (mid & jm_lane_mask(p23)) |
       (0x18001800u & jm_lane_mask(p1))) & fall;
  return (naive ^ ((naive ^ fx) & zall)) | sg;
}

inline std::uint32_t e5m2_pair_f16x2(std::uint32_t x) {
  const std::uint32_t lane = x & 0x007F007Fu;
  const std::uint32_t sg = (x & 0x00800080u) << 8;
  const std::uint32_t naive = lane << 8;
  const std::uint32_t pe1 =
      ((lane + 0x007C007Cu) >> 7) & 0x00010001u;
  const std::uint32_t pv1 =
      ((lane + 0x007F007Fu) >> 7) & 0x00010001u;
  const std::uint32_t zall = jm_lane_mask(pe1 ^ 0x00010001u);
  const std::uint32_t fx = (lane << 8) & jm_lane_mask(pv1);
  return (naive ^ ((naive ^ fx) & zall)) | sg;
}

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

inline float e8m0_to_float(std::uint32_t b) {
  return sycl::exp2(static_cast<float>(static_cast<int>(b) - 127));
}

inline std::uint16_t bf16_to_f16_bits(std::uint16_t b) {
  const float f = sycl::bit_cast<float>(
      static_cast<std::uint32_t>(b) << 16);
  return sycl::bit_cast<std::uint16_t>(static_cast<sycl::half>(f));
}

enum class WeightFormat : int { kF8E4m3 = 0, kF8E5m2 = 1, kF4E2m1 = 2 };

// Verbatim class + two diagnostic template knobs (fold away at false).
template <WeightFormat WF, bool GROUP32, bool A_BF16, bool DIRECT,
          bool CONST_STAGE = false, bool NO_JM = false>
class JmGemmStage1Kernel {
 public:
  JmGemmStage1Kernel(
      const void* a, const std::uint8_t* b, const std::uint8_t* bsc,
      sycl::half* scratch, float* partials, void* out, const float* bias,
      const float* pc_scale, float pt_scale, int m_rows, int n_cols,
      int k_elems, int splits, int k_per_split, int64_t sc_n_stride,
      int64_t sc_g_stride, bool swizzle, int64_t pc_stride)
      : a_(static_cast<const std::uint8_t*>(a)),
        b_(b),
        bsc_(bsc),
        scratch_(scratch),
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
    auto sgc = it.get_sub_group();
    const int gid = static_cast<int>(it.get_global_linear_id());
    const int lane = static_cast<int>(sgc.get_local_linear_id());
    const int sg = gid >> 4;
    const int tiles = n_ >> 4;
    const std::uint32_t tt = static_cast<std::uint32_t>(sg % tiles);
    const int ctile =
        swzl_ ? static_cast<int>((tt * 2654435761ull) % tiles)
              : static_cast<int>(tt);
    const int n0 = ctile * 16;
    const int slice = sg / tiles;
    if (slice >= s_) return;

    const int k_lo = slice * kc_;
    const int k_hi = sycl::min(k_lo + kc_, k_);
    const std::uint8_t* tbase =
        b_ + static_cast<std::size_t>(ctile) * (k_ >> 4) *
                 ((WF == WeightFormat::kF4E2m1) ? 128 : 256);

    sycl::half* sa = scratch_ + static_cast<std::size_t>(sg) * 384;
    sycl::half* sb = sa + 128;

    JmA ja;
    JmB jb;
    JmC c0, c1;
    float tot[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
      stage_a(sa, k0, lane);
      stage_b(sb, tbase, k0, lane);
      sycl::group_barrier(sgc);
      if constexpr (NO_JM) {
        // staging + barrier visibility only; no fragment ops in kernel
        sycl::group_barrier(sgc);
        continue;
      }
      jmx::joint_matrix_load(
          sgc, ja,
          sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space>(
              sa),
          16);
      jmx::joint_matrix_load(
          sgc, jb,
          sycl::multi_ptr<sycl::half, sycl::access::address_space::global_space>(
              sb),
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
    if constexpr (!NO_JM) epilogue_arr(tot, slice, n0, lane);
  }

 private:
  std::uint16_t ld_act(int row, int k) const {
    const std::uint16_t h = *reinterpret_cast<const std::uint16_t*>(
        a_ + (static_cast<std::size_t>(row) * k_ + k) * 2);
    return A_BF16 ? bf16_to_f16_bits(h) : h;
  }

  void stage_a(sycl::half* sa, int k0, int lane) const {
    if constexpr (CONST_STAGE) {
#pragma unroll
      for (int m = 0; m < 8; ++m)
        sa[m * 16 + lane] = sycl::half(float(m + 1));
    } else {
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        const std::uint16_t bits =
            m < m_ ? ld_act(m, k0 + lane) : 0;
        sa[m * 16 + lane] = sycl::bit_cast<sycl::half>(bits);
      }
    }
  }

  void stage_b(sycl::half* sb, const std::uint8_t* tbase, int k0,
               int lane) const {
    if constexpr (CONST_STAGE) {
#pragma unroll
      for (int kk = 0; kk < 16; ++kk)
        sb[lane * 16 + kk] = sycl::half(float(lane + 1));
    } else if constexpr (WF == WeightFormat::kF4E2m1) {
      const sycl::uint2 rw = *reinterpret_cast<const sycl::uint2*>(
          tbase + (static_cast<std::size_t>(k0 >> 4) << 7) +
          static_cast<std::size_t>(lane) * 8);
      const std::uint32_t wds[2] = {rw.x(), rw.y()};
#pragma unroll
      for (int v1 = 0; v1 < 2; ++v1) {
        const std::uint32_t w = wds[v1];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
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

  void epilogue_arr(const float (&tot)[8], int slice, int n0, int lane) const {
    const int nn = n0 + lane;
    for (int m = 0; m < m_ && m < 8; ++m) store_out(tot[m], m, nn, slice);
  }

  void store_out(float val, int m, int nn, int slice) const {
    if (DIRECT && !GROUP32) val *= pc_ ? pc_[nn * pcs_] : pt_;
    if (DIRECT) {
      if (bias_) val += bias_[nn];
      if (A_BF16) {
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
  const std::uint8_t* b_;
  const std::uint8_t* bsc_;
  sycl::half* scratch_;
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

}  // namespace jm
}  // namespace xpu
}  // namespace vllm

// ------------------------------ harness ---------------------------------
static float bf16_to_f(std::uint16_t b) {
  return sycl::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}
static std::uint16_t f_to_bf16(float f) {
  return static_cast<std::uint16_t>(sycl::bit_cast<std::uint32_t>(f) >> 16);
}
static float h16_to_f(std::uint16_t h) {
  return sycl::bit_cast<sycl::half>(h);
}

template <bool CS, bool NJM>
void run_case(sycl::queue &q, std::uint8_t *a, std::uint8_t *b,
              sycl::half *scr, float *partials, std::uint8_t *out, int m,
              int poison, const char *tag, const std::uint16_t *ahost) {
  const int N = 16, K = 32;
  q.memset(out, 0xEE, m * N * 2).wait();
  if (poison)
    q.memset(scr, 0x3C, 768).wait();
  else
    q.memset(scr, 0x00, 768).wait();
  q.memcpy(a, ahost, m * K * 2).wait();
  using KImpl = vllm::xpu::jm::JmGemmStage1Kernel<
      vllm::xpu::jm::WeightFormat::kF8E4m3, false, true, true, CS, NJM>;
  q.parallel_for<KImpl>(
      sycl::nd_range<1>(64, 64),
      KImpl(a, b, nullptr, scr, partials, out, nullptr, nullptr, 1.0f, m, N,
            K, 1, 64, 0, 1, false, 1));
  q.wait();
  std::vector<std::uint16_t> o(m * N);
  q.memcpy(o.data(), out, m * N * 2).wait();
  std::cout << tag << " out m0:";
  for (int n = 0; n < 8; ++n) std::cout << " " << bf16_to_f(o[n]);
  std::cout << std::endl;
  // scratch dump: last k-step staging (k0=16): sa rows 0/1, sb cols 0/3
  std::vector<std::uint16_t> s(384);
  q.memcpy(s.data(), scr, 768).wait();
  std::cout << tag << " sa r0:";
  for (int k = 0; k < 16; ++k)
    std::cout << " " << h16_to_f(s[k]);
  std::cout << "\n" << tag << " sa r1:";
  for (int k = 0; k < 16; ++k)
    std::cout << " " << h16_to_f(s[16 + k]);
  std::cout << "\n" << tag << " sb c0:";
  for (int k = 0; k < 16; ++k)
    std::cout << " " << h16_to_f(s[128 + k]);
  std::cout << "\n" << tag << " sb c3:";
  for (int k = 0; k < 16; ++k)
    std::cout << " " << h16_to_f(s[128 + 3 * 16 + k]);
  std::cout << std::endl;
}


int main(int argc, char **argv) {
  const int poison = argc > 1 ? std::atoi(argv[1]) : 0;
  const int cs = argc > 2 ? std::atoi(argv[2]) : 0;
  const int njm = argc > 3 ? std::atoi(argv[3]) : 0;
  const int m = argc > 4 ? std::atoi(argv[4]) : 1;
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << "dev: " << q.get_device().get_info<sycl::info::device::name>()
            << " poison=" << poison << " const=" << cs << " nojm=" << njm
            << " m=" << m << std::endl;

  const int N = 16, K = 32;
  // weights: 2 k-tiles of all-e4m3-1.0 (0x38), repacked column layout
  std::vector<std::uint8_t> bh(2 * 256, 0x38);
  std::uint8_t *a = sycl::malloc_device<std::uint8_t>(m * K * 2, q);
  std::uint8_t *b = sycl::malloc_device<std::uint8_t>(2 * 256, q);
  sycl::half *scr = sycl::malloc_device<sycl::half>(384, q);
  float *partials = sycl::malloc_device<float>(8 * N, q);
  std::uint8_t *out = sycl::malloc_device<std::uint8_t>(8 * N * 2, q);
  q.memcpy(b, bh.data(), 512).wait();

  // run A: x all 1.0
  std::vector<std::uint16_t> ah(m * K, f_to_bf16(1.0f));
  // run F: x[k] = k+1
  std::vector<std::uint16_t> af(m * K);
  for (int r = 0; r < m; ++r)
    for (int k = 0; k < K; ++k) af[r * K + k] = f_to_bf16(float(k + 1));

  if (cs) {
    if (njm)
      run_case<true, true>(q, a, b, scr, partials, out, m, poison, "A", ah.data());
    else
      run_case<true, false>(q, a, b, scr, partials, out, m, poison, "A", ah.data());
  } else {
    if (njm)
      run_case<false, true>(q, a, b, scr, partials, out, m, poison, "A", ah.data());
    else
      run_case<false, false>(q, a, b, scr, partials, out, m, poison, "A", ah.data());
  }
  if (cs) {
    run_case<true, false>(q, a, b, scr, partials, out, m, poison, "F", af.data());
  } else {
    run_case<false, false>(q, a, b, scr, partials, out, m, poison, "F", af.data());
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(scr, q);
  sycl::free(partials, q);
  sycl::free(out, q);
  std::cout << "REPRO_DONE" << std::endl;
  return 0;
}
