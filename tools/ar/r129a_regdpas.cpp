// r129a_regdpas.cpp -- P6.34 vehicle-(a) feasibility probe: REGISTER-operand
// DPAS (cutlass-sycl cute/arch, BSD-3, in-tree .deps) with INLINE dequant --
// no SLM staging, no global scratch. cut-10 closed vehicle (b) (SLM-staged
// joint_matrix: 0/56 cells, ~2x stock); the open question was whether
// register-DPAS + Marlin-repacked coalesced B streams + branchless decode
// sustains >= stock's 60-90%-of-floor DRAM rate.
//
// VERDICT (2026-11-09, banked Part 8 of results/2026-11-09-p634-jm-ladder):
// UNANSWERABLE -- the register-fed dpas instruction is unreachable on this
// stack. Both lowering forms of subgroup-matrix-multiply KILL the thread at
// the first fma (raw-asm XE_DPAS_TT; XE_8x16x16_F32F16F16F32_TT -> builtin /
// __spirv_SubgroupMatrixMultiplyAccumulateINTEL), under AOT bmg AND JIT,
// with [[reqd_sub_group_size(16)]] and without, and with production vxk
// spirv-ext flags. Pre-fma DBG prints fire; post-fma never (logs r129a_*).
// joint_matrix (OpJointMatrixMadINTEL) is the ONLY dpas vehicle on
// bmg/oneAPI-2026.0 -- and its staging economics already lost cut-10.
// Vehicle (a) CLOSED; the perf question is moot.
//
// Benches per shape (N,K): lane f8 (e4m3) and f4 (e2m1), each with
// CONST_A (pure B-stream+decode+dpas rate) and REAL_A (adds the per-step
// 8-half A gather from the k-blocked a16 buffer); M=1 row stored.
//   B layout (Marlin repack): [nt][kc][t16][16B(f8)/8B(f4)] -- each of the
//   16 subgroup threads loads ONE contiguous uint4/uchar8 per k-step; the
//   SG streams consecutive 256-B lines. Fragment contract (mma_traits_xe
//   wi_interleave, M=8 f16): a[t][j]=A(m=t%8, k0+2j+t/8); b[t][j]=
//   B(n0+t, k0+j); d[t][v]=C(m=v, n0+t) (column-owning epilogue).
//   e4m3 decode FTZ on both sides (host ref flushes subnormals too);
//   e2m1 exact branchless.
// Timing: 50 reps between profiled events after 3 warmups; maxrel vs host
// f64-decoded ref. Compare vs cut-10 stock bars (write-up side).
#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"                     // config umbrella
#include <cute/int_tuple.hpp>                    // ceil_div (sycl_vec dep)
#include <cute/numeric/integral_constant.hpp>    // Int (sycl_vec dep)
#include <cute/arch/mma_xe_legacy.hpp>           // builtin/spirv dpas (raw-asm
                                                 // mma_xe.hpp traps on bmg AOT)
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

using half_u = std::uint16_t;

// ---- host decoders (FTZ e4m3 to mirror the kernel) ----
static inline float e4m3_to_f32(std::uint8_t v) {
  int s = (v >> 7) & 1, e = (v >> 3) & 15, m = v & 7;
  float val = e ? ldexpf((float)(8 + m), e - 10) : 0.f;
  return s ? -val : val;
}
static inline float e2m1_to_f32(std::uint8_t v) {
  static const float t[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6,
                              -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  return t[v & 15];
}
static inline float f16_to_f32(std::uint16_t h) {
  int s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float val;
  if (e == 0) val = (m == 0) ? 0.f : ldexpf((float)m, -24);
  else if (e == 31) val = m ? NAN : INFINITY;
  else val = ldexpf((float)(1024 + m), e - 25);
  return s ? -val : val;
}
// exact f16 bits for m/1024 with integer m, |m| <= 1023 (values exactly
// representable: mantissa <= 10 bits, normal range)
static inline half_u frac1024_to_f16(int m) {
  if (m == 0) return 0;
  std::uint32_t s = m < 0 ? 0x8000u : 0;
  std::uint32_t a = (std::uint32_t)std::abs(m);
  int p = 0;
  while ((a << p) < 1024) ++p;          // a*2^p in [1024,2047]
  std::uint32_t mant = (a << p) & 0x3FFu;
  std::uint32_t e = (std::uint32_t)(10 - p + 15);  // value = a*2^-10
  return (half_u)(s | (e << 10) | mant);
}

static std::uint32_t lcg(std::uint32_t& s) {
  s = s * 1664525u + 1013904223u; return s >> 8;
}

template <int LANE, bool CONST_A>
struct Probe {
  static constexpr int SG = 16;      // subgroup size (sgsize 16)
  static constexpr int WGT = 64;     // threads per WG = 4 SGs
  const std::uint8_t* rp;            // repacked weights
  const half_u* a16;                 // [K/16][8][16] f16 activations
  float* out;                        // [M][N]
  int n, k, m;
  [[sycl::reqd_sub_group_size(16)]]  // fragment contract assumes sg=16 (XE_DPAS
  void operator()(sycl::nd_item<1> it) const {  // + spirv-mma operand 16 ditto)
    using Op = cute::XE_8x16x16_F32F16F16F32_TT;  // legacy builtin dpas
    using F8 = cute::intel::float8;
    using S8 = cute::intel::short8;
    using I8 = cute::intel::int8;
    sycl::sub_group sgc = it.get_sub_group();
    const int gid = (int)(it.get_group(0) * (WGT / SG) + sgc.get_group_id()[0]);
    const int nt = gid;                       // one SG per n-tile
#ifdef R129_DEBUG
    if (nt == 0 && sgc.get_local_id()[0] == 0)
      printf("DBG nt=%d n=%d k=%d kcmax=%d rp0=%02x%02x%02x%02x\n", nt, n, k,
             k / 16, rp[0], rp[1], rp[2], rp[3]);
#endif
    if (nt >= n / 16) return;
    const int t = (int)sgc.get_local_id()[0]; // 0..15
    const int n0 = nt * 16;
    const int kc_max = k / 16;
    F8 d;
    for (int v = 0; v < 8; ++v) d[v] = 0.f;
    S8 ac;
    if (CONST_A)  // row-0..7 pattern of the kc=0 block, reused every step
      for (int j = 0; j < 8; ++j)
        ac[j] = (short)a16[((std::size_t)(t & 7)) * 16 + 2 * j + (t >> 3)];
    for (int kc = 0; kc < kc_max; ++kc) {
      // ---- B: one contiguous 16B (f8) / 8B (f4) vector load per thread ----
      std::uint32_t b16[16];  // decoded f16 halves -> packed into I8 pairs
      if (LANE == 0) {
        sycl::uint4 v = *reinterpret_cast<const sycl::uint4*>(
            rp + ((std::size_t)nt * kc_max + kc) * 256 + (std::size_t)t * 16);
        const std::uint32_t w[4] = {v.x(), v.y(), v.z(), v.w()};
        for (int j = 0; j < 16; ++j) {
          std::uint32_t u = (w[j >> 2] >> ((j & 3) * 8)) & 0xFF;
          std::uint32_t e = (u >> 3) & 15, mm = u & 7, s = (u >> 7) & 1;
          std::uint32_t bits = e ? (((e + 8) << 10) | (mm << 7)) : 0u;
          b16[j] = (s << 15) | bits;
        }
      } else {
        sycl::uchar8 v = *reinterpret_cast<const sycl::uchar8*>(
            rp + ((std::size_t)nt * kc_max + kc) * 128 + (std::size_t)t * 8);
        const std::uint8_t w[8] = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]};
        for (int j = 0; j < 16; ++j) {
          std::uint32_t u = (std::uint32_t)((w[j >> 3] >> ((j & 7) * 4)) & 15);
          std::uint32_t s = u >> 3, e = (u >> 1) & 3, mm = u & 1;
          std::uint32_t bits = (e == 0) ? (mm ? 0x3800u : 0u)
                                        : (((e + 14) << 10) | (mm << 9));
          b16[j] = (s << 15) | bits;
        }
      }
      I8 b;  // 16 f16 halves as 8 u32 pairs (legacy B register form)
      for (int i = 0; i < 8; ++i)
        b[i] = (int)(b16[2 * i] | (b16[2 * i + 1] << 16));
      // ---- A fragment: row m=t%8, k = 2j + t/8 ----
      S8 a;
      if (CONST_A) {
        a = ac;
      } else {
        const half_u* base = a16 + ((std::size_t)kc * 8 + (t & 7)) * 16 + (t >> 3);
        for (int j = 0; j < 8; ++j) a[j] = (short)base[2 * j];
      }
#ifdef R129_DEBUG
      if (nt == 0 && t == 0 && kc == 0)
        printf("DBG pre-fma kc0 b0=%08x b7=%08x a0=%04x\n", (unsigned)b[0],
               (unsigned)b[7], (unsigned)(unsigned short)a[0]);
#endif
#ifndef R129_NOMMA
      Op::fma(d, a, b, d);
#else
      if (b[0] == 0x7FFF7FFFu && a[0] == (short)0x7FFF) d[0] += 1.f;  // keep loads alive
#endif
#ifdef R129_DEBUG
      if (nt == 0 && t == 0 && kc == 0)
        printf("DBG post-fma kc0 d0=%f\n", d[0]);
#endif
    }
    // ---- epilogue: thread t owns column n0+t ----
    const std::size_t col = (std::size_t)n0 + t;
    for (int r = 0; r < m; ++r) out[(std::size_t)r * n + col] = d[r];
  }
};

template <int LANE, bool CONST_A>
static void run(sycl::queue& q, const char* tag, int N, int K, int M,
                const std::vector<std::uint8_t>& w8,
                const std::vector<half_u>& a16h, std::vector<float>& out,
                const std::vector<float>& ref) {
  const int kch = K / 16;
  std::vector<std::uint8_t> rp((size_t)(N / 16) * kch * (LANE == 0 ? 256 : 128), 0);
  for (int nt = 0; nt < N / 16; ++nt)
    for (int kc = 0; kc < kch; ++kc)
      for (int t = 0; t < 16; ++t) {
        std::uint8_t* dst = &rp[((size_t)nt * kch + kc) * (LANE == 0 ? 256 : 128) + (size_t)t * (LANE == 0 ? 16 : 8)];
        for (int j = 0; j < 16; ++j) {
          int kk = kc * 16 + j;
          if (LANE == 0) {
            dst[j] = w8[(size_t)(nt * 16 + t) * K + kk];
          } else {
            std::uint8_t byte = w8[(size_t)(nt * 16 + t) * (K / 2) + kk / 2];
            std::uint8_t nib = (kk & 1) ? (byte >> 4) : (byte & 15);
            dst[j >> 1] |= (std::uint8_t)(nib << ((j & 1) * 4));
          }
        }
      }
  std::uint8_t* rp_d = sycl::malloc_device<std::uint8_t>(rp.size(), q);
  half_u* a_d = sycl::malloc_device<half_u>(a16h.size(), q);
  float* o_d = sycl::malloc_device<float>(out.size(), q);
  q.memcpy(rp_d, rp.data(), rp.size());
  q.memcpy(a_d, a16h.data(), a16h.size() * 2);
  q.memset(o_d, 0, out.size() * 4).wait();
  const int wgs = ((N / 16 + 3) / 4) * Probe<LANE, CONST_A>::WGT;
  auto launch = [&](sycl::handler& cgh) {
    cgh.parallel_for<Probe<LANE, CONST_A>>(
        sycl::nd_range<1>(wgs, Probe<LANE, CONST_A>::WGT),
        Probe<LANE, CONST_A>{rp_d, a_d, o_d, N, K, M});
  };
  for (int i = 0; i < 3; ++i) (void)q.submit(launch);
  sycl::event e0 = q.submit(launch);
  for (int i = 0; i < 48; ++i) (void)q.submit(launch);
  sycl::event e1 = q.submit(launch);
  e1.wait();
  q.memcpy(out.data(), o_d, out.size() * 4).wait();
  double ms = (e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
               e0.get_profiling_info<sycl::info::event_profiling::command_start>()) /
              1e6 / 50.0;
  double maxrel = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    double r = ref[i] ? std::fabs(out[i] - ref[i]) / std::fabs(ref[i]) : 0;
    if (r > maxrel) maxrel = r;
  }
  double bytes = (double)N * K * (LANE == 0 ? 1 : 0.5);
  printf("%-10s %-3s %-3s %8.1f us %7.0f GB/s  maxrel %.2e\n", tag,
              LANE == 0 ? "f8" : "f4", CONST_A ? "cA" : "rA", ms * 1e3,
              bytes / (ms * 1e-3) / 1e9, maxrel);
  sycl::free(rp_d, q); sycl::free(a_d, q); sycl::free(o_d, q);
}

int main(int argc, char** argv) {
  struct SH { int n, k; const char* tag; } shapes[] = {
      {5120, 8704, "qwen.up"}, {5376, 5376, "gemma.up"}, {3584, 18944, "deepN"}};
  sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::enable_profiling(),
                                       sycl::property::queue::in_order()}};
  printf("device %s sgsize-default\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());
  int only = argc > 1 ? std::atoi(argv[1]) : -1;
  for (auto& sh : shapes) {
    if (only >= 0 && (int)(&sh - shapes) != only) continue;
    const int N = sh.n, K = sh.k, M = 1;
    std::uint32_t seed = 12345;
    std::vector<std::uint8_t> w8((size_t)N * K);      // f8 view full, f4 first half
    std::vector<half_u> a16h((size_t)(K / 16) * 8 * 16, 0);
    std::vector<float> out((size_t)M * N), ref((size_t)M * N);
    std::vector<int> act(K);
    for (int i = 0; i < K; ++i) act[i] = (int)(lcg(seed) % 2001) - 1000;
    for (int kb = 0; kb < K / 16; ++kb)
      for (int j = 0; j < 16; ++j)
        a16h[(size_t)kb * 128 + j] = frac1024_to_f16(act[kb * 16 + j]);
    for (auto& b : w8) b = (std::uint8_t)(lcg(seed) & 0xFF);
    // f8 ref
    for (int n_ = 0; n_ < N; ++n_) {
      double s = 0;
      for (int k = 0; k < K; ++k)
        s += (double)e4m3_to_f32(w8[(size_t)n_ * K + k]) *
             f16_to_f32(a16h[(size_t)(k / 16) * 128 + (k % 16)]);
      ref[n_] = (float)s;
    }
    run<0, true>(q, sh.tag, N, K, M, w8, a16h, out, ref);
    run<0, false>(q, sh.tag, N, K, M, w8, a16h, out, ref);
    // f4 ref (nibbles of the same bytes)
    for (int n_ = 0; n_ < N; ++n_) {
      double s = 0;
      for (int k = 0; k < K; ++k) {
        std::uint8_t byte = w8[(size_t)n_ * (K / 2) + k / 2];
        s += (double)e2m1_to_f32((k & 1) ? (byte >> 4) : (byte & 15)) *
             f16_to_f32(a16h[(size_t)(k / 16) * 128 + (k % 16)]);
      }
      ref[n_] = (float)s;
    }
    run<1, true>(q, sh.tag, N, K, M, w8, a16h, out, ref);
    run<1, false>(q, sh.tag, N, K, M, w8, a16h, out, ref);
  }
  printf("R129A_DONE\n");
  return 0;
}
