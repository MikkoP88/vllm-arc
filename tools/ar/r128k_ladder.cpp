// r128k_ladder.cpp -- P6.34 arc7 JM per-step cost ladder (torch-free).
// Stage-B law: jm ~0.44 ms for 160 k-steps/WG (qwen.up M=1) = ~2.8 us
// per 16-k step = ~6700 cycles: no modeled component (subgroup barrier,
// 768 B staging, cooperative loads, mad, applies) is anywhere near that.
// This ladder isolates the dominant term and pre-measures the fix
// candidates. Geometry mirrors jm_launch exactly: lsz 64 (4 sg/WG),
// subs = tiles*splits, scratch slot per GLOBAL sg, golden-ratio swizzle.
// Variants (cumulative V0->V5 = current kernel; V6+ = candidates):
//  V0 loop only            V1 +staging           V2 +2 barriers
//  V3 +loads+mad           V4 +zero applies      V5 +read apply (FULL)
//  V6 reorder [bar;load;stage(next);mad] + zero-once, read/step
//  V7 V6 + alternate-target accumulation, read at end (3 frag ops/step)
//  V8 two n-tiles per sg (current order), shared ja, zero-once,
//     4-C alt-target, read at end -- amortizes A-side frag ops
// GROUP32=false ladder (checkpoint lane): scales fold at epilogue.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#include <sycl/sycl.hpp>

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
  const std::uint32_t pe1 = ((lane + 0x00780078u) >> 7) & 0x00010001u;
  const std::uint32_t pv1 = ((lane + 0x007F007Fu) >> 7) & 0x00010001u;
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
inline std::uint16_t bf16_to_f16_bits(std::uint16_t b) {
  const float f =
      sycl::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
  return sycl::bit_cast<std::uint16_t>(static_cast<sycl::half>(f));
}
using hptr = sycl::multi_ptr<sycl::half,
                             sycl::access::address_space::global_space>;

template <int VAR>
class Lab {
 public:
  Lab(const std::uint8_t* a, const std::uint8_t* b, sycl::half* scratch,
      float* out, int m, int n, int k, int splits, int kc)
      : a_(a), b_(b), scr_(scratch), out_(out), m_(m), n_(n), k_(k),
        s_(splits), kc_(kc) {}

  void operator()(sycl::nd_item<1> it) const {
    auto sgc = it.get_sub_group();
    const int gid = static_cast<int>(it.get_global_linear_id());
    const int lane = static_cast<int>(sgc.get_local_linear_id());
    const int sg = gid >> 4;
    const int tiles = n_ >> 4;
    const std::uint32_t tt = static_cast<std::uint32_t>(sg % tiles);
    const int ctile = static_cast<int>((tt * 2654435761ull) % tiles);
    const int n0 = ctile * 16;
    const int slice = sg / tiles;
    if (slice >= s_) return;
    const int k_lo = slice * kc_;
    const int k_hi = sycl::min(k_lo + kc_, k_);
    const std::size_t tstep = static_cast<std::size_t>(k_ >> 4) * 256;
    const std::uint8_t* tbase = b_ + ctile * tstep;
    const std::uint8_t* tbase2 =
        b_ + ((ctile + 1) % tiles) * tstep;  // V8 second tile
    // slot A at sg*768, slot B (ping-pong / tile-1 B) at +384 halves
    sycl::half* sa0 = scr_ + static_cast<std::size_t>(sg) * 768;
    sycl::half* sb0 = sa0 + 128;
    sycl::half* sa1 = sa0 + 384;
    sycl::half* sb1 = sa1 + 128;

    JmA ja;
    JmB jb, jb2;
    JmC c0, c1, c0b, c1b;
    float tot0[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float tot1[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    if constexpr (VAR >= 6) {  // zero-once: c0 is the read-only addend
      jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
    }
    if constexpr (VAR == 8) {
      jmx::joint_matrix_apply(sgc, c0b, [](float& e) { e = 0.0f; });
    }

    if constexpr (VAR >= 6 && VAR <= 7) {
      // single-barrier ping-pong: pre-stage step 0, then per iteration
      // load(cur); mad; stage(next -> other slot); barrier; swap.
      // Sound: load(cur)@i < bar@end_i < stage(cur)@i+1, and the OTHER
      // slot's staging@i is visible to load@i+1 through the same bar.
      stage_a(sa0, k_lo, lane);
      stage_b(sb0, tbase, k_lo, lane);
      sycl::group_barrier(sgc);
      for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
        jmx::joint_matrix_load(sgc, ja, hptr(sa0), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
        if constexpr (VAR == 7) {  // acc stays in fragments
          if (((k0 - k_lo) & 16) == 0)
            jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
          else
            jmx::joint_matrix_mad(sgc, c0, ja, jb, c1);
        } else {  // V6: mad from the once-zeroed c0 addend
          jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        }
        const int kn = k0 + 16;
        if (kn < k_hi) {
          stage_a(sa1, kn, lane);
          stage_b(sb1, tbase, kn, lane);
        }
        sycl::group_barrier(sgc);
        sycl::half* t = sa0;
        sa0 = sa1;
        sa1 = t;
        t = sb0;
        sb0 = sb1;
        sb1 = t;
        if constexpr (VAR == 6) {
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
      }
    } else {
    for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
      // ---- V0..V5, V8: current-order ladder ----
      if constexpr (VAR == 9) {
        stage_a(sa0, k0, lane);          // bisect: A staging only
      } else if constexpr (VAR == 10) {
        stage_b(sb0, tbase, k0, lane);   // bisect: B staging only
      } else if constexpr (VAR >= 1) {
        stage_a(sa0, k0, lane);
        if constexpr (VAR == 8)
          stage_b(sb0, tbase, k0, lane), stage_b(sb1, tbase2, k0, lane);
        else
          stage_b(sb0, tbase, k0, lane);
      }
      if constexpr (VAR >= 2) {
        sycl::group_barrier(sgc);
        if constexpr (VAR >= 3) {
          jmx::joint_matrix_load(sgc, ja, hptr(sa0), 16);
          jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
          if constexpr (VAR == 8)
            jmx::joint_matrix_load(sgc, jb2, hptr(sb1), 16);
        }
        sycl::group_barrier(sgc);
        if constexpr (VAR >= 4) {
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        }
        if constexpr (VAR >= 3) {
          if constexpr (VAR == 8) {
            if (((k0 - k_lo) & 16) == 0) {
              jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
              jmx::joint_matrix_mad(sgc, c1b, ja, jb2, c0b);
            } else {
              jmx::joint_matrix_mad(sgc, c0, ja, jb, c1);
              jmx::joint_matrix_mad(sgc, c0b, ja, jb2, c1b);
            }
          } else {
            jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
          }
        }
        if constexpr (VAR >= 5) {
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
      }
    }
    }  // else (V0..V5, V8 loop)
    // read-once epilogue for the fragment-accumulation variants
    if constexpr (VAR == 7 || VAR == 8) {
      const int nsteps = (k_hi - k_lo) / 16;
      JmC& ca = (nsteps & 1) ? c1 : c0;
      int i = 0;
      jmx::joint_matrix_apply(sgc, ca,
                              [&](float& e) { tot0[i++] += e; });
      if constexpr (VAR == 8) {
        JmC& cb = (nsteps & 1) ? c1b : c0b;
        i = 0;
        jmx::joint_matrix_apply(sgc, cb,
                                [&](float& e) { tot1[i++] += e; });
      }
    }
    const int nn = n0 + lane;
    if (nn < n_)
      out_[static_cast<std::size_t>(nn)] =
          tot0[0] + tot0[1] + tot0[2] + tot0[3] + tot0[4] + tot0[5] +
          tot0[6] + tot0[7] + tot1[0] + tot1[1] + tot1[2] + tot1[3] +
          tot1[4] + tot1[5] + tot1[6] + tot1[7];
  }

 private:
  void stage_a(sycl::half* sa, int k0, int lane) const {
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      const std::uint16_t bits =
          m < m_
              ? *reinterpret_cast<const std::uint16_t*>(
                    a_ + (static_cast<std::size_t>(m) * k_ + k0 + lane) * 2)
              : 0;
      sa[m * 16 + lane] =
          sycl::bit_cast<sycl::half>(bf16_to_f16_bits(bits));
    }
  }
  void stage_b(sycl::half* sb, const std::uint8_t* tbase, int k0,
               int lane) const {
    const sycl::uint4 rw = *reinterpret_cast<const sycl::uint4*>(
        tbase + (static_cast<std::size_t>(k0 >> 4) << 8) +
        static_cast<std::size_t>(lane) * 16);
    const std::uint32_t wds[4] = {rw.x(), rw.y(), rw.z(), rw.w()};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const std::uint32_t w = wds[j];
      const std::uint32_t x1 = (w & 0x000000FFu) | ((w & 0x0000FF00u) << 8);
      const std::uint32_t x2 =
          ((w >> 16) & 0x000000FFu) | ((w >> 8) & 0x00FF0000u);
      const std::uint32_t p1 = e4m3_pair_f16x2(x1);
      const std::uint32_t p2 = e4m3_pair_f16x2(x2);
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
  const std::uint8_t* a_;
  const std::uint8_t* b_;
  sycl::half* scr_;
  float* out_;
  int m_, n_, k_, s_, kc_;
};

static int pick_splits(int tiles, int k, int sg_floor) {
  int splits = 1;
  if (tiles < sg_floor) {
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
  return splits;
}

#define LAB_CASE(VV)                                                   \
  case VV:                                                             \
    q.parallel_for<Lab<VV>>(                                           \
        sycl::nd_range<1>(wgs, lsz),                                   \
        Lab<VV>((const std::uint8_t*)a, (const std::uint8_t*)b,        \
                (sycl::half*)scr, (float*)out, M, npad, K, splits, kc));       \
    break;

int main(int argc, char** argv) {
  const int K = argc > 1 ? std::atoi(argv[1]) : 5120;
  const int N = argc > 2 ? std::atoi(argv[2]) : 8704;
  const int M = argc > 3 ? std::atoi(argv[3]) : 1;
  const int SGF = argc > 4 ? std::atoi(argv[4]) : 2048;
  const int R = argc > 5 ? std::atoi(argv[5]) : 50;
  const int vlo = argc > 6 ? std::atoi(argv[6]) : 0;   // variant range
  const int vhi = argc > 7 ? std::atoi(argv[7]) : 10;  // (isolate crashes)

  sycl::queue q{sycl::gpu_selector_v};
  std::printf("device: %s\n",
              q.get_device()
                  .get_info<sycl::info::device::name>()
                  .c_str());

  const int tiles = (N + 15) / 16;
  const int npad = tiles * 16;
  const int splits = pick_splits(tiles, K, SGF);
  const int kc = ((K / splits) + 63) / 64 * 64;
  const int subs = tiles * splits;
  const int steps = kc / 16;
  const int lsz = 64;
  const int wgs = ((subs * 16 + lsz - 1) / lsz) * lsz;

  auto a = sycl::malloc_device(8 * K * 2, q);
  auto b = sycl::malloc_device(
      static_cast<std::size_t>(tiles) * (K / 16) * 256, q);
  auto scr = sycl::malloc_device(
      static_cast<std::size_t>(subs) * 768 * 2, q);  // + slot B
  auto out = sycl::malloc_device(npad * 8 * 4, q);
  std::vector<std::uint8_t> h8(static_cast<std::size_t>(8) * K * 2, 0x30);
  q.memcpy(a, h8.data(), h8.size());
  std::vector<std::uint8_t> hw(
      static_cast<std::size_t>(tiles) * (K / 16) * 256, 0x38);
  q.memcpy(b, hw.data(), hw.size());
  q.memset(out, 0, npad * 8 * 4);
  q.wait();

  std::printf(
      "K=%d N=%d M=%d SGF=%d tiles=%d splits=%d kc=%d subs=%d "
      "steps/WG=%d wgs=%d\n",
      K, N, M, SGF, tiles, splits, kc, subs, steps, wgs);

  const char* names[] = {"V0 loop",         "V1 +stage",
                         "V2 +2bar",        "V3 +load+mad",
                         "V4 +2zero",       "V5 FULL now",
                         "V6 rord+zonc",    "V7 altacc+rdend",
                         "V8 twotile",      "V9 stageA only", "V10 stageB only"};
  float ref_out = 0.f;
  for (int v = vlo; v <= vhi; ++v) {
    std::printf("V%d submit\n", v);
    std::fflush(stdout);
    for (int i = 0; i < 3; ++i) {  // warmup
      switch (v) {
        LAB_CASE(0)
        LAB_CASE(1)
        LAB_CASE(2)
        LAB_CASE(3)
        LAB_CASE(4)
        LAB_CASE(5)
        LAB_CASE(6)
        LAB_CASE(7)
        LAB_CASE(8)
        LAB_CASE(9)
        LAB_CASE(10)
      }
    }
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r) {
      switch (v) {
        LAB_CASE(0)
        LAB_CASE(1)
        LAB_CASE(2)
        LAB_CASE(3)
        LAB_CASE(4)
        LAB_CASE(5)
        LAB_CASE(6)
        LAB_CASE(7)
        LAB_CASE(8)
        LAB_CASE(9)
        LAB_CASE(10)
      }
    }
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / R;
    float o0 = 0.f;
    q.memcpy(&o0, out, 4).wait();
    if (v == 5) ref_out = o0;
    std::printf("%-16s %9.1f us/call  %8.1f ns/step   out0=%.2f%s\n",
                names[v], us, us * 1000.0 / steps, o0,
                (v >= 6 && v <= 7)
                    ? (std::fabs(o0 - ref_out) < 1e-3 * std::fabs(ref_out) + 1
                           ? " (matches V5)"
                           : " (DIFFERS)")
                    : "");
    std::fflush(stdout);
  }
  std::printf("LADDER_DONE\n");
  return 0;
}
