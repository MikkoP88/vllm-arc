// r128e2_jointmap.cpp -- P6.34 arc7 JM authoritative layout probe (v2).
// Uses the OFFICIAL joint_matrix LOAD path with sentinel-filled buffers,
// reads every fragment back per work-item via joint_matrix_apply:
//   A row-major 8x16 half, ld=16, Abuf[m][k]   = half(m*16+k+1)      (1..128)
//   B col-major 16x16 half, ld=16, Bmem[n*16+k]= half(128+n*16+k+1)  (129..384)
//   C: mad -> read fragments
// Two runs (B sentinel / B ones) + two more (A sentinel / A ones) give:
//   A map (lane,slot) -> (m,k)   [direct fragment read]
//   B map (lane,slot) -> (n,k)   [direct fragment read]
//   C map (lane,slot) -> (m,n)   [sum-decoded: B-ones run c = 256*m+136,
//                                 A-ones run c = 256*n+2184]
// Compare against cutlass mma_traits_xe.hpp wi_interleave derivations and
// the raw-asm r128d measured law. Run inside arc-jm7 with
// LD_LIBRARY_PATH=/root/build/dpcpp72/dpcpp/lib.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <iostream>

using namespace sycl;
namespace syclex = ext::oneapi::experimental;
namespace matrix = syclex::matrix;

int main() {
  queue q{gpu_selector_v};
  std::cout << "dev: " << q.get_device().get_info<info::device::name>()
            << std::endl;

  // host staging: A 8*16, B 16*16 (col-major: n*16+k), C dump: 4 runs x
  // 16 lanes x slots (A 8, B 16, C 8) as float/int codes.
  constexpr int RUNS = 4;  // 0: Bsent 1: Bones 2: Asent 3: Aones
  constexpr int M = 8, K = 16, N = 16;
  half *hA = new half[M * K];
  half *hB = new half[N * K];
  float *out = malloc_shared<float>(RUNS * (16 * 8 + 16 * 16 + 16 * 8), q);
  for (int i = 0; i < RUNS * 512; ++i) out[i] = 0.0f;
  auto base_a = out, base_b = out + RUNS * 128, base_c = out + RUNS * 384;
  for (int i = 0; i < M * K; ++i) hA[i] = half(0.0f);
  for (int i = 0; i < N * K; ++i) hB[i] = half(0.0f);

  {
    buffer<half, 1> bA{hA, range<1>(M * K)};
    buffer<half, 1> bB{hB, range<1>(N * K)};
    q.submit([&](handler &cgh) {
      auto A = bA.get_access<access::mode::read_write>(cgh);
      auto B = bB.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for(nd_range<1>{range<1>{16}, range<1>{16}},
                       [=](nd_item<1> it) {
        auto sg = it.get_sub_group();
        const int lane = (int)sg.get_local_linear_id();

        matrix::joint_matrix<sub_group, half, matrix::use::a, 8, 16,
                             matrix::layout::row_major>
            ja;
        matrix::joint_matrix<sub_group, half, matrix::use::b, 16, 16,
                             matrix::layout::col_major>
            jb;
        matrix::joint_matrix<sub_group, float, matrix::use::accumulator, 8,
                             16, matrix::layout::dynamic>
            jd, jc;

        for (int run = 0; run < RUNS; ++run) {
          // fill A/B staging per run through global memory (host pre-filled
          // only run0; device re-fills per run here for self-containment)
          // run0: A sent, B sent; run1: A sent, B ones; run2: A sent, B
          // sent; run3: A ones, B sent. We load from global each time.
          for (int i = lane; i < M * K; i += 16) {
            const int m = i / K, k = i % K;
            A[i] = run == 3 ? half(1.0f) : half((float)(m * K + k + 1));
          }
          for (int i = lane; i < N * K; i += 16) {
            const int n = i / K, k = i % K;  // Bmem[n*16+k]
            B[i] = run == 1 ? half(1.0f) : half((float)(128 + n * K + k + 1));
          }
          // fragment-wide barrier equivalent: the loads below read staging
          // written by other lanes, so sync the sub_group.
          sg.barrier();

          matrix::joint_matrix_load(sg, ja, A.get_pointer(), 16);
          matrix::joint_matrix_load(sg, jb, B.get_pointer(), 16);
          matrix::joint_matrix_apply(sg, jd, [](float &e) { e = 0.0f; });
          matrix::joint_matrix_apply(sg, jc, [](float &e) { e = 0.0f; });
          matrix::joint_matrix_mad(sg, jd, ja, jb, jc);

          int i = 0;
          matrix::joint_matrix_apply(sg, ja, [&](half &e) {
            base_a[run * 128 + lane * 8 + i] = (float)e;
            ++i;
          });
          i = 0;
          matrix::joint_matrix_apply(sg, jb, [&](half &e) {
            base_b[run * 256 + lane * 16 + i] = (float)e;
            ++i;
          });
          i = 0;
          matrix::joint_matrix_apply(sg, jd, [&](float &e) {
            base_c[run * 128 + lane * 8 + i] = e;
            ++i;
          });
        }
      });
    });
    q.wait();
  }

  auto decode_a = [&](int run, int lane, int slot) {
    const int v = (int)base_a[run * 128 + lane * 8 + slot];
    return v >= 1 && v <= 128 ? std::to_string(v - 1) + ":" +
        std::to_string((v - 1) / 16) + "," + std::to_string((v - 1) % 16)
        : std::to_string(v);
  };
  auto decode_b = [&](int run, int lane, int slot) {
    const int v = (int)base_b[run * 256 + lane * 16 + slot];
    return v >= 129 && v <= 384
               ? std::to_string(v - 129) + ":" +
                     std::to_string((v - 129) / 16) + "," +
                     std::to_string((v - 129) % 16)
               : std::to_string(v);
  };

  std::cout << "=== A fragments run0 (lane.slot = sent idx m,k) ==="
            << std::endl;
  for (int l = 0; l < 16; ++l) {
    std::cout << "A L" << l << ":";
    for (int s = 0; s < 8; ++s) std::cout << " " << decode_a(0, l, s);
    std::cout << std::endl;
  }
  std::cout << "=== B fragments run0 (lane.slot = sent idx n,k) ==="
            << std::endl;
  for (int l = 0; l < 16; ++l) {
    std::cout << "B L" << l << ":";
    for (int s = 0; s < 16; ++s) std::cout << " " << decode_b(0, l, s);
    std::cout << std::endl;
  }
  std::cout << "=== C run1 (B ones): c=256m+136 -> m | C run3 (A ones): "
               "c=256n+2184 -> n ==="
            << std::endl;
  for (int l = 0; l < 16; ++l) {
    std::cout << "C L" << l << ":";
    for (int s = 0; s < 8; ++s) {
      const float c1 = base_c[1 * 128 + l * 8 + s];
      const float c3 = base_c[3 * 128 + l * 8 + s];
      int m = -1, n = -1;
      if (c1 != 0) m = (int)((c1 - 136) / 256);
      if (c3 != 0) n = (int)((c3 - 2184) / 256);
      std::cout << " " << (m >= 0 && n >= 0 ? std::to_string(m) + "," +
                                                std::to_string(n)
                                            : std::to_string((int)c1) + "/" +
                                                std::to_string((int)c3));
    }
    std::cout << std::endl;
  }
  std::cout << "JOINTMAP2_DONE" << std::endl;
  delete[] hA;
  delete[] hB;
  free(out, q);
  return 0;
}
