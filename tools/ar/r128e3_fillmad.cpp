// r128e3_fillmad.cpp -- P6.34 arc7 JM iterated-load probe (v5).
// The wheel kernel (torch-tensor scratch) drops every source halfword at
// offset == 3 (mod 4) from the LOADED fragments (r128f pinpoint: sums
// 24/32, k=7 and k=11 invisible, diag missing n=3,7,11,15 -- BOTH A and
// B sides). The v4 probe (malloc_shared staging, ONE work group) passed.
// This version pins the differing factor via argv:
//   argv[1] = scratch kind: 0 = malloc_shared, 1 = malloc_device
//   argv[2] = work groups: 1 or 16 (each SG gets its own 384-half slot,
//             indexed by GLOBAL subgroup id exactly like the kernel)
// Expected: every C lane t slot m = 160*(m+1)*(t+1) (4 steps, sum of
// (s+1) = 10). Drops would show as 120 with k%4==3 holes.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>

using namespace sycl;
namespace syclex = ext::oneapi::experimental;
namespace matrix = syclex::matrix;

int main(int argc, char **argv) {
  const int kind = argc > 1 ? std::atoi(argv[1]) : 0;
  const int nwg = argc > 2 ? std::atoi(argv[2]) : 1;
  queue q{gpu_selector_v};
  std::cout << "dev: " << q.get_device().get_info<info::device::name>()
            << " kind=" << (kind ? "device" : "shared") << " nwg=" << nwg
            << std::endl;
  const int nsg = nwg * 4;
  float *out = malloc_shared<float>(4 * 128, q);  // WG0 result check
  for (int i = 0; i < 512; ++i) out[i] = -1.0f;
  half *scr = kind ? malloc_device<half>(nsg * 384, q)
                   : malloc_shared<half>(nsg * 384, q);
  q.memset(scr, 0, sizeof(half) * nsg * 384).wait();

  q.submit([&](handler &cgh) {
    cgh.parallel_for(
        nd_range<1>{range<1>{static_cast<size_t>(nwg) * 64}, range<1>{64}},
                     [=](nd_item<1> it) {
      auto sg = it.get_sub_group();
      const int lane = (int)sg.get_local_linear_id();
      const int gid = (int)it.get_global_linear_id();
      const int gsg = gid >> 4;  // global subgroup id (kernel law)
      half *sa = scr + gsg * 384;
      half *sb = sa + 128;

      matrix::joint_matrix<sub_group, half, matrix::use::a, 8, 16,
                           matrix::layout::row_major>
          ja;
      matrix::joint_matrix<sub_group, half, matrix::use::b, 16, 16,
                           matrix::layout::col_major>
          jb;
      matrix::joint_matrix<sub_group, float, matrix::use::accumulator, 8,
                           16, matrix::layout::dynamic>
          c0, c1;
      float tot[8] = {0, 0, 0, 0, 0, 0, 0, 0};

      for (int step = 0; step < 4; ++step) {
        for (int m = 0; m < 8; ++m)
          sa[m * 16 + lane] = half(float(m + 1));
        for (int kk = 0; kk < 16; ++kk)
          sb[lane * 16 + kk] = half(float(lane + 1) * (step + 1));
        group_barrier(sg);
        matrix::joint_matrix_load(
            sg, ja,
            multi_ptr<half, access::address_space::global_space>(sa), 16);
        matrix::joint_matrix_load(
            sg, jb,
            multi_ptr<half, access::address_space::global_space>(sb), 16);
        group_barrier(sg);
        matrix::joint_matrix_apply(sg, c0, [](float &e) { e = 0.0f; });
        matrix::joint_matrix_apply(sg, c1, [](float &e) { e = 0.0f; });
        matrix::joint_matrix_mad(sg, c1, ja, jb, c0);
        int i = 0;
        matrix::joint_matrix_apply(sg, c1, [&](float &e) {
          tot[i] += e;
          ++i;
        });
      }
      if (gsg < 4) {  // only WG0 reports (all WGs compute the same)
        int i = 0;
        matrix::joint_matrix_apply(sg, c1, [&](float &e) {
          out[gsg * 128 + lane * 8 + i] = tot[i];
          ++i;
        });
      }
    });
  });
  q.wait();

  int bad = 0;
  for (int l = 0; l < 16; ++l) {
    std::cout << "C L" << l << ":";
    for (int s = 0; s < 8; ++s) {
      const float expect = 160.0f * (s + 1) * (l + 1);
      const float got = out[l * 8 + s];
      std::cout << " " << (int)got << (got == expect ? "" : "!");
      if (got != expect) ++bad;
    }
    std::cout << std::endl;
  }
  std::cout << (bad ? "ITER_FAIL " : "ITER_PASS ") << bad << " mismatches"
            << std::endl;
  free(out, q);
  free(scr, q);
  return bad ? 1 : 0;
}
