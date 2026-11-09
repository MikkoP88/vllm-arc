// r128e_jointmap.cpp -- P6.34 arc7 JM authoritative fragment-map probe.
// Runs the OFFICIAL sycl::ext::oneapi::experimental::matrix::joint_matrix
// M8xN16xK16 half/half/float mma with fragments planted/read per
// work-item via joint_matrix_apply, and dumps the complete transfer map:
//   A-plant sweep (128):  A[lane l][slot s] = 1.0 (one at a time),
//                         B = all 1.0  -> which C(lane,slot) respond
//   B-plant sweep (256):  B[lane l][slot s] = 1.0 (one at a time),
//                         A = all 1.0  -> which C(lane,slot) respond
// Result: the TRUE hardware pairing law through Intel's own path, to be
// compared against the raw-asm law measured by r128d (v_D = (va-vb) mod 8,
// lane-local, odd-vb slots unread by the buggy decode). Fragment shapes
// match the raw-asm path exactly: A 8 halves, B 16 halves, C 8 f32 per wi.
// oneAPI 2026 API: joint_matrix_apply + 5-arg joint_matrix_mad(D, A, B, C).
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

  // masks[plant][2]: plant = 0..127 (A) then 128..383 (B): bit (cl*8+cv).
  constexpr int NA = 128, NB = 256, NPLANT = NA + NB;
  std::uint64_t *masks = malloc_host<std::uint64_t>(2 * NPLANT, q);
  for (int i = 0; i < 2 * NPLANT; ++i) masks[i] = 0;

  buffer<std::uint64_t, 1> mb{masks, range<1>(2 * NPLANT)};

  q.submit([&](handler &cgh) {
    accessor m(mb, cgh, write_only, no_init);
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
      matrix::joint_matrix<sub_group, float, matrix::use::accumulator, 8, 16,
                           matrix::layout::dynamic>
          jd;

      for (int pl = 0; pl < NPLANT; ++pl) {
        const bool is_a = pl < NA;
        const int p = is_a ? pl : pl - NA;
        const int slots = is_a ? 8 : 16;
        int i = 0;
        matrix::joint_matrix_apply(sg, ja, [&](half &e) {
          const bool on = is_a && (p / 8 == lane) && (p % 8 == i);
          e = is_a ? (on ? half(1.0f) : half(0.0f)) : half(1.0f);
          ++i;
        });
        i = 0;
        matrix::joint_matrix_apply(sg, jb, [&](half &e) {
          const bool on = !is_a && (p / 16 == lane) && (p % 16 == i);
          e = !is_a ? (on ? half(1.0f) : half(0.0f)) : half(1.0f);
          ++i;
        });
        matrix::joint_matrix_apply(sg, jd, [](float &e) { e = 0.0f; });
        matrix::joint_matrix_mad(sg, jd, ja, jb, jd);
        std::uint64_t lo = 0, hi = 0;
        i = 0;
        matrix::joint_matrix_apply(sg, jd, [&](float &e) {
          if (e != 0.0f) {
            const int bit = lane * 8 + i;
            if (bit < 64)
              lo |= 1ull << bit;
            else
              hi |= 1ull << (bit - 64);
          }
          ++i;
        });
        m[2 * pl] = lo;
        m[2 * pl + 1] = hi;
      }
    });
  });
  q.wait();

  auto print = [&](const char *kind, int base, int count, int slots) {
    std::cout << "=== " << kind << " plant -> C(lane,slot) bits ==="
              << std::endl;
    for (int pl = 0; pl < count; ++pl) {
      const std::uint64_t hi = masks[2 * (base + pl) + 1];
      const std::uint64_t lo = masks[2 * (base + pl)];
      if (!hi && !lo) {
        std::cout << kind << "[" << pl / slots << "][" << pl % slots
                  << "]: DEAD" << std::endl;
        continue;
      }
      std::cout << kind << "[" << pl / slots << "][" << pl % slots << "]:";
      for (int bit = 0; bit < 128; ++bit) {
        const std::uint64_t w = bit < 64 ? lo : hi;
        if ((w >> (bit % 64)) & 1)
          std::cout << " C" << bit / 8 << "." << bit % 8;
      }
      std::cout << std::endl;
    }
  };
  print("A", 0, NA, 8);
  print("B", NA, NB, 16);
  std::cout << "JOINTMAP_DONE" << std::endl;
  free(masks, q);
  return 0;
}
