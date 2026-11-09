// r128k2_ladder.cpp -- P6.34 JM ping-pong NUMERICS ladder (torch-free).
// The r128k ladder ranked V6 ping-pong 41-108x faster but its out0
// cross-check was vacuous (filler bytes decode ~0; tol +1 abs passes
// anything) -- and the build7 wheel gate then failed every GATE-1 cell
// (zeros / e+18 junk). V5 per step is [stage; bar; load; BAR; zero c0;
// zero c1; mad; apply]; V6 collapsed to [load; mad; ...] with NO
// barrier between the cooperative global loads and the mad that
// consumes them -- the r128k "ordering proof" covered load-vs-staging
// visibility, never load ARRIVAL vs mad. This ladder fills REAL data
// and checks every output column against a host f64 reference:
//   V5  = current-order FULL sequence   (must PASS = harness sanity)
//   V6  = r128k ping-pong, no extra ops (expected DIFFERS = repro)
//   V11 = V6 + barrier after loads, before mad
//   V12 = V6 + per-step zero c0/c1 before mad (single barrier kept)
//   V13 = V11 + V12 (V5's exact fragment op sequence, staging one
//         step ahead)
//   V14 = unroll-2, lexically FIXED load sites (the Law-1 workaround)
//   V15 = unroll-4, four fixed slots, ONE barrier per 4-step body,
//         per-quarter zeros (aggressive bars, conservative ops)
//   V16 = V15 minus the per-quarter zeros (c0 zeroed once; the minimal
//         op form at unroll 4)
//   V18 = V5 sequence but the A fragment loads DIRECTLY from the
//         activation buffer at a monotonically varying address --
//         Law-1 SCOPE probe: exact zeros = any varying fragment-load
//         address is dead (A staging is mandatory); nonzero = direct-A
//         loads are live and A staging can vanish from the kernel
//         (verdict is zeros-vs-nonzero; values are garbage by design)
// Geometry mirrors jm_launch exactly (lsz 64, subs = tiles*splits,
// golden-ratio swizzle, scratch slots per GLOBAL sg). GROUP32=false
// (checkpoint lane): no group scales in the ladder.
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

  Lab(const std::uint8_t* a, const sycl::half* ablk,
       const std::uint8_t* b, const sycl::half* wblk, sycl::half* scratch,
      sycl::local_accessor<sycl::half, 1> slm,
      float* out, int m, int n, int k, int splits, int kc)
      : a_(a), ablk_(ablk), b_(b), wblk_(wblk), scr_(scratch),
        slm_(std::move(slm)), out_(out), m_(m), n_(n), k_(k),
        s_(splits), kc_(kc) {}

  void operator()(sycl::nd_item<1> it) const {
    auto sgc = it.get_sub_group();
    const int gid = static_cast<int>(it.get_global_linear_id());
    if (gid == 0)  // runtime sub-group size probe (debug slot)
      out_[n_] = (float)sgc.get_local_range()[0];
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
    // slot j at sg*3072 B + j*768 B (A 256 + B 512 each); V15/V16 use
    // 4 slots (unroll-4), V5/V6/V14 use the first two. Uniform stride
    // keeps every variant off its neighbors.
    sycl::half* sa0 = scr_ + static_cast<std::size_t>(sg) * 1536;
    sycl::half* sb0 = sa0 + 128;
    sycl::half* sa1 = sa0 + 384;
    sycl::half* sb1 = sa1 + 128;

    JmA ja;
    JmB jb;
    JmC c0, c1;
    float tot0[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    if constexpr (VAR != 5 && VAR != 18 && VAR != 19 && VAR != 20 &&
                  VAR != 21 && VAR != 22 && VAR != 23 && VAR != 24) {
      // ping-pong family: zero-once (V19+ keep V5's per-step zeros)
      jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
    }

    if constexpr (VAR == 15 || VAR == 16) {
      // V15/V16: unrolled-by-4 ping-pong, FOUR lexically fixed slots
      // (3072 B/sg), aggressive barrier schedule -- ONE group_barrier
      // per 4-step body (plus the prologue bar) instead of V14's
      // 1.5 bars/step. Per-lane ordering argument: the fragment layouts
      // are lane-local (A lane t reads sa[m*16+t] which lane t wrote;
      // B lane t reads sb[t*16..] which lane t wrote), so same-slot
      // reuse needs only per-lane program order -- the barriers may be
      // JIT-stability artifacts, which is exactly what this measures.
      // V15 keeps V5-style per-quarter zeros (c0+c1); V16 drops them
      // (c0 zeroed once above; mad fully overwrites c1) = the ladder
      // V14 op form at unroll 4.
      sycl::half* sa2 = sa1 + 384;
      sycl::half* sb2 = sa2 + 128;
      sycl::half* sa3 = sa2 + 384;
      sycl::half* sb3 = sa3 + 128;
      stage_a(sa0, k_lo, lane);
      stage_b(sb0, tbase, k_lo, lane);
      stage_a(sa1, k_lo + 16, lane);
      stage_b(sb1, tbase, k_lo + 16, lane);
      stage_a(sa2, k_lo + 32, lane);
      stage_b(sb2, tbase, k_lo + 32, lane);
      stage_a(sa3, k_lo + 48, lane);
      stage_b(sb3, tbase, k_lo + 48, lane);
      sycl::group_barrier(sgc);
      int k0 = k_lo;
      for (; k0 + 48 < k_hi; k0 += 64) {
        jmx::joint_matrix_load(sgc, ja, hptr(sa0), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
        if constexpr (VAR == 15) {
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
        }
        jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        {
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
        stage_a(sa0, k0 + 64, lane);
        stage_b(sb0, tbase, k0 + 64, lane);
        jmx::joint_matrix_load(sgc, ja, hptr(sa1), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb1), 16);
        if constexpr (VAR == 15) {
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
        }
        jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        {
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
        stage_a(sa1, k0 + 80, lane);
        stage_b(sb1, tbase, k0 + 80, lane);
        jmx::joint_matrix_load(sgc, ja, hptr(sa2), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb2), 16);
        if constexpr (VAR == 15) {
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
        }
        jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        {
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
        stage_a(sa2, k0 + 96, lane);
        stage_b(sb2, tbase, k0 + 96, lane);
        jmx::joint_matrix_load(sgc, ja, hptr(sa3), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb3), 16);
        if constexpr (VAR == 15) {
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
        }
        jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        {
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
        stage_a(sa3, k0 + 112, lane);
        stage_b(sb3, tbase, k0 + 112, lane);
        sycl::group_barrier(sgc);  // this body's stages -> next body
      }
      // steps/WG is a multiple of 4 (kc 64-aligned): no tail
    } else if constexpr (VAR == 18) {
      // V18: Law-1 SCOPE probe. V5's exact sequence, but the A
      // fragment loads DIRECTLY from the activation buffer at a
      // MONOTONICALLY VARYING address (base + k0, stride k_) instead
      // of a fixed staging slot. Law 1 (r128k2) was found on the
      // two-site SWAP pattern; if ANY varying load address is
      // mis-lowered this reads exact zeros; if it reads DATA, direct-A
      // loads are live and A staging can vanish from the kernel. The
      // bits are bf16-reinterpreted-as-f16 (wrong VALUES, garbage
      // rows>=M) -- the verdict is zeros-vs-nonzero, not numerics.
      for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
        stage_b(sb0, tbase, k0, lane);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_load(
            sgc, ja,
            hptr(reinterpret_cast<sycl::half*>(
                     const_cast<std::uint8_t*>(a_)) +
                 k0),
            k_);
        jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        int i = 0;
        jmx::joint_matrix_apply(sgc, c1,
                                [&](float& e) { tot0[i++] += e; });
      }
    } else if constexpr (VAR == 19 || VAR == 20) {
      // V19/V20: A-DIRECT restructure -- the one big lever left. V18
      // proved fragment loads from NEVER-WRITTEN input memory work at
      // any address (immune to the scratch store->load lottery), while
      // every cross-iteration scratch form (V6..V16) reads zeros and
      // the only correct scratch form is V5's same-step chain (~5 us
      // /step, 4-9x stock). Here A loads DIRECTLY from a k-blocked
      // activation buffer ablk_[s*128 + m*16 + t] (the fragment's
      // native row-major 8x16 stride-16 tile per step, perfectly
      // coalesced; host-built, never kernel-written = V18's immunity
      // class) -- NO A staging at all. B keeps the V5-exact same-step
      // discipline (the only proven-correct scratch class). V19 = one
      // tile/sg. V20 additionally gives each sg the ADJACENT n-tile
      // sharing the ja load (2 B slots, 2 mads): load-per-math 3/2 vs
      // V5's 2. The ladder keeps full tile coverage (double-computed
      // neighbor tile, identical value -- benign); the wheel form
      // would halve the sg count instead.
      const sycl::half* ap = ablk_ + (k_lo >> 4) * 128;
      sycl::half* sb2 = sa0 + 384;  // V20's 2nd B slot (sa1 slot,
                                    // unused: no A staging here)
      const std::uint8_t* tbase2 =
          (ctile + 1 < tiles) ? tbase + tstep : tbase;
      if constexpr (VAR == 19) {
        for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
          stage_b(sb0, tbase, k0, lane);
          sycl::group_barrier(sgc);
          jmx::joint_matrix_load(
              sgc, ja,
              hptr(const_cast<sycl::half*>(ap + ((k0 - k_lo) >> 4) * 128)),
              16);
          jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
          sycl::group_barrier(sgc);
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
      } else {
        JmB jb2;
        JmC c2;
        float tot1[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
          stage_b(sb0, tbase, k0, lane);
          stage_b(sb2, tbase2, k0, lane);
          sycl::group_barrier(sgc);
          jmx::joint_matrix_load(
              sgc, ja,
              hptr(const_cast<sycl::half*>(ap + ((k0 - k_lo) >> 4) * 128)),
              16);
          jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
          jmx::joint_matrix_load(sgc, jb2, hptr(sb2), 16);
          sycl::group_barrier(sgc);
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_apply(sgc, c2, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
          jmx::joint_matrix_mad(sgc, c2, ja, jb2, c0);
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
          i = 0;
          jmx::joint_matrix_apply(sgc, c2,
                                  [&](float& e) { tot1[i++] += e; });
        }
        const int nn2 = n0 + 16 + lane;
        if (nn2 < n_)
          out_[static_cast<std::size_t>(nn2)] =
              tot1[0] + tot1[1] + tot1[2] + tot1[3] + tot1[4] + tot1[5] +
              tot1[6] + tot1[7];
      }
    } else if constexpr (VAR == 21 || VAR == 22) {
      // V21/V22: the jm-family CEILING probes (perf + numerics). The
      // B fragment loads DIRECTLY from a host-built pre-decoded f16
      // weight buffer wblk_[tile*(K/16)+s)*256 + lane*16 + j] -- the
      // exact layout stage_b writes, decoded on the HOST, never
      // kernel-written (Law-1 immunity class). Pre-decoding to f16 at
      // SERVING is the known 2x-DRAM dead end; these variants exist to
      // measure the pure fragment-load chain floor:
      //   V21 = A staged (V5 discipline) + B direct   [1 store + bar]
      //   V22 = A direct + B direct -- NO scratch, NO barriers: the
      //         minimal [load ja; load jb; zero; zero; mad; apply]
      // If V22 >= ~560 ns/step (stock oneDNN per-step equivalent), no
      // joint_matrix form can beat stock on these shapes and the lane
      // pivots to register-DPAS.
      const sycl::half* ap = ablk_ + (k_lo >> 4) * 128;
      const sycl::half* wb =
          wblk_ + static_cast<std::size_t>(ctile) * (k_ >> 4) * 256;
      if constexpr (VAR == 21) {
        for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
          stage_a(sa0, k0, lane);
          sycl::group_barrier(sgc);
          jmx::joint_matrix_load(sgc, ja, hptr(sa0), 16);
          jmx::joint_matrix_load(
              sgc, jb,
              hptr(const_cast<sycl::half*>(
                  wb + (static_cast<std::size_t>(k0 >> 4) * 256))),
              16);
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
      } else {
        for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
          jmx::joint_matrix_load(
              sgc, ja,
              hptr(const_cast<sycl::half*>(
                  ap + ((k0 - k_lo) >> 4) * 128)),
              16);
          jmx::joint_matrix_load(
              sgc, jb,
              hptr(const_cast<sycl::half*>(
                  wb + (static_cast<std::size_t>(k0 >> 4) * 256))),
              16);
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
      }
    } else if constexpr (VAR == 23 || VAR == 24) {
      // V23/V24: B staged in SLM (shared-local), A direct -- the last
      // untested staging door. V22 proved fragment loads are cheap and
      // EXACT when they read never-written memory; the whole V5/V19
      // wall is the global-scratch store->barrier->load round trip.
      // r128e3 called SLM fragment loads "dead kernels", but that was
      // pre-AOT-ladder and every harness verdict since has been
      // lottery-contaminated -- retest. 1024 halves per WG = 4 sg
      // slots x 256 (B tile). V23 = V5-exact op shape (stage; bar;
      // load ja direct; load jb SLM; bar; zeros; mad; apply).
      // V24 = V23 minus the post-load barrier (is it needed?).
      const sycl::half* ap = ablk_ + (k_lo >> 4) * 128;
      const int sgl = static_cast<int>(sgc.get_group_linear_id());
      sycl::half* sbs =
          slm_.get_pointer().get() + static_cast<std::size_t>(sgl) * 256;
      for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
        stage_b(sbs, tbase, k0, lane);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_load(
            sgc, ja,
            hptr(const_cast<sycl::half*>(ap + ((k0 - k_lo) >> 4) * 128)),
            16);
        jmx::joint_matrix_load(
            sgc, jb,
            sycl::multi_ptr<
                sycl::half,
                sycl::access::address_space::local_space>(sbs),
            16);
        if constexpr (VAR == 23) {
          sycl::group_barrier(sgc);
        }
        jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        int i = 0;
        jmx::joint_matrix_apply(sgc, c1,
                                [&](float& e) { tot0[i++] += e; });
      }
    } else if constexpr (VAR == 14) {
      // V14: unrolled-by-2 ping-pong with LEXICALLY FIXED load pointers.
      // The swap-based ping-pong (V6/V11/V12/V13) reads zero fragments on
      // this stack -- the fragment-load lowering cannot take a load
      // address that varies across loop iterations. Unrolling by 2 gives
      // each load site a constant pointer; staging stays one step ahead.
      // steps/WG is a multiple of 4 (kc is 64-aligned), so pairs are full.
      stage_a(sa0, k_lo, lane);
      stage_b(sb0, tbase, k_lo, lane);
      stage_a(sa1, k_lo + 16, lane);
      stage_b(sb1, tbase, k_lo + 16, lane);
      sycl::group_barrier(sgc);
      int k0 = k_lo;
      for (; k0 + 16 < k_hi; k0 += 32) {
        jmx::joint_matrix_load(sgc, ja, hptr(sa0), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        {
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
        stage_a(sa0, k0 + 32, lane);
        stage_b(sb0, tbase, k0 + 32, lane);
        jmx::joint_matrix_load(sgc, ja, hptr(sa1), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb1), 16);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        {
          int i = 0;
          jmx::joint_matrix_apply(sgc, c1,
                                  [&](float& e) { tot0[i++] += e; });
        }
        stage_a(sa1, k0 + 48, lane);
        stage_b(sb1, tbase, k0 + 48, lane);
        sycl::group_barrier(sgc);
      }
      for (; k0 < k_hi; k0 += 16) {  // tail (never taken: steps % 4 == 0)
        stage_a(sa0, k0, lane);
        stage_b(sb0, tbase, k0, lane);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_load(sgc, ja, hptr(sa0), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        int i = 0;
        jmx::joint_matrix_apply(sgc, c1,
                                [&](float& e) { tot0[i++] += e; });
      }
    } else if constexpr (VAR != 5) {
      // ping-pong: pre-stage step 0, load LAST step's staging
      stage_a(sa0, k_lo, lane);
      stage_b(sb0, tbase, k_lo, lane);
      sycl::group_barrier(sgc);
      for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
        jmx::joint_matrix_load(sgc, ja, hptr(sa0), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
        if constexpr (VAR >= 11 && VAR != 12) {
          sycl::group_barrier(sgc);  // load arrival before the mad
        }
        if constexpr (VAR == 12 || VAR == 13) {
          jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
          jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        }
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
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
        int i = 0;
        jmx::joint_matrix_apply(sgc, c1,
                                [&](float& e) { tot0[i++] += e; });
      }
    } else {
      // ---- V5: current-order FULL sequence (proven in the wheel) ----
      for (int k0 = k_lo; k0 < k_hi; k0 += 16) {
        stage_a(sa0, k0, lane);
        stage_b(sb0, tbase, k0, lane);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_load(sgc, ja, hptr(sa0), 16);
        jmx::joint_matrix_load(sgc, jb, hptr(sb0), 16);
        sycl::group_barrier(sgc);
        jmx::joint_matrix_apply(sgc, c0, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_apply(sgc, c1, [](float& e) { e = 0.0f; });
        jmx::joint_matrix_mad(sgc, c1, ja, jb, c0);
        int i = 0;
        jmx::joint_matrix_apply(sgc, c1,
                                [&](float& e) { tot0[i++] += e; });
      }
    }
    const int nn = n0 + lane;
    if (nn < n_)
      out_[static_cast<std::size_t>(nn)] =
          tot0[0] + tot0[1] + tot0[2] + tot0[3] + tot0[4] + tot0[5] +
          tot0[6] + tot0[7];
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
  const sycl::half* ablk_;  // V19/V20: k-blocked f16 activations
  const std::uint8_t* b_;
  const sycl::half* wblk_;  // V21/V22: pre-decoded f16 weights
  sycl::half* scr_;
  sycl::local_accessor<sycl::half, 1> slm_;  // V23/V24 B slots
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
    q.submit([&](sycl::handler& cgh) {                                 \
      sycl::local_accessor<sycl::half, 1> slm(1024, cgh);              \
      cgh.parallel_for<Lab<VV>>(                                      \
          sycl::nd_range<1>(wgs, lsz),                                 \
          Lab<VV>((const std::uint8_t*)a, (const sycl::half*)ablk,     \
                  (const std::uint8_t*)b, (const sycl::half*)wblk,     \
                  (sycl::half*)scr, std::move(slm), (float*)out, M,    \
                  npad, K, splits, kc));                               \
    });                                                                \
    break;

static float e4m3_to_f32(std::uint8_t c) {
  const int s = (c >> 7) & 1, e = (c >> 3) & 15, m = c & 7;
  float v;
  if (e == 0)
    v = m ? std::ldexp(m / 8.0f, -6) : 0.0f;
  else
    v = std::ldexp(1.0f + m / 8.0f, e - 7);
  return s ? -v : v;
}

int main(int argc, char** argv) {
  const int K = argc > 1 ? std::atoi(argv[1]) : 5120;
  const int N = argc > 2 ? std::atoi(argv[2]) : 8704;
  const int M = argc > 3 ? std::atoi(argv[3]) : 1;
  const int SGF = argc > 4 ? std::atoi(argv[4]) : 2048;
  const int R = argc > 5 ? std::atoi(argv[5]) : 50;
  const int vlo = argc > 6 ? std::atoi(argv[6]) : 5;
  const int vhi = argc > 7 ? std::atoi(argv[7]) : 13;

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
  auto ablk = sycl::malloc_device(8 * K * 2, q);  // V19/V20 k-blocked f16
  auto wblk = sycl::malloc_device(
      static_cast<std::size_t>(tiles) * (K / 16) * 512, q);  // V21/V22 f16
  auto b = sycl::malloc_device(
      static_cast<std::size_t>(tiles) * (K / 16) * 256, q);
  auto scr = sycl::malloc_device(
      static_cast<std::size_t>(subs) * 3072, q);  // 4 slots/sg (V15/16)
  auto out = sycl::malloc_device(static_cast<std::size_t>(npad + 8) * 4, q);

  // ---- REAL data (deterministic LCG; .25-step bf16 x, NaN-free e4m3 w)
  std::vector<std::uint8_t> h8(static_cast<std::size_t>(8) * K * 2);
  std::uint32_t lcg = 0x12345678u;
  for (std::size_t i = 0; i < h8.size(); i += 2) {
    lcg = lcg * 1664525u + 1013904223u;
    const float v = (float)((int)((lcg >> 8) % 7) - 3) * 0.25f;
    const std::uint32_t f = sycl::bit_cast<std::uint32_t>(v);
    h8[i] = (std::uint8_t)(f >> 24);
    h8[i + 1] = (std::uint8_t)(f >> 16);
  }
  q.memcpy(a, h8.data(), h8.size());
  // ablk[s*128 + m*16 + t] = f16(bf16 value (m, s*16+t)) -- the exact
  // conversion stage_a performs, in the fragment's native tile layout.
  // LCG values are multiples of 0.25 in [-0.75, 0.75]: exact in f16.
  // Rows m >= M are ZERO (stage_a zero-pads them in every other variant;
  // production routes zero-pad x to 8 rows the same way).
  {
    std::vector<std::uint16_t> hblk(static_cast<std::size_t>(8) * K, 0);
    for (int m = 0; m < M && m < 8; ++m)
      for (int k = 0; k < K; ++k) {
        const std::uint16_t bits =
            (std::uint16_t)(h8[(static_cast<std::size_t>(m) * K + k) * 2] |
                            (h8[(static_cast<std::size_t>(m) * K + k) * 2 + 1]
                             << 8));
        hblk[(static_cast<std::size_t>(k >> 4) * 8 + m) * 16 + (k & 15)] =
            bf16_to_f16_bits(bits);
      }
    q.memcpy(ablk, hblk.data(), hblk.size() * 2);
  }
  std::vector<std::uint8_t> hw(
      static_cast<std::size_t>(tiles) * (K / 16) * 256);
  for (std::size_t i = 0; i < hw.size(); ++i) {
    lcg = lcg * 1664525u + 1013904223u;
    hw[i] = (std::uint8_t)((lcg >> 9) & 0x7Eu);  // finite e4m3, no NaN
  }
  q.memcpy(b, hw.data(), hw.size());
  // wblk = pre-decoded f16 weights in stage_b's exact fragment layout
  // (host decode = e4m3_to_f32, exact in f16). V21/V22 numerics are
  // CHECKED against the same ref as everyone else.
  {
    std::vector<std::uint16_t> hwb(
        static_cast<std::size_t>(tiles) * (K / 16) * 256);
    for (std::size_t i = 0; i < hwb.size(); ++i)
      hwb[i] = sycl::bit_cast<std::uint16_t>(
          static_cast<sycl::half>(e4m3_to_f32(hw[i])));
    q.memcpy(wblk, hwb.data(), hwb.size() * 2);
  }
  q.memset(out, 0, static_cast<std::size_t>(npad + 8) * 4);
  q.wait();
  // A = bf16 little-endian pair (h8[i], h8[i+1]) = f32 bits >> 16...
  // NOTE: bf16 storage here is (hi, lo) per the generator above -- the
  // kernel stage_a reads ONE uint16 at (m,k): little-endian = lo byte
  // first. Generator wrote h8[i]=f>>24 (hi) then h8[i+1]=f>>16 (lo) --
  // the pair as uint16 LE = (lo<<8)|hi = f>>16 = the bf16. Consistent.
  std::vector<double> aref(static_cast<std::size_t>(8) * K);
  for (int m = 0; m < 8; ++m)
    for (int k = 0; k < K; ++k) {
      const std::uint16_t bits =
          (std::uint16_t)(h8[(static_cast<std::size_t>(m) * K + k) * 2] |
                          (h8[(static_cast<std::size_t>(m) * K + k) * 2 + 1]
                             << 8));
      aref[static_cast<std::size_t>(m) * K + k] =
          (double)sycl::bit_cast<float>((std::uint32_t)bits << 16);
    }
  std::vector<double> ref(npad, 0.0);
  for (int n = 0; n < npad; ++n) {
    const int tile = n >> 4, c = n & 15;
    double acc = 0.0;
    for (int m = 0; m < M; ++m)
      for (int ks = 0; ks < K / 16; ++ks)
        for (int j = 0; j < 16; ++j) {
          const std::uint8_t code =
              hw[(static_cast<std::size_t>(tile) * (K / 16) + ks) * 256 +
                 c * 16 + j];
          acc += aref[static_cast<std::size_t>(m) * K + ks * 16 + j] *
                 (double)e4m3_to_f32(code);
        }
    ref[n] = acc;
  }

  {  // decoder selftest: e4m3_pair_f16x2 (device decoder, host-runnable
     // integer math) vs the semantic e4m3_to_f32, all codes, both halves
    int dm = 0;
    for (int c = 0; c < 256; ++c) {
      if (c == 0x7F || c == 0xFF) continue;
      const std::uint32_t x =
          (std::uint32_t)c | ((std::uint32_t)c << 16);
      const std::uint32_t p = e4m3_pair_f16x2(x);
      for (int h = 0; h < 2; ++h) {
        const std::uint16_t f16 = (std::uint16_t)(p >> (16 * h));
        const int s = (f16 >> 15) & 1, e = (f16 >> 10) & 31,
                  m = f16 & 1023;
        float v;
        if (e == 0)
          v = m ? std::ldexp(m / 1024.0f, -14) : 0.0f;
        else if (e == 31)
          v = m ? NAN : INFINITY;
        else
          v = std::ldexp(1.0f + m / 1024.0f, e - 15);
        if (s) v = -v;
        const float sem = e4m3_to_f32((std::uint8_t)c);
        const bool bad = std::isnan(v) || std::isnan(sem)
                             ? false
                             : std::fabs(v - sem) >
                                   1e-6 * (std::fabs(sem) + 1e-6);
        if (bad) {
          if (dm < 10)
            std::printf("decoder mismatch code=%02x half=%d f16=%f "
                        "sem=%f\n",
                        c, h, v, sem);
          ++dm;
        }
      }
    }
    std::printf("decoder selftest: %d mismatches\n", dm);
  }

  std::printf(
      "K=%d N=%d M=%d SGF=%d tiles=%d splits=%d kc=%d subs=%d "
      "steps/WG=%d wgs=%d  ref[0]=%.6f ref[1]=%.6f\n",
      K, N, M, SGF, tiles, splits, kc, subs, steps, wgs, ref[0], ref[1]);

  const char* names[] = {"V0", "V1", "V2", "V3", "V4",
                         "V5 FULL now", "V6 pp-nobar", "V7",
                         "V8", "V9", "V10", "V11 pp+loadbar",
                         "V12 pp+zerostep", "V13 pp+both",
                         "V14 pp-fixedptr", "V15 u4-zeros",
                         "V16 u4-min", "V17", "V18 directA-probe",
                         "V19 Adirect16", "V20 2tile-Adirect", "V21",
                         "V22 all-direct", "V23 slmB", "V24 slmB-nobar"};
  std::vector<float> hout(npad);
  for (int v = vlo; v <= vhi; ++v) {
    if (v < 5 || (v > 6 && v < 11) || (v > 13 && v != 14 && v != 15 &&
                                       v != 16 && v != 18 && v != 19 &&
                                       v != 20 && v != 21 && v != 22 &&
                                       v != 23 && v != 24))
      continue;
    q.memset(out, 0, static_cast<std::size_t>(npad) * 4);
    q.wait();
    std::printf("V%d submit\n", v);
    std::fflush(stdout);
    for (int i = 0; i < 3; ++i) {
      switch (v) {
        LAB_CASE(5)
        LAB_CASE(6)
        LAB_CASE(11)
        LAB_CASE(12)
        LAB_CASE(13)
        LAB_CASE(14)
        LAB_CASE(15)
        LAB_CASE(16)
        LAB_CASE(18)
        LAB_CASE(19)
        LAB_CASE(20)
        LAB_CASE(21)
        LAB_CASE(22)
        LAB_CASE(23)
        LAB_CASE(24)
      }
    }
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < R; ++r) {
      switch (v) {
        LAB_CASE(5)
        LAB_CASE(6)
        LAB_CASE(11)
        LAB_CASE(12)
        LAB_CASE(13)
        LAB_CASE(14)
        LAB_CASE(15)
        LAB_CASE(16)
        LAB_CASE(18)
        LAB_CASE(19)
        LAB_CASE(20)
        LAB_CASE(21)
        LAB_CASE(22)
        LAB_CASE(23)
        LAB_CASE(24)
      }
    }
    q.wait();
    const auto t1 = std::chrono::steady_clock::now();
    const double us =
        std::chrono::duration<double, std::micro>(t1 - t0).count() / R;
    q.memcpy(hout.data(), out, static_cast<std::size_t>(npad) * 4).wait();
    if (v == vlo) {
      float sgz = 0.f;
      q.memcpy(&sgz, (const char*)out + (std::size_t)npad * 4, 4).wait();
      std::printf("runtime sub_group size = %.0f\n", sgz);
    }
    double maxrel = 0.0;
    double maxabs = 0.0;
    int bad = 0;
    for (int n = 0; n < npad; ++n) {
      const double denom = std::fabs(ref[n]) > 1.0 ? std::fabs(ref[n]) : 1.0;
      const double rel = std::fabs((double)hout[n] - ref[n]) / denom;
      const double a = std::fabs((double)hout[n]);
      if (rel > maxrel) maxrel = rel;
      if (a > maxabs) maxabs = a;
      if (rel > 1e-3) ++bad;
    }
    if (v == vlo)  // first variant: trace dump for ref debugging
      for (int n = 0; n < 8 && n < npad; ++n)
        std::printf("  n=%d out=%.6f ref=%.6f\n", n, hout[n], ref[n]);
    std::printf("%-16s %9.1f us/call %8.1f ns/step  maxrel=%.3g "
                "maxabs=%.3g  badcols=%d/%d  %s\n",
                names[v], us, us * 1000.0 / steps, maxrel, maxabs, bad,
                npad, maxabs == 0.0
                           ? "ZEROS"
                           : (bad == 0 ? "NUMERICS-PASS" : "NUMERICS-FAIL"));
    std::fflush(stdout);
  }
  std::printf("LADDER2_DONE\n");
  return 0;
}
