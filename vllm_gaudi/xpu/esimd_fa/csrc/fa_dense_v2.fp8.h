// Dense causal varlen flash attention v2 — ESIMD/Xe2 (P6.21, branch
// feat/arc-p621c-fa-esimd-v2).
//
// v1 (p72, fa_dense.fp8.h) measured 23-26x slower than the stock vxk op
// and OOB-corrupts adjacent allocations at big bs=1 shapes (r87/r88
// evidence). Both defects are structural in the v1 grid: (qHeads, rows)
// work-groups of size (1,1) = ONE hardware thread per (row, q-head)
// streaming the whole KV as a scalar chain, with dpas A-tiles carrying a
// single real row (M-util 1/8) and every (row, head) re-dequantizing the
// full KV cache.
//
// v2 redesign (r87-r91 decision record, plan P6.21):
//   * Work-group = 8 threads x 64 q-rows of ONE head; each thread OWNS 8
//     rows -> dpas RepeatCount=8 with ALL 8 A-rows real (M-util 8/8, the
//     v1 killer fixed).
//   * KV dequantized ONCE per 32-token tile into SLM by the work-group
//     (4 tokens per thread) — v1 dequantized the whole cache per
//     (row, head); SLM staging amortizes it across the KV walk.
//   * QK runs ROWS-AS-M (v2.1): A = q rows row-major [8 rows x 16 dims]
//     staged once per WG (QRM SLM, one 32-B block store per (row, kb));
//     B = K-tile VNNI [16 tokens x 16 dims] staged directly in VNNI form
//     by one 8-word slm_scatter per (token, kb) — word(kp,n)@(kp*16+n),
//     the v1-proven law; C[m][token]@m*16+n, one 16-token half per dpas
//     chain (N=16). N=8 IS UNUSABLE on BMG: the r92_dbg7 dpas probe
//     proved dpas<8,8> with a 64-lane C executes M4xK8xN16 (B read in
//     the N=16 word layout, only A rows 0..3 computed, A columns 8..15
//     ignored) — that was the v2 numerics defect.
//   * PV: A = p-tile [8 rows x 16 tokens] — a direct fp16 convert of the
//     softmax p half (identical [m][16] lane layout, no lane surgery);
//     B = V-tile VNNI [16 tokens x 16 dims] whose packing is FREE at
//     dequant time (each thread owns a CONSECUTIVE token pair, so the
//     cross-token VNNI words are built in registers and stored
//     contiguously, 64 B per dim-block).
//   * Output accumulators live in SLM (nb-major: 16 dim-blocks x 64
//     rows x 64 B) so the thread's 8-row x 16-dim slice is one 512-B
//     block; the online-softmax rescale folds into the mandatory
//     read-modify-write (no extra pass). fp32 accumulators throughout
//     (the fp16-acc alternative over ~1024 rescale RMWs risks the
//     fro < 5e-3 gate; 64-row slabs keep fp32 inside the SLM cap).
//   * Online softmax vectorized across all 8 rows: one whole-vector
//     h-fold then per-row in-half folds (offsets 8/4/2/1 — vector-halves
//     would pair lanes ACROSS rows in the [m][16] layout); exp on 256
//     lanes (128 per half).
//   * bs=1 ONLY (the serve contract: chunked prefill runs a single
//     sequence; the launcher TORCH_CHECKs it). Causal bottom-right:
//     row r attends kv <= r + (kvLen - qLen). Tiles (32 tokens; 832 and
//     1664 are multiples of 32, so a tile never spans a page) beyond the
//     work-group's last row edge are skipped; in-tile masking per row.
//
// SLM budget — BMG per-work-group cap measured 131072 B exactly (ocloc
// -device bmg bisect, P6.21 build-3: 131072 compiles, 133120 fails "SLM
// size exceeds target limits"; the original 128-row/224 KB design cannot
// exist on this target, hence 64-row slabs):
//   QRM    32 KB  q rows fp16 row-major [kb][thread][row][16 dims]:
//                 kb*2048 + lid*256 + m*32
//   KVNNI  16 KB  K fp16 VNNI words (kp, n): (h*16+kb)*512 + (kp*16+n)*4
//                 (h = token half-tile 0/1, n = in-half token 0..15)
//   VVNNI  16 KB  V VNNI words (tp, d): ks*8192 + nb*512 + (tp*16+d)*4
//   ACC   64 KB   fp32 nb-major [nb][row][16 dims]: nb*4096 + r*64
//
// Laws inherited verbatim from v1/donor: 512B q-head law; non-pow2 page
// law (integer div/mod); SYCL device-lambda pointer law (launcher);
// simd_mask law; lsc byte-offset law; faDequantE4m3 verbatim; log2-domain
// online softmax with -1e30 sentinels; fp32 accumulators; GRF law
// (v2.1 live state ~3.5 KB: sT halves 1 KB + pf halves 1 KB + operands
// ~1.5 + softmax vectors — no spills by construction; probe verifies).
// vectors 0.5 — no spills by construction; probe verifies).
#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::experimental::esimd;

// fp16 is not a top-level ESIMD type name; alias it (donor law).
using fp16 = sycl::half;

// e4m3 byte -> fp16 bits (verbatim from the donor, proven BMG).
template <int N>
ESIMD_INLINE simd<fp16, N> faDequantE4m3(simd<uint8_t, N> bytes) {
  static_assert(N % 2 == 0, "dequant expects an even N");
  simd<uint16_t, N> b16 = 0;
  b16.template bit_cast_view<uint8_t>().template select<N, 2>(0) = bytes;
  simd<uint16_t, N> out = (b16 & 0x7F) << 7;
  out += 0x2000;
  out |= (b16 & 0x80) << 8;
  out.merge((b16 & 0x80) << 8, (b16 & 0x78) == 0);
  return out.template bit_cast_view<fp16>();
}

// Elementwise max via compare+merge (merge(src, mask): lanes where mask
// is true take src). merge with a simd source — if the overload is
// missing on this header set, swap to the scalar-row form.
template <int N>
ESIMD_INLINE simd<float, N> faVMax(simd<float, N> a, simd<float, N> b) {
  auto c = a > b;
  simd<float, N> r = b;
  r.merge(a, c);
  return r;
}

inline constexpr uint32_t V2_HEAD_DIM = 256;
inline constexpr uint32_t V2_THREADS = 8;         // threads per WG
inline constexpr uint32_t V2_RPB = 8;             // rows per thread (= RC)
inline constexpr uint32_t V2_ROWS = V2_THREADS * V2_RPB;  // 64 rows/WG
inline constexpr uint32_t V2_TOK = 32;            // KV tile (832 % 32 == 0)

// SLM byte layout (see header comment). 131072 = the measured BMG cap.
inline constexpr uint32_t SLM_QV = 0;             // 32 KB
inline constexpr uint32_t SLM_KG = 32768;         // 16 KB
inline constexpr uint32_t SLM_VV = 49152;         // 16 KB
inline constexpr uint32_t SLM_AC = 65536;         // 64 KB
inline constexpr uint32_t SLM_SIZE = 131072;      // 128 KB total

void faDenseV2Kernel(
    const uint8_t* __restrict__ qState,   // [rows, qHeads, 256] fp16
    const uint8_t* __restrict__ kState,   // [pages, pageSize, kvHeads, 256] e4m3
    const uint8_t* __restrict__ vState,   // same shape as kState
    const int32_t* __restrict__ cuSeqlensQ,  // [2] = {0, rows} (bs == 1)
    const int32_t* __restrict__ sequsedK,    // [1]
    const int32_t* __restrict__ blockTable,  // [1, tableStride]
    uint8_t* __restrict__ outState,       // [rows, qHeads, 256] fp16
    const float* __restrict__ kScalePtr,  // [1, kvHeads] fp32 or null
    const float* __restrict__ vScalePtr,  // [1, kvHeads] fp32 or null
    const uint32_t rows, const uint32_t qHeads, const uint32_t kvHeads,
    const uint32_t tableStride, const uint32_t numPages,
    const uint64_t pageStride, const uint64_t tokenStride,
    const uint64_t headStride, const uint32_t pageSize,
    const float softmaxScaleLog2, const uint32_t ablate, nd_item<1> ndi) {
  // Single call site, uniform, before any early exit (slm_init law).
  slm_init<SLM_SIZE>();

  const uint32_t wg = ndi.get_group(0);
  const uint32_t lid = ndi.get_local_id(0);
  const uint32_t slabs = (rows + V2_ROWS - 1) / V2_ROWS;
  const uint32_t head = wg / slabs;          // head-major (KV L2 locality)
  const uint32_t slab = wg % slabs;
  const uint32_t rowLo = slab * V2_ROWS;
  const uint32_t r0 = rowLo + lid * V2_RPB;  // this thread's first row
  const bool active = r0 < rows;

  // bs == 1 (launcher-checked): sequence metadata is scalar.
  const uint32_t qLen = (uint32_t)cuSeqlensQ[1];
  const uint32_t kvLen = (uint32_t)sequsedK[0];
  const uint32_t groupSize = qHeads / kvHeads;
  const uint32_t kvHead = head / groupSize;
  const float kDescale = kScalePtr ? kScalePtr[kvHead] : 1.0f;
  const float vDescale = vScalePtr ? vScalePtr[kvHead] : 1.0f;
  const float scoreScale = softmaxScaleLog2 * kDescale;

  // Work-group-uniform walk bound: the LAST row of the slab gates the
  // tile loop (barrier uniformity law — every thread of the WG must hit
  // the same barriers; per-thread causal edges are masked in-tile).
  const uint32_t lastRow = rowLo + V2_ROWS - 1;
  const uint32_t edgeLast =
      (lastRow < rows - 1 ? lastRow : rows - 1) + (kvLen - qLen);
  const uint32_t tiles = (edgeLast / V2_TOK) + 1;

  // ---- per-thread persistent state ----
  simd<float, V2_RPB> runMax = -1.0e30f;
  simd<float, V2_RPB> runSum = 0.0f;
  simd<uint32_t, V2_TOK> tokenIds(0, 1);   // +kvPos0 later, per tile

  if (active) {
    // ---- q -> SLM row-major (QRM; once per WG; the (kb, lid) tile is
    // private to the owning thread — no barrier needed). 512B q-head law
    // on the global load; each (row, kb) 16-dim segment is one 32-B
    // block store, so the QK A operand is a single 256-B SLM load. ----
    for (int m = 0; m < (int)V2_RPB; m++) {
      const uint32_t g = r0 + m;
      const uint32_t gc = (g < rows) ? g : (rows - 1);
      const uint64_t qBase = ((uint64_t)gc * qHeads + head)
                             * V2_HEAD_DIM * 2u;
      simd<uint8_t, 512> qRaw = 0;
#pragma unroll
      for (int c8 = 0; c8 < 8; c8++)
        qRaw.template select<64, 1>(c8 * 64) =
            lsc_block_load<uint8_t, 64, lsc_data_size::default_size,
                           cache_hint::cached, cache_hint::cached>(
                qState + qBase + (uint32_t)(c8 * 64));
      simd<fp16, 256> qH = qRaw.template bit_cast_view<fp16>();
#pragma unroll
      for (int kb = 0; kb < 16; kb++)
        slm_block_store<fp16, 16>(SLM_QV + kb * 2048 + lid * 256 + m * 32,
                                  qH.template select<16, 1>(kb * 16));
    }
    // ---- acc SLM zero (private region — no barrier needed) ----
#pragma unroll
    for (int nb = 0; nb < 16; nb++)
      slm_block_store<float, 128>(SLM_AC + nb * 4096 + lid * 512,
                                  simd<float, 128>(0.0f));
  }

  // ---- KV walk ----
  for (uint32_t c = 0; c < tiles; c++) {
    const uint32_t kvPos0 = c * V2_TOK;
    // One page per tile (pageSize % 32 == 0, launcher-checked; non-pow2
    // page law: integer divide only).
    const uint32_t pgIdx = kvPos0 / pageSize;
    const uint32_t pgOff = kvPos0 - pgIdx * pageSize;
    const uint32_t page = (uint32_t)blockTable[pgIdx];
    const bool pgOk = page < numPages;

    // -- stage 1: cooperative load+dequant (thread lid owns tokens
    // 4*lid..4*lid+3). Tokens beyond kvLen (last partial tile) are
    // clamped to the last valid token; their scores are masked at
    // softmax time (edge < kvLen always). v2.2: every token's K and V
    // is ONE 256-B single-message lsc block load and all 8 messages
    // are in flight before any dequant/store. v2.3 halves the SLM
    // store message count (K 32-lane 4-token scatter, V merged pair
    // stores) — the r92_v22b ablation proved staging scales with store
    // MESSAGE COUNT, not bytes or scatter stride. ablate&128 (floor
    // probe, L0 only): skip the loads entirely.
    if (pgOk) {
      const uint64_t tBase = (uint64_t)page * pageStride +
                             (uint64_t)pgOff * tokenStride +
                             (uint64_t)kvHead * headStride;
      // clamp the in-tile slot for the last partial tile (scores of
      // tokens >= kvLen are masked at softmax; the clamped offset stays
      // in-page: pgOff % 32 == 0, so pgOff + 31 < pageSize).
      uint32_t nC[4];
#pragma unroll
      for (int p = 0; p < 4; p++) {
        const uint32_t n = 4 * lid + p;             // token slot in tile
        nC[p] = (kvPos0 + n < kvLen) ? n : (kvLen - 1 - kvPos0);
      }
      simd<uint8_t, 256> kR0, kR1, kR2, kR3, vR0, vR1, vR2, vR3;
      if (ablate & 128u) {
        // floor probe: zero regs (loads skipped) — isolates the
        // load-issue/latency share of the L96 floor.
        kR0 = 0; kR1 = 0; kR2 = 0; kR3 = 0;
        vR0 = 0; vR1 = 0; vR2 = 0; vR3 = 0;
      } else {
      const uint8_t* kp0 = kState + tBase + (uint64_t)nC[0] * tokenStride;
      const uint8_t* kp1 = kState + tBase + (uint64_t)nC[1] * tokenStride;
      const uint8_t* kp2 = kState + tBase + (uint64_t)nC[2] * tokenStride;
      const uint8_t* kp3 = kState + tBase + (uint64_t)nC[3] * tokenStride;
      const uint8_t* vp0 = vState + tBase + (uint64_t)nC[0] * tokenStride;
      const uint8_t* vp1 = vState + tBase + (uint64_t)nC[1] * tokenStride;
      const uint8_t* vp2 = vState + tBase + (uint64_t)nC[2] * tokenStride;
      const uint8_t* vp3 = vState + tBase + (uint64_t)nC[3] * tokenStride;
        kR0 = lsc_block_load<uint8_t, 256, lsc_data_size::default_size,
                             cache_hint::cached, cache_hint::cached>(kp0);
        vR0 = lsc_block_load<uint8_t, 256, lsc_data_size::default_size,
                             cache_hint::cached, cache_hint::cached>(vp0);
        kR1 = lsc_block_load<uint8_t, 256, lsc_data_size::default_size,
                             cache_hint::cached, cache_hint::cached>(kp1);
        vR1 = lsc_block_load<uint8_t, 256, lsc_data_size::default_size,
                             cache_hint::cached, cache_hint::cached>(vp1);
        kR2 = lsc_block_load<uint8_t, 256, lsc_data_size::default_size,
                             cache_hint::cached, cache_hint::cached>(kp2);
        vR2 = lsc_block_load<uint8_t, 256, lsc_data_size::default_size,
                             cache_hint::cached, cache_hint::cached>(vp2);
        kR3 = lsc_block_load<uint8_t, 256, lsc_data_size::default_size,
                             cache_hint::cached, cache_hint::cached>(kp3);
        vR3 = lsc_block_load<uint8_t, 256, lsc_data_size::default_size,
                             cache_hint::cached, cache_hint::cached>(vp3);
      }
      // K -> VNNI SLM (the QK B operand; v2.1 law): word(kp, n) at
      // (kp*16+n)*4 inside the [h][kb] 512-B block at (h*16+kb)*512,
      // holding this token's dims (2kp, 2kp+1) — the v1-proven N=16 B
      // law. v2.3 (r92_v22b ablation law: staging cost is SLM-store
      // MESSAGE COUNT, not the scatter pattern — K 64x32B msgs cost 2x
      // V 32x64B): the thread's 4 tokens (nl0 = 4*(lid&3) in {0,4,8,12}
      // — never straddle a half-tile) interleave into ONE 32-lane
      // slm_scatter per kb: lane 4*kp+t = word(kp, nl0+t), offsets
      // kp*64 + t*4 — 16 msgs of 128 B per thread per tile (was 64 of
      // 32 B). Bounded unrolls — the fully-unrolled body crashed IGC on
      // bmg (v2 build-1). Timing bits (compose with level 0): ablate&8
      // zero the dequant results; ablate&64 skip the K SLM stores.
      auto stageK4 = [&](const simd<uint8_t, 256>& r0,
                         const simd<uint8_t, 256>& r1,
                         const simd<uint8_t, 256>& r2,
                         const simd<uint8_t, 256>& r3) {
        simd<fp16, 256> kq0 = faDequantE4m3<256>(r0);
        simd<fp16, 256> kq1 = faDequantE4m3<256>(r1);
        simd<fp16, 256> kq2 = faDequantE4m3<256>(r2);
        simd<fp16, 256> kq3 = faDequantE4m3<256>(r3);
        if (ablate & 8u) { kq0 = 0; kq1 = 0; kq2 = 0; kq3 = 0; }
        const uint32_t kh = lid >> 2;         // the 4 tokens share a half
        const uint32_t nl0 = 4 * (lid & 3u);  // token-in-half of token 0
        if (!(ablate & 64u)) {
#pragma unroll 4
          for (int kb = 0; kb < 16; kb++) {
            const uint32_t base =
                SLM_KG + (kh * 16 + kb) * 512 + nl0 * 4;
            // word8: the kb-slice's 8 dim-pairs as VNNI words (seg
            // materialized — simd_view does not deduce into simd
            // template params; ternaries on simd are deleted — hence
            // the helper, not a select chain).
            auto word8 = [&](simd<fp16, 256> kq) {  // by value: select()
              // needs a non-const object (ref-qualified & overload)
              simd<fp16, 16> seg = kq.template select<16, 1>(kb * 16);
              return simd<uint32_t, 8>(seg.template bit_cast_view<uint32_t>());
            };
            simd<uint32_t, 32> W, off;
            W.template select<8, 4>(0) = word8(kq0);
            W.template select<8, 4>(1) = word8(kq1);
            W.template select<8, 4>(2) = word8(kq2);
            W.template select<8, 4>(3) = word8(kq3);
            off.template select<8, 4>(0) = simd<uint32_t, 8>(base, 64);
            off.template select<8, 4>(1) = simd<uint32_t, 8>(base + 4, 64);
            off.template select<8, 4>(2) = simd<uint32_t, 8>(base + 8, 64);
            off.template select<8, 4>(3) = simd<uint32_t, 8>(base + 12, 64);
            slm_scatter<uint32_t, 32>(off, W);
          }
        }
      };
      stageK4(kR0, kR1, kR2, kR3);
      // V -> VNNI SLM: word(tp, d) = V[2tp][d] | V[2tp+1][d]<<16; the
      // thread's pairs P0 = 2*lid, P1 = 2*lid+1 share ks and sit at
      // adjacent tp — v2.3 merges them into one contiguous 128-B store
      // per dim-block (16 msgs of 128 B per tile, was 32 of 64 B).
      auto stageV2 = [&](const simd<uint8_t, 256>& r0,
                         const simd<uint8_t, 256>& r1,
                         const simd<uint8_t, 256>& r2,
                         const simd<uint8_t, 256>& r3) {
        simd<fp16, 256> vq0 = faDequantE4m3<256>(r0);
        simd<fp16, 256> vq1 = faDequantE4m3<256>(r1);
        simd<fp16, 256> vq2 = faDequantE4m3<256>(r2);
        simd<fp16, 256> vq3 = faDequantE4m3<256>(r3);
        if (ablate & 8u) { vq0 = 0; vq1 = 0; vq2 = 0; vq3 = 0; }
        const uint32_t P = 2 * lid;   // first tile token-pair index
        const uint32_t ks = P >> 3;
        const uint32_t tp = P & 7u;
        if (!(ablate & 32u)) {
#pragma unroll 4
          for (int nb = 0; nb < 16; nb++) {
            simd<fp16, 16> pv0 = vq0.template select<16, 1>(nb * 16);
            simd<fp16, 16> pv1 = vq1.template select<16, 1>(nb * 16);
            simd<fp16, 16> pv2 = vq2.template select<16, 1>(nb * 16);
            simd<fp16, 16> pv3 = vq3.template select<16, 1>(nb * 16);
            simd<uint32_t, 32> packed;
            packed.template select<16, 1>(0) =
                (simd<uint32_t, 16>(
                     pv0.template bit_cast_view<uint16_t>()) |
                 (simd<uint32_t, 16>(
                      pv1.template bit_cast_view<uint16_t>())
                  << 16));
            packed.template select<16, 1>(16) =
                (simd<uint32_t, 16>(
                     pv2.template bit_cast_view<uint16_t>()) |
                 (simd<uint32_t, 16>(
                      pv3.template bit_cast_view<uint16_t>())
                  << 16));
            slm_block_store<uint32_t, 32>(
                SLM_VV + ks * 8192 + nb * 512 + tp * 64, packed);
          }
        }
      };
      stageV2(vR0, vR1, vR2, vR3);
    }
    barrier();

    // -- compute (per-thread; inactive threads hold the barrier) --
    if (active) {
      // ---- QK': S[8 rows x 32 tokens] as TWO N=16 dpas chains per kb
      // (v2.1). A = q rows row-major [8 rows][16 dims] (one 256-B SLM
      // block load); B = K-tile VNNI [16 tokens][16 dims] per (kb,
      // half); C lane = m*16 + n in each 128-lane half. The dpas C width
      // (128, not 64) selects N=16 execution — the v1-proven operand
      // laws; a 64-lane C executes M4xK8xN16 on BMG (r92_dbg7 probe).
      // ----
      // ablate instrument (perf bisection only — serve always runs
      // ablate == 4): level = ablate & 7 (0 staging only; 1 +acc RMW;
      // 2 +QK; 3 +softmax; 4 +PV full; numerics valid only at 4);
      // variant bits = ablate & ~7, compose with level 0: 8 zero-dequant,
      // 32 skip V stores, 64 skip K stores, 128 skip global loads (the
      // r92_v22 window: bit values also satisfied threshold guards, so
      // 8/16/24 measured the FULL pipeline — fixed by the & 7 masks).
      simd<float, 128> sT0 = 0, sT1 = 0;
      if ((ablate & 7) >= 2) {
#pragma unroll 8
        for (int kb = 0; kb < 16; kb++) {
          simd<fp16, 128> qv =
              slm_block_load<fp16, 128>(SLM_QV + kb * 2048 + lid * 256);
          simd<fp16, 256> b0 = slm_block_load<fp16, 256>(SLM_KG + kb * 512);
          sT0 = xmx::dpas<8, 8, float, float, fp16, fp16,
                          xmx::dpas_argument_type::fp16,
                          xmx::dpas_argument_type::fp16>(sT0, b0, qv);
          simd<fp16, 256> b1 =
              slm_block_load<fp16, 256>(SLM_KG + (16 + kb) * 512);
          sT1 = xmx::dpas<8, 8, float, float, fp16, fp16,
                          xmx::dpas_argument_type::fp16,
                          xmx::dpas_argument_type::fp16>(sT1, b1, qv);
        }
      }

      // ---- scale + causal mask (bottom-right; per-row edge, one
      // 16-lane merge per row and half in the [m][16] layout) ----
      if ((ablate & 7) >= 2) {
        sT0 = sT0 * scoreScale;
        sT1 = sT1 * scoreScale;
        simd<uint32_t, 32> tokG = tokenIds + kvPos0;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          const uint32_t edge = (r0 + m) + (kvLen - qLen);
          sT0.template select<16, 1>(m * 16).merge(
              -1.0e30f, tokG.template select<16, 1>(0) > edge);
          sT1.template select<16, 1>(m * 16).merge(
              -1.0e30f, tokG.template select<16, 1>(16) > edge);
        }
      }

      // ---- online-softmax max fold: one whole-vector h-fold, then
      // in-row folds (offsets 8/4/2/1) as 8 per-row select pairs — a
      // vector-halves fold would pair lanes ACROSS rows in [m][16].
      // Operands materialized — simd_view does not deduce into simd
      // template params. ----
      simd<float, 128> pf0 = sT0, pf1 = sT1;   // ablate<3 passthrough
      simd<float, 8> rescale = 1.0f;
      if ((ablate & 7) >= 3) {
        simd<float, 128> f0 = faVMax(sT0, sT1);
        simd<float, 64> f1;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          simd<float, 8> xa = f0.template select<8, 1>(m * 16);
          simd<float, 8> xb = f0.template select<8, 1>(m * 16 + 8);
          f1.template select<8, 1>(m * 8) = faVMax(xa, xb);
        }
        simd<float, 32> f2;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          simd<float, 4> xa = f1.template select<4, 1>(m * 8);
          simd<float, 4> xb = f1.template select<4, 1>(m * 8 + 4);
          f2.template select<4, 1>(m * 4) = faVMax(xa, xb);
        }
        simd<float, 16> f3;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          simd<float, 2> xa = f2.template select<2, 1>(m * 4);
          simd<float, 2> xb = f2.template select<2, 1>(m * 4 + 2);
          f3.template select<2, 1>(m * 2) = faVMax(xa, xb);
        }
        simd<float, 8> mv;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          simd<float, 1> xa = f3.template select<1, 1>(m * 2);
          simd<float, 1> xb = f3.template select<1, 1>(m * 2 + 1);
          mv.template select<1, 1>(m) = faVMax(xa, xb);
        }
        simd<float, 8> newMax = runMax;
        {
          auto up = mv > runMax;
          newMax.merge(mv, up);
        }
        // rescale = 2^(runMax - newMax); first-sight rows stay at 1
        // (v1 guard semantics — acc is zero anyway).
        simd<float, 8> diff = (runMax - newMax) * 0.6931471805599453f;
        diff.merge(0.0f, runMax < -1.0e29f);
        rescale = exp(diff);
        runSum = runSum * rescale;
        runMax = newMax;

        // p = 2^(sT - newMax) per half; newMax[m] broadcast to its 16
        // lane slots in each half. pf halves ARE the PV A-tiles ([m][16
        // tokens] row-major — direct fp16 convert, no lane surgery).
        // (Masked lanes: -1e30 - real max -> exp underflows to 0.)
        simd<float, 128> nm0, nm1;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          nm0.template select<16, 1>(m * 16) = simd<float, 16>(newMax[m]);
          nm1.template select<16, 1>(m * 16) = simd<float, 16>(newMax[m]);
        }
        pf0 = exp((sT0 - nm0) * 0.6931471805599453f);
        pf1 = exp((sT1 - nm1) * 0.6931471805599453f);
        // row sums fold (h-fold whole-vector, then per-row like the max).
        simd<float, 128> s0 = pf0 + pf1;
        simd<float, 64> s1;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          simd<float, 8> xa = s0.template select<8, 1>(m * 16);
          simd<float, 8> xb = s0.template select<8, 1>(m * 16 + 8);
          s1.template select<8, 1>(m * 8) = xa + xb;
        }
        simd<float, 32> s2;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          simd<float, 4> xa = s1.template select<4, 1>(m * 8);
          simd<float, 4> xb = s1.template select<4, 1>(m * 8 + 4);
          s2.template select<4, 1>(m * 4) = xa + xb;
        }
        simd<float, 16> s3;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          simd<float, 2> xa = s2.template select<2, 1>(m * 4);
          simd<float, 2> xb = s2.template select<2, 1>(m * 4 + 2);
          s3.template select<2, 1>(m * 2) = xa + xb;
        }
        simd<float, 8> s4;
#pragma unroll
        for (int m = 0; m < (int)V2_RPB; m++) {
          simd<float, 1> xa = s3.template select<1, 1>(m * 2);
          simd<float, 1> xb = s3.template select<1, 1>(m * 2 + 1);
          s4.template select<1, 1>(m) = xa + xb;
        }
        runSum = runSum + s4;
      }

      // ---- PV: A = p-tile [8 rows x 16 tokens] = direct fp16 convert
      // of the pf half (identical [m][16] layout); B = V VNNI [16
      // tokens x 16 dims] (N=16). acc RMW folds the rescale; the
      // thread's nb slice is one 512-B block. ----
#pragma unroll 8
      for (int nb = 0; nb < 16; nb++) {
        simd<float, 128> pv = 0;
        if ((ablate & 7) >= 4) {
#pragma unroll
          for (int ks = 0; ks < 2; ks++) {
            simd<fp16, 128> a;
            if (ks == 0) a = simd<fp16, 128>(pf0);
            else         a = simd<fp16, 128>(pf1);
            simd<fp16, 256> bvv =
                slm_block_load<fp16, 256>(SLM_VV + ks * 8192 + nb * 512);
            pv = xmx::dpas<8, 8, float, float, fp16, fp16,
                           xmx::dpas_argument_type::fp16,
                           xmx::dpas_argument_type::fp16>(pv, bvv, a);
          }
        }
        if ((ablate & 7) >= 1) {
          simd<float, 128> a0v =
              slm_block_load<float, 128>(SLM_AC + nb * 4096 + lid * 512);
#pragma unroll
          for (int m = 0; m < (int)V2_RPB; m++)
            a0v.template select<16, 1>(m * 16) =
                a0v.template select<16, 1>(m * 16) * rescale[m] +
                pv.template select<16, 1>(m * 16);
          slm_block_store<float, 128>(SLM_AC + nb * 4096 + lid * 512, a0v);
        }
      }
    }
    barrier();   // K/V SLM reuse next tile — all threads done reading
  }

  // ---- epilogue: acc -> fp16 out (VALUE conversion — probe-1 law:
  // simd<fp16>(simd<float>) is the donor's hardware-cvt idiom) ----
  if (active) {
#pragma unroll
    for (int m = 0; m < (int)V2_RPB; m++) {
      const uint32_t g = r0 + m;
      if (g >= rows) break;
      float denom = runSum[m];
      if (denom == 0.0f) denom = 1.0f;
      const float norm = vDescale / denom;
      simd<fp16, 256> outH;
#pragma unroll
      for (int nb = 0; nb < 16; nb++) {
        simd<float, 16> ac =
            slm_block_load<float, 16>(SLM_AC + nb * 4096 + (lid * 8 + m) * 64);
        outH.template select<16, 1>(nb * 16) = simd<fp16, 16>(ac * norm);
      }
      simd<uint8_t, 512> bytes = outH.template bit_cast_view<uint8_t>();
      const uint64_t outBase =
          ((uint64_t)g * qHeads + head) * V2_HEAD_DIM * 2u;
#pragma unroll
      for (int c8 = 0; c8 < 8; c8++)
        lsc_block_store<uint8_t, 64, lsc_data_size::default_size,
                        cache_hint::write_back, cache_hint::write_back>(
            outState + outBase + (uint32_t)(c8 * 64),
            bytes.template select<64, 1>(c8 * 64));
    }
  }
}
