// Dense causal varlen flash attention — ESIMD/Xe2 (p72, P6.7 rung B).
//
// Computes the model's 16 dense full-attention layers' prefill:
//   out[row, head, :] = softmax(q[row, head, :] . K^T * scale) V
// against a PAGED fp8-e4m3 KV cache, causal bottom-right aligned, varlen
// batched by cu_seqlens_q. This is the site the r67b capture sized at
// 20.3% of rank0 device time at U32k (_vllm_fa2_C::varlen_fwd, 102 calls
// ~31 ms avg, ~5-7% est. MFU — plausibly KV-bandwidth-bound).
//
// Port of the radiance W14 QSA sparse kernel (qsa.sparse.fp8.h, md5
// 122515dc32097ea39cf8b66ae662c8d6, donor read-only — fork law §13).
// The sparse->dense delta per results/2026-10-31-p72-rungb-dense-fa-esimd/
// RECON.md: DROP the logical_indices walk / tokenToReq / split-G+merge;
// ADD per-row causal masking, GQA kv-head dispatch (qHeads/kvHeads group),
// per-(seq, head) descales, fp16 q/out (donor is bf16 out).
//
// Layout laws inherited verbatim from the donor (W14 banked laws):
//   * 512B q-head law — a fp16 head of 256 dims is 512 BYTES; load all
//     8x64B blocks or dims >= 128 silently drop (probe23 signature).
//   * GRF law — one q-head per work-item, live state ~3.3KB (q 512B +
//     acc 1KB + pp 512B + pads); no scratch, no merge kernel.
//   * Non-pow2 page law — PAGE 832 (serve, dense layer; the probe ladder
//     used 1664, the mamba-padded hybrid page): integer div/mod only,
//     never shift/mask. 832 = 52*16 (and 1664 = 104*16), so a 16-token
//     chunk NEVER spans a page boundary: ONE page lookup per chunk
//     (scalar), lanes walk +tokenStride.
//   * dpas BMG law — RepeatCount=4 keeps M=8; the A-tile carries 1 real
//     row, rows 1-7 zero.
//   * lsc_gather offsets are BYTE offsets; K[token i][seg dim c] lands at
//     gather-result byte (c>>2)*64 + i*4 + (c&3) (donor SoA walk).
//
// Numerics: log2-domain online softmax with -1e30 sentinels (single-pass,
// flash reassociation — bit-exactness vs FA2 impossible; gate is fro/cos,
// donor law). Deterministic: serial chunk chain per item, fp32 accumulators.
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

// One work-item = (q-head, row). The row's KV span is walked serially in
// 16-token chunks with online softmax — donor v2 shape, no split-G (rows x
// qHeads ~ 98k items at a U32k chunk call; no scratch at all). Chunks whose
// first position is beyond the row's causal position break out (causal skip
// halves the average walk); the boundary chunk masks tail lanes.
void faDenseKernel(
    const uint8_t* __restrict__ qState,   // [rows, qHeads, 256] fp16
    const uint8_t* __restrict__ kState,   // [pages, pageSize, kvHeads, 256] e4m3
    const uint8_t* __restrict__ vState,   // same shape as kState
    const int32_t* __restrict__ cuSeqlensQ,  // [bs+1] row starts
    const int32_t* __restrict__ sequsedK,    // [bs] kv lengths
    const int32_t* __restrict__ blockTable,  // [bs, tableStride]
    uint8_t* __restrict__ outState,       // [rows, qHeads, 256] fp16
    const float* __restrict__ kScalePtr,  // [bs, kvHeads] fp32 or null
    const float* __restrict__ vScalePtr,  // [bs, kvHeads] fp32 or null
    const uint32_t rows, const uint32_t qHeads, const uint32_t kvHeads,
    const uint32_t bs, const uint32_t tableStride,
    const uint32_t numPages, const uint32_t pageStride,
    const uint32_t tokenStride, const uint32_t headStride,
    const uint32_t pageSize, const float softmaxScaleLog2,
    nd_item<2> ndi) {
  constexpr uint32_t HEAD_DIM = 256;
  constexpr uint32_t LANES = 16;
  const uint32_t head = ndi.get_global_id(0);
  const uint32_t row = ndi.get_global_id(1);
  if (row >= rows) return;

  // ---- row -> (sequence, q-index) : binary search over cu_seqlens_q.
  // bs <= 64 on this serve (max_num_seqs law), so <= 7 scalar probes.
  uint32_t lo = 0, hi = bs;
  while (lo + 1 < hi) {
    const uint32_t mid = (lo + hi) >> 1;
    if ((uint32_t)cuSeqlensQ[mid] <= row) lo = mid; else hi = mid;
  }
  const uint32_t seq = lo;
  const uint32_t rowBeg = (uint32_t)cuSeqlensQ[seq];
  const uint32_t qLenSeq = (uint32_t)cuSeqlensQ[seq + 1] - rowBeg;
  const uint32_t qIdx = row - rowBeg;
  const uint32_t kvLen = (uint32_t)sequsedK[seq];
  // causal bottom-right: the q chunk sits at the sequence end — the row's
  // global causal position (last visible key index).
  const uint32_t qGlobal = qIdx + (kvLen - qLenSeq);

  const uint32_t groupSize = qHeads / kvHeads;   // 12/2 = 6 at this serve
  const uint32_t kvHead = head / groupSize;
  const float kDescale = kScalePtr ? kScalePtr[seq * kvHeads + kvHead] : 1.0f;
  const float vDescale = vScalePtr ? vScalePtr[seq * kvHeads + kvHead] : 1.0f;
  const float scoreScale = softmaxScaleLog2 * kDescale;

  // ---- Q head: 512 BYTES of fp16 (512B law). Input is already fp16 —
  // bit_cast straight through, no bf16->fp16 widen (donor needed one).
  const uint64_t qBase = ((uint64_t)row * qHeads + head) * HEAD_DIM * 2u;
  simd<uint8_t, 512> qRaw = 0;
#pragma unroll
  for (int c8 = 0; c8 < 8; c8++)
    qRaw.template select<64, 1>(c8 * 64) =
        lsc_block_load<uint8_t, 64, lsc_data_size::default_size,
                       cache_hint::cached, cache_hint::cached>(
            (const uint8_t*)qState + qBase + (uint32_t)(c8 * 64));
  // non-const: select()'s lvalue overload is non-const-qualified (build-2)
  simd<fp16, 256> qH = qRaw.template bit_cast_view<fp16>();

  simd<uint32_t, LANES> laneIds(0, 1);

  // ---- online-softmax state (this head) ----
  float runMax = -1.0e30f;
  float runSum = 0.0f;
  simd<float, 256> acc = 0.0f;

  const uint32_t chunks = (kvLen + 15u) / 16u;
  const uint32_t lastChunk = qGlobal / 16u;   // chunks beyond are all-masked
  for (uint32_t c = 0; c < chunks; c++) {
    if (c > lastChunk) break;                 // causal skip (monotone)
    const uint32_t kvPos0 = c * 16u;

    // ONE page per chunk (832 = 52*16 — 16-token chunks never straddle);
    // non-pow2 page law: divide, never shift/mask.
    const uint32_t pgIdx = kvPos0 / pageSize;
    const uint32_t pgOff = kvPos0 - pgIdx * pageSize;
    const uint32_t page = (uint32_t)blockTable[seq * tableStride + pgIdx];
    if (page >= numPages) break;              // corrupt table: stop, not garbage
    const uint32_t tokenBase =
        page * pageStride + pgOff * tokenStride + kvHead * headStride;

    // lanes: kvPos0+lane <= qGlobal (< kvLen by construction).
    // auto = simd_mask (donor law): lsc_gather/merge want a mask, not a
    // u32 simd (build-2: explicit simd<uint32_t> failed overload resolution).
    auto colOk = laneIds <= (qGlobal - kvPos0);
    simd<uint32_t, LANES> tokBase = tokenBase + laneIds * tokenStride;

    // -- QK: 8 segments x (gather + stride-4 dequant + VNNI + 2 dpas) --
    simd<float, 8 * 16> pp(0.0f);
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
      simd<uint32_t, LANES * 8> raw =
          lsc_gather<uint32_t, 8, lsc_data_size::u32, cache_hint::cached,
                     cache_hint::cached, LANES, uint32_t>(
              (const uint32_t*)kState, tokBase + (uint32_t)(kk * 32), colOk);
      auto rawB = raw.template bit_cast_view<uint8_t>();
#pragma unroll
      for (int kd = 0; kd < 2; kd++) {
        simd<fp16, 16 * 16> bVnni;
        auto bU16 = bVnni.template bit_cast_view<uint16_t>();
#pragma unroll
        for (int kp = 0; kp < 8; kp++) {
          const int c_lo = kd * 16 + 2 * kp;
          const int c_hi = c_lo + 1;
          // 16 stride-4 bytes = K[lane i][c] straight from the SoA bytes
          simd<uint8_t, 16> loB =
              rawB.template select<16, 4>((c_lo >> 2) * 64 + (c_lo & 3));
          simd<uint8_t, 16> hiB =
              rawB.template select<16, 4>((c_hi >> 2) * 64 + (c_hi & 3));
          simd<fp16, 16> lo = faDequantE4m3<16>(loB);
          simd<fp16, 16> hi = faDequantE4m3<16>(hiB);
          lo.merge(fp16(0.0f), !colOk);
          hi.merge(fp16(0.0f), !colOk);
          simd<uint32_t, 16> packed =
              (simd<uint32_t, 16>(lo.template bit_cast_view<uint16_t>()) |
               (simd<uint32_t, 16>(hi.template bit_cast_view<uint16_t>())
                << 16));
          auto dst = bU16.template select<32, 1>(kp * 32);
          dst.template bit_cast_view<uint32_t>() = packed;
        }
        simd<fp16, 8 * 16> aPad = 0;
        aPad.template select<16, 1>(0) =
            qH.template select<16, 1>(kk * 32 + kd * 16);
        pp = sycl::ext::intel::esimd::xmx::dpas<
            8, 8, float, float, fp16, fp16,
            sycl::ext::intel::esimd::xmx::dpas_argument_type::fp16,
            sycl::ext::intel::esimd::xmx::dpas_argument_type::fp16>(
            pp, bVnni, aPad);
      }
    }

    // scores -> log2 domain; masked lanes to the -inf sentinel
    {
      auto r = pp.template select<16, 1>(0);
      r = r * scoreScale;
      r.merge(-1.0e30f, !colOk);
    }

    // -- online softmax update (donor verbatim) --
    float mv = -1.0e30f;
    {
      auto r = pp.template select<16, 1>(0);
#pragma unroll
      for (int l = 0; l < 16; l++) {
        float v = r[l];
        mv = (v > mv) ? v : mv;
      }
    }
    const float newMax = (runMax > mv) ? runMax : mv;
    const float rescale =
        (runMax - newMax) > -1.0e29f && newMax > -1.0e29f
            ? sycl::exp((runMax - newMax) * 0.6931471805599453f)
            : 1.0f;
    simd<fp16, 8 * 16> aPadP = 0;
    if (newMax > -1.0e29f) {
      auto r = pp.template select<16, 1>(0);
      simd<float, 16> pf;
#pragma unroll
      for (int l = 0; l < 16; l++)
        pf[l] = (r[l] - newMax) * 0.6931471805599453f;
      pf = exp(pf);
      aPadP.template select<16, 1>(0) = simd<fp16, 16>(pf);
      float s = 0.0f;
#pragma unroll
      for (int l = 0; l < 16; l++) s += pf[l];
      runSum = runSum * rescale + s;
    }
    acc = acc * rescale;

    // -- PV: 8 segments x 2 col-slices; B-tile token-major (donor walk) --
#pragma unroll
    for (int vk = 0; vk < 8; vk++) {
      simd<uint32_t, LANES * 8> raw =
          lsc_gather<uint32_t, 8, lsc_data_size::u32, cache_hint::cached,
                     cache_hint::cached, LANES, uint32_t>(
              (const uint32_t*)vState, tokBase + (uint32_t)(vk * 32), colOk);
      auto rawB = raw.template bit_cast_view<uint8_t>();
#pragma unroll
      for (int n16 = 0; n16 < 2; n16++) {
        simd<fp16, 16 * 16> bVnni;
        auto bU16 = bVnni.template bit_cast_view<uint16_t>();
#pragma unroll
        for (int kp = 0; kp < 8; kp++) {
          simd<uint8_t, 16> loB = 0, hiB = 0;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            loB.template select<4, 1>(j * 4) =
                rawB.template select<4, 1>((n16 * 4 + j) * 64 + (2 * kp) * 4);
            hiB.template select<4, 1>(j * 4) =
                rawB.template select<4, 1>((n16 * 4 + j) * 64 + (2 * kp + 1) * 4);
          }
          simd<fp16, 16> lo = faDequantE4m3<16>(loB);
          simd<fp16, 16> hi = faDequantE4m3<16>(hiB);
          simd<uint32_t, 16> packed =
              (simd<uint32_t, 16>(lo.template bit_cast_view<uint16_t>()) |
               (simd<uint32_t, 16>(hi.template bit_cast_view<uint16_t>())
                << 16));
          auto dst = bU16.template select<32, 1>(kp * 32);
          dst.template bit_cast_view<uint32_t>() = packed;
        }
        simd<float, 8 * 16> pv = 0;
        pv = sycl::ext::intel::esimd::xmx::dpas<
            8, 8, float, float, fp16, fp16,
            sycl::ext::intel::esimd::xmx::dpas_argument_type::fp16,
            sycl::ext::intel::esimd::xmx::dpas_argument_type::fp16>(
            pv, bVnni, aPadP);
        acc.template select<16, 1>(vk * 32 + n16 * 16) =
            acc.template select<16, 1>(vk * 32 + n16 * 16) +
            pv.template select<16, 1>(0);
      }
    }
    runMax = newMax;
  }

  // ---- normalize + store fp16 (VALUE conversion — probe-1 lesson: a
  // bf16-style bit-drop is illegal here; fp16's 5-bit exponent makes
  // fp32->fp16 truncation produce garbage magnitudes (~1e4 out of ~1).
  // simd<fp16>(simd<float>) is the donor's own conversion idiom (its qH
  // widen), hardware cvt/RNE) ----
  const uint64_t outBase = ((uint64_t)row * qHeads + head) * HEAD_DIM * 2u;
  float denom = runSum;
  if (denom == 0.0f) denom = 1.0f;
  const float norm = vDescale / denom;
  simd<fp16, 256> outH;
#pragma unroll
  for (int s = 0; s < 16; s++)
    outH.template select<16, 1>(s * 16) =
        simd<fp16, 16>(acc.template select<16, 1>(s * 16) * norm);
  simd<uint8_t, 512> bytes = outH.template bit_cast_view<uint8_t>();
#pragma unroll
  for (int c8 = 0; c8 < 8; c8++)
    lsc_block_store<uint8_t, 64, lsc_data_size::default_size,
                    cache_hint::write_back, cache_hint::write_back>(
        (uint8_t*)outState + outBase + (uint32_t)(c8 * 64),
        bytes.template select<64, 1>(c8 * 64));
}
