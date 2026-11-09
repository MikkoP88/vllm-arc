// JM (joint-matrix/DPAS) skinny-M weight-only GEMM entry points (P6.34,
// arc7 wheel).
//
// Three torch ops mirroring the signatures of the stock oneDNN weight-only
// ops (and the arc6 skinny twins) so probes and the python M-split route
// swap the call 1:1 at M <= 8:
//
//   jm_fp8_gemm_w8a16(A, B, B_scale?, bias?)
//       checkpoint-fp8 per-tensor / per-channel scales (fp32)
//   jm_fp8_gemm_w8a16_mxfp8(A, B, B_scale, out_dtype?, bias?)
//       online MXFP8 group-32 e8m0 weight scales
//   jm_fp4_gemm_w4a16(A, B, B_scale, out_dtype?, bias?)
//       online MXFP4 packed e2m1 weights, group-32 e8m0 scales
//
// Weights arrive COLUMN-REPACKED (each lane's B fragment is its column's
// 16/8 consecutive bytes of a tile -- a near-identity reshape of [N, K];
// see jm_gemm_kernel.hpp): f8 u8 [N/16, K/16, 256], f4 u8 [N/16, K/16,
// 128], N zero-padded to a 16 multiple by the one-time python repack
// helper at weight load (the ops return [M, N_pad]; callers slice the
// real N). The
// scale resolvers (group-stride order, pointer-passed per-tensor scales --
// no .item() D2H sync) are lifted from the arc6 skinny interface (P6.33);
// the kernel underneath is the DPAS family (jm_gemm_kernel.hpp).

#include <torch/all.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include "jm_gemm_kernel.hpp"

namespace vllm::xpu::jm {

namespace {

int64_t flatten_m(const torch::Tensor& a) {
  int64_t m = 1;
  for (int64_t i = 0; i < a.dim() - 1; ++i) m *= a.size(i);
  return m;
}

bool valid_epilogue_dtype(c10::ScalarType t) {
  return t == at::ScalarType::BFloat16 || t == at::ScalarType::Half;
}

// Bias pointer (fp32 [N]) or nullptr; an undefined/empty tensor is "none"
// (the fp8_block_w8a16 idiom passes torch.Tensor()).
const float* bias_ptr_or_null(const std::optional<torch::Tensor>& bias,
                              int64_t n) {
  if (!bias.has_value() || !bias.value().defined() ||
      bias.value().numel() == 0) {
    return nullptr;
  }
  const torch::Tensor& b = bias.value();
  TORCH_CHECK(
      b.dim() == 1 && b.size(0) == n && b.is_contiguous() &&
          b.scalar_type() == at::ScalarType::Float,
      "jm gemm bias must be a contiguous fp32 [N] tensor, got ",
      b.sizes(),
      " ",
      b.scalar_type());
  return b.const_data_ptr<float>();
}

// f8 weights: column-repacked tiles (python helper runs ONCE at weight
// load; see jm_gemm_kernel.hpp for the tile byte layout) as u8
// [N/16, K/16, 256]. N is zero-padded up to a 16 multiple by the repack,
// so tiles = size(0) exactly and the op returns [M, N_pad] (the caller
// slices the real N columns; stock-layout weights are rejected -- a
// per-call repack would poison the measurement law).
int64_t resolve_repacked_f8_n(const torch::Tensor& b, int64_t k) {
  TORCH_CHECK(
      b.dim() == 3 && b.scalar_type() == at::ScalarType::Byte &&
          b.is_contiguous(),
      "jm w8a16 weight must be the contiguous column-repacked u8 "
      "[N/16, K/16, 256] tensor (python repack helper), got ",
      b.sizes(),
      " ",
      b.scalar_type());
  TORCH_CHECK(
      b.size(2) == 256 && b.size(1) * 16 == k,
      "jm w8a16 repacked weight shape mismatch: got ",
      b.sizes(),
      " for K=",
      k);
  return b.size(0) * 16;
}

// Packed f4 weights: column-repacked nibble tiles as u8 [N/16, K/16, 128].
int64_t resolve_repacked_f4_n(const torch::Tensor& b, int64_t k) {
  TORCH_CHECK(
      b.dim() == 3 && b.scalar_type() == at::ScalarType::Byte &&
          b.is_contiguous(),
      "jm f4 gemm weight must be the contiguous column-repacked u8 "
      "[N/16, K/16, 128] tensor (python repack helper), got ",
      b.sizes(),
      " ",
      b.scalar_type());
  TORCH_CHECK(
      b.size(2) == 128 && b.size(1) * 16 == k,
      "jm f4 repacked weight shape mismatch: got ",
      b.sizes(),
      " for K=",
      k);
  return b.size(0) * 16;
}

// Group scales: resolve per-column and per-group byte strides for either
// [N, KG] or [KG, N] order (N-axis preference on square ambiguity).
struct ScaleStrides {
  int64_t n_stride;
  int64_t g_stride;
};
ScaleStrides resolve_group_strides(
    const torch::Tensor& sc, int64_t n, int64_t kg) {
  TORCH_CHECK(
      sc.scalar_type() == at::ScalarType::Float8_e8m0fnu,
      "jm gemm group scale must be f8_e8m0, got ", sc.scalar_type());
  if (sc.dim() == 2 && sc.size(0) == n && sc.size(1) == kg) {
    return ScaleStrides{sc.stride(0), sc.stride(1)};
  }
  if (sc.dim() == 2 && sc.size(0) == kg && sc.size(1) == n) {
    return ScaleStrides{sc.stride(1), sc.stride(0)};
  }
  TORCH_CHECK(
      false,
      "jm gemm group scale must be [N, K/32] or [K/32, N], got ",
      sc.sizes(),
      " for N=",
      n,
      " K/32=",
      kg);
}

// Per-tensor / per-channel fp32 scales as POINTER + N-stride (0 = scalar
// broadcast): never a host .item() -- that was a per-call D2H sync (the
// arc6 v2.5.1 lesson; window8b p90 tails).
struct EpilogueScale {
  const float* per_channel;  // nullptr => no scale at all
  int64_t n_stride;          // 0 = scalar (broadcast), 1+ = per-channel
  float per_tensor;
};
EpilogueScale resolve_epilogue_scale(
    const std::optional<torch::Tensor>& sc, int64_t n) {
  if (!sc.has_value() || !sc.value().defined() || sc.value().numel() == 0) {
    return EpilogueScale{nullptr, 0, 1.0f};
  }
  const torch::Tensor& s = sc.value();
  TORCH_CHECK(
      s.scalar_type() == at::ScalarType::Float,
      "jm gemm checkpoint scale must be fp32, got ", s.scalar_type());
  if (s.numel() == 1) {
    return EpilogueScale{s.const_data_ptr<float>(), 0, 1.0f};
  }
  if (s.dim() == 1 && s.size(0) == n) {
    return EpilogueScale{s.const_data_ptr<float>(), s.stride(0), 1.0f};
  }
  if (s.dim() == 2) {
    if (s.size(0) == 1 && s.size(1) == n) {
      return EpilogueScale{s.const_data_ptr<float>(), s.stride(1), 1.0f};
    }
    if (s.size(1) == 1 && s.size(0) == n) {
      return EpilogueScale{s.const_data_ptr<float>(), s.stride(0), 1.0f};
    }
  }
  TORCH_CHECK(
      false,
      "jm gemm checkpoint scale must be scalar or [1, N], got ",
      s.sizes(),
      " for N=",
      n);
}

torch::Tensor make_output(
    const torch::Tensor& a, int64_t n, c10::ScalarType out_dtype) {
  auto sizes = a.sizes().vec();
  sizes.back() = n;
  return at::empty(sizes, a.options().dtype(out_dtype));
}

void check_activation(const torch::Tensor& a) {
  TORCH_CHECK(
      a.is_contiguous(),
      "jm gemm requires contiguous activations, got strides ",
      a.strides());
  TORCH_CHECK(
      a.scalar_type() == at::ScalarType::BFloat16 ||
          a.scalar_type() == at::ScalarType::Half,
      "jm gemm requires bf16/f16 activations, got ", a.scalar_type());
}

void check_weights_aligned16(const torch::Tensor& b) {
  // 128-bit f8 / 64-bit f4 weight loads: K is a multiple of 32 so every
  // column base is aligned iff the storage is; a misaligned view falls
  // back via the python wrapper's try/except.
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(b.const_data_ptr()) % 16 == 0,
      "jm gemm weight pointer must be 16-byte aligned (k-contiguous "
      "layout with odd storage offset -- call .contiguous())");
}

// torch 2.13's XPUStream::queue() yields the sycl queue directly; the copy
// submits to the same underlying in-order queue (arc6 pattern).
sycl::queue current_sycl_queue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

void check_m_small(int64_t m) {
  TORCH_CHECK(
      m >= 1 && m <= 8,
      "jm gemm is an M <= 8 decode-path kernel, got M=", m,
      " (the python route must gate on M)");
}

}  // namespace

// ---------------------------------------------------------------------------
// Checkpoint-fp8 lane (w16pt / w16pc champion classes).
// ---------------------------------------------------------------------------
torch::Tensor jm_fp8_gemm_w8a16(
    const torch::Tensor& A,        // [.., M, K] bf16/f16
    const torch::Tensor& B,        // column-repacked u8 [N/16, K/16, 256]
    const std::optional<torch::Tensor>& B_scale,  // fp32 scalar or [1, N_pad]
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  check_activation(A);
  const int64_t k = A.size(-1);
  const int64_t m = flatten_m(A);
  check_m_small(m);
  TORCH_CHECK(k % 32 == 0, "jm gemm requires K % 32 == 0, got K=", k);
  const int64_t n = resolve_repacked_f8_n(B, k);
  check_weights_aligned16(B);
  // Repacked u8 tiles carry no dtype tag: the checkpoint-fp8 stock lane is
  // e4m3fn (e5m2 remains a kernel-family instantiation, not an op lane).
  const auto wf = WeightFormat::kF8E4m3;

  const auto esc = resolve_epilogue_scale(B_scale, n);
  const float* bias_p = bias_ptr_or_null(bias, n);

  torch::Tensor result = make_output(A, n, A.scalar_type());
  auto queue = current_sycl_queue();
  const int m_i = static_cast<int>(m), n_i = static_cast<int>(n),
            k_i = static_cast<int>(k);
  const auto cfg = jm_pick_cfg(m_i, n_i, k_i);
  torch::Tensor partials;
  float* partials_p = nullptr;
  if (!cfg.direct) {
    partials = at::empty(
        {cfg.splits, 8, n}, A.options().dtype(at::kFloat));
    partials_p = partials.data_ptr<float>();
  }
  jm_launch(
      queue, wf, /*group32=*/false,
      /*a_bf16=*/A.scalar_type() == at::ScalarType::BFloat16,
      A.const_data_ptr(),
      reinterpret_cast<const std::uint8_t*>(B.const_data_ptr()),
      /*bsc=*/nullptr, partials_p, result.data_ptr(), bias_p,
      esc.per_channel, esc.per_tensor,
      /*out_bf16=*/A.scalar_type() == at::ScalarType::BFloat16, m_i, n_i,
      k_i, /*sc_ns=*/0, /*sc_gs=*/1, esc.n_stride);
  return result;
}

// ---------------------------------------------------------------------------
// Online MXFP8 lane (group-32 e8m0 weight scales).
// ---------------------------------------------------------------------------
torch::Tensor jm_fp8_gemm_w8a16_mxfp8(
    const torch::Tensor& A,        // [.., M, K] bf16/f16
    const torch::Tensor& B,        // column-repacked u8 [N/16, K/16, 256]
    const torch::Tensor& B_scale,  // e8m0 [N_pad, K/32] or [K/32, N_pad]
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  check_activation(A);
  const int64_t k = A.size(-1);
  const int64_t m = flatten_m(A);
  check_m_small(m);
  TORCH_CHECK(k % 32 == 0, "jm gemm requires K % 32 == 0, got K=", k);
  const int64_t n = resolve_repacked_f8_n(B, k);
  check_weights_aligned16(B);
  const auto wf = WeightFormat::kF8E4m3;

  const auto gs = resolve_group_strides(B_scale, n, k / 32);
  const float* bias_p = bias_ptr_or_null(bias, n);
  const c10::ScalarType odt = out_dtype.value_or(A.scalar_type());
  TORCH_CHECK(
      valid_epilogue_dtype(odt),
      "jm mxfp8 out_dtype must be bf16/f16, got ", odt);

  torch::Tensor result = make_output(A, n, odt);
  auto queue = current_sycl_queue();
  const int m_i = static_cast<int>(m), n_i = static_cast<int>(n),
            k_i = static_cast<int>(k);
  const auto cfg = jm_pick_cfg(m_i, n_i, k_i);
  torch::Tensor partials;
  float* partials_p = nullptr;
  if (!cfg.direct) {
    partials = at::empty(
        {cfg.splits, 8, n}, A.options().dtype(at::kFloat));
    partials_p = partials.data_ptr<float>();
  }
  jm_launch(
      queue, wf, /*group32=*/true,
      /*a_bf16=*/A.scalar_type() == at::ScalarType::BFloat16,
      A.const_data_ptr(),
      reinterpret_cast<const std::uint8_t*>(B.const_data_ptr()),
      reinterpret_cast<const std::uint8_t*>(B_scale.const_data_ptr()),
      partials_p, result.data_ptr(), bias_p,
      /*pc_scale=*/nullptr, /*pt_scale=*/1.0f,
      /*out_bf16=*/odt == at::ScalarType::BFloat16, m_i, n_i, k_i,
      gs.n_stride, gs.g_stride, /*pc_stride=*/1);
  return result;
}

// ---------------------------------------------------------------------------
// Online MXFP4 lane (packed e2m1 nibbles, group-32 e8m0 scales).
// ---------------------------------------------------------------------------
torch::Tensor jm_fp4_gemm_w4a16(
    const torch::Tensor& A,        // [.., M, K] bf16/f16
    const torch::Tensor& B,        // column-repacked u8 [N/16, K/16, 128]
    const torch::Tensor& B_scale,  // e8m0 [N_pad, K/32] or [K/32, N_pad]
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  check_activation(A);
  const int64_t k = A.size(-1);
  const int64_t m = flatten_m(A);
  check_m_small(m);
  TORCH_CHECK(k % 32 == 0, "jm gemm requires K % 32 == 0, got K=", k);
  const int64_t n = resolve_repacked_f4_n(B, k);
  check_weights_aligned16(B);

  const auto gs = resolve_group_strides(B_scale, n, k / 32);
  const float* bias_p = bias_ptr_or_null(bias, n);
  const c10::ScalarType odt = out_dtype.value_or(A.scalar_type());
  TORCH_CHECK(
      valid_epilogue_dtype(odt),
      "jm mxfp4 out_dtype must be bf16/f16, got ", odt);

  torch::Tensor result = make_output(A, n, odt);
  auto queue = current_sycl_queue();
  const int m_i = static_cast<int>(m), n_i = static_cast<int>(n),
            k_i = static_cast<int>(k);
  const auto cfg = jm_pick_cfg(m_i, n_i, k_i);
  torch::Tensor partials;
  float* partials_p = nullptr;
  if (!cfg.direct) {
    partials = at::empty(
        {cfg.splits, 8, n}, A.options().dtype(at::kFloat));
    partials_p = partials.data_ptr<float>();
  }
  jm_launch(
      queue, WeightFormat::kF4E2m1, /*group32=*/true,
      /*a_bf16=*/A.scalar_type() == at::ScalarType::BFloat16,
      A.const_data_ptr(),
      reinterpret_cast<const std::uint8_t*>(B.const_data_ptr()),
      reinterpret_cast<const std::uint8_t*>(B_scale.const_data_ptr()),
      partials_p, result.data_ptr(), bias_p,
      /*pc_scale=*/nullptr, /*pt_scale=*/1.0f,
      /*out_bf16=*/odt == at::ScalarType::BFloat16, m_i, n_i, k_i,
      gs.n_stride, gs.g_stride, /*pc_stride=*/1);
  return result;
}

}  // namespace vllm::xpu::jm
