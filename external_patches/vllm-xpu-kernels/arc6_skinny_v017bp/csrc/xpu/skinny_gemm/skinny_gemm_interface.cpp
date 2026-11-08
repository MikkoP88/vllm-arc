// Skinny-M weight-only GEMM entry points (P6.33, arc6 wheel).
//
// Three torch ops mirroring the signatures of the stock oneDNN weight-only
// ops so the python M-split patch (vllm_gaudi/xpu/patches/skinny_m.py)
// swaps the call 1:1 at M <= 8:
//
//   skinny_fp8_gemm_w8a16(A, B, B_scale?, bias?)
//       checkpoint-fp8 per-tensor / per-channel scales (fp32)
//   skinny_fp8_gemm_w8a16_mxfp8(A, B, B_scale, out_dtype?, bias?)
//       online MXFP8 group-32 e8m0 weight scales
//   skinny_fp4_gemm_w4a16(A, B, B_scale, out_dtype?, bias?)
//       online MXFP4 packed e2m1 weights, group-32 e8m0 scales
//
// Layout resolution accepts the same tensors the stock ops receive:
// f8 lanes get [K, N] k-contiguous (a .t() view or its [N, K] C-order
// twin -- identical memory: element (k, n) at n * K + k), the f4 lane gets
// packed [N, K/2] row-major (byte (k, n) at n * (K/2) + (k >> 1), low
// nibble = even k). Group scales resolve in either [N, K/32] or [K/32, N]
// order via strides. Anything else TORCH_CHECK-fails loudly and the python
// wrapper falls back to the stock route.

#include <torch/all.h>
#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

#include "skinny_gemm_kernel.hpp"

namespace vllm::xpu::skinny {

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
      "skinny gemm bias must be a contiguous fp32 [N] tensor, got ",
      b.sizes(),
      " ",
      b.scalar_type());
  return b.const_data_ptr<float>();
}

// f8 weights: accept [K, N] Fortran (k-contiguous rows) or its [N, K]
// C-order twin -- both place element (k, n) at n * K + k.
int64_t resolve_f8_n(const torch::Tensor& b, int64_t k) {
  TORCH_CHECK(b.dim() == 2, "skinny gemm weight must be 2-D, got ", b.sizes());
  if (b.stride(0) == 1 && b.size(0) == k) {
    return b.size(1);
  }
  if (b.stride(1) == 1 && b.size(1) == k) {
    return b.size(0);
  }
  TORCH_CHECK(
      false,
      "skinny gemm weight must be k-contiguous [K, N] (or [N, K] twin), "
      "got sizes ",
      b.sizes(),
      " strides ",
      b.strides(),
      " for K=",
      k);
}

// Packed f4 weights: float4_e2m1fn_x2 (1 byte = 2 nibbles) as [K/2, N]
// Fortran (the w.t() view the stock op receives; byte (k, n) at
// n * (K/2) + (k >> 1)) or the [N, K/2] row-major twin. A raw uint8 tensor
// with the same packing is accepted too (kbench feedstock).
int64_t resolve_f4_n(const torch::Tensor& b, int64_t k) {
  TORCH_CHECK(
      b.scalar_type() == at::ScalarType::Float4_e2m1fn_x2 ||
          b.scalar_type() == at::ScalarType::Byte,
      "skinny f4 gemm weight must be float4_e2m1fn_x2 (or packed uint8), "
      "got ",
      b.scalar_type());
  TORCH_CHECK(
      b.dim() == 2 && (b.size(0) == k / 2 || b.size(1) == k / 2),
      "skinny f4 gemm weight must be [N, K/2] packed, got ",
      b.sizes(),
      " for K=",
      k);
  if (b.size(1) == k / 2 && b.stride(1) == 1) {
    return b.size(0);
  }
  if (b.size(0) == k / 2 && b.stride(0) == 1) {
    return b.size(1);
  }
  TORCH_CHECK(
      false,
      "skinny f4 gemm weight must have the packed K/2 axis contiguous, got "
      "sizes ",
      b.sizes(),
      " strides ",
      b.strides());
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
      "skinny gemm group scale must be f8_e8m0, got ", sc.scalar_type());
  if (sc.dim() == 2 && sc.size(0) == n && sc.size(1) == kg) {
    return ScaleStrides{sc.stride(0), sc.stride(1)};
  }
  if (sc.dim() == 2 && sc.size(0) == kg && sc.size(1) == n) {
    return ScaleStrides{sc.stride(1), sc.stride(0)};
  }
  TORCH_CHECK(
      false,
      "skinny gemm group scale must be [N, K/32] or [K/32, N], got ",
      sc.sizes(),
      " for N=",
      n,
      " K/32=",
      kg);
}

// Per-tensor / per-channel fp32 scales: nullptr + scalar for per-tensor,
// column pointer + stride for per-channel.
struct EpilogueScale {
  const float* per_channel;  // nullptr => per-tensor
  int64_t n_stride;
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
      "skinny gemm checkpoint scale must be fp32, got ", s.scalar_type());
  if (s.numel() == 1) {
    return EpilogueScale{nullptr, 0, s.item<float>()};
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
      "skinny gemm checkpoint scale must be scalar or [1, N], got ",
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
      "skinny gemm requires contiguous activations, got strides ",
      a.strides());
  TORCH_CHECK(
      a.scalar_type() == at::ScalarType::BFloat16 ||
          a.scalar_type() == at::ScalarType::Half,
      "skinny gemm requires bf16/f16 activations, got ", a.scalar_type());
  // the kernel issues aligned 128-bit activation loads
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(a.const_data_ptr()) % 16 == 0,
      "skinny gemm activation pointer must be 16-byte aligned");
}

void check_weights_aligned16(const torch::Tensor& b) {
  // 128-bit weight loads (r126 v2): K is a multiple of 32 so every column
  // base is 16-byte aligned iff the storage is; a misaligned view falls
  // back to the stock route via the python wrapper's try/except.
  TORCH_CHECK(
      reinterpret_cast<std::uintptr_t>(b.const_data_ptr()) % 16 == 0,
      "skinny gemm weight pointer must be 16-byte aligned (k-contiguous "
      "layout with odd storage offset -- call .contiguous())");
}

// The SYCL queue of the current torch XPU stream -- the same acquisition
// pattern the stock onednn headers use (c10::xpu::getCurrentXPUStream).
// torch 2.13's XPUStream::queue() yields the sycl queue directly (not a
// pointer); sycl::queue is a shared handle, so the copy submits to the
// same underlying in-order queue and the skinny kernels stay ordered
// with torch work on this stream.
sycl::queue current_sycl_queue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

void check_m_small(int64_t m) {
  TORCH_CHECK(
      m >= 1 && m <= 8,
      "skinny gemm is an M <= 8 decode-path kernel, got M=", m,
      " (the python route must gate on M)");
}

}  // namespace

// ---------------------------------------------------------------------------
// Checkpoint-fp8 lane (w16pt / w16pc champion classes).
// ---------------------------------------------------------------------------
torch::Tensor skinny_fp8_gemm_w8a16(
    const torch::Tensor& A,        // [.., M, K] bf16/f16
    const torch::Tensor& B,        // f8 weights, k-contiguous (see above)
    const std::optional<torch::Tensor>& B_scale,  // fp32 scalar or [1, N]
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  check_activation(A);
  const int64_t k = A.size(-1);
  const int64_t m = flatten_m(A);
  check_m_small(m);
  TORCH_CHECK(k % 32 == 0, "skinny gemm requires K % 32 == 0, got K=", k);
  const int64_t n = resolve_f8_n(B, k);
  check_weights_aligned16(B);
  const auto wf = B.scalar_type() == at::ScalarType::Float8_e5m2
                      ? WeightFormat::kF8E5m2
                      : WeightFormat::kF8E4m3;
  TORCH_CHECK(
      B.scalar_type() == at::ScalarType::Float8_e4m3fn ||
          B.scalar_type() == at::ScalarType::Float8_e5m2,
      "skinny w8a16 weight must be f8_e4m3fn or f8_e5m2, got ",
      B.scalar_type());

  const auto esc = resolve_epilogue_scale(B_scale, n);
  TORCH_CHECK(
      esc.per_channel == nullptr || esc.n_stride == 1,
      "skinny w8a16 per-channel scale must be contiguous along N");
  const float* bias_p = bias_ptr_or_null(bias, n);

  torch::Tensor result = make_output(A, n, A.scalar_type());
  auto queue = current_sycl_queue();
  const int m_i = static_cast<int>(m), n_i = static_cast<int>(n),
            k_i = static_cast<int>(k);
  const auto cfg = skinny_pick_cfg(m_i, n_i, k_i);
  torch::Tensor partials;
  float* partials_p = nullptr;
  if (!cfg.direct) {
    partials = at::empty(
        {cfg.splits, 8, n}, A.options().dtype(at::kFloat));
    partials_p = partials.data_ptr<float>();
  }
  skinny_launch(
      queue, wf, /*group32=*/false,
      /*a_bf16=*/A.scalar_type() == at::ScalarType::BFloat16,
      A.const_data_ptr(), reinterpret_cast<const std::uint8_t*>(B.const_data_ptr()),
      /*bsc=*/nullptr, partials_p, result.data_ptr(), bias_p,
      esc.per_channel, esc.per_tensor,
      /*out_bf16=*/A.scalar_type() == at::ScalarType::BFloat16, m_i, n_i, k_i);
  return result;
}

// ---------------------------------------------------------------------------
// Online MXFP8 lane (group-32 e8m0 weight scales).
// ---------------------------------------------------------------------------
torch::Tensor skinny_fp8_gemm_w8a16_mxfp8(
    const torch::Tensor& A,        // [.., M, K] bf16/f16
    const torch::Tensor& B,        // f8 weights, k-contiguous
    const torch::Tensor& B_scale,  // e8m0 [N, K/32] or [K/32, N]
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  check_activation(A);
  const int64_t k = A.size(-1);
  const int64_t m = flatten_m(A);
  check_m_small(m);
  TORCH_CHECK(k % 32 == 0, "skinny gemm requires K % 32 == 0, got K=", k);
  const int64_t n = resolve_f8_n(B, k);
  check_weights_aligned16(B);
  const auto wf = B.scalar_type() == at::ScalarType::Float8_e5m2
                      ? WeightFormat::kF8E5m2
                      : WeightFormat::kF8E4m3;
  TORCH_CHECK(
      B.scalar_type() == at::ScalarType::Float8_e4m3fn ||
          B.scalar_type() == at::ScalarType::Float8_e5m2,
      "skinny mxfp8 weight must be f8_e4m3fn or f8_e5m2, got ",
      B.scalar_type());

  const auto gs = resolve_group_strides(B_scale, n, k / 32);
  const float* bias_p = bias_ptr_or_null(bias, n);
  const c10::ScalarType odt = out_dtype.value_or(A.scalar_type());
  TORCH_CHECK(
      valid_epilogue_dtype(odt),
      "skinny mxfp8 out_dtype must be bf16/f16, got ", odt);

  torch::Tensor result = make_output(A, n, odt);
  auto queue = current_sycl_queue();
  const int m_i = static_cast<int>(m), n_i = static_cast<int>(n),
            k_i = static_cast<int>(k);
  const auto cfg = skinny_pick_cfg(m_i, n_i, k_i);
  torch::Tensor partials;
  float* partials_p = nullptr;
  if (!cfg.direct) {
    partials = at::empty(
        {cfg.splits, 8, n}, A.options().dtype(at::kFloat));
    partials_p = partials.data_ptr<float>();
  }
  skinny_launch(
      queue, wf, /*group32=*/true,
      /*a_bf16=*/A.scalar_type() == at::ScalarType::BFloat16,
      A.const_data_ptr(),
      reinterpret_cast<const std::uint8_t*>(B.const_data_ptr()),
      reinterpret_cast<const std::uint8_t*>(B_scale.const_data_ptr()),
      partials_p, result.data_ptr(), bias_p,
      /*pc_scale=*/nullptr, /*pt_scale=*/1.0f,
      /*out_bf16=*/odt == at::ScalarType::BFloat16, m_i, n_i, k_i, gs.n_stride,
      gs.g_stride);
  return result;
}

// ---------------------------------------------------------------------------
// Online MXFP4 lane (packed e2m1 nibbles, group-32 e8m0 scales).
// ---------------------------------------------------------------------------
torch::Tensor skinny_fp4_gemm_w4a16(
    const torch::Tensor& A,        // [.., M, K] bf16/f16
    const torch::Tensor& B,        // packed uint8 [N, K/2]
    const torch::Tensor& B_scale,  // e8m0 [N, K/32] or [K/32, N]
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  check_activation(A);
  const int64_t k = A.size(-1);
  const int64_t m = flatten_m(A);
  check_m_small(m);
  TORCH_CHECK(k % 32 == 0, "skinny gemm requires K % 32 == 0, got K=", k);
  const int64_t n = resolve_f4_n(B, k);
  check_weights_aligned16(B);

  const auto gs = resolve_group_strides(B_scale, n, k / 32);
  const float* bias_p = bias_ptr_or_null(bias, n);
  const c10::ScalarType odt = out_dtype.value_or(A.scalar_type());
  TORCH_CHECK(
      valid_epilogue_dtype(odt),
      "skinny mxfp4 out_dtype must be bf16/f16, got ", odt);

  torch::Tensor result = make_output(A, n, odt);
  auto queue = current_sycl_queue();
  const int m_i = static_cast<int>(m), n_i = static_cast<int>(n),
            k_i = static_cast<int>(k);
  const auto cfg = skinny_pick_cfg(m_i, n_i, k_i);
  torch::Tensor partials;
  float* partials_p = nullptr;
  if (!cfg.direct) {
    partials = at::empty(
        {cfg.splits, 8, n}, A.options().dtype(at::kFloat));
    partials_p = partials.data_ptr<float>();
  }
  skinny_launch(
      queue, WeightFormat::kF4E2m1, /*group32=*/true,
      /*a_bf16=*/A.scalar_type() == at::ScalarType::BFloat16,
      A.const_data_ptr(),
      reinterpret_cast<const std::uint8_t*>(B.const_data_ptr()),
      reinterpret_cast<const std::uint8_t*>(B_scale.const_data_ptr()),
      partials_p, result.data_ptr(), bias_p,
      /*pc_scale=*/nullptr, /*pt_scale=*/1.0f,
      /*out_bf16=*/odt == at::ScalarType::BFloat16, m_i, n_i, k_i, gs.n_stride,
      gs.g_stride);
  return result;
}

}  // namespace vllm::xpu::skinny
