#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

// MXFP4 W4A8 matmul: e4m3 activations with per-token f32 scales [m,1]
// (cheap SRC attr, group {1,k}), packed f4_e2m1 weights with group-32
// e8m0 scales [k/32,n] on the weight-decomposition path (cheap WEIGHTS
// attr, group {32,1}). Motivation (p630 residual ledger): w4a16's
// bf16-src GEMM cell is the fastest at every decode M, yet e2e solo
// decode sits at 74% of the fp8 checkpoint champion -- the delta is
// per-step overhead; the candidate escalation is e4m3 activations,
// which sit in the gemmstone counter-operand list natively (f4 does
// NOT, which is why stock w4a4 lands matmul:ref). Whether oneDNN
// v3.13's md validation accepts the (e4m3, f4_e2m1) pair at all is
// exactly what this op probes -- a create-time throw is the clean
// closed-negative answer.
static inline void dnnl_matmul_w4a8_fp4(
    torch::Tensor& result,       // dst, [b, m, n] bf16/f16
    const torch::Tensor& mat1,   // src, [b, m, k] f8_e4m3
    const torch::Tensor& m1_sc,  // src scales, [m,1] f32 (per token)
    const torch::Tensor& mat2,   // packed f4 weight, [k, n] transpose
    bool is_nt,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m2_sc) {  // wei scales, [k/32,n] e8m0
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  joint_dtypes_t jd;
  auto out_st = result.scalar_type();
  if (out_st == at::ScalarType::BFloat16) {
    jd = joint_dtypes_t::f8_e4m3_mxfp4_bf16;
  } else if (out_st == at::ScalarType::Half) {
    jd = joint_dtypes_t::f8_e4m3_mxfp4_f16;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported output dtype for mxfp4 w4a8 matmul: ", out_st);
  }
  TORCH_CHECK(
      mat1.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "mxfp4 w4a8 expects e4m3 activations, got ", mat1.scalar_type());
  TORCH_CHECK(
      mat2.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
      "mxfp4 w4a8 expects packed f4 weights, got ", mat2.scalar_type());
  TORCH_CHECK(
      m1_sc.scalar_type() == at::ScalarType::Float,
      "mxfp4 w4a8 src scales must be f32 per-token [m,1], got ",
      m1_sc.sizes(), " ", m1_sc.scalar_type());
  TORCH_CHECK(
      m1_sc.dim() == 2 && m1_sc.size(0) == m && m1_sc.size(1) == 1,
      "mxfp4 w4a8 src scales must be [m,1], got ", m1_sc.sizes());
  TORCH_CHECK(
      m2_sc.scalar_type() == at::ScalarType::Float8_e8m0fnu,
      "mxfp4 w4a8 wei scales must be e8m0 [k/32,n], got ",
      m2_sc.scalar_type());
  TORCH_CHECK(
      m2_sc.dim() == 2 && m2_sc.size(0) == k / 32 && m2_sc.size(1) == n,
      "mxfp4 w4a8 wei scales must be [k/32,n], got ", m2_sc.sizes());

  // get bias type
  bias_type_t b_type = get_bias_type(bias, m, n);

  trans_type_t tt = trans_type_t::nn;
  if (is_nt) {
    // transpose mat2
    tt = trans_type_t::nt;
  }

  // get lda ldb and ldc (f4 weights are packed: ldb doubles, matching
  // fp4_gemm_w4a4/w4a16)
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false, "Unsupported input dimension for mxfp4 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : 2 * (mat2.strides()[mat2.dim() - 1]);
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    // SRC: per-token f32 scales (cheap attr class)
    pattr.set_scales(
        DNNL_ARG_SRC,
        /* mask */ (1 << 0) + (1 << 1),
        {1, k},
        get_onednn_dtype(m1_sc));
    // WEIGHTS: group-32 e8m0 along K (the proven-cheap attr)
    pattr.set_scales(
        DNNL_ARG_WEIGHTS,
        /* mask */ (1 << 0) + (1 << 1),
        {32, 1},
        get_onednn_dtype(m2_sc));
  };

  int arg_off = 0;

  // ************************************************************
  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

  int m1_sc_group_size = m1_sc.numel();
  int m2_sc_group_size = m2_sc.numel();
  int sc_group_size = (m1_sc_group_size << 8) | m2_sc_group_size;
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, dev_id, f_attr, sc_group_size);

  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      m2_sc.data_ptr(),
      [&]() {
        return make_onednn_memory(
            get_onednn_md(m2_sc), engine, m2_sc.data_ptr());
      });
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
        return make_onednn_memory(
            get_onednn_md(m1_sc), engine, m1_sc.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);

  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (get_shape(b_type) != bias_shape_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }
  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  auto qfp4_matmul_event =
      matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
}  // namespace oneDNN
