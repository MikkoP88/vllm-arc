#!/usr/bin/env python3
"""apply_arc5.py -- rig-side source patcher for the arc5 wheel section
(P6.31 FIX-Q lanes). Applies the arc5 additions to a COPY of the arc4
source tree (/root/build/vxk-arc-src -> /root/build/vxk-arc5-src):

  1. onednn_ext.h: append joint enum values f8_e4m3_mxfp4_bf16/_f16 +
     their type mappers (after the bf16_mxfp4/f16_mxfp4 block).
  2. ops.h: append the two op declarations (torch_bindings.cpp
     compiles against these).
  3. onednn_matmul.cpp: include the two arc5 headers + append the
     two entry points (fp8_gemm_w8a16_mxfp8, fp4_gemm_w4a8).
  4. torch_bindings.cpp: append the two op registrations.
  5. Copy the two headers into csrc/xpu/onednn/.

W8A8-grpw was DESIGNED then DROPPED before build: r122 kbench measured
every f8xf8 fp8_gemm scale flavor at 2.05-2.64x slower than bf16 at
prefill Ms (pt 48.10 / tok 48.20 / mx8 60.74 vs bf16 22.98 ms,
qwen.up M=32768) while (bf16,f8) w16pc runs 1.03x bf16 -- the f8xf8
joint rate, not the scale flavor, is the wall, so grpw is strictly
dominated by w8a16. Kept here as the provenance note (r122 evidence:
/root/build/p631/kbench/r122_kbench.out).

Anchors are exact-match and the script REFUSES (exit 1) on any miss or
double-apply -- byte-exact review discipline.
"""
import shutil
import sys
from pathlib import Path

SRC = Path("/root/build/vxk-arc-src")
DST = Path("/root/build/vxk-arc5-src")
KIT = Path("/root/build/arc5kit")  # headers shipped here


def patch_file(path: Path, edits, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"[skip] {path.name}: already patched (marker present)")
        return
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(
                f"FATAL: anchor not unique in {path.name}: "
                f"{old[:60]!r} (count={text.count(old)})"
            )
        text = text.replace(old, new)
    path.write_text(text)
    print(f"[ok] {path.name} patched")


def main() -> None:
    if DST.exists():
        sys.exit(f"FATAL: {DST} already exists -- remove it to re-apply")
    # .deps (onednn-src + build cache) MUST come -- warm-cache reuse is
    # the point; build/ stale objects are dead weight (inner script does
    # `rm -rf build` first thing anyway).
    shutil.copytree(
        SRC, DST, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"))
    print(f"[ok] tree copied {SRC} -> {DST} (build/ excluded)")

    ext = DST / "csrc/xpu/onednn/onednn_ext.h"
    # 1. enum append (after the arc4 f16_mxfp4 value closes the enum)
    patch_file(
        ext,
        [(
            "  bf16_mxfp4,\n  f16_mxfp4,\n};",
            "  bf16_mxfp4,\n  f16_mxfp4,\n"
            "  // arc5 (P6.31 FIX-Q): e4m3 activations on packed f4 "
            "weights (W4A8)\n"
            "  f8_e4m3_mxfp4_bf16,\n"
            "  f8_e4m3_mxfp4_f16,\n};",
        )],
        "f8_e4m3_mxfp4_bf16",
    )

    # 2. mappers append -- anchor on the f16_mxfp4 mapper tail is
    # fragile; instead append before the closing of the mappers block:
    # anchor on the mapper namespace end is not uniquely identifiable,
    # so anchor on the LAST mapper struct (f16_mxfp4) by its content.
    text = ext.read_text()
    anchor = (
        "struct onednn_types_mapper<joint_dtypes_t::f16_mxfp4> {"
    )
    if text.count(anchor) != 1:
        sys.exit("FATAL: f16_mxfp4 mapper anchor not unique")
    # find the end of that struct: the next "};" after the anchor
    i = text.index(anchor)
    j = text.index("};", i) + 2
    mappers = (
        "\n\n// arc5 (P6.31 FIX-Q): e4m3 activations, packed f4 weights "
        "(W4A8).\n"
        "template <>\n"
        "struct onednn_types_mapper<joint_dtypes_t::f8_e4m3_mxfp4_bf16> {\n"
        "  static inline std::\n"
        "      tuple<memory::data_type, memory::data_type, "
        "memory::data_type>\n"
        "      get() {\n"
        "    return std::make_tuple(\n"
        "        memory::data_type::f8_e4m3,\n"
        "        memory::data_type::f4_e2m1,\n"
        "        memory::data_type::bf16);\n"
        "  }\n"
        "};\n"
        "\ntemplate <>\n"
        "struct onednn_types_mapper<joint_dtypes_t::f8_e4m3_mxfp4_f16> {\n"
        "  static inline std::\n"
        "      tuple<memory::data_type, memory::data_type, "
        "memory::data_type>\n"
        "      get() {\n"
        "    return std::make_tuple(\n"
        "        memory::data_type::f8_e4m3,\n"
        "        memory::data_type::f4_e2m1,\n"
        "        memory::data_type::f16);\n"
        "  }\n"
        "};\n"
    )
    if "joint_dtypes_t::f8_e4m3_mxfp4_bf16> {" in text:
        sys.exit("FATAL: mappers already present (double-apply)")
    text = text[:j] + mappers + text[j:]
    ext.write_text(text)
    print("[ok] onednn_ext.h mappers appended")

    # 3. copy headers
    for h in (
        "fp8_gemm_w8a16_mxfp8.h",
        "fp4_gemm_w4a8.h",
    ):
        src_f = KIT / h
        dst_f = DST / "csrc/xpu/onednn" / h
        if not src_f.exists():
            sys.exit(f"FATAL: kit header missing: {src_f}")
        shutil.copy2(src_f, dst_f)
        print(f"[ok] header copied {h}")

    # 3.5 ops.h: torch_bindings.cpp compiles against THESE declarations
    # (csrc/xpu/ops.h) — without them the bindings TU fails with "use of
    # undeclared identifier" (the 08:39 UTC first-build failure; the arc4
    # kit patched this file, arc5 v1 missed it).
    ops = DST / "csrc/xpu/ops.h"
    patch_file(
        ops,
        [(
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\ntorch::Tensor int4_gemm_w4a16(",
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\n// arc5 (P6.31 FIX-Q) lane ops\n"
            "torch::Tensor fp8_gemm_w8a16_mxfp8(\n"
            "    const torch::Tensor& A,\n"
            "    const torch::Tensor& B,\n"
            "    const torch::Tensor& B_scale,\n"
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\ntorch::Tensor fp4_gemm_w4a8(\n"
            "    const torch::Tensor& A,\n"
            "    const torch::Tensor& A_scale,\n"
            "    const torch::Tensor& B,\n"
            "    const torch::Tensor& B_scale,\n"
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\ntorch::Tensor int4_gemm_w4a16(",
        )],
        "fp8_gemm_w8a16_mxfp8",
    )

    # 4. onednn_matmul.cpp: includes + entry points
    mm = DST / "csrc/xpu/onednn/onednn_matmul.cpp"
    patch_file(
        mm,
        [
            (
                '#include "fp4_gemm_w4a16.h"',
                '#include "fp4_gemm_w4a16.h"\n'
                '#include "fp8_gemm_w8a16_mxfp8.h"\n'
                '#include "fp4_gemm_w4a8.h"',
            ),
            (
                "torch::Tensor int4_gemm_w4a16(",
                """// arc5 (P6.31 FIX-Q) ------------------------------------------------
// MXFP8 weight-only: bf16/f16 activations straight in, group-32 e8m0
// weight scales [k/32, n]. Skips the per-call mxfp8 activation quant.
torch::Tensor fp8_gemm_w8a16_mxfp8(
    const torch::Tensor& A,        // src, [b, m, k] bf16/f16
    const torch::Tensor& B,        // quantized weight, [k, n] transpose
    const torch::Tensor& B_scale,  // [k/32, n] f8_e8m0
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  TORCH_CHECK(
      B_scale.is_contiguous(),
      "B_scale must be contiguous for mxfp8 w8a16 matmul");
  TORCH_CHECK(
      !bias.has_value() || bias.value().is_contiguous(),
      "bias must be contiguous for mxfp8 w8a16 matmul");
  torch::Tensor result = check_and_create_output_tensor(
      A, B, out_dtype.value_or(A.scalar_type()));
  TORCH_CHECK(
      is_supported_fp8(B.scalar_type()),
      "weight must be f8_e5m2 or f8_e4m3fn for mxfp8 w8a16 matmul");
  TORCH_CHECK(
      A.scalar_type() == at::ScalarType::BFloat16 ||
          A.scalar_type() == at::ScalarType::Half,
      "input must be bfloat16 or float16 for mxfp8 w8a16 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  oneDNN::dnnl_matmul_w8a16_mxfp8(result, A, B, is_nt, bias, B_scale);
  return result;
}

// MXFP4 W4A8: e4m3 activations with per-token f32 scales, packed f4
// weights with group-32 e8m0 scales (weight-decomposition path).
torch::Tensor fp4_gemm_w4a8(
    const torch::Tensor& A,        // src, [b, m, k] f8_e4m3
    const torch::Tensor& A_scale,  // [m, 1] f32 (per token)
    const torch::Tensor& B,        // packed f4 weight, [k, n] transpose
    const torch::Tensor& B_scale,  // [k/32, n] f8_e8m0
    std::optional<c10::ScalarType> out_dtype,
    const std::optional<torch::Tensor>& bias) {
  const at::DeviceGuard device_guard(A.device());
  TORCH_CHECK(
      A_scale.is_contiguous() && B_scale.is_contiguous(),
      "scales must be contiguous for mxfp4 w4a8 matmul");
  TORCH_CHECK(
      !bias.has_value() || bias.value().is_contiguous(),
      "bias must be contiguous for mxfp4 w4a8 matmul");
  torch::Tensor result =
      check_and_create_output_tensor(A, B, out_dtype.value_or(torch::kBFloat16));
  TORCH_CHECK(
      A.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "input must be f8_e4m3fn for mxfp4 w4a8 matmul");
  TORCH_CHECK(
      B.scalar_type() == at::ScalarType::Float4_e2m1fn_x2,
      "weight must be f4_e2m1x2 for mxfp4 w4a8 matmul");
  // check if nt format
  bool is_nt = B.strides()[B.dim() - 2] == 1;

  oneDNN::dnnl_matmul_w4a8_fp4(
      result, A, A_scale, B, is_nt, bias, B_scale);
  return result;
}

torch::Tensor int4_gemm_w4a16(""",
            ),
        ],
        "fp8_gemm_w8a16_mxfp8",
    )

    # 5. torch_bindings.cpp: registrations
    tb = DST / "csrc/xpu/torch_bindings.cpp"
    text = tb.read_text()
    if "fp8_gemm_w8a16_mxfp8" in text:
        print("[skip] torch_bindings.cpp already patched")
    else:
        # anchor: the arc4 w4a16 registration block end -- find the
        # impl line for fp4_gemm_w4a16 and insert AFTER its statement
        anchor = 'xpu_ops.impl("fp4_gemm_w4a16", torch::kXPU, &fp4_gemm_w4a16);'
        if text.count(anchor) != 1:
            sys.exit(
                f"FATAL: bindings anchor not unique "
                f"(count={text.count(anchor)})"
            )
        reg = anchor + """

  // arc5 (P6.31 FIX-Q) mxfp8/mxfp4 lane ops
  xpu_ops.def(
      "fp8_gemm_w8a16_mxfp8(Tensor A, Tensor B, Tensor B_scale, "
      "ScalarType? out_dtype, Tensor? bias_) -> Tensor");
  xpu_ops.impl(
      "fp8_gemm_w8a16_mxfp8", torch::kXPU, &fp8_gemm_w8a16_mxfp8);

  xpu_ops.def(
      "fp4_gemm_w4a8(Tensor A, Tensor A_scale, Tensor B, "
      "Tensor B_scale, ScalarType? out_dtype, Tensor? bias_) -> Tensor");
  xpu_ops.impl("fp4_gemm_w4a8", torch::kXPU, &fp4_gemm_w4a8);"""
        text = text.replace(anchor, reg)
        tb.write_text(text)
        print("[ok] torch_bindings.cpp patched")

    print("apply_arc5: DONE")


if __name__ == "__main__":
    main()
