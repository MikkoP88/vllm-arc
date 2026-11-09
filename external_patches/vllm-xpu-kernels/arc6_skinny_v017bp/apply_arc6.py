#!/usr/bin/env python3
"""apply_arc6.py -- rig-side source patcher for the arc6 wheel section
(P6.33 skinny-M lanes + v0.1.17 Option-B backports). Applies to a COPY of
the live post-arc5 tree (/root/build/vxk-arc-src -> /root/build/vxk-arc6-src):

  1. Backports (upstream vllm-xpu-kernels PRs, csrc-only; tests/ hunks
     dropped -- the rig tree carries no tests/):
       #611  FP8 block GEMM K==block_k routing (fp8_gemm_w8a8.h,
             fp8_gemm_w8a16.h)
       #645  MXFP8 K==32 e8m0 -> block-scale routing (fp8_gemm_w8a8.h;
             CONTEXT DEPENDS ON #611 -- apply order is load-bearing)
       #650  per-tensor fp8 quant scale rounding (quantization/fp8/
             fp8_quant.h)
       #600  GDN ragged spec-token traversal (causal_conv1d.hpp,
             gated_delta_rule.hpp, gdn_attn_interface.cpp)
     Deferred to Phase 3 (rebase): #658/#660/#683 (GDN causal-conv
     offsets/NULL_BLOCK/DeepSeek layout) -- they refactor the xe_2
     chunk_causal_conv1d split that does not exist in our 0.1.14.1 base;
     porting the intermediate refactor chain is an unbounded cascade
     (master plan §P6.32 item 8 future-release policy: a port that fails
     clean anchoring defers to the rebase, it is not hand-carried).
  2. onednn_ext.h: W4A8 dispatch case arms for the two arc5 enum values
     (f8_e4m3_mxfp4_bf16/_f16) -- the arc5 mappers existed but had no case
     arms, so the vxk wrapper's own default: threw "Only support int4 and
     fp8 gemm" before oneDNN was ever asked (r123 mis-attribution,
     corrected §P6.32 item 3; the lane is OPEN, not oneDNN-refused).
  3. Copy csrc/xpu/skinny_gemm/ (kernel hpp + interface cpp): the
     skinny-M decode GEMM family (M <= 8, dequant-in-register, fixed-order
     split-K; see skinny_gemm_kernel.hpp header comment for the design
     and the r123 bytes-floor economics).
  4. ops.h: the three skinny op declarations.
  5. torch_bindings.cpp: the three skinny op registrations.
  6. CMakeLists.txt: one source-list line after "csrc/xpu/utils.cpp"
     (the _xpu_C target is an explicit VLLM_EXT_XPU_SRC list; new SYCL
     module = one appended entry).

Anchors are exact-match and the script REFUSES (exit 1) on any miss or
double-apply -- byte-exact review discipline (arc5 kit law).
"""
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("/root/build/vxk-arc-src")
DST = Path("/root/build/vxk-arc6-src")
KIT = Path("/root/build/arc6kit")  # this kit's payload ships here

# (patch file, marker file, marker string) -- apply order is load-bearing:
# 645's context contains 611's added lines.
BACKPORTS = [
    ("kit_611.patch", "csrc/xpu/onednn/fp8_gemm_w8a8.h", "single_k_block"),
    ("kit_645.patch", "csrc/xpu/onednn/fp8_gemm_w8a8.h", "mx_single_k_block"),
    (
        "kit_650.patch",
        "csrc/quantization/fp8/fp8_quant.h",
        "Device fp32 divide is not correctly rounded",
    ),
    (
        "kit_600.patch",
        "csrc/xpu/gdn_attn/causal_conv1d.hpp",
        "token_start = query_start_loc[batch_id]",
    ),
]


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


def apply_backports() -> None:
    for diff_name, marker_file, marker in BACKPORTS:
        diff = KIT / "patches" / diff_name
        if not diff.exists():
            sys.exit(f"FATAL: backport diff missing: {diff}")
        already = (DST / marker_file).read_text().find(marker) >= 0
        check = subprocess.run(
            ["git", "apply", "--check", str(diff)],
            cwd=DST,
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            if already:
                print(f"[skip] {diff_name}: already applied")
                continue
            sys.exit(f"FATAL: {diff_name} failed --check:\n{check.stderr}")
        if already:
            sys.exit(f"FATAL: {diff_name} marker present but --check passes")
        run = subprocess.run(
            ["git", "apply", str(diff)], cwd=DST, capture_output=True, text=True
        )
        if run.returncode != 0:
            sys.exit(f"FATAL: {diff_name} apply failed:\n{run.stderr}")
        print(f"[ok] backport {diff_name} applied")


def main() -> None:
    if DST.exists():
        sys.exit(f"FATAL: {DST} already exists -- remove it to re-apply")
    # .deps (onednn-src + build cache) MUST come -- warm-cache reuse is
    # the point; build/ stale objects are dead weight (inner script does
    # `rm -rf build` first thing anyway).
    shutil.copytree(
        SRC, DST, ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__")
    )
    print(f"[ok] tree copied {SRC} -> {DST} (build/ excluded)")

    apply_backports()

    # 2. onednn_ext.h: W4A8 dispatch arms (arc5 enum values gain cases)
    ext = DST / "csrc/xpu/onednn/onednn_ext.h"
    arm = """      return matmul_primitive_create_and_cache<joint_dtypes_t::{E}, F>(
          Tt,
          b_type,
          m,
          n,
          k,
          lda,
          ldb,
          ldc,
          device_id,
          attr,
          scale_group_size,
          zp_group_size,
          batch_dim);
"""
    patch_file(
        ext,
        [(
            '    default:\n'
            '      throw std::runtime_error("Only support int4 and fp8 gemm ...");',
            "    // arc6 (P6.33): W4A8 dispatch arms for the arc5 enum values\n"
            "    // (e4m3 activations, packed f4 weights). The mappers\n"
            "    // (onednn_types_mapper<f8_e4m3_mxfp4_*>) exist since arc5;\n"
            "    // without these arms the wrapper's own default: threw\n"
            "    // before oneDNN was ever asked (r123 mis-attribution,\n"
            "    // P6.32 item 3).\n"
            "    case joint_dtypes_t::f8_e4m3_mxfp4_bf16:\n"
            + arm.format(E="f8_e4m3_mxfp4_bf16")
            + "    case joint_dtypes_t::f8_e4m3_mxfp4_f16:\n"
            + arm.format(E="f8_e4m3_mxfp4_f16")
            + '    default:\n'
            '      throw std::runtime_error("Only support int4 and fp8 gemm ...");',
        )],
        "case joint_dtypes_t::f8_e4m3_mxfp4_bf16:",
    )

    # 3. skinny module sources
    skinny_dst = DST / "csrc/xpu/skinny_gemm"
    if skinny_dst.exists():
        sys.exit("FATAL: csrc/xpu/skinny_gemm already present")
    skinny_src = KIT / "csrc/xpu/skinny_gemm"
    if not (skinny_src / "skinny_gemm_interface.cpp").exists():
        sys.exit(f"FATAL: kit skinny sources missing under {skinny_src}")
    shutil.copytree(skinny_src, skinny_dst)
    print("[ok] skinny_gemm module copied")

    # 4. ops.h declarations (torch_bindings.cpp compiles against these --
    # arc5 v1 lesson: without them the bindings TU fails)
    ops = DST / "csrc/xpu/ops.h"
    patch_file(
        ops,
        [(
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\ntorch::Tensor int4_gemm_w4a16(",
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\n// arc6 (P6.33) skinny-M decode GEMM ops (M <= 8 route).\n"
            "// Same signatures/layouts as the stock ops they replace at\n"
            "// small M; see csrc/xpu/skinny_gemm/skinny_gemm_kernel.hpp.\n"
            "namespace vllm::xpu::skinny {\n"
            "torch::Tensor skinny_fp8_gemm_w8a16(\n"
            "    const torch::Tensor& A,\n"
            "    const torch::Tensor& B,\n"
            "    const std::optional<torch::Tensor>& B_scale,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\ntorch::Tensor skinny_fp8_gemm_w8a16_mxfp8(\n"
            "    const torch::Tensor& A,\n"
            "    const torch::Tensor& B,\n"
            "    const torch::Tensor& B_scale,\n"
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\ntorch::Tensor skinny_fp4_gemm_w4a16(\n"
            "    const torch::Tensor& A,\n"
            "    const torch::Tensor& B,\n"
            "    const torch::Tensor& B_scale,\n"
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "}  // namespace vllm::xpu::skinny\n"
            "\ntorch::Tensor int4_gemm_w4a16(",
        )],
        "skinny_fp8_gemm_w8a16",
    )

    # 5. torch_bindings.cpp registrations
    tb = DST / "csrc/xpu/torch_bindings.cpp"
    anchor = 'xpu_ops.impl("fp4_gemm_w4a8", torch::kXPU, &fp4_gemm_w4a8);'
    text = tb.read_text()
    if "skinny_fp8_gemm_w8a16" in text:
        print("[skip] torch_bindings.cpp already patched")
    else:
        if text.count(anchor) != 1:
            sys.exit(
                f"FATAL: bindings anchor not unique "
                f"(count={text.count(anchor)})"
            )
        reg = anchor + """

  // arc6 (P6.33) skinny-M decode GEMM ops (M <= 8 route)
  xpu_ops.def(
      "skinny_fp8_gemm_w8a16(Tensor A, Tensor B, Tensor? B_scale, "
      "Tensor? bias_) -> Tensor");
  xpu_ops.impl(
      "skinny_fp8_gemm_w8a16",
      torch::kXPU,
      &vllm::xpu::skinny::skinny_fp8_gemm_w8a16);

  xpu_ops.def(
      "skinny_fp8_gemm_w8a16_mxfp8(Tensor A, Tensor B, Tensor B_scale, "
      "ScalarType? out_dtype, Tensor? bias_) -> Tensor");
  xpu_ops.impl(
      "skinny_fp8_gemm_w8a16_mxfp8",
      torch::kXPU,
      &vllm::xpu::skinny::skinny_fp8_gemm_w8a16_mxfp8);

  xpu_ops.def(
      "skinny_fp4_gemm_w4a16(Tensor A, Tensor B, Tensor B_scale, "
      "ScalarType? out_dtype, Tensor? bias_) -> Tensor");
  xpu_ops.impl(
      "skinny_fp4_gemm_w4a16",
      torch::kXPU,
      &vllm::xpu::skinny::skinny_fp4_gemm_w4a16);"""
        text = text.replace(anchor, reg)
        tb.write_text(text)
        print("[ok] torch_bindings.cpp patched")

    # 6. CMakeLists.txt: the _xpu_C target is an explicit source list; one
    # line after utils.cpp adds the skinny module TU.
    cm = DST / "CMakeLists.txt"
    patch_file(
        cm,
        [(
            '      "csrc/xpu/utils.cpp")',
            '      "csrc/xpu/utils.cpp"\n'
            '      "csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp")',
        )],
        "skinny_gemm_interface.cpp",
    )

    print("apply_arc6: DONE")


if __name__ == "__main__":
    main()
