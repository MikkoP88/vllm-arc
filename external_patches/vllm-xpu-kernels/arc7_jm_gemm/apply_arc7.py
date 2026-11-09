#!/usr/bin/env python3
"""apply_arc7.py -- rig-side source patcher for the arc7 wheel section
(P6.34 JM/DPAS skinny family). Applies to the applied-arc6 tree at
/root/build/vxk-arc-src (IN-PLACE by default -- the builder bind
references the directory inode, see below; --copy for a sibling tree):

  1. Copy csrc/xpu/jm_gemm/ (kernel hpp + interface cpp): the
     joint-matrix (tensor-core) decode GEMM family -- dpas.hf.hf.8.8
     register fragments, packed pair decoders, MX group-flush scaling,
     formula split-K; see jm_gemm_kernel.hpp header for the design and
     the r128a bar economics. Additive ops: stock + arc6 skinny + JM all
     coexist in one wheel (probe A/B in-process).
  2. ops.h: the three jm op declarations (after the skinny block).
  3. torch_bindings.cpp: the three jm op registrations.
  4. CMakeLists.txt: one source-list line (the _xpu_C target is an
     explicit VLLM_EXT_XPU_SRC list).

Anchors are exact-match and the script REFUSES (exit 1) on any miss or
double-apply -- byte-exact review discipline (arc5/arc6 kit law).
"""
import shutil
import subprocess
import sys
from pathlib import Path

# In-place mode (the certified builder bind /root/build/vxk-arc-src ->
# /src/vllm-xpu-kernels references the directory INODE: replacing the
# path with a new tree would leave the container seeing the old/deleted
# one. Edits therefore go directly into the live tree, AFTER a tar
# archive snapshot for rollback.) Copy mode (--copy) builds a sibling
# tree instead, for reference builds.
TREE = Path("/root/build/vxk-arc-src")
ARCHIVE = Path("/root/build/vxk-arc6-cert22-src.tgz")
KIT = Path("/root/build/arc7kit")  # this kit's payload ships here


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
    copy_mode = "--copy" in sys.argv
    dst = TREE.parent / "vxk-arc7-src" if copy_mode else TREE
    if copy_mode:
        if dst.exists():
            sys.exit(f"FATAL: {dst} already exists -- remove it to re-apply")
    if not (TREE / "csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp").exists():
        sys.exit(f"FATAL: {TREE} is not the applied-arc6 tree")
    if not copy_mode and not ARCHIVE.exists():
        subprocess.run(
            ["tar", "czf", str(ARCHIVE),
             "--exclude=build", "--exclude=*.egg-info",
             "--exclude=__pycache__",
             "-C", str(TREE.parent), TREE.name],
            check=True)
        print(f"[ok] rollback archive {ARCHIVE}")
    if copy_mode:
        shutil.copytree(
            TREE, dst,
            ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"))
        print(f"[ok] tree copied {TREE} -> {dst} (build/ excluded)")

    # 1. jm module sources
    jm_dst = dst / "csrc/xpu/jm_gemm"
    if jm_dst.exists():
        sys.exit("FATAL: csrc/xpu/jm_gemm already present")
    jm_src = KIT / "csrc/xpu/jm_gemm"
    if not (jm_src / "jm_gemm_interface.cpp").exists():
        sys.exit(f"FATAL: kit jm sources missing under {jm_src}")
    shutil.copytree(jm_src, jm_dst)
    print("[ok] jm_gemm module copied")

    # 2. ops.h declarations (anchored after the arc6 skinny namespace
    # block; torch_bindings.cpp compiles against these)
    ops = dst / "csrc/xpu/ops.h"
    patch_file(
        ops,
        [(
            "}  // namespace vllm::xpu::skinny\n",
            "}  // namespace vllm::xpu::skinny\n"
            "\n"
            "// arc7 (P6.34) JM (joint-matrix/DPAS) skinny-M decode GEMM\n"
            "// ops. Same signatures/layouts as the arc6 skinny ops (and the\n"
            "// stock ops they replace at small M); the math runs on the Xe2\n"
            "// tensor cores. See csrc/xpu/jm_gemm/jm_gemm_kernel.hpp.\n"
            "namespace vllm::xpu::jm {\n"
            "torch::Tensor jm_fp8_gemm_w8a16(\n"
            "    const torch::Tensor& A,\n"
            "    const torch::Tensor& B,\n"
            "    const std::optional<torch::Tensor>& B_scale,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\n"
            "torch::Tensor jm_fp8_gemm_w8a16_mxfp8(\n"
            "    const torch::Tensor& A,\n"
            "    const torch::Tensor& B,\n"
            "    const torch::Tensor& B_scale,\n"
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "\n"
            "torch::Tensor jm_fp4_gemm_w4a16(\n"
            "    const torch::Tensor& A,\n"
            "    const torch::Tensor& B,\n"
            "    const torch::Tensor& B_scale,\n"
            "    std::optional<c10::ScalarType> out_dtype,\n"
            "    const std::optional<torch::Tensor>& bias);\n"
            "}  // namespace vllm::xpu::jm\n",
        )],
        "jm_fp8_gemm_w8a16",
    )

    # 3. torch_bindings.cpp registrations
    tb = dst / "csrc/xpu/torch_bindings.cpp"
    anchor = (
        'xpu_ops.impl(\n'
        '      "skinny_fp4_gemm_w4a16",\n'
        '      torch::kXPU,\n'
        '      &vllm::xpu::skinny::skinny_fp4_gemm_w4a16);'
    )
    text = tb.read_text()
    if "jm_fp8_gemm_w8a16" in text:
        print("[skip] torch_bindings.cpp already patched")
    else:
        if text.count(anchor) != 1:
            sys.exit(
                f"FATAL: bindings anchor not unique "
                f"(count={text.count(anchor)})"
            )
        reg = anchor + """

  // arc7 (P6.34) JM (joint-matrix/DPAS) skinny-M decode GEMM ops
  xpu_ops.def(
      "jm_fp8_gemm_w8a16(Tensor A, Tensor B, Tensor? B_scale, "
      "Tensor? bias_) -> Tensor");
  xpu_ops.impl(
      "jm_fp8_gemm_w8a16",
      torch::kXPU,
      &vllm::xpu::jm::jm_fp8_gemm_w8a16);

  xpu_ops.def(
      "jm_fp8_gemm_w8a16_mxfp8(Tensor A, Tensor B, Tensor B_scale, "
      "ScalarType? out_dtype, Tensor? bias_) -> Tensor");
  xpu_ops.impl(
      "jm_fp8_gemm_w8a16_mxfp8",
      torch::kXPU,
      &vllm::xpu::jm::jm_fp8_gemm_w8a16_mxfp8);

  xpu_ops.def(
      "jm_fp4_gemm_w4a16(Tensor A, Tensor B, Tensor B_scale, "
      "ScalarType? out_dtype, Tensor? bias_) -> Tensor");
  xpu_ops.impl(
      "jm_fp4_gemm_w4a16",
      torch::kXPU,
      &vllm::xpu::jm::jm_fp4_gemm_w4a16);"""
        text = text.replace(anchor, reg)
        tb.write_text(text)
        print("[ok] torch_bindings.cpp patched")

    # 4. CMakeLists.txt: one appended source-list entry
    cm = dst / "CMakeLists.txt"
    patch_file(
        cm,
        [(
            '      "csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp")',
            '      "csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp"\n'
            '      "csrc/xpu/jm_gemm/jm_gemm_interface.cpp")',
        )],
        "jm_gemm_interface.cpp",
    )

    print("apply_arc7: DONE")


if __name__ == "__main__":
    main()
