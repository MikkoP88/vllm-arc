#!/usr/bin/env python3
"""apply_arc7mx.py -- rig-side source patcher for the arc7_mx_attn wheel
section (P6.47 Phase 8: MX-native paged-decode kernel). Builds a NEW tree
at /root/build/vxk-arc7mx-src from the CERTIFIED arc6 base and applies:

  1. Base: extract /root/build/vxk-arc6-cert22-src.tgz (the tar snapshot
     of the applied-arc6 tree taken BEFORE the jm edits -- i.e. exactly
     the content the certified 0.1.14.1+arc6 wheel was built from,
     including the .deps warm cache). Provenance asserts: skinny module
     present, jm module ABSENT. The live /root/build/vxk-arc-src tree
     (arc6+jm) is NOT touched here -- the build script swaps trees.
  2. Copy csrc/xpu/attn/xe_2/mx/ (kernel + entry TU): the xe_2 kernel
     library GLOBs *.cpp recursively (cmake/utils.cmake
     add_xe2_kernel_library), so the module joins the build with ZERO
     CMake changes -- the vendor's own add-a-file idiom.
  3. Copy vllm_xpu_kernels/mx_attn_interface.py (python surface; pure
     addition, the package __init__ stays untouched -- callers import
     the module directly).
  4. csrc/xpu/attn/attn_interface.h: cutlass_mx_paged_decode_interface
     declaration (mirrors the stock decode decl; the MX deltas are the
     single uint8 slot cache, the fmt id, and decode-fixed flags).
  5. csrc/xpu/attn/attn_interface.cpp: mx include inside the XE2 ifdef +
     the arch-guarded forwarder to cutlass_mx_paged_decode_xe2 (the
     vendor layering: fa2 op -> attn_interface -> xe2 entry).
  6. csrc/flash_attn/flash_api.cpp: mx_paged_decode_op in
     FLASH_NAMESPACE (queue via vllmGetQueue; decode buffers left
     undefined -- the xe2 entry allocates per the vendor shapes) + the
     TORCH_LIBRARY def/impl via make_pytorch_shim (int/float/optional
     conversions, same as varlen_fwd).

Anchors are exact-match against the tree's bytes and the script REFUSES
(exit 1) on any miss or double-apply -- byte-exact review discipline
(arc5/arc6/arc7 kit law). Anchor texts verified against
/root/build/vxk-arc-src's files 2026-11-19 (kit _vendor_anchors/,
tarball md5 8274aa7a399eb514f088ba20ec84fc43 / 7de4139f5365edee52f22259af8f0722).
"""
import shutil
import sys
import tarfile
from pathlib import Path

BUILD = Path("/root/build")
TAR = BUILD / "vxk-arc6-cert22-src.tgz"  # certified arc6 base (pre-jm)
NEW = BUILD / "vxk-arc7mx-src"
KIT = BUILD / "arc7mxkit"  # this kit's payload ships here


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
    if NEW.exists():
        sys.exit(f"FATAL: {NEW} already exists -- remove it to re-apply")
    if not TAR.exists():
        sys.exit(f"FATAL: certified arc6 base tarball missing: {TAR}")
    if not (KIT / "csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp").exists():
        sys.exit(f"FATAL: kit mx sources missing under {KIT}")
    if not (KIT / "vllm_xpu_kernels/mx_attn_interface.py").exists():
        sys.exit(f"FATAL: kit python surface missing under {KIT}")

    # 1. certified base tree (strip the leading vxk-arc-src/ component so
    # the tree root IS the new dir)
    staging = NEW.with_name(NEW.name + ".staging")
    if staging.exists():
        sys.exit(f"FATAL: stale staging dir {staging} -- remove it")
    staging.mkdir(parents=True)
    with tarfile.open(TAR, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.startswith("vxk-arc-src/")]
        if not members:
            sys.exit(f"FATAL: {TAR} has no vxk-arc-src/ root")
        tf.extractall(path=staging, members=members)  # noqa: S202 (our tarball)
    (staging / "vxk-arc-src").rename(NEW)
    shutil.rmtree(staging)
    print(f"[ok] base extracted {TAR} -> {NEW}")

    # provenance asserts: the certified arc6 content, pre-jm
    if not (NEW / "csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp").exists():
        sys.exit(f"FATAL: {NEW} lacks the arc6 skinny module (bad base)")
    if (NEW / "csrc/xpu/jm_gemm").exists():
        sys.exit(f"FATAL: {NEW} carries the jm module (base is post-jm)")
    if not (NEW / ".deps").exists():
        print("[warn] base lacks .deps -- first build compiles oneDNN cold")

    # 2. mx module sources
    mx_dst = NEW / "csrc/xpu/attn/xe_2/mx"
    if mx_dst.exists():
        sys.exit("FATAL: csrc/xpu/attn/xe_2/mx already present")
    shutil.copytree(KIT / "csrc/xpu/attn/xe_2/mx", mx_dst)
    print("[ok] mx module copied")

    # 3. python surface
    py_dst = NEW / "vllm_xpu_kernels/mx_attn_interface.py"
    if py_dst.exists():
        sys.exit("FATAL: vllm_xpu_kernels/mx_attn_interface.py already present")
    shutil.copy2(KIT / "vllm_xpu_kernels/mx_attn_interface.py", py_dst)
    print("[ok] python surface copied")

    # 4. attn_interface.h: the mx declaration after the stock decode decl
    hdr = NEW / "csrc/xpu/attn/attn_interface.h"
    patch_file(
        hdr,
        [(
            "    std::optional<const at::Tensor>& is_prefill,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list);",
            "    std::optional<const at::Tensor>& is_prefill,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list);\n"
            "\n"
            "// arc7 (P6.47 Phase 8): MX-native paged decode -- the MX KV\n"
            "// lanes' vendor-class kernel. MX deltas vs the stock decl: one\n"
            "// uint8 TQ-AoS slot cache in place of separate K/V caches, a\n"
            "// format id in place of the dtype pair, and decode-fixed flags\n"
            "// (paged, varlen, no causal -- the vendor decode law). The\n"
            "// decode buffers may arrive undefined; the xe2 entry allocates\n"
            "// per the vendor shapes.\n"
            "void cutlass_mx_paged_decode_interface(\n"
            "    sycl::queue& queue,\n"
            "    const at::Tensor& query,      // [total_q, heads_q, D] bf16\n"
            "    const at::Tensor& mx_cache,   // [num_block, block, Hk, slot_bytes] u8\n"
            "    int64_t fmt_id,               // 0=mxfp8 1=mxfp4 2=mxk8v4\n"
            "    at::Tensor& out,\n"
            "    at::Tensor& temp_out,\n"
            "    at::Tensor& exp_sums,\n"
            "    at::Tensor& max_logits,\n"
            "    const at::Tensor& block_table,\n"
            "    const at::Tensor& cu_seqlens_q,\n"
            "    const at::Tensor& seqused_k,  // per-seq used KV length (int32)\n"
            "    int max_seqlen_q,\n"
            "    int max_seqlen_k,\n"
            "    double sm_scale,\n"
            "    // P-C2 (2026-11-25): sliding-window bounds, -1/-1 = none.\n"
            "    int window_size_left,\n"
            "    int window_size_right,\n"
            "    int num_kv_splits,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list);",
        )],
        "cutlass_mx_paged_decode_interface",
    )

    # 5a. attn_interface.cpp: mx include inside the XE2 ifdef
    iface = NEW / "csrc/xpu/attn/attn_interface.cpp"
    patch_file(
        iface,
        [(
            '  #include "csrc/xpu/attn/xe_2/paged_decode_xe2.h"\n'
            "#endif",
            '  #include "csrc/xpu/attn/xe_2/paged_decode_xe2.h"\n'
            '  #include "csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.h"\n'
            "#endif",
        )],
        "mx/mx_paged_decode_xe2.h",
    )

    # 5b. attn_interface.cpp: the forwarder (arch-guarded, vendor layering)
    patch_file(
        iface,
        [(
            "        splits_per_seq,\n"
            "        work_list);\n"
            "#else\n"
            '    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");\n'
            "#endif\n"
            "  } else {\n"
            '    TORCH_CHECK(false, "Only XE2/XE3 cutlass kernel is supported currently.");\n'
            "  }\n"
            "}",
            "        splits_per_seq,\n"
            "        work_list);\n"
            "#else\n"
            '    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");\n'
            "#endif\n"
            "  } else {\n"
            '    TORCH_CHECK(false, "Only XE2/XE3 cutlass kernel is supported currently.");\n'
            "  }\n"
            "}\n"
            "\n"
            "void cutlass_mx_paged_decode_interface(\n"
            "    sycl::queue& queue,\n"
            "    const at::Tensor& query,\n"
            "    const at::Tensor& mx_cache,\n"
            "    int64_t fmt_id,\n"
            "    at::Tensor& out,\n"
            "    at::Tensor& temp_out,\n"
            "    at::Tensor& exp_sums,\n"
            "    at::Tensor& max_logits,\n"
            "    const at::Tensor& block_table,\n"
            "    const at::Tensor& cu_seqlens_q,\n"
            "    const at::Tensor& seqused_k,\n"
            "    int max_seqlen_q,\n"
            "    int max_seqlen_k,\n"
            "    double sm_scale,\n"
            "    int window_size_left,\n"
            "    int window_size_right,\n"
            "    int num_kv_splits,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list) {\n"
            "  if (vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch()) {\n"
            "#ifdef VLLM_XPU_ENABLE_XE2\n"
            "    cutlass_mx_paged_decode_xe2(\n"
            "        queue,\n"
            "        query,\n"
            "        mx_cache,\n"
            "        fmt_id,\n"
            "        out,\n"
            "        temp_out,\n"
            "        exp_sums,\n"
            "        max_logits,\n"
            "        block_table,\n"
            "        cu_seqlens_q,\n"
            "        seqused_k,\n"
            "        max_seqlen_q,\n"
            "        max_seqlen_k,\n"
            "        sm_scale,\n"
            "        window_size_left,\n"
            "        window_size_right,\n"
            "        num_kv_splits,\n"
            "        splits_per_seq,\n"
            "        work_list);\n"
            "#else\n"
            '    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");\n'
            "#endif\n"
            "  } else {\n"
            '    TORCH_CHECK(false, "Only XE2/XE3 cutlass kernel is supported currently.");\n'
            "  }\n"
            "}",
        )],
        "cutlass_mx_paged_decode_xe2(",
    )

    # 6a. flash_api.cpp: the op fn at the FLASH_NAMESPACE close
    fa = NEW / "csrc/flash_attn/flash_api.cpp"
    patch_file(
        fa,
        [(
            "}  // namespace FLASH_NAMESPACE",
            "// arc7 (P6.47 Phase 8): MX-native paged decode op. Python surface:\n"
            "// vllm_xpu_kernels/mx_attn_interface.py (per-seq split plan mirrors\n"
            "// the stock decode path). Decode buffers stay undefined here --\n"
            "// the xe2 entry allocates them per the vendor shapes; the split\n"
            "// default (num_kv_splits < 1) is the vendor get_num_splits rule,\n"
            "// mirrored in the entry. The kernel reads per-seq KV lengths\n"
            "// from the cu_seqlens_k slot (seqused_k) -- the stock decode\n"
            "// contract.\n"
            "void mx_paged_decode_op(\n"
            "    const at::Tensor& q,\n"
            "    const at::Tensor& mx_cache,\n"
            "    int fmt_id,\n"
            "    const at::Tensor& cu_seqlens_q,\n"
            "    const at::Tensor& seqused_k,\n"
            "    const at::Tensor& block_table,\n"
            "    int max_seqlen_q,\n"
            "    int max_seqlen_k,\n"
            "    float softmax_scale,\n"
            "    int window_size_left,\n"
            "    int window_size_right,\n"
            "    int num_kv_splits,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list,\n"
            "    at::Tensor& out) {\n"
            "  auto& queue = vllm::xpu::vllmGetQueue(q.device().index());\n"
            "  at::Tensor temp_out, exp_sums, max_logits;  // entry allocates\n"
            "  cutlass_mx_paged_decode_interface(\n"
            "      queue,\n"
            "      q,\n"
            "      mx_cache,\n"
            "      fmt_id,\n"
            "      out,\n"
            "      temp_out,\n"
            "      exp_sums,\n"
            "      max_logits,\n"
            "      block_table,\n"
            "      cu_seqlens_q,\n"
            "      seqused_k,\n"
            "      max_seqlen_q,\n"
            "      max_seqlen_k,\n"
            "      softmax_scale,\n"
            "      window_size_left,\n"
            "      window_size_right,\n"
            "      num_kv_splits,\n"
            "      splits_per_seq,\n"
            "      work_list);\n"
            "}\n"
            "}  // namespace FLASH_NAMESPACE",
        )],
        "void mx_paged_decode_op(",
    )

    # 6b. flash_api.cpp: registration after varlen_fwd
    patch_file(
        fa,
        [(
            "  ops.impl(\n"
            '      "varlen_fwd",\n'
            "      torch::kXPU,\n"
            "      make_pytorch_shim(&FLASH_NAMESPACE::mha_varlen_fwd));\n"
            "}",
            "  ops.impl(\n"
            '      "varlen_fwd",\n'
            "      torch::kXPU,\n"
            "      make_pytorch_shim(&FLASH_NAMESPACE::mha_varlen_fwd));\n"
            "\n"
            "  // arc7 (P6.47 Phase 8): MX-native paged decode (MX KV lanes).\n"
            "  // P-C2 (2026-11-25): window pair after softmax_scale —\n"
            "  // sliding decode = (W-1, 0); -1/-1 = the windowless lane.\n"
            "  ops.def(\n"
            '      "mx_paged_decode(Tensor q, Tensor mx_cache, int fmt_id, Tensor "\n'
            '      "cu_seqlens_q, Tensor seqused_k, Tensor block_table, int "\n'
            '      "max_seqlen_q, int max_seqlen_k, float softmax_scale, int "\n'
            '      "window_size_left, int window_size_right, int "\n'
            '      "num_kv_splits, Tensor? splits_per_seq, Tensor? work_list, "\n'
            '      "Tensor! out) -> ()");\n'
            "  ops.impl(\n"
            '      "mx_paged_decode",\n'
            "      torch::kXPU,\n"
            "      make_pytorch_shim(&FLASH_NAMESPACE::mx_paged_decode_op));\n"
            "}",
        )],
        '"mx_paged_decode(Tensor q, Tensor mx_cache, int fmt_id, Tensor "',
    )

    print("apply_arc7mx: DONE")
    print(f"next: bash build_vxk_wheel_arc7mx.sh (swaps {NEW} in as the builder tree)")


if __name__ == "__main__":
    main()
