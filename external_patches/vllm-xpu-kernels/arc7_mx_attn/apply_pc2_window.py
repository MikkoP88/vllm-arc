#!/usr/bin/env python3
"""apply_pc2_window.py -- P-C2 window threading onto a LIVE arc7mx tree.

apply_arc7mx.py carries the P-C2 strings for FRESH wheel sources (it
builds the mx surface from nothing and refuses an already-patched tree).
The rig's /root/build/vxk-arc-src is that applied state -- cert26-era
strings on attn_interface.{h,cpp} / flash_api.cpp plus the r212c..r225
kernel iterations shipped as files. This script re-applies ONLY the
P-C2 deltas to such a tree:

  1. attn_interface.h: the mx decl gains int window_size_left/right
     after double sm_scale (mirrors the stock decl order).
  2. attn_interface.cpp: the mx forwarder signature + its
     cutlass_mx_paged_decode_xe2 call gain the pair after sm_scale.
  3. flash_api.cpp: mx_paged_decode_op signature + its
     cutlass_mx_paged_decode_interface call gain the pair after
     softmax_scale.
  4. flash_api.cpp: the TORCH_LIBRARY schema gains the two ints after
     softmax_scale.

The kernel-side sources (csrc/.../mx/*) and the python surface
(vllm_xpu_kernels/mx_attn_interface.py) ship as FILES (scp), not
patches -- this script never touches them. Every pattern asserts a
single exact occurrence (the patch_file discipline); a mismatch
aborts loudly before writing.

Usage: python3 apply_pc2_window.py /root/build/vxk-arc-src
Idempotent: patterns replaced here no longer match on a second run,
which the marker audit reports as already-applied (exit 0) -- never a
silent double-apply.
"""

from __future__ import annotations

import sys
from pathlib import Path

W = "    // P-C2 (2026-11-25): sliding-window bounds, -1/-1 = none.\n"


def patch_file(path: Path, edits, marker: str) -> None:
    text = path.read_text()
    for old, new in edits:
        n = text.count(old)
        if n == 0:
            if text.count(new) > 0:
                print(f"[skip] {path.name}: already applied ({marker})")
                return
            sys.exit(f"FATAL: pattern not found in {path} ({marker}):\n{old}")
        if n != 1:
            sys.exit(
                f"FATAL: pattern x{n} (expected 1) in {path} "
                f"({marker}):\n{old}")
        text = text.replace(old, new)
    path.write_text(text)
    print(f"[ok] {path.name}: {marker}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

    # 1. attn_interface.h: the mx decl (anchored on the seqused_k row --
    #    the stock decl above rides cu_seqlens_k and different args).
    patch_file(
        root / "csrc/xpu/attn/attn_interface.h",
        [(
            "    const at::Tensor& seqused_k,  // per-seq used KV length (int32)\n"
            "    int max_seqlen_q,\n"
            "    int max_seqlen_k,\n"
            "    double sm_scale,\n"
            "    int num_kv_splits,\n",
            "    const at::Tensor& seqused_k,  // per-seq used KV length (int32)\n"
            "    int max_seqlen_q,\n"
            "    int max_seqlen_k,\n"
            "    double sm_scale,\n"
            + W
            + "    int window_size_left,\n"
            "    int window_size_right,\n"
            "    int num_kv_splits,\n",
        )],
        "mx decl window pair",
    )

    # 2. attn_interface.cpp: the forwarder signature (is_xe2_arch body
    #    follows immediately -- unique to the mx forwarder) and its xe2
    #    call (the stock call rides sm_sink_ + the stock window args).
    patch_file(
        root / "csrc/xpu/attn/attn_interface.cpp",
        [(
            "    double sm_scale,\n"
            "    int num_kv_splits,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list) {\n"
            "  if (vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch()) {\n",
            "    double sm_scale,\n"
            "    int window_size_left,\n"
            "    int window_size_right,\n"
            "    int num_kv_splits,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list) {\n"
            "  if (vllm::xpu::is_xe2_arch() || vllm::xpu::is_xe3_arch()) {\n",
        )],
        "mx forwarder signature",
    )
    patch_file(
        root / "csrc/xpu/attn/attn_interface.cpp",
        [(
            "        max_seqlen_k,\n"
            "        sm_scale,\n"
            "        num_kv_splits,\n"
            "        splits_per_seq,\n"
            "        work_list);\n",
            "        max_seqlen_k,\n"
            "        sm_scale,\n"
            "        window_size_left,\n"
            "        window_size_right,\n"
            "        num_kv_splits,\n"
            "        splits_per_seq,\n"
            "        work_list);\n",
        )],
        "mx forwarder xe2 call",
    )

    # 3. flash_api.cpp: the op fn signature and its interface call.
    patch_file(
        root / "csrc/flash_attn/flash_api.cpp",
        [(
            "    float softmax_scale,\n"
            "    int num_kv_splits,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list,\n"
            "    at::Tensor& out) {\n",
            "    float softmax_scale,\n"
            "    int window_size_left,\n"
            "    int window_size_right,\n"
            "    int num_kv_splits,\n"
            "    std::optional<at::Tensor>& splits_per_seq,\n"
            "    std::optional<at::Tensor>& work_list,\n"
            "    at::Tensor& out) {\n",
        )],
        "mx_paged_decode_op signature",
    )
    patch_file(
        root / "csrc/flash_attn/flash_api.cpp",
        [(
            "      max_seqlen_k,\n"
            "      softmax_scale,\n"
            "      num_kv_splits,\n"
            "      splits_per_seq,\n"
            "      work_list);\n"
            "}\n"
            "}  // namespace FLASH_NAMESPACE",
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
        "mx_paged_decode_op interface call",
    )

    # 4. flash_api.cpp: the TORCH_LIBRARY schema (comment lines mirror
    #    apply_arc7mx.py's fresh-apply state byte-for-byte).
    patch_file(
        root / "csrc/flash_attn/flash_api.cpp",
        [(
            "  // arc7 (P6.47 Phase 8): MX-native paged decode (MX KV lanes).\n"
            "  ops.def(\n",
            "  // arc7 (P6.47 Phase 8): MX-native paged decode (MX KV lanes).\n"
            "  // P-C2 (2026-11-25): window pair after softmax_scale --\n"
            "  // sliding decode = (W-1, 0); -1/-1 = the windowless lane.\n"
            "  ops.def(\n",
        )],
        "mx_paged_decode schema comment",
    )
    patch_file(
        root / "csrc/flash_attn/flash_api.cpp",
        [(
            '      "max_seqlen_q, int max_seqlen_k, float softmax_scale, int "\n'
            '      "num_kv_splits, Tensor? splits_per_seq, Tensor? work_list, "\n'
            '      "Tensor! out) -> ()");\n',
            '      "max_seqlen_q, int max_seqlen_k, float softmax_scale, int "\n'
            '      "window_size_left, int window_size_right, int "\n'
            '      "num_kv_splits, Tensor? splits_per_seq, Tensor? work_list, "\n'
            '      "Tensor! out) -> ()");\n',
        )],
        "mx_paged_decode schema",
    )

    print("apply_pc2_window: DONE")


if __name__ == "__main__":
    main()
