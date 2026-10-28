#!/usr/bin/env python3
"""vxk-int64fix — int64 pool-offset fix for GDN conv-state kernels.

Port of donor wedgefix-v88 patcher (iexperimental kit, read-only import per
plan §6 / fork law; provenance in README.md). The v88 law at the 0.26 pin
covered 8 sites; upstream refactored before v0.1.14 — the pinned lineage
carries 6 bare sites (all `states_id`):

    csrc/xpu/gdn_attn/causal_conv1d.hpp:232,452
    csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp:203,627
    csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp:220,585

Each computes the per-request state pointer as

    conv_states + states_id * conv_states_stride_0        (int * int)

The deployed mamba pool has stride(0) = 524,288 elements (padded 1-MiB
unified page), so the product overflows signed int32 for every state
id >= 4097 (4096 wraps to INT32_MIN): wrapped-negative offsets give silent
state corruption when mapped, UR_RESULT_ERROR_DEVICE_LOST when unmapped
(the lsv-test wedge; donor RCA 2026-10-18/19, cap727 aim row 4343).
mamba_pool_ratio 2 pushes dense qwen3.8-27b rows past 4096 — the target
model IS exposed (plan §6).

Fix: widen the id to int64 before the multiply, matching the existing safe
pattern in chunk_gated_delta_rule_kernels_xe2.hpp / gated_delta_rule.hpp.
Stride itself stays int (524,288 fits).

Idempotent (skips files carrying the mark); writes .vxki64bak backups.

Usage:
    patch_vxk_int64fix.py <vxk-source-root> [--expect N]   # patch (N=6)
    patch_vxk_int64fix.py <vxk-source-root> --audit        # report only
"""
import argparse
import re
import shutil
import sys
from pathlib import Path

FILES = [
    "csrc/xpu/gdn_attn/causal_conv1d.hpp",
    "csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_xe2.hpp",
    "csrc/xpu/gdn_attn/xe_2/chunk_causal_conv1d_tiled_xe2.hpp",
]

# (id variable) — donor v88 PATTERNS (init_state_id/save_state_id kept for
# lineage parity; the v0.1.14 lineage uses states_id at all 6 sites)
PATTERNS = ["states_id", "init_state_id", "save_state_id"]

MARK = "[VXKI64]"
EXPECTED_SITES = 6  # v88 law: 8 at the 0.26 pin; 0.1.14 lineage refactored to 6


def audit(root: Path) -> dict:
    """Per-file {bare, fixed} counts of id*stride sites."""
    out = {}
    for rel in FILES:
        src = (root / rel).read_text()
        bare = fixed = 0
        for var in PATTERNS:
            fixed += len(re.findall(
                r"static_cast<int64_t>\(" + re.escape(var)
                + r"\) \* conv_states_stride_0", src))
            bare += len(re.findall(
                r"(?<!static_cast<int64_t>\()" + re.escape(var)
                + r" \* conv_states_stride_0", src))
        out[rel] = {"bare": bare, "fixed": fixed, "marked": MARK in src}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="vllm-xpu-kernels source root")
    ap.add_argument("--expect", type=int, default=EXPECTED_SITES,
                    help="fail unless exactly this many bare sites patched")
    ap.add_argument("--audit", action="store_true",
                    help="report site counts without patching")
    args = ap.parse_args()
    root = Path(args.root)
    if not (root / FILES[0]).is_file():
        print(f"FATAL: {root}/{FILES[0]} not found — wrong root?")
        return 2

    counts = audit(root)
    for rel, c in counts.items():
        print(f"{rel}: bare={c['bare']} fixed={c['fixed']} "
              f"marked={c['marked']}")
    total_bare = sum(c["bare"] for c in counts.values())
    if args.audit:
        print(f"TOTAL_BARE={total_bare} -> "
              + ("BARE (contingency applies)" if total_bare
                 else "CLEAN"))
        return 0

    total = 0
    for rel in FILES:
        path = root / rel
        src = path.read_text()
        if MARK in src:
            print(f"{rel}: ALREADY_PATCHED")
            continue
        out, n = src, 0
        for var in PATTERNS:
            bare = re.compile(r"(?<!static_cast<int64_t>\()"
                              + re.escape(var)
                              + r" \* conv_states_stride_0")
            out, k = bare.subn(
                f"static_cast<int64_t>({var}) * conv_states_stride_0", out)
            n += k
        if n == 0:
            print(f"{rel}: NO_SITES_FOUND (unexpected)")
            return 1
        out = out.replace(
            "const int conv_states_stride_0 = conv_states.stride(0);",
            "const int conv_states_stride_0 = conv_states.stride(0);  "
            f"// {MARK} int64 pool offsets (id*stride overflows int32 at "
            "id>=4097; wedgefix-v88 lineage, plan §6)",
            1,
        )
        shutil.copy2(path, str(path) + ".vxki64bak")
        path.write_text(out)
        print(f"{rel}: patched {n} multiply sites + 1 mark")
        total += n
    print(f"TOTAL_SITES_PATCHED={total}")
    if total != args.expect:
        print(f"FATAL: expected {args.expect} sites, patched {total} — "
              "upstream layout changed; re-audit before trusting the wheel")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
