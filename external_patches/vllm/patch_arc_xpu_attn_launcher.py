#!/usr/bin/env python3
"""arc-xpu-attn-tune — B70 prefill launch-config table for the triton
unified-attention launcher (vllm wheel file
vllm/v1/attention/ops/triton_unified_attention.py).

Defect (P6.28, rig r108j/r108k, results/2026-11-06-p628-gemma4/): the 2D
prefill kernel runs at ~0.3% of achievable matmul peak (1.1 TFLOP in 8.3 s
vs a 43.2 TFLOPS fp16 matmul reference on the same device) at the stock
launch config — BLOCK_M=16 / TILE=32 / 4 warps. Gemma-4's 60-layer
5:1 mix of head-512 full / head-256 sliding prefill then burns ~20 min of
device time per 32k-token face, tripping the 300 s
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS rpc kill — the whole "gemma freeze".

Fix shape = upstream vLLM PR#60418 ([ROCm][RDNA][Perf] Add tuned configs
for TRITON_ATTN kernel, merged 2026-11-03): module-level per-platform
config tables keyed by padded head size x max_seqlen_q band
(<=64 / 65-4095 / >=4096) -> (BLOCK_M, TILE_SIZE_PREFILL, num_warps,
num_stages) or None (keep stock), selected in the launcher before the
grid upper bound is computed. Only launch parameters change — kernel
source and numerics untouched (r108k numeric dumps allclose(1e-2) vs
stock). The kernel's own authors describe this as the designed tuning
surface ("autotuning selects appropriate block sizes for each platform";
KV-side tiles are page-constrained so "optimization focuses on the query
side" — vllm.ai Triton backend deep dive, 2026-03-04).

B70 sweep evidence (r108k, ntok=1024 / seqlen=8192 / fp8-KV, per-rank
GQA 16Q:8KV): head-512 stock 2.002 s -> (M64, T32, w8) 0.428 s = 4.67x;
TILE>32 at head-512 is infeasible on B70 (SM-side memory wall — the same
LDS boundary vllm#52585 documented for TILE=128 on gfx1100). num_stages
stays None = triton default (that is what was measured).

Decode / 3D split-KV path is untouched: the band guard
(max_seqlen_q > 64) keeps every decode-shaped call on stock configs.

Idempotent (skips when the mark is present); writes .arcxpubak backup.

Usage:
    patch_arc_xpu_attn_launcher.py [launcher.py]        # patch
    patch_arc_xpu_attn_launcher.py [launcher.py] --audit
    (no path: locate vllm on sys.path — the image build does this)
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

MARK = "[ARC-P628-XPU-TUNE]"
REL = "vllm/v1/attention/ops/triton_unified_attention.py"

# --- B70 prefill table (rig sweep r108k, 2026-11-06) ---
# band tuples: (BLOCK_M, TILE_SIZE_PREFILL, num_warps, num_stages);
# None band entry = keep stock config for that band.
CFG_256_MID = "(64, 64, 8, None)"
CFG_256_HIGH = "(64, 64, 8, None)"
CFG_512_MID = "(64, 32, 8, None)"
CFG_512_HIGH = "(64, 32, 8, None)"

# --- byte-exact anchors of the pinned wheel (vllm 0.30.0+xpu, cert11) ---
# module top, right after fp8 dtype info: table + band selector
ANCHOR_MODULE = "float8_info = torch.finfo(current_platform.fp8_dtype())\n"
# prefill branch: right after the tuned_large_head BLOCK_M block, BEFORE
# total_num_q_blocks (the 2D grid upper bound scales with BLOCK_Q)
ANCHOR_PREFILL = (
    "    if tuned_large_head:\n"
    "        BLOCK_M = 32\n"
    "        BLOCK_Q = BLOCK_M // num_queries_per_kv\n"
    "        launch_num_warps = 8\n"
    "        launch_num_stages = 2\n"
)
# tile override: right after the tuned_large_head TILE block (which itself
# sits after the _get_tile_size calls), before the use_td clamp
ANCHOR_TILE = (
    "    if tuned_large_head:\n"
    "        TILE_SIZE_PREFILL = 128\n"
)

MODULE_BLOCK = f'''

# {MARK} B70 prefill launch-config table -------------------------
# Upstream pattern: PR#60418 ([ROCm][RDNA][Perf] tuned configs for
# TRITON_ATTN, 2026-11-03) — per-platform tables keyed by padded head
# size x max_seqlen_q band (<=64 / 65-4095 / >=4096) ->
# (BLOCK_M, TILE_SIZE_PREFILL, num_warps, num_stages) or None (stock).
# XPU/B70 values: rig sweep r108k 2026-11-06
# (results/2026-11-06-p628-gemma4/phaseB/tile_sweep/) — stock BLOCK_M=16
# leaves 2D prefill at ~0.3% of achievable matmul peak (r108j microbench:
# 43.2 TFLOPS fp16 reference); TILE>32 infeasible at head 512 on B70
# (SM memory wall; same LDS boundary class as vllm#52585 TILE=128 on
# gfx1100). num_stages None = triton default (as measured). Decode / 3D
# split-KV untouched (band guard max_seqlen_q > 64 in the launcher).
_ARC_XPU_TUNE = current_platform.is_xpu()
_ARC_XPU_PREFILL_CONFIGS = {{
    # padded head size: (<=64, 65-4095, >=4096)
    256: (None, {CFG_256_MID}, {CFG_256_HIGH}),
    512: (None, {CFG_512_MID}, {CFG_512_HIGH}),
}}


def _arc_xpu_prefill_config(head_size_padded, max_seqlen_q):
    """Band-select a B70 prefill config (PR#60418 _prefill_band_config)."""
    bands = _ARC_XPU_PREFILL_CONFIGS.get(head_size_padded)
    if bands is None:
        return None
    return bands[0] if max_seqlen_q <= 64 else (
        bands[1] if max_seqlen_q < 4096 else bands[2])
'''

PREFILL_BLOCK = f'''
    # {MARK} B70 prefill launch config (PR#60418 pattern). Guards mirror
    # the RDNA branch: GQA groups <= 16, Q not sub-fp16, batch-invariant
    # off, and only the >=4096 / 65-4095 bands are table-covered (the
    # <=64 band is None). Runs BEFORE total_num_q_blocks so the 2D grid
    # shrinks with the wider BLOCK_Q. Only launch parameters change —
    # kernel numerics untouched (r108k allclose(1e-2) vs stock).
    _arc_tile_prefill = None
    if (_ARC_XPU_TUNE and num_queries_per_kv <= 16
            and max_seqlen_q > 64 and not is_batch_invariant
            and q.element_size() <= 2):
        # 1 << (n-1).bit_length() == triton.next_power_of_2(n), no import
        _arc_cfg = _arc_xpu_prefill_config(
            1 << (head_size - 1).bit_length(), max_seqlen_q)
        if _arc_cfg is not None:
            (BLOCK_M, _arc_tile_prefill, launch_num_warps,
             launch_num_stages) = _arc_cfg
            BLOCK_Q = BLOCK_M // num_queries_per_kv
'''

TILE_BLOCK = f'''
    # {MARK} table tile override; max(32, .) keeps fp8-KV legal
    # (PR#60418 guard); the stock use_td clamp below still applies.
    if _arc_tile_prefill is not None:
        TILE_SIZE_PREFILL = max(32, _arc_tile_prefill)
'''


def find_launcher() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("FATAL: vllm not importable — pass the launcher "
                         "path explicitly")
    pkg = Path(spec.origin).parent
    target = pkg / "v1/attention/ops/triton_unified_attention.py"
    if not target.is_file():
        raise SystemExit(f"FATAL: {target} not found — layout changed?")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("launcher", nargs="?",
                    help=f"path to {REL} (default: locate vllm on sys.path)")
    ap.add_argument("--audit", action="store_true",
                    help="report anchor/mark state without patching")
    args = ap.parse_args()
    path = Path(args.launcher) if args.launcher else find_launcher()

    src = path.read_text(encoding="utf-8")
    print(f"target: {path}")
    state = {
        "mark": MARK in src,
        "anchor_module": ANCHOR_MODULE in src,
        "anchor_prefill": ANCHOR_PREFILL in src,
        "anchor_tile": ANCHOR_TILE in src,
    }
    for k, v in state.items():
        print(f"  {k}: {'present' if v else 'MISSING'}")

    if state["mark"]:
        print("ALREADY_PATCHED")
        return 0
    if args.audit:
        missing = [k for k, v in state.items()
                   if not v and k != "mark"]
        print("AUDIT:", "PATCHABLE (all anchors present)"
              if not missing else f"NOT PATCHABLE (missing: {missing})")
        return 0 if not missing else 1
    missing = [k for k, v in state.items() if not v and k != "mark"]
    if missing:
        print(f"FATAL: anchors missing: {missing} — wheel layout drifted; "
              "re-audit before trusting the image")
        return 1

    out = src.replace(ANCHOR_MODULE, ANCHOR_MODULE + MODULE_BLOCK, 1)
    out = out.replace(ANCHOR_PREFILL, ANCHOR_PREFILL + PREFILL_BLOCK, 1)
    out = out.replace(ANCHOR_TILE, ANCHOR_TILE + TILE_BLOCK, 1)
    if out.count(MARK) != 3:
        print(f"FATAL: expected 3 {MARK} insertions, got {out.count(MARK)}")
        return 1
    shutil.copy2(path, str(path) + ".arcxpubak")
    path.write_text(out, encoding="utf-8", newline="")
    print(f"PATCHED: 3 insertions ({len(out) - len(src)} bytes) "
          f"+ backup {path.name}.arcxpubak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
