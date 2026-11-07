#!/usr/bin/env python3
"""ARC P6.28 kit patch #3 (DIAGNOSTIC — debug overlays only, never a cert
pin): per-call launch-key trace for the triton unified-attention wrapper.

Why (r108r, banked results/2026-11-06-p628-gemma4/phaseC/cert14_boot/):
cert14's engine-exact warm still left ONE fresh kernel_unified_attention
compile at the first 98-token request (jit_monitor warn 4 s after health,
TP0; engine watchdog killed the step 300 s later). Static diff of the
wrapper's full launch list against the warm signature finds no differing
dimension — so the miss is empirical: some value the ENGINE holds at
request time differs from every warm point, or the request-time call
reaches a different JITFunction object (cache-namespace split). This
patch settles it with data instead of a third guess: every call logs the
complete triton cache-key-relevant tuple right before the kernel launch.

What this patch does: one guarded block immediately before
``kernel_unified_attention[grid](...)`` in the wheel's
vllm/v1/attention/ops/triton_unified_attention.py, mark
``[ARC-P628-XPU-TRACE]``. Gated by ARC_XPU_TRACE_UNIFIED (default ON in
the debug overlay image that carries it; set 0 to silence). Log line
prefix ``ARCXPU_TRACE`` — one line per attention call, both warmup and
inference, so a single wedge boot yields the warm key-set AND the
engine's request-time key for an exact diff.

Usage:
    python patch_arc_xpu_attn_trace.py [path/to/triton_unified_attention.py] [--audit]

Without a path, the wheel location is found via importlib. --audit only
reports. Idempotent (mark count checked); .arcxpubak backup left next to
the original.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import py_compile
import shutil
import sys

MARK = "[ARC-P628-XPU-TRACE]"

# Anchor — the single kernel launch site of the 2D/3D unified kernel.
ANCHOR = "    kernel_unified_attention[grid](\n"

TRACE_BLOCK = '''    # ''' + MARK + ''' per-call launch-key trace (diagnostic overlay;
    # default ON here, ARC_XPU_TRACE_UNIFIED=0 silences). One line per
    # call: every triton cache-key-relevant value + the JITFunction
    # object id, so warmup vs inference call signatures diff exactly.
    import os as _arc_os
    if _arc_os.environ.get("ARC_XPU_TRACE_UNIFIED", "1") != "0":
        import logging as _arc_log
        _arc_log.getLogger(__name__).info(
            "ARCXPU_TRACE fnid=%s hs=%d nqh=%d nkvh=%d npq=%d "
            "qshape=%s qstride=%s qdtype=%s kstride=%s kdtype=%s "
            "vstride=%s btstride=%s btdtype=%s bs=%d ns=%d "
            "mxq=%d mxk=%d BM=%d BQ=%d TILE=%d warps=%s stages=%s "
            "grid=%s 3d=%s td=%s tdqo=%s win=%s swv=%d cl=%d cs=%d "
            "kqm=%d Q8=%s FP8o=%s csw=%s useg=%d causal=%s "
            "pseq=%s alibi=%s asqrt=%s qq=%s sinks=%s oscale=%s "
            "ostride=%s odtype=%s cuqdt=%s sqdt=%s mmr=%d rswa=%s "
            "kq=%s vq=%s ks=%s",
            id(kernel_unified_attention), head_size, num_query_heads,
            num_kv_heads, num_queries_per_kv, tuple(q.shape),
            q.stride(), q.dtype, k.stride(), k.dtype, v.stride(),
            block_table.stride(0), block_table.dtype, block_size,
            num_seqs, max_seqlen_q, max_seqlen_k, BLOCK_M, BLOCK_Q,
            tile_size, launch_num_warps, launch_num_stages, grid,
            use_3d, use_td, use_td_qo, window_size, sliding_window_val,
            chunk_lookback, chunk_size, kv_quant_mode,
            q.dtype == current_platform.fp8_dtype(),
            output_scale is not None, mm_prefix_clamp_sliding_window,
            num_segments, use_causal, use_per_seq_causal,
            use_alibi_slopes, use_alibi_sqrt, use_qq_bias,
            sinks is not None, output_scale, out.stride(), out.dtype,
            cu_seqlens_q.dtype, seqused_k.dtype, max_mm_ranges,
            use_rswa, q_descale, k_descale, v_descale)
    kernel_unified_attention[grid](
'''


def find_target() -> pathlib.Path:
    spec = importlib.util.find_spec(
        "vllm.v1.attention.ops.triton_unified_attention")
    if spec is None or not spec.origin:
        raise SystemExit("FATAL: triton_unified_attention not importable")
    return pathlib.Path(spec.origin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    help="triton_unified_attention.py (default: locate wheel)")
    ap.add_argument("--audit", action="store_true", help="report only")
    args = ap.parse_args()

    path = pathlib.Path(args.path) if args.path else find_target()
    src = path.read_text(encoding="utf-8")
    marks = src.count(MARK)

    if marks:
        if marks == 1:
            print(f"AUDIT: already patched ({marks} mark): {path}")
            return 0
        print(f"FATAL: unexpected mark count {marks} (expected 0 or 1): {path}")
        return 1
    if src.count(ANCHOR) != 1:
        print(f"FATAL: launch anchor not unique ({src.count(ANCHOR)}): {path}")
        return 1

    print(f"AUDIT: pristine (0 marks), anchor unique: {path}")
    if args.audit:
        return 0

    out = src.replace(ANCHOR, TRACE_BLOCK, 1)
    if out.count(MARK) != 1:
        print(f"FATAL: post-patch mark count {out.count(MARK)} != 1")
        return 1
    if out.count(ANCHOR) != 1:
        print("FATAL: anchor disturbed by patch")
        return 1

    backup = path.with_suffix(".py.arcxpubak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(out, encoding="utf-8", newline="")
    py_compile.compile(str(path), doraise=True)
    print(f"PATCHED: 1 hunk, 1 mark, py_compile OK, backup {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
