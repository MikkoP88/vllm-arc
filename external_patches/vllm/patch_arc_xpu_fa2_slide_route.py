#!/usr/bin/env python3
"""arc-xpu-fa2-slide-route — route sliding-window head-256 calls to the
vllm_xpu_kernels ESVD FA2 kernel (vllm wheel file
vllm/v1/attention/ops/triton_unified_attention.py, CERT18 v3 state).

Defect (P6.29, rig r109a/r109a3,
results/2026-11-07-p629/phaseA/): after cert18 routed gemma-4's
10 full-attention hd512 layers to FA2 (34x), the remaining prefill wall
is the 50/60 SLIDING hd256 layers (16Q/8KV per rank, window (1023,0)):
the triton unified-attention path costs 0.147-0.158 s/call at q8192
fp8-KV, k-INDEPENDENT (0.1466 k8192 / 0.1580 k32k / 0.1576 k98k,
r109a3) while the same vxk ESVD FA2 kernel windowed runs 0.006-0.009 s
(window-bounded) = 18-25x; 50 layers x (0.155 - 0.007) s ~= 7.4 s per
q8192 chunk — the bulk of face8k's 14.2 s and the dominant share of
face32k/face120k. Decode S8x32k: triton 4.38 ms vs FA2 0.3-0.4 ms
= 11-15x (r109a/r109a3 anchors).

Fix shape = a SECOND wrapper-level dispatch in unified_attention,
inserted immediately after the cert18 hd512-full dispatch (same surface,
same kit law): when a call is sliding-window head-256 (window_size[0] >=
0, window_size[1] <= 0 — the causal sliding form the backend passes as
(w-1, 0); the bidirectional encoder form (w-1, w-1) keeps triton), with
the paged block_table, scalar causal, none of the triton-only features
in play, it calls flash_attn_varlen_func with
window_size=[left, 0] — the r108ac/r109a-proven call form — sharing the
cert18 module block (_ARC_FA2_ROUTE env gate, lazy _ARC_FA2_IS_XPU,
_arc_fa2_scalar descale normalization). Every other call keeps triton.

Numeric guards (r109a/r109a2/r109a3): kernels agree 1.2e-4..9.8e-4
max-abs on the windowed class (fp16/fp8/bf16-q, prefill and decode);
vs GLOBAL-softmax table-faithful window-masked GQA torch references
(query i at absolute position seqlen-ntok+i; K/V gathered through the
block table) both kernels match at 1.3e-4..1.1e-3 prefill / 8e-5
decode (r109a6/r109a3). The r108ad..r109a4 ref misses (~4, full class
too) were a harness bug — those refs normalized softmax per 512-chunk —
not a kernel issue; softcap eliminated separately (the triton wrapper
cannot receive None: TypeError, r109a5; production passes 0.0 and
USE_SOFTCAP=(0.0 > 0) skips it).

Env ARC_ATTN_FA2=0 (the cert18 gate) disables BOTH routes (shared
module flag). Boot marker [ARC-P629-SLIDE] prints once per process on
first sliding dispatch.

Operates ONLY on the cert18 v3 state (asserts the v3 dispatch anchor);
idempotent on this patch's mark; writes .arcxpubak2 backup.

Usage:
    patch_arc_xpu_fa2_slide_route.py [ops.py]        # patch
    patch_arc_xpu_fa2_slide_route.py [ops.py] --audit
    (no path: locate vllm on sys.path — the image build does this)
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

MARK = "[ARC-P629-SLIDE]"
V3_MARK = "[ARC-P628-FA2]"
REL = "vllm/v1/attention/ops/triton_unified_attention.py"

# cert18 v3 state anchors (byte-exact — v3 wrote these):
# module tail: the slide logger binds right after the v3 logger
ANCHOR_MODULE = "_ARC_FA2_LOGGED = False\n"
# dispatch tail: the v3 hd512-full route's fa2 call + return — the slide
# dispatch inserts immediately after it (before the sinks assert)
ANCHOR_V3_TAIL = (
    "        _arc_fa2(q=q, k=k, v=v, out=out, max_seqlen_q=max_seqlen_q,\n"
    "                 cu_seqlens_q=cu_seqlens_q, max_seqlen_k=max_seqlen_k,\n"
    "                 seqused_k=seqused_k, softmax_scale=softmax_scale,\n"
    "                 causal=True, window_size=None, block_table=block_table,\n"
    "                 softcap=softcap, k_descale=_arc_fa2_scalar(k_descale),\n"
    "                 v_descale=_arc_fa2_scalar(v_descale))\n"
    "        return\n"
)

MODULE_BLOCK = (
    "_ARC_SLIDE_LOGGED = False\n"
)

DISPATCH_BLOCK = f'''
    # {MARK} sliding-window head-256 route (gemma-4's 50/60 layers,
    # window (1023,0) class) to the same vxk ESVD FA2 kernel, windowed:
    # the triton path costs 0.156 s/call at q8192/k32k fp8 while FA2
    # windowed runs 0.006-0.009 s k-independent (18-25x; r109a/r109a3)
    # and decode S8x32k 0.3-0.4 ms — the remaining prefill wall after
    # cert18. Guards: hd256 + causal sliding window (left >= 0,
    # right <= 0; the bidirectional encoder form (w-1, w-1) keeps
    # triton) + every cert18 triton-only exclusion. Window passes to
    # vxk as [left, 0] (the r108ac/r109a-proven call form).
    if (_ARC_FA2_ROUTE and _ARC_FA2_IS_XPU()
            and use_causal and not use_per_seq_causal
            and block_table is not None and q.shape[-1] == 256
            and 0 <= window_size[0] and window_size[1] <= 0
            and sinks is None
            and mm_prefix_range is None and rswa_prefix_lens is None
            and rswa_window is None and chunk_lookback < 0
            and alibi_slopes is None and not use_alibi_sqrt
            and q_descale is None and output_scale is None
            and qq_bias is None and k_scale_cache is None
            and v_scale_cache is None
            and (k_descale is None) == (v_descale is None)
            and kv_quant_mode not in (KVQuantMode.INT4_PER_TOKEN_HEAD,
                                      KVQuantMode.INT8_PER_TOKEN_HEAD,
                                      KVQuantMode.FP8_PER_TOKEN_HEAD)):
        from vllm_xpu_kernels.flash_attn_interface import (
            flash_attn_varlen_func as _arc_fa2w)
        global _ARC_SLIDE_LOGGED
        if not _ARC_SLIDE_LOGGED:
            _ARC_SLIDE_LOGGED = True
            print("[ARC-P629-SLIDE] hd256 sliding -> vxk FA2 windowed "
                  "(r109 18x prefill)", flush=True)
        _arc_fa2w(q=q, k=k, v=v, out=out, max_seqlen_q=max_seqlen_q,
                  cu_seqlens_q=cu_seqlens_q, max_seqlen_k=max_seqlen_k,
                  seqused_k=seqused_k, softmax_scale=softmax_scale,
                  causal=True,
                  window_size=[int(window_size[0]), 0],
                  block_table=block_table, softcap=softcap,
                  k_descale=_arc_fa2_scalar(k_descale),
                  v_descale=_arc_fa2_scalar(v_descale))
        return

'''


def find_ops() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise SystemExit("FATAL: vllm not importable — pass the ops file "
                         "path explicitly")
    pkg = Path(spec.origin).parent
    target = pkg / "v1/attention/ops/triton_unified_attention.py"
    if not target.is_file():
        raise SystemExit(f"FATAL: {target} not found — layout changed?")
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ops", nargs="?",
                    help=f"path to {REL} (default: locate vllm on sys.path)")
    ap.add_argument("--audit", action="store_true",
                    help="report anchor/mark state without patching")
    args = ap.parse_args()
    path = Path(args.ops) if args.ops else find_ops()

    src = path.read_text(encoding="utf-8")
    print(f"target: {path}")
    state = {
        "mark": MARK in src,
        "v3_mark": V3_MARK in src,
        "anchor_module": ANCHOR_MODULE in src,
        "anchor_v3_tail": ANCHOR_V3_TAIL in src,
    }
    for k, v in state.items():
        print(f"  {k}: {'present' if v else 'MISSING'}")

    if state["mark"]:
        print("ALREADY_PATCHED")
        return 0
    if args.audit:
        missing = [k for k, v in state.items()
                   if not v and k != "mark"]
        print("AUDIT:", "PATCHABLE (cert18 v3 state, all anchors present)"
              if not missing else f"NOT PATCHABLE (missing: {missing})")
        return 0 if not missing else 1
    # hard preconditions: the file MUST be in cert18 v3 state (the slide
    # route builds on the v3 module block: env gate, lazy XPU check,
    # _arc_fa2_scalar)
    if not (state["v3_mark"] and state["anchor_module"]
            and state["anchor_v3_tail"]):
        print("FATAL: file is not in the cert18 v3 state — apply "
              "patch_arc_xpu_fa2_route.py first (base must be cert18)")
        return 1

    out = src.replace(ANCHOR_MODULE, ANCHOR_MODULE + MODULE_BLOCK, 1)
    out = out.replace(ANCHOR_V3_TAIL, ANCHOR_V3_TAIL + DISPATCH_BLOCK, 1)
    # 1 comment + 1 runtime boot marker in the slide block
    if out.count(MARK) != 2:
        print(f"FATAL: expected 2 {MARK} occurrences (comment + boot "
              f"marker), got {out.count(MARK)}")
        return 1
    # the cert18 v3 marks must carry unchanged (2 comments + boot marker)
    if out.count(V3_MARK) != 3:
        print(f"FATAL: v3 marks drifted: expected 3 {V3_MARK}, "
              f"got {out.count(V3_MARK)}")
        return 1
    shutil.copy2(path, str(path) + ".arcxpubak2")
    path.write_text(out, encoding="utf-8", newline="")
    print(f"PATCHED: 2 insertions ({len(out) - len(src)} bytes) "
          f"+ backup {path.name}.arcxpubak2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
