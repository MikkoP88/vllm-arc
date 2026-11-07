#!/usr/bin/env python3
"""arc-xpu-fa2-route — route full-attention head-512 calls to the
vllm_xpu_kernels ESVD FA2 kernel (vllm wheel file
vllm/v1/attention/ops/triton_unified_attention.py).

Defect (P6.28, rig r108aa/r108ab/r108ac/r108ad,
results/2026-11-06-p628-gemma4/): gemma-4's 10 full-attention layers
(16Q/2KV head-512 per rank, no window) run the triton unified-attention
2D prefill path at 10.81 s/call (0.71 TF/s) at q8192/k32768 fp8-KV —
the entire face32k budget. The r108ab sweep closed the launch-config
lane at the ceiling (certified (64,32,8) is optimal; M128 cannot even
launch — SLM), and the fp16-KV control (10.28 s vs 10.81 s at 2x bytes)
refuted byte-boundedness: the kernel is slow per K-tile iteration. The
engine ships a second attention kernel — the vllm_xpu_kernels ESVD FA2
kernel (_vllm_fa2_C.abi3.so) — but the triton backend never dispatches
to it, so gemma-4 never reaches it.

Fix shape = a wrapper-level dispatch in unified_attention (same surface
as the cert12 launcher table, which tunes launch params INSIDE the
triton path): when a call is full-attention head-512 (window left < 0,
scalar causal, per-tensor fp8-KV with scalar descales or fp16-KV, paged
block_table, and none of the triton-only features are in play), call
vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func directly
(same paged KV layout [num_blocks, block_size, num_kv_heads, head_size]
the triton path already consumes; the engine's 1-element-storage scalar
descale views pass its _normalize_descale_tensor exactly). Every other
call — sliding windows, INT4 / per-token-head KV, alibi, sinks,
mm_prefix, rswa, chunked attention, per-seq causal, q_descale — stays
on the triton kernel untouched.

B70 evidence (r108ac/r108ad, phaseC/micro_fa2ab/): FA2 k32k fp8-KV
softcap-50 = 0.3197 s/call (24.1 TF/s) vs triton 10.81 s = 34x;
k98k 1.06 s ~ 30x; decode S8x32k 1.5 ms/call; sliding hd256 win1023
0.0087 s. Head-512 + fp8-e4m3-KV + descale + softcap is compiled in
the stock _vllm_fa2_C.abi3.so (no "not compiled" fallback). Numeric
guards vs a GQA torch reference: green (r108ad).

Descale form (r108ae boot crash, fixed v3): vxk _normalize_descale_tensor
asserts the descale has 1-element storage. The engine path passes
layer._k_scale.expand(...) (scalar storage, passes through untouched);
the cert12 warm gate passes torch.ones(nkvh).expand((1, nkvh)) —
nkvh-element storage, rejected. The dispatch normalizes full-storage
callers by cloning element 0 into fresh storage (exact for per-tensor
scales — per-token-head modes are guarded off).

Env ARC_ATTN_FA2 (default 1) restores the pure-triton path when set
to 0 (diagnostic escape hatch).

Idempotent (skips when the mark is present); writes .arcxpubak backup.

Usage:
    patch_arc_xpu_fa2_route.py [ops.py]        # patch
    patch_arc_xpu_fa2_route.py [ops.py] --audit
    (no path: locate vllm on sys.path — the image build does this)
"""
import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

MARK = "[ARC-P628-FA2]"
REL = "vllm/v1/attention/ops/triton_unified_attention.py"

# --- byte-exact anchors of the cert17 image file (cert11 wheel +
# patch_arc_xpu_attn_launcher.py applied — anchors verified unique in
# the pulled image copy, r108ac prep) ---
# module top, right after fp8 dtype info (same surface the cert12 kit
# patch anchors on; the cert12 table block follows after it)
ANCHOR_MODULE = "float8_info = torch.finfo(current_platform.fp8_dtype())\n"
# end of the INT4 early path inside unified_attention — the dispatch goes
# between the INT4 return and the sinks assert
ANCHOR_DISPATCH = (
    "            softmax_segm_expsum=softmax_segm_expsum,\n"
    "        )\n"
    "        return\n"
)

MODULE_BLOCK = f'''

# {MARK} ESVD FA2 route enable --------------------------------------
# Full-attention head-512 calls (gemma-4 global layers: window left < 0,
# scalar causal, per-tensor fp8-KV scalar descales or fp16-KV, paged
# block_table) route to the vllm_xpu_kernels ESVD FA2 kernel — 34x
# prefill / ~30x k98k / 1.5 ms-class decode vs the triton 2D path at
# production geometry (r108ac/r108ad,
# results/2026-11-06-p628-gemma4/phaseC/micro_fa2ab/). Env
# ARC_ATTN_FA2=0 restores the pure-triton path (escape hatch).
# NOTE: the XPU check is LAZY (called per dispatch) — platform detection
# in a GPU-less build container reports not-XPU, but the engine
# initializes the ARC platform before any model runs (same lazy pattern
# the cert12 table's dispatch-time guards rely on).
import os as _arc_fa2_os

_ARC_FA2_ROUTE = _arc_fa2_os.environ.get("ARC_ATTN_FA2", "1") == "1"
_ARC_FA2_IS_XPU = current_platform.is_xpu
_ARC_FA2_LOGGED = False


def _arc_fa2_scalar(ds):
    # vxk FA2 requires the descale to be a 1-element-storage scalar view
    # (per-tensor scale). The engine path passes layer._k_scale.expand
    # (scalar storage, passes through); the cert12 warm gate passes
    # torch.ones(nkvh).expand((1, nkvh)) — nkvh-element storage, rejected
    # (r108ae boot crash at the first warm hd512 fp8 point). Per-tensor
    # scales are uniform by construction (per-token-head modes are
    # guarded off in the dispatch), so element 0 into fresh storage is
    # exact.
    if ds is None or ds.untyped_storage().nbytes() == ds.element_size():
        return ds
    return ds.reshape(-1)[:1].clone()
'''

DISPATCH_BLOCK = f'''
    # {MARK} full-attention head-512 route (gemma-4 global layers) to the
    # vllm_xpu_kernels ESVD FA2 kernel: the triton 2D prefill path is
    # 10.81 s/call (0.71 TF/s) at q8192/k32768 fp8-KV per rank while FA2
    # does the same call in 0.32 s (24.1 TF/s) = 34x (r108ac; config lane
    # closed at ceiling per r108ab — the kernel, not the config, is the
    # defect). Every feature the ESVD kernel does not model keeps triton:
    # sliding windows, INT4 / per-token-head KV, alibi, sinks, mm_prefix,
    # rswa, chunked attention, per-seq causal, q_descale.
    if (_ARC_FA2_ROUTE and _ARC_FA2_IS_XPU()
            and use_causal and not use_per_seq_causal
            and block_table is not None and q.shape[-1] == 512
            and window_size[0] < 0 and sinks is None
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
            flash_attn_varlen_func as _arc_fa2)
        global _ARC_FA2_LOGGED
        if not _ARC_FA2_LOGGED:
            _ARC_FA2_LOGGED = True
            print("[ARC-P628-FA2] hd512 full-attn -> vxk FA2 "
                  "(r108ac 34x prefill)", flush=True)
        _arc_fa2(q=q, k=k, v=v, out=out, max_seqlen_q=max_seqlen_q,
                 cu_seqlens_q=cu_seqlens_q, max_seqlen_k=max_seqlen_k,
                 seqused_k=seqused_k, softmax_scale=softmax_scale,
                 causal=True, window_size=None, block_table=block_table,
                 softcap=softcap, k_descale=_arc_fa2_scalar(k_descale),
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
        "anchor_module": ANCHOR_MODULE in src,
        "anchor_dispatch": ANCHOR_DISPATCH in src,
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
    out = out.replace(ANCHOR_DISPATCH, ANCHOR_DISPATCH + DISPATCH_BLOCK, 1)
    # 2 comment insertions + 1 runtime boot marker in the dispatch print
    if out.count(MARK) != 3:
        print(f"FATAL: expected 3 {MARK} occurrences (2 comments + boot "
              f"marker), got {out.count(MARK)}")
        return 1
    shutil.copy2(path, str(path) + ".arcxpubak")
    path.write_text(out, encoding="utf-8", newline="")
    print(f"PATCHED: 2 insertions ({len(out) - len(src)} bytes) "
          f"+ backup {path.name}.arcxpubak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
