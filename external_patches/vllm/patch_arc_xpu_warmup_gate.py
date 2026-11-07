#!/usr/bin/env python3
"""ARC P6.28 kit patch #2: rank-serialized boot warmup for the tuned
triton prefill kernels (companion to patch_arc_xpu_attn_launcher.py).

Why (r108o, banked in results/2026-11-06-p628-gemma4/phaseC/
hang_capture_c12/): with the ARC prefill launch-config table live, a
tuned config is a kernel specialization the engine has never loaded.
Its first module creation (CompiledKernel._init_handles ->
SpirvUtils.load_binary -> zeModuleCreate + IGC compile) wedges BOTH TP
ranks when they reach it inside a booted engine — identical py-spy
stacks on both ranks, GIL held, ~166% CPU each, indefinitely: in the
certified graph posture the wedge lands on the first real prefill
request (98-token warmup, engine rpc watchdog kills the step = the
"engine freeze"); in the default eager posture it lands inside
compile_or_warm_up_model's dummy run ("Compile and warming up model for
size 8192"). Single-process loads of the exact same binaries are clean
(r108n micro). vLLM's own jit-monitor names the fix class at warn time:
"consider extending warmup to cover this shape/config".

What this patch does: inserts one module-level helper pair plus a single
guarded call at the top of the stock kernel-warmup section of
Worker.compile_or_warm_up_model (vllm/v1/worker/gpu_worker.py, right
before ``kernel_warmup(self)`` — the wheel's own "warmup and tune the
kernels ... before cuda graph capture" section). The helper:

  * no-ops unless the ARC tune table is present (stock wheels) and the
    loaded model actually has attention geometries the table covers;
  * walks the model's attention modules to collect the real per-rank
    (head_dim, num_heads, num_kv_heads) tuples — no model-name
    hardcoding;
  * loads every (geometry, band, divisibility, window-style) kernel a
    serving step can produce: high band = max_num_batched_tokens chunk,
    mid band = arbitrary remainder (98-class), each against div16 and
    non-div16 KV lengths, for full (-1,-1) and sliding (1023,0) windows;
  * serializes the loads across TP ranks (rank 0 first, barrier between
    ranks) so no two ranks ever module-create concurrently;
  * runs before cuda-graph capture and before the stock kernel_warmup,
    is fail-loud on error, and frees its transient tensors afterwards.

Usage:
    python patch_arc_xpu_warmup_gate.py [path/to/gpu_worker.py] [--audit]

Without a path, the wheel location is found via importlib. --audit only
reports. Patches are idempotent (mark count checked); a .arcxpubak
backup is left next to the original.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import py_compile
import shutil
import sys

MARK = "[ARC-P628-XPU-WARMUP]"

# Anchor 1 — module level: right after the kernel_warmup import (unique
# line in the stock wheel).
ANCHOR_MODULE = "from vllm.model_executor.warmup.kernel_warmup import kernel_warmup\n"

# Anchor 2 — call site: the stock warmup section inside
# compile_or_warm_up_model, immediately before kernel_warmup(self).
ANCHOR_CALL = (
    "        # Warmup and tune the kernels used during model execution before\n"
    "        # cuda graph capture.\n"
    "        kernel_warmup(self)\n"
)

HELPER_BLOCK = '''

def _arc_xpu_warm_points(worker, geoms) -> None:
    """Load one kernel module per (geometry, band, divisibility, window)
    combination a serving prefill produces (see triton_unified_attention
    _ARC_XPU_PREFILL_CONFIGS), replicating TritonAttentionImpl.forward's
    call pattern EXACTLY: the backend allocates the softmax segment
    tensors unconditionally and passes them on every call, XPU enables
    tensor-descriptor mode by default, per-head descales arrive as
    expanded views, and Gemma-4-style layers pair the sliding window
    with mm_prefix_clamp_sliding_window=True. r108q proved a warm that
    omits any of these leaves a fresh variant for the first request to
    load — which wedges. Points: high band = max_num_batched_tokens
    chunk (div16 query), mid band = 98-class chunk remainder (non-div16),
    each against div16 and non-div16 KV lengths, for full and sliding
    windows."""
    import torch

    from vllm.v1.attention.ops import triton_unified_attention as _tua

    mnbt = 8192
    try:
        mnbt = int(worker.vllm_config.scheduler_config.max_num_batched_tokens)
    except Exception:
        pass
    mnbt = max(128, min(mnbt, 32768))
    mid = min(98, mnbt - 1)
    nseg = 16  # triton_attn.NUM_PAR_SOFTMAX_SEGMENTS
    points = (
        (mnbt, mnbt),      # high band: q div16, k div16
        (mnbt, mnbt + 1),  # high band: q div16, k non-div16
        (mid, mid),        # mid band: q non-div16, k non-div16
        (mid, mnbt),       # mid band over long div16 KV (prefix cache)
    )
    # (window, mm_prefix_clamp_sliding_window) — engine pairing: sliding
    # layers set the clamp flag, full-attention layers do not (gemma4.py).
    windows = (((-1, -1), False), ((1023, 0), True))

    dev = torch.device("xpu", torch.xpu.current_device())
    for head, nqh, nkvh in geoms:
        head_pow2 = 1 << (head - 1).bit_length()
        segm_out = torch.empty((16, nqh, nseg, head_pow2),
                               dtype=torch.float32, device=dev)
        segm_max = torch.empty((16, nqh, nseg),
                               dtype=torch.float32, device=dev)
        segm_sum = torch.empty((16, nqh, nseg),
                               dtype=torch.float32, device=dev)
        kd = torch.ones(nkvh, device=dev).expand((1, nkvh))
        vd = torch.ones(nkvh, device=dev).expand((1, nkvh))
        for ntok, seqlen in points:
            nblk = (seqlen + 63) // 64
            q = torch.randn(ntok, nqh, head, dtype=torch.float16, device=dev)
            kc = torch.randn(nblk, 64, nkvh, head, dtype=torch.float16,
                             device=dev).to(torch.float8_e4m3fn)
            vc = torch.randn(nblk, 64, nkvh, head, dtype=torch.float16,
                             device=dev).to(torch.float8_e4m3fn)
            out = torch.empty_like(q)
            bt = torch.arange(nblk - 1, -1, -1, dtype=torch.int32,
                              device=dev).reshape(1, nblk)
            cuq = torch.tensor([0, ntok], dtype=torch.int32, device=dev)
            sq = torch.tensor([seqlen], dtype=torch.int32, device=dev)
            for win, clamp in windows:
                logger.info(
                    "ARC XPU tune warmup: load head=%d nqh=%d nkvh=%d "
                    "ntok=%d kv=%d win=%s", head, nqh, nkvh, ntok, seqlen, win)
                _tua.unified_attention(
                    q=q, k=kc, v=vc, out=out, cu_seqlens_q=cuq,
                    max_seqlen_q=ntok, seqused_k=sq, max_seqlen_k=seqlen,
                    softmax_scale=head ** -0.5, causal=True, alibi_slopes=None,
                    window_size=win, block_table=bt, softcap=0.0,
                    q_descale=None, kv_quant_mode=1,
                    k_descale=kd, v_descale=vd,
                    seq_threshold_3D=16, num_par_softmax_segments=nseg,
                    softmax_segm_output=segm_out, softmax_segm_max=segm_max,
                    softmax_segm_expsum=segm_sum, sinks=None,
                    output_scale=None, mm_prefix_range=None,
                    use_td=True, mm_prefix_clamp_sliding_window=clamp)
                torch.xpu.synchronize()
            del q, kc, vc, out, bt, cuq, sq
        del segm_out, segm_max, segm_sum, kd, vd
    torch.xpu.empty_cache()


def _arc_xpu_warm_tuned_prefill_kernels(worker) -> None:
    """''' + MARK + ''' rank-serialized pre-load of the tuned prefill
    kernels (companion of the _ARC_XPU_PREFILL_CONFIGS table). The first
    in-engine module creation of a tuned config wedges both TP ranks in
    SpirvUtils.load_binary (r108o/r108q: identical py-spy stacks; the
    wedge strikes whenever both ranks concurrently load a variant that is
    fresh for the engine — solo and cache-hit loads are clean, so the
    warm loads every engine-exact variant rank-by-rank inside
    compile_or_warm_up_model, before the stock kernel_warmup and
    cuda-graph capture). No-op when the table is absent (stock wheel) or
    the model has no covered geometry."""
    import torch

    if not getattr(torch, "xpu", None) or not torch.xpu.is_available():
        return
    try:
        from vllm.v1.attention.ops import triton_unified_attention as _tua
    except Exception:
        return
    table = getattr(_tua, "_ARC_XPU_PREFILL_CONFIGS", None)
    if not table:
        return
    model = getattr(worker.model_runner, "model", None)
    if model is None:
        return

    geoms: dict = {}
    for mod in model.modules():
        nq = getattr(mod, "num_heads", None)
        nkv = getattr(mod, "num_kv_heads", None)
        hd = getattr(mod, "head_dim", None)
        if (isinstance(nq, int) and isinstance(nkv, int)
                and isinstance(hd, int) and nq > 0 and 0 < nkv <= nq
                and nq % nkv == 0 and hd in table):
            geoms[(hd, nq, nkv)] = True
    if not geoms:
        return

    import torch.distributed as dist
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    t0 = time.time()
    try:
        for r in range(world):
            if r == rank:
                _arc_xpu_warm_points(worker, sorted(geoms))
            if world > 1:
                dist.barrier()
    except Exception:
        logger.exception("ARC XPU tune warmup FAILED; refusing to serve on "
                         "an engine whose tuned kernels are not loaded")
        raise
    logger.info("ARC XPU tune warmup complete on rank %d/%d in %.1fs "
                "(%d geometries)", rank, world, time.time() - t0, len(geoms))

'''

CALL_BLOCK = (
    "        # " + MARK + " Pre-load the ARC tuned triton prefill kernels\n"
    "        # before any other warmup or capture can reach them: the first\n"
    "        # in-engine module load of a tuned config wedges both TP ranks\n"
    "        # in SpirvUtils.load_binary (r108o); loading during worker\n"
    "        # warmup is what the jit-monitor prescribes. No-op on stock\n"
    "        # wheels and uncovered geometries.\n"
    "        _arc_xpu_warm_tuned_prefill_kernels(self)\n"
)


def find_gpu_worker() -> pathlib.Path:
    spec = importlib.util.find_spec("vllm.v1.worker.gpu_worker")
    if spec is None or not spec.origin:
        raise SystemExit("FATAL: vllm.v1.worker.gpu_worker not importable")
    return pathlib.Path(spec.origin)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="gpu_worker.py (default: locate wheel)")
    ap.add_argument("--audit", action="store_true", help="report only")
    args = ap.parse_args()

    path = pathlib.Path(args.path) if args.path else find_gpu_worker()
    src = path.read_text(encoding="utf-8")
    marks = src.count(MARK)

    if marks:
        if marks == 2:
            print(f"AUDIT: already patched ({marks} marks): {path}")
            return 0
        print(f"FATAL: unexpected mark count {marks} (expected 0 or 2): {path}")
        return 1
    if src.count(ANCHOR_MODULE) != 1:
        print(f"FATAL: module anchor not unique: {path}")
        return 1
    if src.count(ANCHOR_CALL) != 1:
        print(f"FATAL: call anchor not unique: {path}")
        return 1

    print(f"AUDIT: pristine (0 marks), anchors unique: {path}")
    if args.audit:
        return 0

    out = src.replace(ANCHOR_MODULE, ANCHOR_MODULE + HELPER_BLOCK, 1)
    out = out.replace(ANCHOR_CALL, CALL_BLOCK + ANCHOR_CALL, 1)

    if out.count(MARK) != 2:
        print(f"FATAL: post-patch mark count {out.count(MARK)} != 2")
        return 1
    if out.count(ANCHOR_MODULE) != 1 or out.count(ANCHOR_CALL) != 1:
        print("FATAL: anchors disturbed by patch")
        return 1

    backup = path.with_suffix(".py.arcxpubak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(out, encoding="utf-8", newline="")
    py_compile.compile(str(path), doraise=True)
    print(f"PATCHED: 2 hunks, 2 marks, py_compile OK, backup {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
