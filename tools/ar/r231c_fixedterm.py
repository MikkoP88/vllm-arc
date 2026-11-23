#!/usr/bin/env python3
"""r231c_fixedterm.py -- decompose the MX-native decode call's
walk-independent fixed term (r231b verdict next-rung).

Model state (r231a/r231b, build25): the 2K auto call is ~0.51 ms of
which the ns-sweep asymptote (ns32) is ~0.38 -- a per-call constant
that x16 layers = 6.1 ms of the 6.5 ms r230 e2e intercept. This probe
isolates its structure along two axes, xpu-event timed on an idle GPU:

  A. sl ladder at fixed best-ns (32): sl in {1, 64, 256, 1024, 2048}.
     sl=1 walks ~nothing -> the floor = launch + fill + epilogue +
     recover; the slope above it = engaged walk machinery per page.
  B. xpu-graph replay at sl=2048 and sl=1: capture one call, replay
     it -- strips every host cost (allocs, python, plan) the engine's
     ARC_GRAPH=1 posture also strips. replay-vs-direct delta = the
     host share of the "fixed" term (expected small; the r230
     intercept says the term is real GPU work).

Prints R231C_FT rows; exits 0. One payload bank file supplies the
fill bytes (its fmt field wins).
"""

import sys

import torch

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

D = 256
REPS = 50
WARM = 10


def build_inputs(path, dev, sl):
    p = torch.load(path, map_location="cpu", weights_only=False)
    fmt = str(p["fmt"])
    r = int(p["q"].shape[0])
    block_stride = int(p["kv_stride"][0])
    shape = tuple(p["kv_shape"])
    stride = tuple(p["kv_stride"])
    off = int(p["kv_off"])
    bt_head = p["bt_head"].to(torch.long)
    win = p["kv_win"].to(torch.uint8)
    nbt = int(bt_head.numel())
    storage = torch.zeros(
        off + shape[0] * block_stride, dtype=torch.uint8, device=dev)
    for i, b in enumerate(bt_head.tolist()):
        storage[
            off + b * block_stride: off + (b + 1) * block_stride
        ] = win[i].reshape(-1)
    pool = torch.as_strided(storage, shape, stride, off)
    q = p["q"].to(torch.bfloat16).to(dev)
    sl_t = torch.full((r,), sl, dtype=torch.int32, device=dev)
    # block table at the banked engine width; rows walk only sl tokens
    bt = torch.zeros(r, nbt, dtype=p["bt"].dtype, device=dev)
    bt[:, :nbt] = bt_head.unsqueeze(0).repeat(r, 1).to(p["bt"].dtype).to(dev)
    cu_q = torch.arange(r + 1, device=dev, dtype=torch.int32)
    scale = float(p["scale"])
    return fmt, q, pool, storage, sl_t, bt, cu_q, scale, nbt


def time_direct(fn, dev):
    for _ in range(WARM):
        fn()
    torch.xpu.synchronize(dev)
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(REPS):
        fn()
    e.record()
    torch.xpu.synchronize(dev)
    return s.elapsed_time(e) / REPS * 1000.0  # -> us


def time_graph(fn, dev):
    g = torch.xpu.CUDAGraph() if hasattr(torch.xpu, "CUDAGraph") \
        else torch.cuda.CUDAGraph()
    stream = torch.xpu.Stream() if hasattr(torch.xpu, "Stream") \
        else torch.cuda.Stream()
    with torch.xpu.stream(stream) if hasattr(torch.xpu, "stream") \
        else torch.cuda.stream(stream):
        fn()  # warm on the side stream
        torch.xpu.synchronize(dev)
        g.capture_begin()
        fn()
        g.capture_end()
    torch.xpu.synchronize(dev)
    for _ in range(WARM):
        g.replay()
    torch.xpu.synchronize(dev)
    s = torch.xpu.Event(enable_timing=True)
    e = torch.xpu.Event(enable_timing=True)
    s.record()
    for _ in range(REPS):
        g.replay()
    e.record()
    torch.xpu.synchronize(dev)
    return s.elapsed_time(e) / REPS * 1000.0


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    dev = torch.device("xpu")
    print(f"R231C_FT payload={path} reps={REPS}")
    for sl in (1, 64, 256, 1024, 2048):
        fmt, q, pool, storage, sl_t, bt, cu_q, scale, nbt = (
            build_inputs(path, dev, sl))
        max_sk = int(bt.shape[1]) * 64

        def call_ns(
            ns,
            _q=q, _pool=pool, _fmt=fmt, _cu_q=cu_q,
            _sl_t=sl_t, _bt=bt, _max_sk=max_sk, _scale=scale,
        ):
            return lambda: mx_paged_decode(
                _q, _pool, _fmt, _cu_q, _sl_t, _bt,
                max_seqlen_k=_max_sk, softmax_scale=_scale,
                num_splits_kv=ns,
            )

        t_auto = time_direct(call_ns(0), dev)
        t_ns32 = time_direct(call_ns(32), dev)
        t_ns1 = time_direct(call_ns(1), dev)
        print(
            f"R231C_FT sl={sl:5d} auto={t_auto:8.1f}us "
            f"ns32={t_ns32:8.1f}us ns1={t_ns1:9.1f}us"
        )
        if sl in (1, 2048):
            try:
                tg_auto = time_graph(call_ns(0), dev)
                tg_ns32 = time_graph(call_ns(32), dev)
                print(
                    f"R231C_FT sl={sl:5d} GRAPH auto={tg_auto:8.1f}us "
                    f"ns32={tg_ns32:8.1f}us"
                )
            except Exception as ex:  # noqa: BLE001 -- report, keep ladder
                print(f"R231C_FT sl={sl:5d} GRAPH_UNAVAILABLE {type(ex).__name__}")
        del pool, storage
        torch.xpu.empty_cache()
    print("R231C_FT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
