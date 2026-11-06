#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ar_overlap_bench.py — P6.25 r101: the overlap-door microbench (ceiling-first).

ONE question: can the row-parallel AR (store-bound on the wire, EUs idle)
ride a side stream concurrently with GEMM compute, and does k-sub-batch
interleaving then take it off the layer critical path? r83 closed overlap
ON ONE STREAM (queue never idle); r100 closed fabric rd+wr concurrency and
AR-INTERNAL pipelining; neither closed compute-vs-wire co-tenancy.

Modes (torchrun --standalone --nproc-per-node 2, backend xccl — the serve
engine's own backend; real qwen3.8-27b TP2 per-rank shapes, M=7488 = the
dominant prefill chunk; hidden 5120, intermediate 17408, 24/4 heads × 256):

  async    host-enqueue latency of dist.all_reduce (does it block? overlap
           is dead on arrival if the call host-syncs)
  census   the four per-rank linear GEMMs solo — TF/s vs the P6.21b
           measured fp16 dense peak (73.5 TF/s/rank): GEMM-lane close or
           follow-on
  ar       dist.all_reduce([M, H] fp16) solo — stock vs FP8-wire (env set
           by the runner leg)
  coten    GEMM loop on the compute stream + AR loop on a side stream;
           per-side inflation factor vs solo = the co-tenancy answer
  mock     L-layer mock block {qkv -> o -> AR -> gate_up -> down -> AR}
           (attention abstracted: the overlap target is the AR, FA is a
           separate falsified lane): serial vs k=2/3/4 sub-batch
           interleaved (side stream + events, per-sub row split — the
           row-parallel AR is per-token independent, so the split is
           numerics-transparent). Two-phase schedule per layer = the
           LOWER bound of achievable overlap (r102 schedules tighter).

PRE-REGISTERED GATE (plan §P6.25): best-k mock overlap >= 1.08x serial
-> r102 in-situ prototype; else the overlap door closes measured.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

H = 5120
I = 17408
I_HALF = 8704   # 17408 / TP2 — GLU output width = down_proj input
QKV_N = 4096    # (24*256 q + 2*4*256 kv) / TP2
O_IN = 3072     # 12*256 local attention out
GU_N = 17408    # 2 * 17408 / TP2 (gate || up concat)
D_IN = 8704     # 17408 / TP2
PEAK_TFS = 150.0  # r101-measured practical dense fp16 ceiling, rank-local
# (the earlier 73.5/rank assumption was WRONG — P6.21b's "147 TF/s dual"
#  phrasing was per-card; this census's own torch.matmul = the practical
#  dense ceiling reference, so pct_peak ≈ 100% means "no GEMM headroom")

GEMMS = {  # name: (K, N) at M rows — the per-rank prefill linears
    "qkv": (H, QKV_N),
    "o": (O_IN, H),
    "gate_up": (H, GU_N),
    "down": (D_IN, H),
}


def med(fn, iters=7, warmup=3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    evs = []
    for _ in range(iters):
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        evs.append((s, e))
    torch.xpu.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in evs])


class W:
    """One decoder layer's per-rank weights (fp16, ~340 MB)."""

    def __init__(self, dev, seed=0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        def rnd(r, c):
            return ((torch.randn(r, c, generator=g) * 0.02)
                    .to(device=dev, dtype=torch.float16))
        self.qkv = rnd(H, QKV_N)
        self.o = rnd(O_IN, H)
        self.gu = rnd(H, GU_N)
        self.d = rnd(D_IN, H)


def layer_serial(x, w, ar):
    """k=1 reference: the layer critical path as it runs today.
    Attention abstracted as the q-slice pass-through (qkv output cols
    [:3072] = local q heads; FA is a separate, already-falsified lane —
    this mock isolates the GEMM+AR structure only)."""
    q = x @ w.qkv
    o = q[:, :O_IN] @ w.o
    ar(o)
    g = o @ w.gu                       # [M, 17408] = gate || up
    act = torch.nn.functional.silu(g[:, :I_HALF]) * g[:, I_HALF:]
    d = act @ w.d
    ar(d)
    return d


def layer_overlap(x, w, ar, k, s_comm):
    """k-sub-batch interleaved: AR(sub-i) on s_comm flies while the compute
    stream runs the next sub's GEMMs. Row split only — every element's
    reduction order is unchanged (per-token independent AR)."""
    M = x.shape[0]
    ms = M // k
    # phase 1: qkv+o per sub on compute; AR(o_i) on comm as each lands
    pend = []
    for i in range(k):
        xi = x[i * ms:(i + 1) * ms]
        oi = (xi @ w.qkv)[:, :O_IN] @ w.o
        e = torch.xpu.Event()
        e.record()
        with torch.xpu.stream(s_comm):
            s_comm.wait_event(e)
            ar(oi)
            e2 = torch.xpu.Event()
            e2.record()
        pend.append((oi, e2))
    # phase 2: gate_up+down per sub on compute; AR(d_i) on comm —
    # AR(d_i) overlaps the compute of sub i+1
    outs = []
    for oi, e2 in pend:
        torch.xpu.current_stream().wait_event(e2)
        g = oi @ w.gu
        di = (torch.nn.functional.silu(g[:, :I_HALF]) * g[:, I_HALF:]) @ w.d
        e3 = torch.xpu.Event()
        e3.record()
        with torch.xpu.stream(s_comm):
            s_comm.wait_event(e3)
            ar(di)
            e4 = torch.xpu.Event()
            e4.record()
        outs.append((di, e4))
    for _, e4 in outs:
        torch.xpu.current_stream().wait_event(e4)
    return torch.cat([d for d, _ in outs])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True,
                   choices=["async", "census", "ar", "coten", "mock"])
    p.add_argument("--out", required=True)
    p.add_argument("--m", type=int, default=7488)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--label", default="stock")
    args = p.parse_args()

    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    torch.xpu.set_device(rank)
    dist.init_process_group(backend="xccl")  # the serve path's own backend

    def ar(t):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    dev = f"xpu:{rank}"
    rep = {"label": args.label, "rank": rank, "m": args.m, "mode": args.mode}

    if args.mode == "async":
        x = torch.randn(args.m, H, device=dev, dtype=torch.float16)
        ar(x); torch.xpu.synchronize()
        ts = []
        for _ in range(9):
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            ar(x)
            ts.append((time.perf_counter() - t0) * 1e3)
        rep["enqueue_ms_med"] = round(statistics.median(ts), 3)
        rep["verdict"] = ("async-enqueue (overlap viable at the API level)"
                          if statistics.median(ts) < 1.0 else
                          "HOST-BLOCKS (overlap dead at the API level)")

    elif args.mode == "census":
        rows = {}
        for name, (kk, nn) in GEMMS.items():
            a = torch.randn(args.m, kk, device=dev, dtype=torch.float16) * 0.1
            b = torch.randn(kk, nn, device=dev, dtype=torch.float16) * 0.1
            t = med(lambda a=a, b=b: a @ b)
            tfs = 2 * args.m * kk * nn / (t * 1e-3) / 1e12
            rows[name] = {"ms": round(t, 3), "tflops": round(tfs, 1),
                          "pct_peak": round(100 * tfs / PEAK_TFS, 1)}
        rep["gemms"] = rows
        tot_flop = sum(2 * args.m * kk * nn for kk, nn in GEMMS.values())
        tot_ms = sum(r["ms"] for r in rows.values())
        rep["layer_gemm_ms"] = round(tot_ms, 3)
        rep["layer_gemm_tfs"] = round(tot_flop / (tot_ms * 1e-3) / 1e12, 1)
        rep["pct_peak"] = round(100 * rep["layer_gemm_tfs"] / PEAK_TFS, 1)

    elif args.mode == "ar":
        # numerics guard lives in ar_ceiling.py (r95) — here: rate only
        x = torch.randn(args.m, H, device=dev, dtype=torch.float16)
        ar(x); torch.xpu.synchronize()
        rep["ar_ms"] = round(med(lambda: ar(x)), 3)

    elif args.mode == "coten":
        a = torch.randn(args.m, H, device=dev, dtype=torch.float16) * 0.1
        b = torch.randn(H, GU_N, device=dev, dtype=torch.float16) * 0.1
        x = torch.randn(args.m, H, device=dev, dtype=torch.float16)
        solo_g = med(lambda: a @ b, iters=9)
        solo_a = med(lambda: ar(x), iters=9)
        s_comp = torch.xpu.current_stream()
        s_comm = torch.xpu.Stream()
        R = 9
        # concurrent region: GEMM loop on compute, AR loop on comm (forked
        # via event; BOTH start/end events recorded ON THEIR OWN stream —
        # elapsed_time is only valid within one stream)
        torch.xpu.synchronize()
        gs = torch.xpu.Event(enable_timing=True)
        ge = torch.xpu.Event(enable_timing=True)
        cs = torch.xpu.Event(enable_timing=True)
        ce = torch.xpu.Event(enable_timing=True)
        fork = torch.xpu.Event()
        gs.record(s_comp)
        fork.record(s_comp)
        with torch.xpu.stream(s_comm):
            s_comm.wait_event(fork)
            cs.record()  # current stream here = s_comm
            for _ in range(R):
                ar(x)
            ce.record()
        for _ in range(R):
            (a @ b)
        ge.record(s_comp)
        torch.xpu.synchronize()
        g_ms = gs.elapsed_time(ge) / R
        c_ms = cs.elapsed_time(ce) / R
        rep.update({"gemm_solo_ms": round(solo_g, 3),
                    "ar_solo_ms": round(solo_a, 3),
                    "gemm_conc_ms": round(g_ms, 3),
                    "ar_conc_ms": round(c_ms, 3),
                    "gemm_inflation": round(g_ms / solo_g, 3),
                    "ar_inflation": round(c_ms / solo_a, 3)})

    elif args.mode == "mock":
        w = W(dev)
        x0 = torch.randn(args.m, H, device=dev, dtype=torch.float16) * 0.05
        s_comm = torch.xpu.Stream()
        L = args.layers

        def run_serial():
            x = x0.clone()
            for _ in range(L):
                x = layer_serial(x, w, ar)
            torch.xpu.synchronize()

        def run_overlap(k):
            x = x0.clone()
            for _ in range(L):
                x = layer_overlap(x, w, ar, k, s_comm)
            torch.xpu.synchronize()

        t_serial = med(run_serial, iters=5, warmup=2)
        rep["serial_ms"] = round(t_serial, 2)
        best = None
        for k in (2, 3, 4):
            t = med(lambda k=k: run_overlap(k), iters=5, warmup=2)
            sp = t_serial / t
            rep[f"overlap_k{k}_ms"] = round(t, 2)
            rep[f"overlap_k{k}_speedup"] = round(sp, 3)
            if best is None or sp > best[1]:
                best = (k, sp)
        rep["best_k"] = best[0]
        rep["best_speedup"] = round(best[1], 3)
        rep["gate_1p08"] = "PASS" if best[1] >= 1.08 else "FAIL"
        # correctness: overlapped == serial within fp16 AR associativity
        # (identical per-element reduction order — expect bitwise)
        with torch.no_grad():
            xs = layer_serial(x0.clone(), w, ar)
            xo = layer_overlap(x0.clone(), w, ar, 2, s_comm)
            torch.xpu.synchronize()
            rep["overlap_bitwise"] = bool(torch.equal(xs, xo))

    dist.barrier()
    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1)
        print(json.dumps(rep, indent=1))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
