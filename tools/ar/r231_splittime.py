#!/usr/bin/env python3
"""r231_splittime.py -- split-overprovisioning timing decomposition
(r231 Phase A, the MX ITL constant).

r230 measured a format-independent ITL intercept vs ctrl (~6.4-6.8 ms
per step at n1) plus a format-scaled ctx slope. The bridge python runs
only at capture; the replayed cost is device-side in the frozen
launches, and the capture telemetry prices max_seqlen_k at the bucket
CAPACITY (bt_width 4114 x 64 = 262,336 tokens -- the maxlen posture)
on every graph bucket regardless of the face ctx: the auto heuristic
freezes ~128 splits, so a step at 8K real ctx replays ~123 early-exit
splits + a 128-partial combine per row. Hypothesis: that frozen
over-provisioning IS the constant.

This probe replays the wheel kernel on the banked AB payload
geometries with a FORCED num_splits_kv ladder against a ctx ladder,
timed with xpu events:

  time(rows, sl, ns)  ->  const(ns) = time(sl -> small, ns)
                         slope(ns)  = d time / d sl
                         recoverable(sl) = time(auto) - min_ns time(sl)

Pool built at real geometry; block tables scattered (stride prime to
the block count -- real tables are scattered, sequential ids would
read prefetch-friendly and understate the walk). Rows share tables
per request group like the engine's verify plan.

Usage: r231_splittime.py <out.txt> <payload.pt> [payload2.pt ...]
"""

import sys

import torch

BLOCK = 64
BLOCK_STRIDE = 51200
ENGINE_BT_WIDTH = 4114
D = 256
SL_LADDER = (2048, 8192, 32768, 131072, 262144)
NS_LADDER = (0, 1, 2, 4, 8, 16, 32, 64, 128)  # 0 = auto heuristic
WARMUP = 3
REPS = 10


class Timer:
    """xpu-event wall timer around a callable (device time)."""

    def __init__(self, dev):
        self.s = torch.xpu.Event(enable_timing=True)
        self.e = torch.xpu.Event(enable_timing=True)
        self.dev = dev

    def time(self, fn):
        for _ in range(WARMUP):
            fn()
        torch.xpu.synchronize(self.dev)
        best = None
        samples = []
        for _ in range(REPS):
            self.s.record()
            fn()
            self.e.record()
            torch.xpu.synchronize(self.dev)
            t = self.s.elapsed_time(self.e)  # ms
            samples.append(t)
            best = t if best is None else min(best, t)
        samples.sort()
        p50 = samples[len(samples) // 2]
        return best, p50


def build_pool(template, win, dev, n_blocks):
    """Pool at real geometry; blocks = window tiled (distinct
    addresses carry the timing; dequant/dot cost is data-independent)."""
    # offset 0: the payload's kv_off (multi-GB base inside the engine
    # pool) is not reproduced -- only block-addressing math matters.
    # Block byte stride is FORMAT-dependent (mxk8v4 51200, mxfp8
    # 14336, mxfp4 ...): derive from the payload's kv_stride[0].
    bs = int(template["kv_stride"][0])
    storage = torch.zeros(n_blocks * bs,
                          dtype=torch.uint8, device=dev)
    src = win.reshape(-1).to(dev)
    nb_src = src.numel() // bs
    for b in range(n_blocks):
        storage[b * bs:(b + 1) * bs] = \
            src[(b % nb_src) * bs:(b % nb_src) * bs + bs]
    return torch.as_strided(storage, tuple(template["kv_shape"]),
                            tuple(template["kv_stride"]), 0)


def scattered_bt(r, n_blocks, sl_t, dtype, dev, rows_per_req=4):
    """Per-request-group tables of distinct scattered block ids."""
    bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=dtype, device=dev)
    need = (sl_t + BLOCK - 1) // BLOCK
    n_req = max(1, r // rows_per_req)
    stride = 787  # prime to typical block counts
    for g in range(n_req):
        rows = range(g * rows_per_req,
                     min((g + 1) * rows_per_req, r))
        ids = torch.tensor(
            [(int(g) * stride * 31 + int(j) * stride) % n_blocks
             for j in range(need)], dtype=dtype, device=dev)
        for i in rows:
            bt[i, :need] = ids
    return bt, need


def main() -> int:
    out_path = sys.argv[1]
    dev = torch.device("xpu")
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

    for arg in sys.argv[2:]:
        # payloads CARRY fmt (verified: r224a=mxk8v4 r229a=mxfp8
        # r229b=mxfp4) and it wins; "path:fmt" is only a fallback for
        # a payload missing the field. The r231a4/a5 lessons: an arg
        # override miscast the mxfp8 cell (a4) and the banks' colliding
        # basenames made the wrapper's second cp overwrite the first
        # (a5) -- copy to distinct names upstream.
        path, _, fmt_arg = arg.rpartition(":")
        if "/" not in path:
            path, fmt_arg = arg, ""
        p = torch.load(path, map_location="cpu", weights_only=False)
        fmt = str(p.get("fmt") or fmt_arg or "mxfp4")
        r = int(p["q"].shape[0])
        scale = float(p["scale"])
        q = p["q"].to(torch.bfloat16).to(dev)
        win = p["kv_win"].to(torch.uint8)
        bt_dtype = p["bt"].dtype
        # pool: enough blocks for the widest ladder table (4096) plus
        # pool at the payload's real block geometry (offset 0); the
        # as_strided extent check needs storage >= kv_shape[0] blocks
        n_pool = int(p["kv_shape"][0])
        if n_pool > 32768 or n_pool * 64 < SL_LADDER[-1]:
            emit(f"== {path.split('/')[-1]}: SKIP (pool blocks {n_pool})")
            continue
        pool = build_pool(p, win, dev, n_pool)
        sl32_base = torch.full((r,), 64, dtype=torch.int32, device=dev)
        cu_q = torch.arange(r + 1, dtype=torch.int32, device=dev)
        tm = Timer(dev)
        name = path.split("/")[-1]
        emit(f"== {name} fmt={fmt} rows={r} pool_blocks={n_pool}")
        emit(f" {'sl':>7} | " + " | ".join(
            f"{('auto' if ns == 0 else f'ns{ns}'):>8}" for ns in NS_LADDER)
            + " | best_ns recover_ms")

        for sl_t in SL_LADDER:
            bt, need = scattered_bt(r, n_pool, sl_t, bt_dtype, dev)
            if need > n_pool:
                emit(f" {sl_t:>7} | SKIP (need {need} blocks > pool)")
                continue
            sl32 = sl32_base.clone()
            sl32[:] = sl_t
            row = []
            times = {}
            for ns in NS_LADDER:
                def call(q=q, pool=pool, fmt=fmt, cu_q=cu_q, sl32=sl32,
                         bt=bt, ns=ns, scale=scale):
                    return mx_paged_decode(
                        q, pool, fmt, cu_q, sl32, bt,
                        max_seqlen_k=ENGINE_BT_WIDTH * 64,
                        softmax_scale=scale, num_splits_kv=ns)
                best, p50 = tm.time(call)
                row.append(p50)
                times[ns] = p50
            auto_t = times[0]
            best_ns = min((v, k) for k, v in times.items() if k != 0)
            emit(f" {sl_t:>7} | " + " | ".join(
                f"{times[ns]:8.3f}" for ns in NS_LADDER)
                + f" | ns{best_ns[1]:<3} {auto_t - best_ns[0]:9.3f}")
        pool = None
        torch.xpu.empty_cache()

    emit("R231_SPLITTIME_DONE")
    with open(out_path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
