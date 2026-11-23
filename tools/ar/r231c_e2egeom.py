#!/usr/bin/env python3
"""r231c_e2egeom.py -- fixed-term probe v2: the e2e geometry (r231c v1
next step).

v1 findings (plan tail): floor ~65-105 us; single-split cliff below 32
k-tiles (512 tokens); 96-block window keeps reps L2-hot (228 us @2K vs
splittime's 510). This v2 runs the two missing axes:

  - POOL: thousands of blocks with the banked window bytes tiled in
    and the walked block ids stride-scattered -- L2-defeating,
    cold-DRAM walk class like the engine's 48 GB pool.
  - MAX_SEQLLEN_K ARM: "win" = window capacity (nbt*64, v1 posture)
    vs "engine" = bt padded to width 4114 -> max_sk 263,296 -> the
    r212f frozen capacity-priced split posture the serving graph
    replays.

sl ladder {512, 2048, 8192, 32768} x both arms x {auto, ns32, ns1}.
Prints R231C_E2E rows.
"""

import sys

import torch

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

REPS = 30
WARM = 5
ENGINE_BT_WIDTH = 4114
BLOCK = 64


def build_inputs(path, dev, sl, arm):
    p = torch.load(path, map_location="cpu", weights_only=False)
    fmt = str(p["fmt"])
    r = int(p["q"].shape[0])
    shape = tuple(p["kv_shape"])
    stride = tuple(p["kv_stride"])
    off = int(p["kv_off"])
    block_stride = int(p["kv_stride"][0])
    win = p["kv_win"].to(torch.uint8).reshape(p["kv_win"].shape[0], -1)
    nbt = int(win.shape[0])
    needed = (sl + BLOCK - 1) // BLOCK
    # thousands of blocks, tiled window content (timing-only; bytes are
    # walked, values never gated on)
    pool_blocks = max(3072, needed + 64)
    storage = torch.zeros(
        off + pool_blocks * block_stride, dtype=torch.uint8, device=dev)
    tile = win.repeat((pool_blocks + nbt - 1) // nbt, 1)[:pool_blocks]
    storage[off:] = tile.reshape(-1).to(dev)
    pool = torch.as_strided(storage, (pool_blocks, *shape[1:]), stride, off)
    # scattered unique block ids: stride coprime with pool_blocks
    ids = ((torch.arange(needed) * 97 + 11) % pool_blocks).to(torch.int64)
    bt_width = nbt if arm == "win" else max(ENGINE_BT_WIDTH, needed)
    bt = torch.zeros(r, bt_width, dtype=torch.int32, device=dev)
    bt[:, :needed] = ids.unsqueeze(0).repeat(r, 1).to(torch.int32)
    max_sk = bt_width * BLOCK
    q = p["q"].to(torch.bfloat16).to(dev)
    sl_t = torch.full((r,), sl, dtype=torch.int32, device=dev)
    cu_q = torch.arange(r + 1, device=dev, dtype=torch.int32)
    scale = float(p["scale"])
    return fmt, q, pool, storage, sl_t, bt, cu_q, scale, max_sk


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


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    dev = torch.device("xpu")
    print(f"R231C_E2E payload={path} reps={REPS}")
    for arm in ("win", "engine"):
        for sl in (512, 2048, 8192, 32768):
            needed = (sl + BLOCK - 1) // BLOCK
            if arm == "win" and needed > 96:
                # the win arm's bt width caps the walk at the banked
                # 96-block window (6144 tokens) -- rows past that would
                # silently measure the clamped walk
                print(f"R231C_E2E arm=win    sl={sl:6d} SKIP (window 6144 cap)")
                continue
            fmt, q, pool, storage, sl_t, bt, cu_q, scale, max_sk = (
                build_inputs(path, dev, sl, arm))

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
            t_ns128 = time_direct(call_ns(128), dev)
            print(
                f"R231C_E2E arm={arm:6s} sl={sl:6d} max_sk={max_sk:7d} "
                f"auto={t_auto:9.1f}us ns32={t_ns32:9.1f}us "
                f"ns128={t_ns128:9.1f}us"
            )
            del pool, storage
            torch.xpu.empty_cache()
    print("R231C_E2E_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
