#!/usr/bin/env python3
"""r225p_orderprobe.py — order/allocator-history probe (r225 fourth
source).

r225o replayed the EXACT banked cell (bt 391..537, kv_off 5222400000,
build24 wheel) CLEAN at the grid floor — while gate3's r225a replay of
the identical configuration measured 5.158e-2. The harness delta is
ORDER: r225a runs the Triton oracle FIRST on the same pool (the A/B
replay contract), r225o/others call the native op alone. Both are
self-consistent across runs — a stable per-process difference: the
GPU allocator state at native-call time. The native op's internal
at::empty buffers (temp_out 6.3 MB, exp_sums/max_logits 24 KB) draw
from different cached blocks per process; if ANY read touches bytes
the kernel did not write, recycled contents become the output's error.

Cells (payload 2447.30, banked geometry, auto splits):
  native-only      control (r225o parity — expect the grid floor)
  triton-first     r225a replica (expect the 5e-2 defect if order is it)
  poison+1-first   free +1.0-filled blocks of the op's internal sizes,
                   then native (wrong => live stale reads)
  poison-1-first   same with -1.0 (clean vs poison+1 separates the
                   exp_sums<=0 guard from raw partial reads)
  poison0-first    same with 0.0 (fresh-page parity control)

Usage: r225p_orderprobe.py <payload.pt>
"""

import sys

import torch

sys.path.insert(0, "/tmp")
from r225j_iso import dmax, grid_stats, model_row  # noqa: E402
from r225o_pageprobe import Pool  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode import mx_decode  # noqa: E402

D = 256


def poison(dev, value: float) -> None:
    """Free cached device blocks holding a known filler pattern."""
    blocks = []
    for mb in (32.0, 6.3, 1.0):
        blocks.append(torch.full((int(mb * 1024 * 1024),), value,
                                 dtype=torch.float32, device=dev))
    blocks.append(torch.full((24 * 1024,), value, dtype=torch.float32,
                             device=dev))
    del blocks
    torch.xpu.empty_cache()


def main() -> int:
    dev = torch.device("xpu")
    off = _mx_off(D, "mxk8v4")
    p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    sl = p["sl"][:r].to(torch.long).tolist()
    scale = float(p["scale"])
    q_dev = p["q"].to(dev)  # banked dtype (the oracle contract)
    q = p["q"].to(torch.bfloat16)
    rep = Pool(p, dev)
    K, V = decode_kv(p["kv_win"][..., : off["v"]],
                     p["kv_win"][..., off["v"]:], "mxk8v4", D)
    models = [model_row(q[i].double(), K.double(), V.double(), sl[i], scale)
              for i in range(r)]
    pages = rep.bt_head.tolist()
    bank_off = int(p["kv_off"])
    print(f"== {sys.argv[1].split('/')[-1]} sl={sl} pages="
          f"[{pages[0]}..{pages[-1]}] off={bank_off}")

    def run_native():
        n_out = rep(bank_off, pages, q, sl, scale)
        worst = (0, 0.0)
        dm = 0.0
        for i in range(r):
            mism, mul = grid_stats(n_out[i], models[i])
            worst = (max(worst[0], mism), max(worst[1], mul))
            dm = max(dm, dmax(n_out[i], models[i]))
        print(f"   dmax={dm:.3e}  mism={worst[0]}/{12 * D}"
              f"  max_ulp={worst[1]:.2f}")
        return n_out

    print("   native-only:")
    a = run_native()

    print("   native-only (repeat, stability):")
    b = run_native()
    print(f"   bitwise-stable={all(torch.equal(a[i], b[i]) for i in range(r))}")

    print("   triton-first (r225a replica):")
    pool, storage = None, None
    # r225a order: build pool, triton oracle on it, then native on it
    nbt = len(pages)
    bt_t = torch.zeros(r, 4114, dtype=p["bt"].dtype, device=dev)
    bt_t[:, :nbt] = (rep.bt_head.unsqueeze(0).repeat(r, 1)
                     .to(dev).to(p["bt"].dtype))
    sl_dev = torch.tensor(sl, device=dev).to(torch.int32)
    # rebuild the pool exactly as r225a's rebuild_pool
    shape, stride = tuple(p["kv_shape"]), tuple(p["kv_stride"])
    storage = torch.zeros(bank_off + shape[0] * 51200, dtype=torch.uint8,
                          device=dev)
    win = p["kv_win"].to(torch.uint8)
    for i, pg in enumerate(pages):
        storage[bank_off + pg * 51200: bank_off + (pg + 1) * 51200] = \
            win[i].reshape(-1).to(dev)
    pool = torch.as_strided(storage, shape, stride, bank_off)
    _ = mx_decode(q_dev, pool, bt_t, sl_dev, scale)
    c = run_native()
    print(f"   triton-first vs native-only bitwise="
          f"{all(torch.equal(a[i], c[i]) for i in range(r))}")
    del pool, storage
    torch.xpu.empty_cache()

    for label, val in (("poison+1", 1.0), ("poison-1", -1.0),
                       ("poison0", 0.0)):
        print(f"   {label}-first:")
        poison(dev, val)
        d = run_native()
        print(f"   {label} vs native-only bitwise="
              f"{all(torch.equal(a[i], d[i]) for i in range(r))}")
    print("R225P_ORDERPROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
