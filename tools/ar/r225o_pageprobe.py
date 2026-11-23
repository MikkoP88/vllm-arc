#!/usr/bin/env python3
"""r225o_pageprobe.py — page-induction probe (r225 fourth source).

r225l (build24 wheel) flipped the picture: with an IDENTITY block table
(window compacted into pages 0..95) all three payloads sit at the bf16
grid floor -- auto/nsplit2/nsplit8 clean -- while gate3's replay at the
BANKED table (pages 391..537, one 52-page hole) keeps 2447.30 at
5.16e-2. The payloads share one bt; the per-payload difference is the
pool view's storage offset: 2447.30 kv_off = 5222400000 bytes (> 2^32),
the clean pair 0. r225l's identity replay USED that 5.22e9 offset and
stayed clean -> the failure needs BOTH a large base and real page ids:
the base + page*stride sum crosses something the kernel's addressing
truncates.

This probe replays ONE payload at a matrix of (kv_off, page placement)
cells and scores each against the fp64 identity-walk model:

  banked@off      banked bt (391..537 + gap), window at banked pages
  identity@off    bt = arange(96), window at pages 0..95
  compact P@off   bt = P + arange(96), window at pages P..P+95 (sweep)
  off sweep       compact at P=391 across offsets (incl. 2^31, 2^32)

A cliff in P (at fixed off) or in off (at fixed P) pins the truncating
arithmetic; a smooth/no cliff indicts the table's GAP structure instead
(cells banked vs compact-at-391 isolate exactly that).

Usage: r225o_pageprobe.py <payload.pt>
"""

import sys

import torch

sys.path.insert(0, "/tmp")
from r225j_iso import dmax, grid_stats, model_row  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402
from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

D = 256
BLOCK = 64
BLOCK_STRIDE = 51200
ENGINE_BT_WIDTH = 4114


class Pool:
    """Engine-geometry replay with an explicit base offset and page
    placement for the banked window bytes."""

    def __init__(self, template, dev):
        self.shape = tuple(template["kv_shape"])
        self.stride = tuple(template["kv_stride"])
        self.bt_dtype = template["bt"].dtype
        self.dev = dev
        self.win = template["kv_win"].to(torch.uint8)
        self.bt_head = template["bt_head"].to(torch.long)

    def __call__(self, off, pages, q, sl, scale):
        r = len(sl)
        storage = torch.zeros(off + self.shape[0] * BLOCK_STRIDE,
                              dtype=torch.uint8, device=self.dev)
        for i, pg in enumerate(pages):
            storage[off + pg * BLOCK_STRIDE:
                    off + (pg + 1) * BLOCK_STRIDE] = \
                self.win[i].reshape(-1).to(self.dev)
        pool = torch.as_strided(storage, self.shape, self.stride, off)
        bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=self.bt_dtype,
                         device=self.dev)
        for i, pg in enumerate(pages):
            bt[:, i] = pg
        sl32 = torch.tensor(sl, dtype=torch.int32, device=self.dev)
        cu_q = torch.arange(r + 1, dtype=torch.int32, device=self.dev)
        out = mx_paged_decode(
            q.to(self.dev), pool, "mxk8v4", cu_q, sl32, bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale, num_splits_kv=0,
        )[:r].cpu()
        del pool, storage
        torch.xpu.empty_cache()
        return out


def main() -> int:
    dev = torch.device("xpu")
    off = _mx_off(D, "mxk8v4")
    p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    sl = p["sl"][:r].to(torch.long).tolist()
    scale = float(p["scale"])
    q = p["q"].to(torch.bfloat16)
    rep = Pool(p, dev)
    K, V = decode_kv(p["kv_win"][..., : off["v"]],
                     p["kv_win"][..., off["v"]:], "mxk8v4", D)
    models = [model_row(q[i].double(), K.double(), V.double(), sl[i], scale)
              for i in range(r)]
    banked = rep.bt_head.tolist()
    print(f"== {sys.argv[1].split('/')[-1]} sl={sl} banked_pages="
          f"[{banked[0]}..{banked[-1]}] n={len(banked)}")

    def cell(label, off_v, pages):
        n_out = rep(off_v, pages, q, sl, scale)
        worst = (0, 0.0)
        dm = 0.0
        for i in range(r):
            mism, mul = grid_stats(n_out[i], models[i])
            worst = (max(worst[0], mism), max(worst[1], mul))
            dm = max(dm, dmax(n_out[i], models[i]))
        print(f"   {label:<26}: dmax={dm:.3e}  mism={worst[0]}/{12 * D}"
              f"  max_ulp={worst[1]:.2f}")

    bank_off = int(p["kv_off"])
    ident = list(range(len(banked)))
    cell(f"banked@{bank_off}", bank_off, banked)
    cell("banked@0", 0, banked)
    cell(f"identity@{bank_off}", bank_off, ident)
    cell("identity@0", 0, ident)
    cell("compact391@0", 0, [391 + i for i in ident])

    print("   -- page-base sweep at fixed off "
          f"{bank_off} (compact placement)")
    for base in (64, 128, 192, 256, 320, 352, 368, 384, 391, 400,
                 416, 448, 512):
        cell(f"compact{base}@{bank_off}", bank_off,
             [base + i for i in ident])

    print("   -- offset sweep at fixed compact page base 391")
    for off_v in (1073741824, 2147483648, 2500000000, 3000000000,
                  3500000000, 4000000000, 4294967296, 4500000000,
                  5000000000, bank_off):
        cell(f"compact391@{off_v}", off_v, [391 + i for i in ident])
    print("R225O_PAGEPROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
