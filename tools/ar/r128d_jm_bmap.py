# r128d_jm_bmap.py -- P6.34 arc7 JM full B-fragment transfer map.
# Sweeps every B half-slot (lane tb = 0..15, half vb = 0..15): plant ONE
# e4m3 1.0 (0x38) at repacked-tile byte 32*tb + 2*vb (which my decode
# routes to bv[vb] of lane tb), x = single nonzero row with k-sentinels
# x[r][k] = 1 + k/16 (bf16-exact). Each printed nonzero cell
# out[pr][pc] = x-sentinel reveals k; the pattern over the sweep is the
# hardware's (A-pairing, B, D) transfer map -- invertible to the TRUE
# fragment geometry. Two passes: row 0 (my aload even halves) and row 1
# (odd halves). N=16 K=32 M=8, tiles=1, splits resolve 1 direct (w16pt).
import torch

import vllm_xpu_kernels  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

ops = torch.ops._xpu_C
dev = "xpu"
print("xpu:", torch.xpu.get_device_name(0), flush=True)

K, N, M = 32, 16, 8
w_pt = torch.ones(1, device=dev)
sent = torch.tensor([1 + k / 16 for k in range(16)],
                    dtype=torch.bfloat16, device=dev)


def sweep(row):
    x = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
    x[row, :16] = sent
    print(f"=== pass row={row}: plant (tb, vb) -> [(pr, pc, k)...] ===",
          flush=True)
    for tb in range(16):
        line = []
        for vb in range(16):
            tl = torch.zeros(1, 2, 512, dtype=torch.uint8, device=dev)
            tl[0, 0, 32 * tb + 2 * vb] = 0x38
            out = ops.jm_fp8_gemm_w8a16(x, tl, w_pt, None).float()
            hits = []
            nz = (out != 0).nonzero().tolist()
            for pr, pc in nz:
                k = round((out[pr, pc].item() - 1.0) * 16)
                hits.append(f"({pr},{pc},k{k})")
            line.append(f"v{vb}:{'|'.join(hits) if hits else '-'}")
        print(f"t{tb:2d} " + " ".join(line), flush=True)


sweep(0)
sweep(1)
print("BMAP_DONE", flush=True)
