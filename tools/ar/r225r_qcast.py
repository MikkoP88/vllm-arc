#!/usr/bin/env python3
"""r225r_qcast.py — quantify the q fp16-vs-bf16 cast sensitivity (CPU).

r225q collapsed the "fourth source": all 8 harness cells produce
byte-identical native output (md5 51736f834358) at the fp64 floor
(dmax 1.556e-2, mism 1/3072, max_ulp 0.54), while native-vs-oracle
still prints 5.078e-2. The banked payload explains it: q is
torch.float16, and the two lanes consume DIFFERENT inputs:

  triton oracle: mx_decode(q16, ...)         — fp16 q, no cast
  native op:     mx_paged_decode(q16.to(bf16)) — the engine's bf16 cast

and the two fp64 arbiters inherit the same split:
  r225b (gate):      q = p["q"].double()            — fp16 precision
  r225j (my floor):  q = p["q"].to(bf16).double()   — native's input

Each lane matches its own walk to sub-ulp; the 5e-2 "defect" is the
q-cast sensitivity of the attention output on that payload. This
script measures it directly: the SAME r224d-proven identity walk with
ONLY q's precision varied.

  qA = fp16 -> double          (oracle-side reference)
  qB = fp16 -> bf16 -> double  (the native op's actual input)

Per payload: dmax(walkA, walkB) — if it reproduces the observed
native-vs-oracle ladder (2447.18 ~7.8e-3, 2447.30 ~5.1e-2, 2547.18
~5.9e-3), the residual gate gap is INPUT sensitivity, not a kernel
defect; the gate needs a q-cast-matched oracle, not a kernel fix.

Usage: r225r_qcast.py <payload.pt> [...]
"""

import sys

import torch

sys.path.insert(0, "/tools")
from r225j_iso import dmax, grid_stats, model_row  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402

D = 256


def main() -> int:
    off = _mx_off(D, "mxk8v4")
    for path in sys.argv[1:]:
        p = torch.load(path, map_location="cpu", weights_only=False)
        r = p["q"].shape[0]
        scale = float(p["scale"])
        K, V = decode_kv(p["kv_win"][..., : off["v"]], p["kv_win"][..., off["v"] :], "mxk8v4", D)
        K, V = K.double(), V.double()
        qA = p["q"][:r].double()  # fp16 precision
        qB = p["q"][:r].to(torch.bfloat16).double()  # native's input
        dm = 0.0
        worst = (0, 0.0)
        dmA = 0.0
        for i in range(r):
            sl_i = int(p["sl"][i])
            mA = model_row(qA[i], K, V, sl_i, scale)
            mB = model_row(qB[i], K, V, sl_i, scale)
            dm = max(dm, dmax(mA, mB))
            mism, mul = grid_stats(mA.to(torch.bfloat16), mB)
            worst = (max(worst[0], mism), max(worst[1], mul))
            dmA = max(dmA, dmax(p["out_triton"][i].to(mA.dtype), mA))
        name = path.split("/")[-1]
        print(
            f"R225R {name}: dmax(walkF16,walkBF16)={dm:.3e} "
            f"mism={worst[0]}/{12 * D} max_ulp={worst[1]:.2f} | "
            f"walkF16-vs-banked-triton={dmA:.3e}"
        )
    print("R225R_QCAST_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
