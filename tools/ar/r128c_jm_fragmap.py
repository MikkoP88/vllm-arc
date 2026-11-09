# r128c_jm_fragmap.py -- P6.34 arc7 JM fragment-map probe (hardware truth).
# One tile (N=16, K=32, M=8), one subgroup, direct w16pt lane, no scales
# (splits resolve to 1 direct: tiles=1, groups=1). Empirically maps:
#   B: plant ONE e4m3 1.0 code (0x38) at repacked-tile even byte
#      32*t + 2*h (half h of lane t, kt=0); x[0][k] = 1 + k/16 sentinels
#      (k < 16). The single nonzero output cell out[0, c] = 1 + k/16
#      reveals BOTH the column c and the k the hardware assigns to that
#      half-slot. Sweep h = 0..15 at lanes 0/5/10/15.
#   A: plant x[r][k] = 1 + (8r + k)/128 (all rows, k < 16) with the B half
#      at (t=0, h=0): out[:, c_b] values reveal which (row, k) my aload
#      actually fed into each hardware output row (expected under my
#      geometry: out[m, c_b] = 1 + (8m + k_b)/128).
# Prints assumed-vs-measured verdicts; nonzero-diff = geometry defect.
import torch

import vllm_xpu_kernels  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401

ops = torch.ops._xpu_C
dev = "xpu"
print("xpu:", torch.xpu.get_device_name(0), flush=True)

K, N, M = 32, 16, 8
w_pt = torch.ones(1, device=dev)

x_sent = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
x_sent[0, :16] = torch.tensor([1 + k / 16 for k in range(16)],
                              dtype=torch.bfloat16, device=dev)


def run(x, tiles):
    return ops.jm_fp8_gemm_w8a16(x, tiles, w_pt, None).float()


print("=== B half map: (lane t, half h) -> measured (col, k); "
      "assumed col=(2t+(h&1))&15 k=2*(h>>1)+(t>>3) ===", flush=True)
bmap = {}
for t in (0, 5, 10, 15):
    for h in range(16):
        tl = torch.zeros(1, 2, 512, dtype=torch.uint8, device=dev)
        tl[0, 0, 32 * t + 2 * h] = 0x38  # e4m3 1.0 at half h of lane t
        out = run(x_sent, tl)
        nz = (out[0] != 0).nonzero().flatten().tolist()
        if len(nz) != 1:
            print(f"t={t:2d} h={h:2d}: NONZEROS {nz} "
                  f"vals {[round(out[0, c].item(), 4) for c in nz]}",
                  flush=True)
            continue
        c = nz[0]
        v = out[0, c].item()
        k = round((v - 1.0) * 16)
        ac, ak = (2 * t + (h & 1)) & 15, 2 * (h >> 1) + (t >> 3)
        ok = "OK " if (c == ac and k == ak) else "DIFF"
        bmap[(t, h)] = (c, k)
        print(f"t={t:2d} h={h:2d}: measured (col {c:2d}, k {k:2d}) "
              f"assumed (col {ac:2d}, k {ak:2d}) {ok}", flush=True)

print("=== A routing: B half at (t=0, h=0), x[r][k] = 1+(8r+k)/128 ===",
      flush=True)
if (0, 0) in bmap:
    c_b, k_b = bmap[(0, 0)]
    x_a = torch.zeros(M, K, dtype=torch.bfloat16, device=dev)
    for r in range(M):
        x_a[r, :16] = torch.tensor(
            [1 + (8 * r + k) / 128 for k in range(16)],
            dtype=torch.bfloat16, device=dev)
    tl = torch.zeros(1, 2, 512, dtype=torch.uint8, device=dev)
    tl[0, 0, 0] = 0x38
    out = run(x_a, tl)
    col = out[:, c_b]
    print(f"B anchor (col {c_b}, k {k_b}); out[:, c_b]:", flush=True)
    bad = 0
    for m in range(M):
        got = round(col[m].item(), 4)
        want = round(1 + (8 * m + k_b) / 128, 4)
        ok = "OK " if abs(got - want) < 1e-3 else "DIFF"
        bad += ok == "DIFF"
        # DIFF reveals the fed (r', k'): got = 1 + (8r' + k')/128
        rpk = round((got - 1) * 128)
        print(f"  row {m}: got {got:.4f} want {want:.4f} {ok}"
              + (f"  (fed r'={rpk // 8} k'={rpk % 8})" if ok == "DIFF"
                 else ""), flush=True)
    print(f"A_ROUTING diffs={bad}", flush=True)
else:
    print("B anchor missing; A test skipped", flush=True)
print("FRAGMAP_DONE", flush=True)
