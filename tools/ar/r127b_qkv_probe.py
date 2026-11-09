# r127b_qkv_probe.py -- P6.33: bench the SERVE shapes the r126 matrix missed.
# r127 leg D regressed gemma m816 solo decode -38% (17.1 vs 27.7 tps ref) with
# the only routed marker at M=2 (size-1 graph capture pads to 2 tokens), so
# every solo step replays skinny m816 at M=2 across ALL 240 quantized layers
# -- and 60 of those are qkv_proj (K=5376 N=8192/rank), a shape the r126
# matrix never benched (up/down/o only). This probe benches qkv + the real
# o_proj shape at M=1/2/4 m816 and g16, stock vs skinny, paired per-iteration.
import statistics
import torch

torch.manual_seed(0)
dev = "xpu"
ROUNDS, WARM = 30, 12
import vllm_xpu_kernels  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401  (dlopen law)

ops = torch.ops._xpu_C
# (tag, K, N) -- gemma4-31b per-rank TP2: qkv N=(32*256+2*16*256)/2=8192;
# o input=(32/2)*256=4096 N=5376; plus kbench o (square) for continuity
SHAPES = [
    ("gemma.qkv", 5376, 8192),
    ("gemma.o_real", 4096, 5376),
    ("gemma.o", 5376, 5376),
]
MS = [1, 2, 4]


def e8m0_depow(sc_u8):
    return torch.pow(2.0, sc_u8.view(torch.uint8).to(torch.float32) - 127.0)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2,
                     dtype=torch.float32)
_E2M1[8:] *= -1.0


def bench_pair(fn_st, fn_sk):
    for _ in range(WARM):
        fn_st(); fn_sk()
    torch.xpu.synchronize()
    ts, tk = [], []
    for _ in range(ROUNDS):
        a = torch.xpu.Event(enable_timing=True); b = torch.xpu.Event(enable_timing=True)
        c = torch.xpu.Event(enable_timing=True); d = torch.xpu.Event(enable_timing=True)
        a.record(); fn_st(); b.record()
        c.record(); fn_sk(); d.record()
        torch.xpu.synchronize()
        ts.append(a.elapsed_time(b)); tk.append(c.elapsed_time(d))
    return statistics.median(ts), statistics.median(tk)


print("xpu:", torch.xpu.is_available(), torch.xpu.get_device_name(0))
for tag, K, N in SHAPES:
    for M in MS:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
        Bt8 = w8.t()
        ws8_kn = torch.randint(120, 136, (K // 32, N), dtype=torch.uint8,
                               device=dev).view(torch.float8_e8m0fnu)
        w = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8,
                          device=dev).view(torch.float4_e2m1fn_x2)
        ws = torch.randint(120, 136, (N, K // 32), dtype=torch.uint8,
                           device=dev).view(torch.float8_e8m0fnu)
        ws_kn = ws.t().contiguous()
        Bt = w.t()
        st_m, sk_m = bench_pair(
            lambda: ops.fp8_gemm_w8a16_mxfp8(x, Bt8, ws8_kn, torch.bfloat16, None),
            lambda: ops.skinny_fp8_gemm_w8a16_mxfp8(x, Bt8, ws8_kn, torch.bfloat16, None))
        st_g, sk_g = bench_pair(
            lambda: ops.fp4_gemm_w4a16(x, Bt, ws_kn, torch.bfloat16, None),
            lambda: ops.skinny_fp4_gemm_w4a16(x, Bt, ws_kn, torch.bfloat16, None))
        bwm = K * N / sk_m / 1e6; bwg = K * N // 2 / sk_g / 1e6
        print(f"{tag:14s} M={M} m816 {st_m:.4f}->{sk_m:.4f} ({sk_m/st_m:.3f}x,{bwm:.0f}GB/s)"
              f"  g16 {st_g:.4f}->{sk_g:.4f} ({sk_g/st_g:.3f}x,{bwg:.0f}GB/s)", flush=True)
print("PROBE_DONE")
