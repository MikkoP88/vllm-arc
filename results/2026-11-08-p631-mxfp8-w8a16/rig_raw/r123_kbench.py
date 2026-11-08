"""r123_kbench.py -- P6.31 arc5 smoke + superiority head-to-head kbench.
Runs on an ephemeral container over the arc4test image with the arc5
wheel pip-installed (image bytes untouched). New ops under test:

  fp8_gemm_w8a16_mxfp8  -- MXFP8 W8A16 (bf16 acts, group-32 e8m0 {32,1}
                           WEIGHTS attr [K/32,N]) -- the P6.31 main lane
  fp4_gemm_w4a8         -- MXFP4 W4A8 probe (e4m3 acts + per-token f32,
                           packed f4 weights) -- the owner-addendum gate
                           question; a create-time throw here is the
                           clean closed-negative answer

Head-to-head vs stock on identical shapes/Ms (the dispatch finding:
champion fp8 = XPUW8A16FP8LinearKernel = w16pc/w16pt class, so THE
fight for "MXFP8 superior vs FP8" is m816 vs w16pc/w16pt):

  bf16  = torch.matmul reference
  m816  = NEW fp8_gemm_w8a16_mxfp8(x, Bt8, ws8_kn e8m0, bf16, None)
  w16pc = stock fp8_gemm_w8a16(x, Bt8, w_pc f32 [1,N], None)  [champion]
  w16pt = stock fp8_gemm_w8a16(x, Bt8, w_pt f32 [1], None)
  mx8   = stock fp8_gemm(x8g, Bt8, bf16, x_sc e8m0, ws8_kn)   [2x-class]
  g16   = stock fp4_gemm_w4a16(x, Bt, ws_kn, bf16, None)      [cert20]
  w4a8  = NEW fp4_gemm_w4a8(x8tok, a_tok f32 [M,1], Bt, ws_kn, bf16, None)
          + qtok quant cost (the apply total = qtok + w4a8)
  q8/qpt/qtok = quant-op costs

M=8192 is the e2e-relevant prefill regime (champion posture:
max_num_batched_tokens=8192 -> chunked-prefill GEMM M).

Numerics: every quant cell gets a rel-err vs the fp32 dequant reference
built from the ACTUAL stored quantized values (scale-mask
misinterpretation explodes rel).

Timing: torch.xpu.Event pairs, 15 warmup, 50 iters, median.
ONEDNN_VERBOSE=2 for impl-selection lines (jit vs ref names the winner).
"""
import gc
import statistics
import traceback

import torch

torch.manual_seed(0)
dev = "xpu"
ITERS, WARM = 50, 15

SHAPES = [
    ("gemma.up", 5376, 10752),
    ("gemma.down", 10752, 5376),
    ("gemma.o", 5376, 5376),
    ("qwen.up", 5120, 8704),
    ("qwen.down", 8704, 5120),
    ("qwen.o", 5120, 5120),
]
MS = [1, 8, 4096, 8192, 32768]

# e2m1 nibble LUT: [0, .5, 1, 1.5, 2, 3, 4, 6] with sign in bit 3
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
     ] * 2, dtype=torch.float32
)
_E2M1[8:] *= -1.0

E4M3_MAX = 448.0


def bench(fn):
    for _ in range(WARM):
        fn()
    torch.xpu.synchronize()
    times = []
    for _ in range(ITERS):
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.xpu.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def e8m0_depow(sc_u8):
    """e8m0 scale bytes -> f32 powers of two."""
    return torch.pow(2.0, sc_u8.view(torch.uint8).to(torch.float32) - 127.0)


print("=== [0] env ===")
print("xpu:", torch.xpu.is_available(), torch.xpu.get_device_name(0))
import vllm_xpu_kernels  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401  (dlopen law, r120b)
import vllm._xpu_ops  # noqa: F401

for op in (
    "fp8_gemm_w8a16_mxfp8",
    "fp4_gemm_w4a8",
    "fp8_gemm",
    "fp8_gemm_w8a16",
    "fp4_gemm_w4a16",
):
    assert hasattr(torch.ops._xpu_C, op), f"FATAL: {op} absent"
print("ops PRESENT: fp8_gemm_w8a16_mxfp8 fp4_gemm_w4a8 (arc5) + stock set")

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    xpu_mxfp4_quantize as qop4,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as qop8,
)


def qpt(x):
    """per-tensor e4m3 dynamic quant."""
    a = x.detach().abs().amax().float().clamp(min=1e-6) / E4M3_MAX
    x8 = (x.float() / a).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return x8, a.reshape(1)


def qtok(x):
    """per-token (row) e4m3 dynamic quant."""
    a = (
        x.detach().abs().amax(dim=-1, keepdim=True).float()
        .clamp(min=1e-6) / E4M3_MAX
    )  # [M,1]
    x8 = (x.float() / a).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return x8, a.contiguous()


rows = []
w4a8_gate_open = None  # None = untried, True = md accepted, False = threw
for tag, K, N in SHAPES:
    for M in MS:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
        wref = torch.randn(K, N, dtype=torch.bfloat16, device=dev) * 0.02
        w8 = torch.randn(N, K, device=dev).to(torch.float8_e4m3fn)
        Bt8 = w8.t()  # nt view [K, N]
        w_pt = (torch.rand(1, device=dev, dtype=torch.float32) * 0.01 + 0.005)
        w_pc = (torch.rand(1, N, device=dev, dtype=torch.float32) * 0.01 + 0.005)
        ws8_kn = torch.randint(
            120, 136, (K // 32, N), dtype=torch.uint8, device=dev
        ).view(torch.float8_e8m0fnu)

        row = {"tag": tag, "M": M, "K": K, "N": N}

        row["bf16"] = bench(lambda: torch.matmul(x, wref))

        def _rel(y, ref):
            return float((y.float() - ref).norm() / ref.norm())

        # ---- m816: NEW mxfp8 w8a16 (the P6.31 main lane) ----
        try:
            def m816():
                return torch.ops._xpu_C.fp8_gemm_w8a16_mxfp8(
                    x, Bt8, ws8_kn, torch.bfloat16, None
                )
            row["m816"] = bench(m816)
            with torch.no_grad():
                wdq = w8.float().t() * e8m0_depow(
                    ws8_kn
                ).repeat_interleave(32, dim=0)  # [K, N]
                row["rel_m816"] = _rel(m816(), x.float() @ wdq)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} m816] FAILED: {type(ex).__name__}: {ex}")
            row["m816"] = row["rel_m816"] = None

        # ---- w16pc / w16pt: champion class (stock) ----
        for cell, sc in (("w16pc", w_pc), ("w16pt", w_pt)):
            try:
                def g16f(sc=sc):
                    return torch.ops._xpu_C.fp8_gemm_w8a16(x, Bt8, sc, None)
                row[cell] = bench(g16f)
                with torch.no_grad():
                    ref = x.float() @ (w8.float().t() * sc)
                    row["rel_" + cell] = _rel(g16f(), ref)
            except Exception as ex:  # noqa: BLE001
                print(f"[{tag} M={M} {cell}] FAILED: {type(ex).__name__}: {ex}")
                row[cell] = row["rel_" + cell] = None

        # ---- mx8: stock mxfp8 W8A8 (the slow class, reference) ----
        try:
            x8g = qop8(x)  # (x8, sc e8m0 [M, K/32])

            def gmx():
                return torch.ops._xpu_C.fp8_gemm(
                    x8g[0], Bt8, torch.bfloat16, x8g[1], ws8_kn, None
                )
            row["mx8"] = bench(gmx)
            row["q8"] = bench(lambda: qop8(x))
            with torch.no_grad():
                if x8g[1].dim() == 2 and x8g[1].shape[1] == K // 32:
                    xdq = x8g[0].float() * e8m0_depow(
                        x8g[1]
                    ).repeat_interleave(32, dim=1)
                    wdq = w8.float().t() * e8m0_depow(
                        ws8_kn
                    ).repeat_interleave(32, dim=0)
                    row["rel_mx8"] = _rel(gmx(), xdq @ wdq)
                else:
                    row["rel_mx8"] = None
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} mx8] FAILED: {type(ex).__name__}: {ex}")
            row["mx8"] = row["q8"] = row["rel_mx8"] = None

        # ---- quant costs (champion-class + w4a8 feedstock) ----
        try:
            row["qpt"] = bench(lambda: qpt(x)[0])
            row["qtok"] = bench(lambda: qtok(x)[0])
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} quant] FAILED: {type(ex).__name__}: {ex}")
            row["qpt"] = row["qtok"] = None

        # ---- g16: cert20 incumbent w4a16 (cross-width reference) ----
        w = torch.randint(
            0, 255, (N, K // 2), dtype=torch.uint8, device=dev
        ).view(torch.float4_e2m1fn_x2)
        ws = torch.randint(
            120, 136, (N, K // 32), dtype=torch.uint8, device=dev
        ).view(torch.float8_e8m0fnu)
        ws_kn = ws.t().contiguous()
        Bt = w.t()
        try:
            def g16():
                return torch.ops._xpu_C.fp4_gemm_w4a16(
                    x, Bt, ws_kn, torch.bfloat16, None
                )
            row["g16"] = bench(g16)
            with torch.no_grad():
                y16 = g16()
                b = w.view(torch.uint8).to(torch.int16)
                lo = (b & 0xF).to(torch.int64)
                hi = ((b >> 4) & 0xF).to(torch.int64)
                lut = _E2M1.to(dev)
                wdq = torch.empty(N, K, dtype=torch.float32, device=dev)
                wdq[:, 0::2] = lut[lo]
                wdq[:, 1::2] = lut[hi]
                sc = torch.pow(
                    2.0, ws.view(torch.uint8).to(torch.float32) - 127.0
                )
                wdq *= sc.repeat_interleave(32, dim=1)
                row["rel_g16"] = _rel(y16, x.float() @ wdq.t())
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} w4a16] FAILED: {type(ex).__name__}: {ex}")
            row["g16"] = row["rel_g16"] = None

        # ---- w4a8: NEW probe (e4m3 acts + per-token f32, packed f4) ----
        try:
            x8tok = qtok(x)

            def w4a8():
                return torch.ops._xpu_C.fp4_gemm_w4a8(
                    x8tok[0], x8tok[1], Bt, ws_kn, torch.bfloat16, None
                )
            row["w4a8"] = bench(w4a8)
            if w4a8_gate_open is None:
                w4a8_gate_open = True
                print(
                    "[gate] (e4m3, f4_e2m1) joint ACCEPTED by oneDNN md "
                    "validation -- primitive created"
                )
            with torch.no_grad():
                b = w.view(torch.uint8).to(torch.int16)
                lo = (b & 0xF).to(torch.int64)
                hi = ((b >> 4) & 0xF).to(torch.int64)
                lut = _E2M1.to(dev)
                wdq = torch.empty(N, K, dtype=torch.float32, device=dev)
                wdq[:, 0::2] = lut[lo]
                wdq[:, 1::2] = lut[hi]
                sc = torch.pow(
                    2.0, ws.view(torch.uint8).to(torch.float32) - 127.0
                )
                wdq *= sc.repeat_interleave(32, dim=1)
                ref = (x8tok[0].float() * x8tok[1]) @ wdq.t()
                row["rel_w4a8"] = _rel(w4a8(), ref)
        except Exception as ex:  # noqa: BLE001
            if w4a8_gate_open is None:
                w4a8_gate_open = False
                print(
                    f"[gate] (e4m3, f4_e2m1) joint REJECTED: "
                    f"{type(ex).__name__}: {ex}"
                )
            row["w4a8"] = row["rel_w4a8"] = None

        rows.append(row)
        print(
            "%-11s M=%-6d bf16=%7.2f | m816=%7.2f(r%.0e) w16pc=%7.2f(r%.0e)"
            " w16pt=%7.2f(r%.0e) mx8=%7.2f | q8=%6.3f qpt=%6.3f qtok=%6.3f"
            " | g16=%7.2f(r%.0e) w4a8=%7.2f(r%.0e)"
            % (
                tag, M, row["bf16"],
                *[v if v is not None else -1 for v in (
                    row["m816"], row["rel_m816"], row["w16pc"],
                    row["rel_w16pc"], row["w16pt"], row["rel_w16pt"],
                    row["mx8"])],
                *[v if v is not None else -1 for v in (
                    row["q8"], row["qpt"], row["qtok"])],
                *[v if v is not None else -1 for v in (
                    row["g16"], row["rel_g16"], row["w4a8"],
                    row["rel_w4a8"])],
            )
        )
        del x, wref, w8, w, x8g, x8tok
        gc.collect()
        torch.xpu.empty_cache()

print("\n=== [summary] medians ===")


def med(key, mfilter):
    vals = [r[key] for r in rows
            if mfilter(r["M"]) and r.get(key) is not None]
    return statistics.median(vals) if vals else float("nan")


for label, mf in (("decode (M<=8)", lambda m: m <= 8),
                  ("prefill (M>=4096)", lambda m: m >= 4096),
                  ("chunk M=8192", lambda m: m == 8192)):
    print(
        "%s: bf16=%.3f | m816=%.3f w16pc=%.3f w16pt=%.3f mx8(+q8)=%.3f"
        " g16=%.3f w4a8(+qtok)=%.3f"
        % (label, med("bf16", mf), med("m816", mf), med("w16pc", mf),
           med("w16pt", mf), med("mx8", mf) + med("q8", mf),
           med("g16", mf), med("w4a8", mf) + med("qtok", mf))
    )
print(f"w4a8 gate open: {w4a8_gate_open}")
print("=== r123_kbench done ===")
