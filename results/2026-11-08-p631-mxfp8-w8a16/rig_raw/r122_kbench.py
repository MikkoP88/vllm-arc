"""r122_kbench.py -- P6.31 FIX-Q lane-selection kbench on the arc4test
image (vxk 0.1.14.1+arc4, oneDNN v3.13). ALL cells ride STOCK wheel ops
-- no build needed. Answers the P6.31 physics questions before any arc5
work:

  (a) does the (bf16, f8) joint W8A16 matmul run f8-rate or bf16-rate
      at prefill Ms?  (cells w16pc/w16pt vs pt/tok)
  (b) what does the champion static class cost per cell, measured for
      the first time?  (cell pt: per-tensor/per-tensor f8xf8)
  (c) the mxfp8 group act-quant cost per M (cell q8 -- the missing
      decode-tax column from r120d) + the cheap quant alternatives
      (qpt per-tensor, qtok per-token e4m3)
  (d) stock mxfp8 W8A8 group/group re-run (mx8) + w4a16/w4a4 reruns
      (g16/q4/g4) for same-window comparability with r120d.

Cells per shape/M (layouts wheel-exact per r120d: B = w.t() nt view,
group scales [K/32, N] contiguous e8m0):
  bf16   = torch.matmul reference
  mx8    = fp8_gemm(x8g, w.t(), bf16, x_sc_g e8m0 [M,K/32], ws_kn e8m0)   [stock mxfp8]
  pt     = fp8_gemm(x8pt, w.t(), bf16, a_pt f32[1], w_pt f32[1])          [champion static class]
  tok    = fp8_gemm(x8tok, w.t(), bf16, a_tok f32[M,1], w_pc f32[1,N])    [llmcompressor dynamic class]
  w16pc  = fp8_gemm_w8a16(x, w.t(), w_pc f32[1,N], None)                  [(bf16,f8) joint, cheap scales]
  w16pt  = fp8_gemm_w8a16(x, w.t(), w_pt f32[1], None)
  q8     = qop8(x) cost alone        qpt/qtok = cheap e4m3 quant costs
  g16    = fp4_gemm_w4a16(x, Bt, ws_kn, bf16, None)                       [cert20 incumbent]
  q4/g4  = w4a4 quant/gemm (stock)
Numerics: every quant cell gets a rel-err vs the fp32 dequant reference
built from the ACTUAL stored quantized values (catches scale-mask
misinterpretation -- a misindexed attr explodes rel).

Timing: torch.xpu.Event pairs, 15 warmup, 50 iters, median. Run with
ONEDNN_VERBOSE=2 for impl-selection lines.
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
MS = [1, 8, 4096, 32768]

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
# package __init__ does NOT dlopen the extension -- explicit submodule
# import or the op-presence probe false-negatives (r120b FATAL)
import vllm_xpu_kernels._xpu_C  # noqa: F401
# _xpu_ops registers torch.ops.vllm::xpu_mxfp*_quantize; ORDER matters
import vllm._xpu_ops  # noqa: F401

for op in ("fp8_gemm", "fp8_gemm_w8a16", "fp4_gemm", "fp4_gemm_w4a16"):
    assert hasattr(torch.ops._xpu_C, op), f"FATAL: {op} absent"
print("ops PRESENT: fp8_gemm fp8_gemm_w8a16 fp4_gemm fp4_gemm_w4a16")

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    xpu_mxfp4_quantize as qop4,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    xpu_mxfp8_quantize as qop8,
)


def qpt(x):
    """per-tensor e4m3 dynamic quant (champion act-quant class)."""
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
        with torch.no_grad():
            yref_bf = torch.matmul(x.float(), wref.float())

        # ---- quantized activations (built ONCE; gemm cells bench on
        # these pre-quantized tensors, mirroring r120d semantics) ----
        x8g = qop8(x)           # (x8, sc e8m0 [M, K/32])
        x8pt = qpt(x)           # (x8, sc f32 [1])
        x8tok = qtok(x)         # (x8, sc f32 [M, 1])

        def _rel(y, ref):
            return float((y.float() - ref).norm() / ref.norm())

        # ---- mx8: stock mxfp8 W8A8 group/group ----
        try:
            def gmx():
                return torch.ops._xpu_C.fp8_gemm(
                    x8g[0], Bt8, torch.bfloat16, x8g[1], ws8_kn, None
                )
            row["mx8"] = bench(gmx)
            with torch.no_grad():
                if x8g[1].dim() == 2 and x8g[1].shape[1] == K // 32:
                    xdq = x8g[0].float() * e8m0_depow(
                        x8g[1]
                    ).repeat_interleave(32, dim=1)
                    wdq = w8.float().t() * e8m0_depow(
                        ws8_kn
                    ).repeat_interleave(32, dim=0)  # [K, N]
                    row["rel_mx8"] = _rel(gmx(), xdq @ wdq)
                else:
                    print(
                        f"[{tag} M={M} mx8] scale geom {tuple(x8g[1].shape)}"
                        " != [M, K/32] -- rel skipped"
                    )
                    row["rel_mx8"] = None
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} mx8] FAILED: {type(ex).__name__}: {ex}")
            row["mx8"] = None
            row["rel_mx8"] = None

        # ---- pt: champion static class (per-tensor/per-tensor) ----
        try:
            def gpt():
                return torch.ops._xpu_C.fp8_gemm(
                    x8pt[0], Bt8, torch.bfloat16, x8pt[1], w_pt, None
                )
            row["pt"] = bench(gpt)
            with torch.no_grad():
                ref = (x8pt[0].float() * x8pt[1]) @ (w8.float().t() * w_pt)
                row["rel_pt"] = _rel(gpt(), ref)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} pt] FAILED: {type(ex).__name__}: {ex}")
            row["pt"] = row["rel_pt"] = None

        # ---- tok: llmcompressor dynamic class (per-token/per-channel) ----
        try:
            def gtok():
                return torch.ops._xpu_C.fp8_gemm(
                    x8tok[0], Bt8, torch.bfloat16, x8tok[1], w_pc, None
                )
            row["tok"] = bench(gtok)
            with torch.no_grad():
                ref = (x8tok[0].float() * x8tok[1]) @ (w8.float().t() * w_pc)
                row["rel_tok"] = _rel(gtok(), ref)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} tok] FAILED: {type(ex).__name__}: {ex}")
            row["tok"] = row["rel_tok"] = None

        # ---- w16pc / w16pt: (bf16, f8) joint, cheap weight scales ----
        for cell, sc in (("w16pc", w_pc), ("w16pt", w_pt)):
            try:
                def g16f(sc=sc):
                    return torch.ops._xpu_C.fp8_gemm_w8a16(x, Bt8, sc, None)
                row[cell] = bench(g16f)
                with torch.no_grad():
                    # sc is [1,N] (w16pc) or [1] (w16pt); both broadcast
                    # along K against the [K,N] dequantized weight
                    ref = x.float() @ (w8.float().t() * sc)
                    row["rel_" + cell] = _rel(g16f(), ref)
            except Exception as ex:  # noqa: BLE001
                print(f"[{tag} M={M} {cell}] FAILED: {type(ex).__name__}: {ex}")
                row[cell] = row["rel_" + cell] = None

        # ---- quant costs ----
        try:
            row["q8"] = bench(lambda: qop8(x))
            row["qpt"] = bench(lambda: qpt(x)[0])
            row["qtok"] = bench(lambda: qtok(x)[0])
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} quant] FAILED: {type(ex).__name__}: {ex}")
            row["q8"] = row["qpt"] = row["qtok"] = None

        # ---- w4a16 incumbent + w4a4 stock (same-window comparability) ----
        w = torch.randint(
            0, 255, (N, K // 2), dtype=torch.uint8, device=dev
        ).view(torch.float4_e2m1fn_x2)
        ws = torch.randint(
            120, 136, (N, K // 32), dtype=torch.uint8, device=dev
        ).view(torch.float8_e8m0fnu)
        ws_kn = ws.t().contiguous()
        Bt = w.t()
        try:
            x4 = qop4(x)

            def g4():
                return torch.ops._xpu_C.fp4_gemm(
                    x4[0], Bt, x4[1], ws_kn, torch.bfloat16, None
                )
            row["q4"] = bench(lambda: qop4(x))
            row["g4"] = bench(g4)
        except Exception as ex:  # noqa: BLE001
            print(f"[{tag} M={M} w4a4] FAILED: {type(ex).__name__}: {ex}")
            row["q4"] = row["g4"] = None
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

        rows.append(row)
        print(
            "%-11s M=%-6d bf16=%7.2f | mx8=%7.2f(r%.0e) pt=%7.2f(r%.0e)"
            " tok=%7.2f(r%.0e) w16pc=%7.2f(r%.0e) w16pt=%7.2f(r%.0e)"
            " | q8=%6.3f qpt=%6.3f qtok=%6.3f | g16=%7.2f(r%.0e)"
            " q4=%6.3f g4=%7.2f"
            % (
                tag, M, row["bf16"],
                *[v if v is not None else -1 for v in (
                    row["mx8"], row["rel_mx8"], row["pt"], row["rel_pt"],
                    row["tok"], row["rel_tok"], row["w16pc"],
                    row["rel_w16pc"], row["w16pt"], row["rel_w16pt"])],
                *[v if v is not None else -1 for v in (
                    row["q8"], row["qpt"], row["qtok"])],
                *[v if v is not None else -1 for v in (
                    row["g16"], row["rel_g16"], row["q4"], row["g4"])],
            )
        )
        del x, wref, w8, w, x8g, x8pt, x8tok
        gc.collect()
        torch.xpu.empty_cache()

print("\n=== [summary] medians ===")


def med(key, mfilter):
    vals = [r[key] for r in rows
            if mfilter(r["M"]) and r.get(key) is not None]
    return statistics.median(vals) if vals else float("nan")


for label, mf in (("decode (M<=8)", lambda m: m <= 8),
                  ("prefill (M>=4096)", lambda m: m >= 4096)):
    apply_mx8 = med("q8", mf) + med("mx8", mf)
    apply_pt = med("qpt", mf) + med("pt", mf)
    apply_tok = med("qtok", mf) + med("tok", mf)
    print(
        "%s: bf16=%.3f | APPLY mx8=%.3f pt=%.3f tok=%.3f w16pc=%.3f"
        " w16pt=%.3f g16=%.3f w4a4=%.3f | GEMM-only mx8=%.3f pt=%.3f"
        " tok=%.3f w16pc=%.3f"
        % (label, med("bf16", mf), apply_mx8, apply_pt, apply_tok,
           med("w16pc", mf), med("w16pt", mf), med("g16", mf),
           med("q4", mf) + med("g4", mf),
           med("mx8", mf), med("pt", mf), med("tok", mf), med("w16pc", mf))
    )
print("=== r122_kbench done ===")
