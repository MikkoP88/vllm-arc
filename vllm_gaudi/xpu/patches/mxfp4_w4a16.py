# SPDX-License-Identifier: Apache-2.0
"""MXFP4 linear reroute to fp4_gemm_w4a16 (p630 FIX-P) — bf16 activations,
no per-call activation quant, oneDNN weight-decomposition GEMM.

Root cause this fixes (r116-r119 diagnosis, banked
``results/2026-11-08-p630-mxfp4-perf/``): stock
``XPUMxFp4LinearKernel.apply_weights`` quantizes activations to mxfp4
group-32 every call, then calls ``_xpu_C.fp4_gemm`` — a joint-dtypes
(f4, f4, bf16) matmul whose f4 ACTIVATION operand fails the oneDNN
gemmstone JIT eligibility gate (v3.13 ``src/gpu/intel/gemm/jit/pd.cpp``
``wei_decomp()``: the counter-operand type must be one of
f16/f32/bf16/f8_e5m2/f8_e4m3 — f4 is not in that list), so the primitive
lands on the matmul ``ref`` kernel while fp8 W8A8 rides JIT DPAS.
Measured on cert19 (oneDNN v3.12 runtime, r118/r119b microbench): the
per-call activation quant alone costs about as much as the GEMM at decode
Ms (~15-20 ms/token e2e over ~60 layers x 3-4 projections), and the ref
kernel is shape-asymmetric at prefill (down_proj 1.9x SLOWER than bf16 at
M=32768 while up_proj is 2.1x faster) — the r113 e2e mxfp4 ~1.9x
prefill/TTFT penalty vs fp8.

The w4a16 op (arc4 wheel, ``csrc/xpu/onednn/fp4_gemm_w4a16.h``) keeps
activations bf16/f16 — the gemmstone gate now passes (weights f4_e2m1 is
in the decomp type list, activations bf16 in the counter-operand list,
4 < 16 bits) — and the group-32 e8m0 weight scales ride the same
production weight-decomposition route ``int4_gemm_w4a16`` uses.
Numerics strictly improve: the activation quantization step disappears.
PWAL layouts are unchanged — stock W4A4 and this route share them
(weight = packed f4 ``.t()`` view, weight_scale = [k/32, n] e8m0).

Gate: ``ARC_MXFP4_MODE`` — unset/``w4a16`` installs (when the op exists
in the loaded wheel), ``w4a4`` keeps stock (A/B control), any other value
WARNs and keeps stock. On wheels without the op (pre-arc4) the install
WARNs and keeps stock — a silent non-install is indistinguishable from
the deferred patch (visibility law).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc")

_MODE_ENV = "ARC_MXFP4_MODE"


def _install(mod=None) -> None:
    """Install the reroute.

    Product form: this module listed in ``ArcXpuPlatform._XPU_PATCH_MODULES``
    (called with no args, imports the kernel module itself). The ``mod``
    arg mirrors the fp8_block_w8a16 two-caller shape for leg overlays.
    """
    mode = os.environ.get(_MODE_ENV, "w4a16").strip().lower()
    if mode == "w4a4":
        logger.info(
            "ArcXpuPlatform: mxfp4 w4a16 reroute OFF (%s=w4a4 — stock W4A4)",
            _MODE_ENV)
        return
    if mode != "w4a16":
        logger.warning(
            "ArcXpuPlatform: unknown %s=%r — keeping stock W4A4 (valid: "
            "w4a16, w4a4)", _MODE_ENV, mode)
        return

    # ``vllm_xpu_kernels.__init__`` does NOT dlopen the extension — a bare
    # package import leaves the torch.ops namespace empty (verified in the
    # arc4test image: even stock fp4_gemm is absent until the submodule is
    # imported). Force the load before probing or the probe false-negatives.
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        logger.warning(
            "ArcXpuPlatform: mxfp4 w4a16 reroute SKIPPED — "
            "vllm_xpu_kernels._xpu_C not importable; stock W4A4 stays")
        return

    try:
        torch.ops._xpu_C.fp4_gemm_w4a16  # noqa: B018 — presence probe
    except AttributeError:
        logger.warning(
            "ArcXpuPlatform: mxfp4 w4a16 reroute SKIPPED — "
            "_xpu_C.fp4_gemm_w4a16 absent from the loaded wheel (pre-arc4); "
            "stock W4A4 stays")
        return

    if mod is not None:
        XPUMxFp4LinearKernel = mod.XPUMxFp4LinearKernel
    else:
        from vllm.model_executor.kernels.linear.mxfp4.xpu import (
            XPUMxFp4LinearKernel, )

    if getattr(XPUMxFp4LinearKernel, "_arc_w4a16_patched", False):
        return  # idempotent: platform hooks run at config time AND per-worker

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # bf16/f16 activations straight into the weight-decomp GEMM — the
        # per-call group-32 activation quant is the decode killer this
        # patch removes. PWAL layouts are identical to stock W4A4's.
        return torch.ops._xpu_C.fp4_gemm_w4a16(
            x,
            layer.weight,
            layer.weight_scale,
            x.dtype,
            bias,
        )

    XPUMxFp4LinearKernel.apply_weights = apply_weights
    XPUMxFp4LinearKernel._arc_w4a16_patched = True
    logger.info(
        "ArcXpuPlatform: mxfp4 w4a16 reroute ACTIVE — "
        "XPUMxFp4LinearKernel.apply_weights -> _xpu_C.fp4_gemm_w4a16 "
        "(no per-call activation quant; stock w4a4 via %s=w4a4)",
        _MODE_ENV)
