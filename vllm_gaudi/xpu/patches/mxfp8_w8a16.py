# SPDX-License-Identifier: Apache-2.0
"""MXFP8 linear reroute to fp8_gemm_w8a16_mxfp8 (p631) — bf16 activations,
no per-call activation quant, WEIGHTS-side group-32 e8m0 scales.

Root cause this addresses (r122 kbench, banked under
``results/2026-11-08-p631-mxfp8-lanes/``): stock
``XPUMxFp8LinearKernel.apply_weights`` quantizes activations to mxfp8
group-32 e8m0 every call, then calls ``_xpu_C.fp8_gemm`` — a joint
(f8, f8) matmul. Measured on the arc4test image (oneDNN v3.13, 6 real
layer shapes x M in {1, 8, 4096, 32768}): the f8xf8 joint rate itself
— not the scale flavor — is the wall. Per-tensor scales 48.10 ms,
per-token 48.20 ms, group-e8m0 60.74 ms vs bf16 22.98 ms at
qwen.up M=32768 (2.09-2.64x slower), while the (bf16, f8) joint
W8A16 class runs 23.69 ms = 1.03x bf16 with per-channel scales. At
decode M=1 the stock path pays quant + GEMM (0.34-0.40 ms) where a
weight-only apply costs 0.17-0.26 ms with zero quant.

The w8a16 op (arc5 wheel, ``csrc/xpu/onednn/fp8_gemm_w8a16_mxfp8.h``)
keeps activations bf16/f16 — bf16 is in the gemmstone JIT
counter-operand list, so the primitive JITs at bf16-class rate — and
rides the WEIGHTS-side group-32 e8m0 scale attr {32,1}, the proven-
cheap side (same attr shape ``fp4_gemm_w4a16`` uses). Numerics change:
activation quantization to e4m3 group-32 disappears from every linear
(it strictly improves over mxfp8 W8A8's e4m3 activations; weight
quantization and scales are identical). PWAL layouts are unchanged —
stock and this route share them (weight = [N, K] f8, ``.t()`` view at
apply; weight_scale = [K//32, N] contiguous e8m0 stored as a ``.t()``
view).

Gate: ``ARC_MXFP8_MODE`` — unset/``w8a16`` installs (when the op
exists in the loaded wheel), ``w8a8`` keeps stock (A/B control), any
other value WARNs and keeps stock. On wheels without the op (pre-arc5)
the install WARNs and keeps stock — a silent non-install is
indistinguishable from the deferred patch (visibility law).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc")

_MODE_ENV = "ARC_MXFP8_MODE"


def _install(mod=None) -> None:
    """Install the reroute.

    Product form: this module listed in ``ArcXpuPlatform._XPU_PATCH_MODULES``
    (called with no args, imports the kernel module itself). The ``mod``
    arg mirrors the mxfp4_w4a16 two-caller shape for leg overlays.
    """
    mode = os.environ.get(_MODE_ENV, "w8a16").strip().lower()
    if mode == "w8a8":
        logger.info(
            "ArcXpuPlatform: mxfp8 w8a16 reroute OFF (%s=w8a8 — stock W8A8)",
            _MODE_ENV)
        return
    if mode != "w8a16":
        logger.warning(
            "ArcXpuPlatform: unknown %s=%r — keeping stock W8A8 (valid: "
            "w8a16, w8a8)", _MODE_ENV, mode)
        return

    # ``vllm_xpu_kernels.__init__`` does NOT dlopen the extension — a bare
    # package import leaves the torch.ops namespace empty (verified in the
    # arc4test image: even stock fp8_gemm is absent until the submodule is
    # imported). Force the load before probing or the probe false-negatives.
    try:
        import vllm_xpu_kernels._xpu_C  # noqa: F401
    except ImportError:
        logger.warning(
            "ArcXpuPlatform: mxfp8 w8a16 reroute SKIPPED — "
            "vllm_xpu_kernels._xpu_C not importable; stock W8A8 stays")
        return

    try:
        torch.ops._xpu_C.fp8_gemm_w8a16_mxfp8  # noqa: B018 — presence probe
    except AttributeError:
        logger.warning(
            "ArcXpuPlatform: mxfp8 w8a16 reroute SKIPPED — "
            "_xpu_C.fp8_gemm_w8a16_mxfp8 absent from the loaded wheel "
            "(pre-arc5); stock W8A8 stays")
        return

    if mod is not None:
        XPUMxFp8LinearKernel = mod.XPUMxFp8LinearKernel
    else:
        from vllm.model_executor.kernels.linear.mxfp8.xpu import (
            XPUMxFp8LinearKernel, )

    if getattr(XPUMxFp8LinearKernel, "_arc_w8a16_patched", False):
        return  # idempotent: platform hooks run at config time AND per-worker

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # bf16/f16 activations straight into the JIT-eligible (bf16, f8)
        # joint GEMM — the per-call group-32 activation quant and the
        # f8xf8 joint rate it fed are what this patch removes. PWAL
        # layouts are identical to stock W8A8's: weight [N, K] ``.t()``
        # view, weight_scale [K//32, N] contiguous e8m0 via ``.t()``.
        return torch.ops._xpu_C.fp8_gemm_w8a16_mxfp8(
            x,
            layer.weight.t(),
            layer.weight_scale.t(),
            x.dtype,
            bias,
        )

    XPUMxFp8LinearKernel.apply_weights = apply_weights
    XPUMxFp8LinearKernel._arc_w8a16_patched = True
    logger.info(
        "ArcXpuPlatform: mxfp8 w8a16 reroute ACTIVE — "
        "XPUMxFp8LinearKernel.apply_weights -> "
        "_xpu_C.fp8_gemm_w8a16_mxfp8 (no per-call activation quant; "
        "stock w8a8 via %s=w8a8)", _MODE_ENV)
