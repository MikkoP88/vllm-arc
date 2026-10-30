# SPDX-License-Identifier: Apache-2.0
"""Block-fp8 linear reroute to fp8_gemm_w8a16 (p44) — fp16 activations,
no per-call input quant, oneDNN block-scale native GEMM.

Evidence (r40, results/2026-10-29-p4-r40-forward-gemm-capture): same-silicon
two-image microbench at the six per-rank TP2 projections of qwen3.8-27b-fp8
(128x128 block scales) shows the stock block path — per-token-group input
quant + ``_xpu_C.fp8_gemm`` W8A8 — is a FIXED ~0.33 ms path per GEMM at
decode Ms (48-268 GB/s effective) on every layer shape, while
``_xpu_C.fp8_gemm_w8a16`` with fp16 activations and the block scales runs
0.064-0.167 ms (247-579 GB/s, the champion's achieved ~537 GB/s class):
3-8x per GEMM at decode Ms, 2.2-2.4x at prefill Ms, relerr 4-5e-4 vs the
fp32 dequant-block reference (strictly better numerics — no activation
quantization error). The op is in the stock 0.30 wheel; only the dispatch
differs. The D champion wheel (iexperimental v1.0.0) routes exactly this way
(``use_onednn -> fp8_gemm_w8a16(a16, B.t(), scale.t().contiguous(), None)``,
``apply_input_quant=False`` — "optimized XPU paths consume the original fp16
activation"), with its custom ESIMD GEMV reserved for M<=2 corners (~0.1
ms/step upside; not ported). Champion's own W8A8 probe is the same slow
class, so this is routing, not a torch-build delta (r40 closed that
suspect).

Mechanism: upstream 0.30's ``Fp8BlockScaledMMLinearKernel.apply_weights``
already ships the hook — ``apply_input_quant: ClassVar[bool]``; when False
the activation is passed through unquantized and ``As`` is a placeholder
that the subclass must not use. This patch flips that flag on
``XPUFp8BlockScaledMMKernel`` and reroutes ``apply_block_scaled_mm`` to
``fp8_gemm_w8a16`` when the incoming activation is fp16/bf16:

    fp8_gemm_w8a16(A, B.t(), Bs.t(), torch.Tensor())

B stays [N, K] (stock block layout; ``.t()`` is the [K, N] view — the exact
argument shapes the r40 microbench timed and numerics-validated). Bs is the
stock ``[n_blocks, k_blocks]`` view over the contiguous [k_blocks, n_blocks]
buffer, so ``Bs.t()`` is a zero-copy view recovery — no per-call copies. Any
fp8 activation (non-fp16 input, bmm layers) falls back to the stock W8A8
path unchanged; output dtype and bias handling stay the base class's.

Gate: ``VLLM_ARC_FP8_W8A16_DISABLE=1`` skips the install (A/B control).
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc")

_DISABLE_ENV = "VLLM_ARC_FP8_W8A16_DISABLE"


def _install(mod=None) -> None:
    """Install the reroute.

    Two callers share this one source: the product form — this module listed
    in ``ArcXpuPlatform._XPU_PATCH_MODULES`` (called with no args, imports
    the kernel module itself) — and the leg-overlay form — the serve-time
    sitecustomize meta-path hook, which passes the just-executed
    ``vllm.model_executor.kernels.linear.scaled_mm.xpu`` module so the
    patch lands inside the same import that resolves the class.
    """
    if os.environ.get(_DISABLE_ENV, "0") == "1":
        logger.info("ArcXpuPlatform: fp8 block w8a16 reroute DISABLED (%s)", _DISABLE_ENV)
        return

    if mod is not None:
        XPUFp8BlockScaledMMKernel = mod.XPUFp8BlockScaledMMKernel
    else:
        from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
            XPUFp8BlockScaledMMKernel,
        )

    if getattr(XPUFp8BlockScaledMMKernel, "_arc_w8a16_block_patched", False):
        return  # idempotent: platform hooks run at config time AND per-worker

    stock_apply = XPUFp8BlockScaledMMKernel.apply_block_scaled_mm

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # Reroute only the by-far-dominant case: dense block-fp8 linear with
        # an unquantized (fp16/bf16) activation — apply_input_quant=False
        # passthrough. Everything else keeps the stock W8A8 path.
        if A.dtype in (torch.float16, torch.bfloat16) and not getattr(
            self, "is_bmm", False
        ):
            return torch.ops._xpu_C.fp8_gemm_w8a16(
                A, B.t(), Bs.t(), torch.Tensor()
            )
        return stock_apply(self, A, B, As, Bs)

    XPUFp8BlockScaledMMKernel.apply_input_quant = False
    XPUFp8BlockScaledMMKernel.apply_block_scaled_mm = apply_block_scaled_mm
    XPUFp8BlockScaledMMKernel._arc_w8a16_block_patched = True
    logger.info(
        "ArcXpuPlatform: fp8 block linear reroute ACTIVE — "
        "XPUFp8BlockScaledMMKernel.apply_input_quant=False, "
        "apply_block_scaled_mm -> _xpu_C.fp8_gemm_w8a16 (stock fallback kept; "
        "r40: 3-8x/GEMM decode Ms, relerr 4-5e-4)"
    )
