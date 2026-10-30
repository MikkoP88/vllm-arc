"""Speculator FULL-capture coercion (defect-a family, plan Appendix C).

vLLM 0.30's v1 default cudagraph mode is ``FULL_AND_PIECEWISE``
(config/compilation.py: "(v1 default)"). When a speculator
(``BaseSpeculator.init_cudagraph_manager``) receives a PIECEWISE-family
mode, its PREFILL graph manager captures the draft routine piecewise:
compiled regions become graph segments and the rest — compute_logits,
gumbel sampling — runs eagerly BETWEEN segments
(``BreakableCUDAGraphCapture.add_eager``), with offloader copy-stream
joins recorded INSIDE the capture context
(``breakable_cudagraph.py: _capture`` → ``join_after_forward``).

On XPU (L0 command graphs) that route boot-crashes: the piecewise capture
path performs device-sync-class operations inside an open command graph
— the same error family as defect (b) — observed on the radiance v0.30.0
image running qwen3.8-27b-fp8 (plan Appendix C defect (a): "speculator
capture() routes to PIECEWISE, which performs a device sync inside graph
capture → boot crash"). Additionally, inductor-piecewise × spec-decode
has a wrong-logits history on this model family (0.26: acceptance
collapse 22.5 → 79.7 solo recovered only by TORCH_COMPILE_DISABLE — plan
falsified-path #1), so piecewise brings no value here even where it
boots.

Fix (plan Appendix C fix direction, first arm): force the FULL capture
path for the speculator. Every speculator family (autoregressive, MTP,
multi-module MTP, eagle-line consumers of the autoregressive machinery)
builds its managers through ``SpeculatorCudaGraphManager`` — coercing
the mode in its ``__init__`` covers all three instantiation sites with
one hook:

    PIECEWISE | FULL_AND_PIECEWISE  ->  FULL

``FULL_DECODE_ONLY`` / ``FULL`` / ``NONE`` pass through untouched (the
autoregressive speculator already downgrades its own decode manager to
FULL_DECODE_ONLY/NONE before construction —
spec_decode/autoregressive/speculator.py:136-141, "PIECEWISE cudagraphs
are not supported for draft decodes").

Note: FULL draft-prefill graphs capture gumbel sampling inside one
graph. If the P2 spec legs show XPU RNG-in-graph failures, the coercion
target iterates to NONE (eager draft prefill) — evidence-driven, per
patch law; never without a leg.

Provenance: plan Appendix C defect (a) + falsified-path #1 REVAL note;
fix direction "force the FULL capture path for the speculator".
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger("vllm.platforms.arc")

_INSTALLED = False


def _coerce_mode(cudagraph_mode):
    """PIEWISE-family -> FULL (defect-a law); everything else unchanged."""
    try:
        from vllm.config.compilation import CUDAGraphMode

        if cudagraph_mode in (
                CUDAGraphMode.PIECEWISE,
                CUDAGraphMode.FULL_AND_PIECEWISE,
        ):
            return CUDAGraphMode.FULL, True
    except Exception:  # noqa: BLE001 — never break speculator init
        pass
    return cudagraph_mode, False


def _install() -> None:
    """Patch SpeculatorCudaGraphManager in place (idempotent, per-process).

    Call sites: the same platform hooks that install graph_safe_output
    (check_and_update_config / check_if_supports_dtype — the worker-side
    hook is the guaranteed one; speculators are built in worker
    processes).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils \
        import SpeculatorCudaGraphManager

    _orig_init = SpeculatorCudaGraphManager.__init__
    _coerced_logs = 0

    def __init__(self, *args, **kwargs):
        # cudagraph_mode is the 3rd positional (vllm_config, device,
        # cudagraph_mode, ...) or a kwarg — coerce in place, pass the rest
        # through untouched.
        nonlocal _coerced_logs
        coerced = False
        if "cudagraph_mode" in kwargs:
            mode, coerced = _coerce_mode(kwargs["cudagraph_mode"])
            if coerced:
                kwargs["cudagraph_mode"] = mode
        elif len(args) >= 3:
            mode, coerced = _coerce_mode(args[2])
            if coerced:
                args = (*args[:2], mode, *args[3:])
        if coerced and _coerced_logs < 3:
            _coerced_logs += 1
            logger.info(
                "ArcXpuPlatform: speculator graph mode coerced "
                "PIEWISE-family -> FULL (defect-a law, plan Appendix C "
                "— piecewise draft capture boot-crashes on XPU command "
                "graphs). mode_in=%s", mode)
        _orig_init(self, *args, **kwargs)

    SpeculatorCudaGraphManager.__init__ = __init__
    _INSTALLED = True
    logger.info(
        "ArcXpuPlatform: speculator FULL-capture patch installed "
        "(pid %d, SpeculatorCudaGraphManager.__init__ wrapped).", os.getpid())
