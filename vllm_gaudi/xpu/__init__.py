# SPDX-License-Identifier: Apache-2.0
"""vllm-arc XPU device tree (Intel Arc / Xe2 GPUs).

Device-wide XPU home for the vllm-arc fork — the generalized successor of
the radiance fork's model-scoped ``models/qwen4_exp/xpu/`` homes. The HPU
plugin surface stays in the parent ``vllm_gaudi`` package and stays dormant
on XPU hosts (see ``vllm_gaudi/__init__.py``).
"""

from __future__ import annotations


def register():
    """Platform-plugin probe for Intel Arc (Xe2/BMG) XPU devices.

    Returns the ``ArcXpuPlatform`` qualname when the host carries an Arc/Xe2
    GPU, so the plugin overrides core's builtin XPUPlatform for Arc-specific
    posture. Returns ``None`` on every other host so vLLM keeps resolving:
    non-Arc XPU hosts fall back to the builtin XPUPlatform, Gaudi hosts are
    served by the sibling ``vllm_gaudi:register`` (HpuPlatform) entry point,
    and CPU/CUDA/ROCm hosts never touch this module's platform.

    A platform plugin returning ``None`` counts as "not activated" in
    ``resolve_current_platform_cls_qualname`` — the dormant-plugin mechanism
    proven on the 0.30 stack on the dual-B70 rig.
    """
    try:
        import torch

        if not torch.xpu.is_available():
            return None
        from vllm_gaudi.xpu.platform import is_xe2_arch

        if not is_xe2_arch():
            return None
    except Exception:
        # A probe failure must never break platform resolution: defer to
        # whatever core resolves on its own.
        return None

    # Wheel-module exec hooks (P3 glues' product wiring, platform.py
    # _WHEEL_HOOKS): inserted at the earliest per-process moment — before
    # any v1/lmcache worker module imports — and importing nothing, so it
    # cannot trip the circularity the in-place patches above hit. Each hook
    # fires once at its target's first import.
    try:
        from vllm_gaudi.xpu.platform import _install_wheel_hooks

        _install_wheel_hooks()
    except Exception as e:  # noqa: BLE001 — must not break platform resolution
        try:
            from vllm.logger import init_logger

            init_logger("vllm.platforms.arc").warning(
                "ArcXpuPlatform: wheel-hook install failed at register(): "
                "%r", e)
        except Exception:  # noqa: BLE001
            pass

    # XPU graph-safety patches (defect-b family, plan Appendix C): installed
    # in every vllm process on Arc hosts before worker modules bind their
    # classes. In-place class patching survives later ``from ... import``
    # rebinding. Expected to FAIL here — this runs mid platform-resolution,
    # where importing vllm.v1.worker.gpu is circular; the platform hooks
    # (check_and_update_config / check_if_supports_dtype) retry post-import.
    # Logged, never silent, and never fatal to resolution.
    try:
        from vllm_gaudi.xpu.patches import graph_safe_output

        graph_safe_output._install()
    except Exception as e:  # noqa: BLE001 — must not break platform resolution
        try:
            from vllm.logger import init_logger

            init_logger("vllm.platforms.arc").warning(
                "ArcXpuPlatform: graph-safe output install failed at "
                "register() (expected mid-resolution; platform hooks retry): "
                "%r", e)
        except Exception:  # noqa: BLE001 — logging must never break resolution
            pass

    return "vllm_gaudi.xpu.platform.ArcXpuPlatform"
