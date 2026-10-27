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
    return "vllm_gaudi.xpu.platform.ArcXpuPlatform"
