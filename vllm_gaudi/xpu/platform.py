# SPDX-License-Identifier: Apache-2.0
"""Arc / Xe2 XPU platform for vllm-arc.

``ArcXpuPlatform`` refines core vLLM's builtin ``XPUPlatform``
(``vllm/platforms/xpu.py``) for Intel Arc discrete GPUs (BMG-G31 die: Arc
Pro B60/B70 — Xe2 cores, 32 GB GDDR6 each, one per socket, TP2 on the
project rig). Core's XPU posture is inherited unchanged (XPUWorker,
``VLLM_XPU_ENABLE_XPU_GRAPH`` gating, XPU-unsupported fusion passes off,
spawn multiprocessing, graceful oneCCL shutdown); Arc-specific behavior is
layered on top. Non-Arc XPU hosts never see this class — the platform-plugin
probe (``vllm_gaudi.xpu:register``) defers to core's XPUPlatform for them,
and Gaudi hosts resolve ``HpuPlatform`` via the sibling entry point.

Registration is by package name + entry point (upstream v0.30.0 tag has no
pyproject entry-points; the group used here is ``vllm.platform_plugins``,
declared in setup.py alongside upstream's ``hpu`` entry).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms.xpu import XPUPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Log under the vllm namespace: vLLM's serve-time dictConfig attaches its
# handler only to the "vllm" logger (propagate=False, no root handler), so a
# vllm_gaudi.* logger's INFO records are silently dropped in serve processes
# (warnings alone survive via logging.lastResort). The posture lines are P1
# gate evidence and must appear in serve logs.
logger = init_logger("vllm.platforms.arc")

# BMG (Battle Mage) discrete-Arc PCI device-id range. The Arc Pro B70 on the
# project rig reports device_id 0xe223 and the friendly name is only
# "Intel(R) Graphics [0xe223]" (rig-verified 2026-10-27), so the id — not a
# name string, and not the opaque integer `architecture` property — is the
# stable gate.
_BMG_DEVICE_ID_MIN = 0xE200
_BMG_DEVICE_ID_MAX = 0xE2FF


def is_xe2_arch(device_id: int = 0) -> bool:
    """True when the XPU device at ``device_id`` is a BMG/Xe2 discrete Arc.

    ``torch.xpu.get_device_properties().architecture`` returns an opaque
    integer enum rather than a name string on this stack, so the PCI
    device-id range is the gate: every BMG-G31 part (incl. Arc Pro B70,
    id 0xe223) lives in 0xE200..0xE2FF. The probe never raises — callers use
    it during platform resolution where an exception would count as an
    activated-but-broken plugin.
    """
    try:
        props = torch.xpu.get_device_properties(device_id)
    except Exception:
        return False
    devid = getattr(props, "device_id", -1)
    return _BMG_DEVICE_ID_MIN <= devid <= _BMG_DEVICE_ID_MAX


class ArcXpuPlatform(XPUPlatform):
    """Intel Arc (Xe2/BMG) refinement of core's builtin XPUPlatform.

    Inherits ``PlatformEnum.XPU`` and every core XPU behavior; overrides
    only add Arc/Xe2 posture. Keeping the XPU enum is deliberate — core
    checks that treat the platform as XPU keep working unchanged.
    """

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Core XPU posture first: XPU-graph env law, fusion-pass disabling,
        # XPUWorker selection, spawn multiproc, oneCCL shutdown timeout.
        super().check_and_update_config(vllm_config)
        logger.info_once("ArcXpuPlatform: Intel Arc Xe2 posture active (%s).",
                         torch.xpu.get_device_name(0))
