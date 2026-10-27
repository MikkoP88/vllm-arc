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

import os
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


# Monkey-patch torch.accelerator.get_memory_info for Arc/Xe2 XPU — the same
# idiom as upstream gaudi's HPU patch at the top of vllm_gaudi/platform.py.
#
# On Arc B70 the accelerator-level query flips to a static device-properties
# view once vLLM's package has been imported: free memory reports exactly 0
# and the total shrinks to torch.xpu.get_device_properties().total_memory
# (31.89 -> 30.3 GiB). MemorySnapshot.measure (vllm/v1/worker) reads that
# query, so request_memory rejects an idle GPU ("Free memory on device
# xpu:N (0.0/30.3 GiB) ... is less than desired GPU memory utilization") and
# every EngineCore boot aborts in XPUWorker.init_device. torch.xpu
# .mem_get_info keeps the live driver view in every context probed
# (before/after allocations, after import vllm, after platform resolution,
# after xccl init). The wrapper keeps free/total self-consistent from one
# live source and passes non-XPU devices through to the original.
#
# Imported at platform-resolution time — always before any worker
# init_device. Idempotent by construction (module import happens once;
# the previous symbol, core's _C_cache_ops wrapper from
# vllm/platforms/xpu.py, is kept for passthrough). Implementation
# radiance-proven on the 0.30 stack (patch_xpu_accelerator_memory_info).
def _xpu_get_memory_info(device=None, /):
    if device is None:
        return torch.xpu.mem_get_info()
    if isinstance(device, int):
        return torch.xpu.mem_get_info(device)
    parsed = device if isinstance(device, torch.device) else torch.device(device)
    if parsed.type == "xpu" or (parsed.type == "" and parsed.index is not None):
        index = parsed.index if parsed.index is not None else torch.xpu.current_device()
        return torch.xpu.mem_get_info(index)
    return _ORIGINAL_GET_MEMORY_INFO(device)


import torch.accelerator as _torch_accelerator  # noqa: E402
import torch.accelerator.memory as _torch_accelerator_memory  # noqa: E402

# The pre-patch symbol (core's _C_cache_ops wrapper once vllm.platforms.xpu
# above has imported) is kept for non-XPU passthrough.
_ORIGINAL_GET_MEMORY_INFO = _torch_accelerator.get_memory_info

if torch.xpu.is_available():
    _torch_accelerator.get_memory_info = _xpu_get_memory_info
    # Patch the defining module too, so call sites that imported the symbol
    # from there (rather than the package re-export) see the same wrapper.
    _torch_accelerator_memory.get_memory_info = _xpu_get_memory_info


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
        # Dual-B70 ZE_AFFINITY law (rig law, plan Appendix F): one Arc per
        # socket, TP2 needs an explicit ZE_AFFINITY_MASK=0,1 so the level-zero
        # drivers of both cards are visible to the workers. Warn only — an
        # operator-set mask is always respected.
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        if tp_size > 1 and not os.environ.get("ZE_AFFINITY_MASK"):
            logger.warning(
                "ArcXpuPlatform: tensor_parallel_size=%d but ZE_AFFINITY_MASK "
                "is unset. The validated dual-B70 posture sets "
                "ZE_AFFINITY_MASK=0,1 (one Arc Pro B70 per socket); without "
                "it level-zero device enumeration can silently degrade.", tp_size)
        logger.info_once("ArcXpuPlatform: Intel Arc Xe2 posture active (%s).",
                         torch.xpu.get_device_name(0))
