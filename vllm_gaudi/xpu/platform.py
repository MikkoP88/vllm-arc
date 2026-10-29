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
import sys
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

# In-place XPU core patches installed from both platform hooks (config time
# + per-worker init_device). Each module owns its idempotence and logs its
# own state (visibility law — a silent non-install is indistinguishable
# from a deferred patch, p2r2 lesson); an install failure must never break
# platform resolution.
_XPU_PATCH_MODULES = (
    "graph_safe_output",        # defect-b: graph-safe worker output copies
    "speculator_full_capture",  # defect-a: PIECEWISE-family -> FULL coercion
    "first_request_warmup",     # defect-b fix (lane W): warm the first eager
                                # large TP AR before serving traffic (r16/r17)
    "gdn_metadata_fastpath",    # p10: steady spec-decode GDN metadata builds
                                # without blocking H2D (r19-r22a; <=5ms law)
    "fp8_block_w8a16",          # p44: block-fp8 linear -> fp8_gemm_w8a16 with
                                # fp16 activations, input quant skipped (r40:
                                # 3-8x/GEMM decode Ms, fixed-cost W8A8 path)
)


def _install_xpu_patches(site: str) -> None:
    from importlib import import_module

    for name in _XPU_PATCH_MODULES:
        try:
            module = import_module(f"vllm_gaudi.xpu.patches.{name}")
            module._install()
        except Exception as e:  # noqa: BLE001 — visibility law: leg logs must show it
            logger.warning(
                "ArcXpuPlatform: %s install failed at %s: %r", name, site, e)


# Wheel-module exec hooks (P3 cache glues + p44 lane): patches that must
# rebind symbols INSIDE a just-imported core/lmcache module can't be
# in-place class patches — they wrap the module's exec. Product form of the
# r30 cap sitecustomize hooks (one source, two callers: each patch module's
# ``_install(mod)`` receives the executed target module). The meta-path
# finder is inserted at register() time — the earliest moment in every vllm
# process, before any v1/lmcache worker module imports — and inserting it
# imports nothing (circular-import safe, unlike the in-place patches above).
# Each hook fires once at its target's first import; the finder removes
# itself after the last target has fired.
_WHEEL_HOOKS = (
    # (target module, patches submodule under vllm_gaudi.xpu.patches)
    ("vllm.v1.worker.mamba_utils", "mamba_lmcache_glue"),          # p41
    ("vllm.v1.worker.gpu.model_states.mamba_hybrid", "mamba_hybrid_snap_glue"),  # p41c
    ("vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector", "lmcache_hma_connector"),  # p41b
    ("lmcache.v1.platform.xpu.device_ops", "lmc_xpu_pinned_env"),  # p40c
    ("lmcache.v1.gpu_connector.kv_format.detectors.vllm", "lmcint.lmc_kvfmt_detector"),  # r43
    ("lmcache.v1.platform.torch_ops", "lmcint.lmc_twohs_transfer"),  # r43
    ("vllm.v1.core.sched.scheduler", "sched_tail_split"),          # r35
)
# gdn_metadata_fastpath and fp8_block_w8a16 are NOT here: they self-install
# in-place (_XPU_PATCH_MODULES) — no exec-wrap needed.


def _load_wheel_patch(patch_name: str):
    """Import a _WHEEL_HOOKS patch, standalone when it lives in a sub-package.

    The lmcint kit's ``__init__`` IS the lmcache_integration entry (it
    imports the heavy vllm_v1_adapter -> lmcache.v1.cache_engine at module
    level), and two of its files hook lmcache-internal targets that exec
    MID lmcache package init (detectors.vllm finishes while cache_engine is
    still executing; torch_ops while lmcache/__init__ is). Importing those
    patches as regular package submodules would first execute the package
    ``__init__`` and re-enter the partially initialized lmcache package —
    the circular-import death of the first p45 image boot (r42). The r30
    cap sitecustomize loaded these files standalone via
    spec_from_file_location for eleven legs; same semantics here — one
    source, two callers. Top-level patch modules (no dot) import normally:
    their package __init__ is the light patches/__init__.
    """
    from importlib import import_module
    from importlib.util import module_from_spec, spec_from_file_location

    if "." not in patch_name:
        return import_module(f"vllm_gaudi.xpu.patches.{patch_name}")
    qualname = f"vllm_gaudi.xpu.patches.{patch_name}"
    if qualname in sys.modules:
        return sys.modules[qualname]
    import vllm_gaudi.xpu.patches as _patches

    path = os.path.join(os.path.dirname(_patches.__file__),
                        *patch_name.split(".")) + ".py"
    spec = spec_from_file_location(qualname, path)
    module = module_from_spec(spec)
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    return module


class _WheelHookFinder:
    """One-shot meta-path finder wrapping the exec of each _WHEEL_HOOKS
    target so its patch module can rebind symbols inside it."""

    def __init__(self) -> None:
        self._targets = {t for t, _ in _WHEEL_HOOKS}

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102
        if fullname not in self._targets:
            return None
        from importlib.util import find_spec

        sys.meta_path.remove(self)
        try:
            spec = find_spec(fullname)
        finally:
            if self._targets:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            self._targets.discard(fullname)
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            self._targets.discard(fullname)
            if not self._targets and self in sys.meta_path:
                sys.meta_path.remove(self)  # last hook fired — retire
            patch_name = dict(_WHEEL_HOOKS)[fullname]
            try:
                patch = _load_wheel_patch(patch_name)
                patch._install(module)
                logger.info(
                    "ArcXpuPlatform: wheel hook %s <- patches.%s OK",
                    fullname.rsplit(".", 1)[-1], patch_name)
            except Exception as e:  # noqa: BLE001 — visibility law
                logger.warning(
                    "ArcXpuPlatform: wheel hook %s <- patches.%s FAILED: %r",
                    fullname, patch_name, e)

        spec.loader.exec_module = exec_module
        return spec


def _install_wheel_hooks() -> None:
    """Idempotent install of the exec-wrap hooks (register()-time, every
    vllm process on Arc hosts)."""
    if any(isinstance(f, _WheelHookFinder) for f in sys.meta_path):
        return
    if not _WHEEL_HOOKS:
        return
    sys.meta_path.insert(0, _WheelHookFinder())


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
        # XPU in-place patches (defect-b graph-safe outputs, defect-a
        # speculator FULL-capture): idempotent per-process install — this
        # runs at config time in APIServer + EngineCore; the per-worker
        # guarantee comes from check_if_supports_dtype below.
        _install_xpu_patches("check_and_update_config")
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
        # GMU conditional relax (VLLM_ARC_GMU_RELAX): when the engine default
        # utilization is still in place and the env pins a sanctioned Arc
        # fraction, relax to it instead of failing boot on headroom. 0.8 is
        # the D-champion certified value (rig law, plan Appendix I #11);
        # 0 (default) disables. A user-passed --gpu-memory-utilization is
        # always respected (only the engine default 0.92 is relaxed).
        from vllm_gaudi.envs import VLLM_ARC_GMU_RELAX

        cache_config = vllm_config.cache_config
        if 0.0 < VLLM_ARC_GMU_RELAX <= 1.0 and cache_config.gpu_memory_utilization == 0.92:
            logger.info(
                "ArcXpuPlatform: relaxing gpu_memory_utilization %.2f -> %.2f "
                "(VLLM_ARC_GMU_RELAX; D-champion certified B70 posture).",
                cache_config.gpu_memory_utilization, VLLM_ARC_GMU_RELAX)
            cache_config.gpu_memory_utilization = VLLM_ARC_GMU_RELAX
        logger.info("ArcXpuPlatform: Intel Arc Xe2 posture active (%s).",
                         torch.xpu.get_device_name(0))
        # Graph posture visibility (one line per phase, upstream lmcache
        # logging style): core honors VLLM_XPU_ENABLE_XPU_GRAPH above; log
        # the effective mode so every rig leg's serve log states its graph
        # posture explicitly. Standing rig law: all graph legs run with
        # VLLM_XPU_ENABLE_XPU_GRAPH=1 (plan Appendix F).
        from vllm.envs import VLLM_XPU_ENABLE_XPU_GRAPH

        compilation_config = vllm_config.compilation_config
        logger.info(
            "ArcXpuPlatform: graph posture cudagraph_mode=%s "
            "capture_sizes=%s (VLLM_XPU_ENABLE_XPU_GRAPH=%s).",
            getattr(compilation_config.cudagraph_mode, "name",
                    compilation_config.cudagraph_mode),
            tuple(compilation_config.cudagraph_capture_sizes)
            if compilation_config.cudagraph_capture_sizes else None,
            VLLM_XPU_ENABLE_XPU_GRAPH)

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        # Per-worker install point: XPUWorker.init_device calls this after
        # the import is complete — the guaranteed post-import moment in every
        # WorkerProc (the register()-time attempt can hit a half-initialized
        # vllm package). Idempotent; failure surfaces in the gate leg.
        _install_xpu_patches("check_if_supports_dtype")
        super().check_if_supports_dtype(dtype)
