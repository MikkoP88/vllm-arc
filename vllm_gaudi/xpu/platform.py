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
    from vllm.v1.attention.selector import AttentionSelectorConfig

    AttentionBackendEnum = str  # wheel type, annotation-only on this path

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
    "graph_safe_output",  # defect-b: graph-safe worker output copies
    "speculator_full_capture",  # defect-a: PIECEWISE-family -> FULL coercion
    "first_request_warmup",  # defect-b fix (lane W): warm the first eager
    # large TP AR before serving traffic (r16/r17)
    "gdn_metadata_fastpath",  # p10: steady spec-decode GDN metadata builds
    # without blocking H2D (r19-r22a; <=5ms law)
    "fp8_block_w8a16",  # p44: block-fp8 linear -> fp8_gemm_w8a16 with
    # fp16 activations, input quant skipped (r40:
    # 3-8x/GEMM decode Ms, fixed-cost W8A8 path)
    "step_trace",  # p620: per-step host timeline (ARC_STEP_TRACE,
    # silent-off diagnostic — envs.md diagnostics class)
    "ar_trace",  # p622: per-call collective census (ARC_AR_TRACE,
    # silent-off diagnostic — message sizes for the TP2-AR lane)
    "mxfp4_w4a16",  # p630 FIX-P: mxfp4 linear -> fp4_gemm_w4a16 with bf16
    # activations, per-call activation quant skipped (r118/r119b: quant op
    # ~= gemm cost at decode Ms; ARC_MXFP4_MODE gate, stock w4a4 kept)
    "mxfp8_w8a16",  # p631: mxfp8 linear -> fp8_gemm_w8a16_mxfp8 with bf16
    # activations, per-call activation quant + f8xf8 joint rate skipped
    # (r122: f8xf8 2.1-2.6x bf16 at prefill Ms, quant ~= gemm at decode;
    # ARC_MXFP8_MODE gate, stock w8a8 kept)
    "skinny_m",  # p633: M<=8 decode GEMMs -> arc6 skinny kernels (GEMV
    # regime: one work-item streams TN weight columns, dequant in
    # registers, fixed-order split-K) when the wheel carries the ops;
    # LAST in the chain — captures the apply_weights the lanes above
    # installed and only intercepts M<=8 (r123: stock routes 3-10x the
    # bytes floor at decode Ms; ARC_SKINNY_M gate, per-lane fallback)
    "graph_exec_doublebuffer",  # p635 D1: capture every XPU graph twice
    # into the shared pool, alternate replays — torch 2.13 same-exec
    # XPUGraph.replay() blocks until its prior launch retires (~24
    # ms/step serialized under async scheduling; leg J: XRP max 24 ->
    # 0.37 ms, device busy 27.7% -> ~80%; VLLM_ARC_GRAPH_EXEC_DB=0
    # opt-out)
)


def _install_xpu_patches(site: str) -> None:
    from importlib import import_module

    for name in _XPU_PATCH_MODULES:
        try:
            module = import_module(f"vllm_gaudi.xpu.patches.{name}")
            module._install()
        except Exception as e:  # noqa: BLE001 — visibility law: leg logs must show it
            logger.warning("ArcXpuPlatform: %s install failed at %s: %r", name, site, e)


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
    ("vllm.v1.worker.mamba_utils", "mamba_lmcache_glue"),  # p41
    ("vllm.v1.worker.gpu.model_states.mamba_hybrid", "mamba_hybrid_snap_glue"),  # p41c
    ("vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector", "lmcache_hma_connector"),  # p41b
    # p40c/p40d (lmc_xpu_pinned_env) ABSORBED into the lmcache wheel patch
    # e01-xpu-platform-ops.patch at 0.5.6 (P6.2): the 0.5.5 wrap targeted
    # lmcache.v1.platform.xpu.device_ops, a module the 0.5.6 platform
    # restructure deleted; the env-gated pinned-alloc re-pin and the
    # unconditional KV-transfer torch-baseline re-pin now live wheel-side
    # in XpuDeviceOps.ensure_native.
    ("lmcache.v1.gpu_connector.kv_format.detectors.vllm", "lmcint.lmc_kvfmt_detector"),  # r43
    ("lmcache.v1.platform.torch_ops", "lmcint.lmc_twohs_transfer"),  # r43
    ("vllm.v1.core.sched.scheduler", "sched_tail_split"),  # r35
    ("vllm.v1.attention.backends.turboquant_attn", "tq_spec_fix"),  # p619
    ("vllm.utils.torch_utils", "mx_dtype_map"),  # p647 leg B: MX kv dtypes
    # -> torch.uint8 allocation dtype (packed uint8-cache class)
    ("vllm.v1.kv_cache_interface", "mx_kv_mode"),  # p647 leg B: mode alias
    # element-class alias onto TURBOQUANT members (mode conveys only
    # packed-uint8-class semantics; format identity travels on the string)
)
# gdn_metadata_fastpath and fp8_block_w8a16 are NOT here: they self-install
# in-place (_XPU_PATCH_MODULES) — no exec-wrap needed.


def _widen_cache_dtype_literal(src: str) -> str:
    """P6.47 leg B: append mxfp4/mxfp8/mxk8v4 to the CacheDType Literal in
    vllm/config/cache.py — a TRUE schema widening (CLI admission, help
    text, config round-trip all see the real members), not an alias.
    Post-exec rebinding cannot widen it: pydantic builds the dataclass
    schema at class-decoration time inside the module exec, and pydantic
    2.13.5 has no rebuild for dataclass validators (rig-proven
    2026-11-11: ValidationError on construct, AttributeError on
    pydantic.rebuild). The anchor is the Literal's last member; the
    count assert keeps the edit surgical and fails loud on wheel drift.
    """
    anchor = '    "fp8_per_token_head",\n'
    add = '    "mxfp4",\n    "mxfp8",\n    "mxk8v4",\n'
    n = src.count(anchor)
    if n != 1:
        raise RuntimeError(
            f"CacheDType Literal anchor count {n} != 1 in vllm/config/cache.py "
            "(wheel drift — re-survey before shipping the MX KV lane)"
        )
    return src.replace(anchor, anchor + add)


# MX KV lane (P6.47): the three --kv-cache-dtype strings and the backend
# qualname both the enum member and get_attn_backend_cls resolve to.
_MX_KV_DTYPES = ("mxfp4", "mxfp8", "mxk8v4")
_MX_BACKEND_PATH = "vllm_gaudi.xpu.patches.mx_kv.engine.MxAttentionBackend"


def _add_mxkv_backend_member(src: str) -> str:
    """P6.47 leg B: give the MX KV lane its OWN backend-family member —
    MX (OCP microscaling) is not TurboQuant (Hadamard + Lloyd-Max MSE),
    and the from-scratch lane keeps its identity end-to-end: the enum
    label, logs, and get_path() all say MXKV
    (2026-11-11: MXFP4/MXFP8 are their own family). The member value is
    our backend's qualname, so even the enum-native get_class() path
    resolves to the real MX backend. Attention.__init__ resolves
    AttentionBackendEnum[get_name()] and the wheel enum is closed —
    the first MX boot failed on exactly that lookup (r149 finding #1).

    This transform is the LATE-IMPORT path only: the registry module is
    already in sys.modules by the time this platform registers (the
    base-class import at module scope — vllm.platforms.xpu — pulls it),
    so the serve path is covered by the register-time member injection
    (_install_mxkv_enum_member). Rig-proven injection recipe: register
    in _member_map_ + _value2member_map_; EnumType.__getattr__ then
    serves attribute access and ["MXKV"] lookups with no setattr.
    """
    anchor = (
        '    TURBOQUANT = "vllm.v1.attention.backends.turboquant_attn.'
        'TurboQuantAttentionBackend"\n'
    )
    add = '    MXKV = "vllm_gaudi.xpu.patches.mx_kv.engine.MxAttentionBackend"\n'
    n = src.count(anchor)
    if n != 1:
        raise RuntimeError(
            f"AttentionBackendEnum anchor count {n} != 1 in registry.py "
            "(wheel drift — re-survey before shipping the MX KV lane)"
        )
    return src.replace(anchor, anchor + add)


def _install_mxkv_enum_member() -> None:
    """Register-time MXKV member injection into the already-imported
    AttentionBackendEnum (the common case — see _add_mxkv_backend_member).
    Idempotent via the member map; wheel-drift loud if the enum internals
    the recipe touches are gone.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    member_map = AttentionBackendEnum.__dict__.get("_member_map_")
    value_map = AttentionBackendEnum.__dict__.get("_value2member_map_")
    if member_map is None or value_map is None:
        raise RuntimeError(
            "AttentionBackendEnum lacks _member_map_/_value2member_map_ "
            "(wheel drift — re-survey the MXKV member injection recipe)"
        )
    if "MXKV" in member_map:
        return
    member = object.__new__(AttentionBackendEnum)
    member._name_ = "MXKV"
    member._value_ = _MX_BACKEND_PATH
    member_map["MXKV"] = member
    value_map[_MX_BACKEND_PATH] = member
    logger.info(
        "ArcXpuPlatform: injected AttentionBackendEnum.MXKV -> %s", _MX_BACKEND_PATH
    )


# Source transforms (P6.47 leg B): patches that must edit a wheel
# module's SOURCE before Python compiles it. The finder wraps the
# loader's get_code so the module compiles from the transformed text —
# bypassing the stale-__pycache__ short-circuit (a fresh .pyc would
# otherwise pass mtime validation and hide the edit). Targets here are
# disjoint from _WHEEL_HOOKS (no exec-wrap patch; the transform IS the
# patch).
_SOURCE_XFORMS = {
    # target module -> callable(source) -> source
    "vllm.config.cache": _widen_cache_dtype_literal,
    "vllm.v1.attention.backends.registry": _add_mxkv_backend_member,
}


def _install_mx_dtype_admission_fallback() -> None:
    """Late admission when vllm.config.cache was already imported before
    the finder installed (host tools importing vllm ahead of platform
    resolution — every serve process boots through the source transform
    instead). Validation alias: the constructor validates the
    element-class analogue preset, then the true MX string is restored
    on the instance before any consumer reads it — every downstream
    dispatch (dtype map, mode alias, backend routing, impl init) sees
    the true string. kwargs-only (the engine constructs CacheConfig by
    keyword); idempotent via the module marker.
    """
    import vllm.config.cache as cc

    if getattr(cc, "_arc_mx_admitted", False):
        return
    cc._arc_mx_admitted = True
    alias = {
        "mxfp4": "turboquant_4bit_nc",
        "mxfp8": "turboquant_k8v4",
        "mxk8v4": "turboquant_k8v4",
    }
    orig_init = cc.CacheConfig.__init__

    def __init__(self, *args, **kwargs):
        true_dtype = kwargs.get("cache_dtype")
        mapped = alias.get(true_dtype)
        if mapped is None:
            return orig_init(self, *args, **kwargs)
        kwargs["cache_dtype"] = mapped
        orig_init(self, *args, **kwargs)
        # pydantic dataclasses validate at __init__ only — a plain
        # attribute write after it restores the engine-visible string.
        self.cache_dtype = true_dtype

    cc.CacheConfig.__init__ = __init__
    logger.warning(
        "ArcXpuPlatform: vllm.config.cache imported before platform "
        "install — MX kv dtypes admitted via CacheConfig validation alias "
        "(boot path uses the true Literal source transform)."
    )


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

    path = os.path.join(os.path.dirname(_patches.__file__), *patch_name.split(".")) + ".py"
    spec = spec_from_file_location(qualname, path)
    module = module_from_spec(spec)
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    return module


class _WheelHookFinder:
    """One-shot meta-path finder wrapping the exec of each _WHEEL_HOOKS
    target so its patch module can rebind symbols inside it, and the
    compile of each _SOURCE_XFORMS target so its patch edits the module
    source before Python ever builds it."""

    def __init__(self) -> None:
        self._targets = {t for t, _ in _WHEEL_HOOKS}
        self._xform_targets = set(_SOURCE_XFORMS)
        self._pending = self._targets | self._xform_targets

    def find_spec(self, fullname, path=None, target=None):  # noqa: D102
        if fullname not in self._pending:
            return None
        from importlib.util import find_spec

        sys.meta_path.remove(self)
        try:
            spec = find_spec(fullname)
        finally:
            if self._pending:
                sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            self._pending.discard(fullname)
            return None

        xform = _SOURCE_XFORMS.get(fullname)
        if xform is not None and hasattr(spec.loader, "path"):
            # Compile-from-transformed-source: shadow the loader's
            # get_code with an instance attribute so the pristine
            # __pycache__ (mtime-valid against the untouched file) can
            # not short-circuit the edit.
            loader = spec.loader
            orig_get_code = loader.get_code

            def get_code(name):
                if name == fullname:
                    src = xform(loader.get_data(loader.path).decode("utf-8"))
                    return compile(src, loader.path, "exec", dont_inherit=True)
                return orig_get_code(name)

            loader.get_code = get_code

        if fullname in self._targets:
            orig_exec = spec.loader.exec_module
            patch_name = dict(_WHEEL_HOOKS)[fullname]

            def exec_module(module):
                orig_exec(module)
                self._pending.discard(fullname)
                if not self._pending and self in sys.meta_path:
                    sys.meta_path.remove(self)  # last hook fired — retire
                try:
                    patch = _load_wheel_patch(patch_name)
                    patch._install(module)
                    logger.info(
                        "ArcXpuPlatform: wheel hook %s <- patches.%s OK",
                        fullname.rsplit(".", 1)[-1], patch_name)
                except Exception as e:  # noqa: BLE001 — visibility law
                    logger.warning("ArcXpuPlatform: wheel hook %s <- patches.%s FAILED: %r", fullname, patch_name, e)

            spec.loader.exec_module = exec_module
        elif xform is not None:
            # Transform-only target: retire from pending only AFTER a
            # successful exec. A find_spec-time discard is not
            # retry-safe — r148 run2: the admission fallback's
            # premature vllm.config.cache import died inside the
            # plugin-load circular window, and the already-discarded
            # target then re-imported PRISTINE (untransformed) on the
            # retry, silently dropping the CacheDType widening.
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                self._pending.discard(fullname)
                if not self._pending and self in sys.meta_path:
                    sys.meta_path.remove(self)  # last transform fired — retire

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
    # Targets already imported before the finder installed (host tools,
    # import-order drift) fire their patch in-place now: exec-wrap can
    # only see FUTURE imports, and a silently-skipped patch is
    # indistinguishable from an applied one (p2r2 visibility law). Each
    # patch module owns its idempotence.
    for target, patch_name in _WHEEL_HOOKS:
        mod = sys.modules.get(target)
        if mod is None:
            continue
        try:
            _load_wheel_patch(patch_name)._install(mod)
            logger.info(
                "ArcXpuPlatform: wheel hook %s <- patches.%s OK (already imported)",
                target.rsplit(".", 1)[-1], patch_name)
        except Exception as e:  # noqa: BLE001 — visibility law
            logger.warning(
                "ArcXpuPlatform: in-place wheel hook %s <- patches.%s FAILED: %r",
                target, patch_name, e)
    # Source transforms cannot apply to an already-imported module —
    # the late-admission fallback covers that ordering for the cache
    # Literal. Scope the trigger to the module the fallback actually
    # patches: vllm.v1.attention.backends.registry is ALWAYS imported
    # by the platform base-class chain (vllm.platforms.xpu) before this
    # install runs, and an any()-over-targets check made that pre-import
    # fire the fallback prematurely inside the plugin-load
    # circular-import window (r148 run2: the fallback import of
    # vllm.config.cache died on a partial vllm.utils.torch_utils, and
    # the failed first exec un-primed the transform — literal+admit
    # regression).
    pre = [t for t in _SOURCE_XFORMS if t in sys.modules]
    if pre:
        logger.info(
            "ArcXpuPlatform: source-transform targets already imported at "
            "install: %s", pre)
    if "vllm.v1.attention.backends.registry" in sys.modules:
        try:
            _install_mxkv_enum_member()
        except Exception as e:  # noqa: BLE001 — visibility law
            logger.warning(
                "ArcXpuPlatform: MXKV enum member injection FAILED: %r", e)
    if "vllm.config.cache" in sys.modules:
        try:
            _install_mx_dtype_admission_fallback()
        except Exception as e:  # noqa: BLE001 — visibility law
            logger.warning("ArcXpuPlatform: MX dtype admission fallback FAILED: %r", e)


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

    # P6.26 (2026-11-06): quark joins the supported set. Core's XPU list
    # omits it — upstream, the quark export format is AMD/ROCm-targeted —
    # but the wheel's QuarkConfig routes its OCP-MX (W4A4 mxfp4) scheme
    # through init_mxfp4_linear_kernel, the same platform-neutral entry
    # CompressedTensorsW4A4Mxfp4 uses, which resolves to the native Xe2
    # XPUMxFp4LinearKernel (fp4_gemm, vllm-xpu-kernels) on this platform —
    # proven live by the RedHatAI MXFP4xFP8 phase-A boot ("Using
    # XPUMxFp4LinearKernel for MXFP4 GEMM", results/2026-11-06-p626-mxfp4/).
    # Without this override Platform.verify_quantization
    # (platforms/interface.py:988) rejects quant_method=quark checkpoints
    # before any scheme code runs. Same pattern as HpuPlatform's own
    # supported_quantization (vllm_gaudi/platform.py).
    supported_quantization = [*XPUPlatform.supported_quantization, "quark"]

    @classmethod
    def supports_mx(cls) -> bool:
        # Truthful Xe2 capability statement (this class is only ever
        # selected on BMG parts — the register gate checks the 0xE200
        # device-id range): BMG computes MX types natively —
        # float4_e2m1fn_x2 weights + e8m0 scales + the vxk fp4_gemm kernel
        # the XPU MXFP4 linear table resolves to. Upstream platforms
        # declare this the same way (rocm.py returns True on gfx95).
        # Effect on this stack: quark_ocp_mx.py:73's "Simulated weight
        # dequantization" warning (false on Arc — kernel selection is
        # table-keyed, _POSSIBLE_MXFP4_KERNELS[XPU] = [XPUMxFp4LinearKernel],
        # and never consults supports_mx; r102 recon, plan §P6.26) stops
        # firing and the honest mismatch warning at :81 takes over for
        # genuinely unsupported weight/act dtype combos.
        return True

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # Core XPU posture first: XPU-graph env law, fusion-pass disabling,
        # XPUWorker selection, spawn multiproc, oneCCL shutdown timeout.
        super().check_and_update_config(vllm_config)
        # TurboQuant KV-cache dtypes serve clean since P6.19 (2026-11-04):
        # the P6.18 corruption was NOT the 4-bit precision ceiling then
        # dispositioned — it was two XPU-graph defects in the spec-verify
        # path (capture baking the raw-KV flash fast path into verify
        # graphs, #TQ-blind-verify; and a replay-broken multi-row
        # synthetic-decode dispatch in the continuation branch), both now
        # fixed wheel-side by the tq_spec_fix hook (patches/tq_spec_fix.py;
        # turboquant_k8v4 clean under graphs in leg t5, all-preset gate in
        # results/2026-11-04-p619-tqverify/). The cert7 boot refusal is
        # lifted; fp8_e4m3 stays the certified default. Preset precision
        # characteristics (upstream's own table rates these presets
        # +2.71..+20.59 pct PPL on gpt-oss) remain — choose per deployment.
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
                "(VLLM_ARC_GMU_RELAX; D-champion certified B70 posture).", cache_config.gpu_memory_utilization,
                VLLM_ARC_GMU_RELAX)
            cache_config.gpu_memory_utilization = VLLM_ARC_GMU_RELAX
        logger.info("ArcXpuPlatform: Intel Arc Xe2 posture active (%s).", torch.xpu.get_device_name(0))
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
            getattr(compilation_config.cudagraph_mode, "name", compilation_config.cudagraph_mode),
            tuple(compilation_config.cudagraph_capture_sizes) if compilation_config.cudagraph_capture_sizes else None,
            VLLM_XPU_ENABLE_XPU_GRAPH)

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        # Per-worker install point: XPUWorker.init_device calls this after
        # the import is complete — the guaranteed post-import moment in every
        # WorkerProc (the register()-time attempt can hit a half-initialized
        # vllm package). Idempotent; failure surfaces in the gate leg.
        _install_xpu_patches("check_if_supports_dtype")
        super().check_if_supports_dtype(dtype)

    # MX KV dtypes (P6.47): the dtype string reaches backend selection
    # on AttentionSelectorConfig — route it to the MX backend and stamp
    # the process-wide format (the spec layer carries no dtype string,
    # so the resolved backend class is the format authority for
    # customize_spec). Dtypes + path live at module scope (shared with
    # the register-time enum-member injection).
    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> str:
        kv = getattr(attn_selector_config, "kv_cache_dtype", None)
        if kv in _MX_KV_DTYPES:
            from vllm_gaudi.xpu.patches.mx_kv.engine import MxAttentionBackend

            MxAttentionBackend.active_kv_dtype = kv
            logger.info_once("Using MX KV attention backend (%s).", kv)
            return _MX_BACKEND_PATH
        return super().get_attn_backend_cls(
            selected_backend, attn_selector_config, num_heads=num_heads
        )
