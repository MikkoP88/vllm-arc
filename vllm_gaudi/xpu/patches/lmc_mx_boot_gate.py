# SPDX-License-Identifier: Apache-2.0
"""lmc_mx_boot_gate.py — P-D LMC x MX boot-gate hardening (P6.47,
vllm_gaudi/xpu/patches/).

Exec-hook target: ``vllm.distributed.kv_transfer.kv_connector.factory``.
The r240b deaths were posture-confounded — a gate script's inherited
env set ARC_LMC=1 over MX lanes and the LMCache connector rode the MX
KV cache; r241's fix was harness-side only (runners export ARC_LMC=0;
boot posture greps serve logs). This gate makes the ENGINE refuse the
combination at boot, beside the factory's own HMA refusal (same
method, same shape: ``hma_enabled and not supports_hma_config(...)``
-> ValueError at connector construction — wheel-proven anchor,
2026-11-26). No byte-preserved C-donor file is touched (the connector
glue ``lmcache_hma_connector.py`` / ``lmcint/`` stay pristine).

Refusal scope (no-unevidenced-customization law): LMCache-named
connectors over the MX KV dtypes ONLY. NIXL/offloading/other
connectors over MX are NOT refused (no evidence basis), and LMCache
over non-MX dtypes stays the lawful certified lane.

Disable knob: VLLM_ARC_LMC_MX_GATE=0 bypasses the refusal — A/B
evidence only (VLLM_ARC_LMC_HMA=0 precedent). Registered in
vllm_gaudi/envs.py.

Validated by the P-D leg (three offline boots: LMC+MX refuses, MX
clean, LMC+non-MX constructs; e2e posture bank) — see the plan's
"P-D LMC x MX BOOT-GATE HARDENING -- DESIGN" section.
"""

from __future__ import annotations

import contextlib
import os

# The MX KV dtype strings (CLI --kv-cache-dtype members widened by
# platform._widen_cache_dtype_literal). Kept in lockstep with
# mx_kv_mode._MX_TQ_ALIAS and platform._MX_KV_DTYPES — the admission
# set, not the alias map: only the raw strings arrive on the cache
# config. vllm 0.30 renamed CacheConfig.kv_cache_dtype -> cache_dtype
# (config/cache.py:111; no legacy alias — the CLI kwarg and the
# AttentionSelectorConfig tuple field kept the old name, which is why
# platform.get_attn_backend_cls's selector read predates-proof worked
# while the first gate revision leaked with kv_dtype=None over the
# real serve path, r246b run 2). The read below covers both surfaces.
_MX_KV_DTYPES = ("mxfp4", "mxfp8", "mxk8v4")


def _install(mod) -> None:
    factory = getattr(mod, "KVConnectorFactory", None)
    if factory is None:
        raise AttributeError(
            "lmc_mx_boot_gate: target module has no KVConnectorFactory "
            "(wheel drift — re-survey before re-pointing this hook)"
        )
    orig_cm = factory.__dict__.get("create_connector")
    if orig_cm is None:
        raise AttributeError(
            "lmc_mx_boot_gate: KVConnectorFactory has no create_connector "
            "classmethod (wheel drift — re-survey before re-pointing this hook)"
        )
    orig = orig_cm.__func__
    if getattr(orig, "_arc_mx_gate", False):
        return

    logger = getattr(mod, "logger", None)

    def _log(fmt: str, *args: object) -> None:
        if logger is not None:
            logger.info(fmt, *args)

    def create_connector(cls, config, role, kv_cache_config):
        ktc = getattr(config, "kv_transfer_config", None)
        if ktc is not None:
            cc = getattr(config, "cache_config", None)
            # legacy surface first (pre-rename/stub objects), then the
            # vllm-0.30 field (config/cache.py:111) the live engine
            # actually carries.
            kv_dtype = getattr(cc, "kv_cache_dtype", None)
            if kv_dtype is None:
                kv_dtype = getattr(cc, "cache_dtype", None)
            configured = str(getattr(ktc, "kv_connector", "") or "")
            resolved = ""
            # Registry miss / import failure: the original call below
            # raises on the same defect — never pre-empt it here.
            with contextlib.suppress(Exception):
                resolved = cls.get_connector_class(ktc).__name__
            is_lmc = ("lmcache" in configured.lower()
                      or "lmcache" in resolved.lower())
            if kv_dtype in _MX_KV_DTYPES and is_lmc:
                if os.environ.get("VLLM_ARC_LMC_MX_GATE", "1") != "0":
                    raise ValueError(
                        "ARC_LMC_MX_REFUSED: LMCache connector "
                        f"'{configured or resolved}' over MX KV dtype "
                        f"'{kv_dtype}' is an unsupported posture — every "
                        "r240b death was LMCache-forced-over-MX (posture "
                        "confound; re-measured clean with LMC off: 68/68 "
                        "faces, results/2026-11-25-p647-r241-mxkv-lmcfree). "
                        "Drop the kv-transfer config for MX lanes "
                        "(ARC_LMC=0). A/B evidence only: "
                        "VLLM_ARC_LMC_MX_GATE=0."
                    )
                _log("ARC_LMC_MX_POSTURE gate=bypassed connector=%s "
                     "kv_dtype=%s", configured or resolved, kv_dtype)
            else:
                _log("ARC_LMC_MX_POSTURE gate=on connector=%s kv_dtype=%s",
                     configured or resolved or "n/a", kv_dtype)
        return orig(cls, config, role, kv_cache_config)

    create_connector._arc_mx_gate = True
    factory.create_connector = classmethod(create_connector)
