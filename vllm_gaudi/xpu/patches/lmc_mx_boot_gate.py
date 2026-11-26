# SPDX-License-Identifier: Apache-2.0
"""lmc_mx_boot_gate.py — P-D LMC x MX boot-gate hardening, P-H4
posture-aware (P6.47, vllm_gaudi/xpu/patches/).

Exec-hook target: ``vllm.distributed.kv_transfer.kv_connector.factory``.
The r240b deaths were posture-confounded — a gate script's inherited
env set ARC_LMC=1 over MX lanes and the LMCache connector rode the MX
KV cache; r241's fix was harness-side only (runners export ARC_LMC=0;
boot posture greps serve logs). This gate makes the ENGINE decide the
posture at boot, beside the factory's own HMA refusal (same method,
same shape: ``hma_enabled and not supports_hma_config(...)`` ->
ValueError at connector construction — wheel-proven anchor,
2026-11-26). No byte-preserved C-donor file is touched (the connector
glue ``lmcache_hma_connector.py`` / ``lmcint/`` stay pristine).

P-H4 (r253) made the gate POSTURE-AWARE instead of blanket: the
r249 packed-row interop (fork-owned ``lmcint/vllm_v1_adapter.py``,
capability marker ``ARC_LMC_MX_INTEROP``) fixed the root cause P-H1
located — the adapter classed MX lanes as standard geometry while the
pool view carried packed rows, a 32 B/token cumulative misaddress on
every store/restore (one mechanism for the r168 essay garble and the
r240b sustained-sweep wedge). r249 validated the fixed interop on
five gates (re-hit A/A byte-equal 6.18->1.89 s wall; the exact
r240b kill sequence clean; zero transfer errors). So:

  interop present  -> ALLOW with a gate=interop-live posture line
  interop absent   -> still REFUSED (pre-r249 images / drifted
                      installs — the r240b posture confound class)

Refusal scope (no-unevidenced-customization law): LMCache-named
connectors over the MX KV dtypes ONLY. NIXL/offloading/other
connectors over MX are NOT refused (no evidence basis), and LMCache
over non-MX dtypes stays the lawful certified lane.

Disable knob: VLLM_ARC_LMC_MX_GATE=0 bypasses the refusal — A/B
evidence only (VLLM_ARC_LMC_HMA=0 precedent). Registered in
vllm_gaudi/envs.py.

Validated by the P-D leg (r246 probe 11/11 + r246b e2e both cells)
and the P-H4 flip leg (r253 probe: interop-live pass-through x3
dtypes, stock-adapter refusal x3 + legacy surface, bypass, scope,
lawful lane, stock behavior, idempotence) — see the plan's
"P-D LMC x MX BOOT-GATE HARDENING -- DESIGN" section and the r253
bank.
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


def _interop_live() -> tuple[bool, str]:
    """P-H4: is the r249 packed-row MX interop present in this
    process? The fork-owned adapter publishes the capability marker;
    a failed import or a missing attribute both read as absent (the
    gate never blocks on its own probe — the connector would fail to
    construct anyway)."""
    try:
        from vllm_gaudi.xpu.patches.lmcint import vllm_v1_adapter as _ad
    except Exception as exc:  # noqa: BLE001 -- any import failure = absent
        return False, f"adapter import failed: {exc.__class__.__name__}"
    ver = getattr(_ad, "ARC_LMC_MX_INTEROP", None)
    if not ver:
        return False, "no ARC_LMC_MX_INTEROP (pre-r249 adapter)"
    return True, str(ver)


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
                # P-H4 (r253): posture-aware. The r249 packed-row
                # interop is the capability; its presence admits the
                # lane with telemetry, its absence keeps the P-D
                # refusal (the r240b posture-confound class).
                live, why = _interop_live()
                if os.environ.get("VLLM_ARC_LMC_MX_GATE", "1") == "0":
                    _log("ARC_LMC_MX_POSTURE gate=bypassed connector=%s "
                         "kv_dtype=%s interop=%s", configured or resolved,
                         kv_dtype, "live" if live else "absent")
                elif live:
                    _log("ARC_LMC_MX_POSTURE gate=interop-live "
                         "connector=%s kv_dtype=%s (%s; r249 INTEROP_PASS: "
                         "re-hit byte-equal, zero transfer errors)",
                         configured or resolved, kv_dtype, why)
                else:
                    raise ValueError(
                        "ARC_LMC_MX_REFUSED: LMCache connector "
                        f"'{configured or resolved}' over MX KV dtype "
                        f"'{kv_dtype}' needs the r249 packed-row interop "
                        f"({why}) — the pre-r249 adapter classed MX lanes "
                        "as standard geometry, the 32 B/token cumulative "
                        "misaddress behind every r240b death (posture "
                        "confound; re-measured clean with LMC off: 68/68 "
                        "faces, results/2026-11-25-p647-r241-mxkv-lmcfree) "
                        "and the r168 essay garble. Use a cert29+ image ("
                        "or serve with the P-H adapter overlay) or drop "
                        "the kv-transfer config for MX lanes (ARC_LMC=0). "
                        "A/B evidence only: VLLM_ARC_LMC_MX_GATE=0."
                    )
            else:
                _log("ARC_LMC_MX_POSTURE gate=on connector=%s kv_dtype=%s",
                     configured or resolved or "n/a", kv_dtype)
        return orig(cls, config, role, kv_cache_config)

    create_connector._arc_mx_gate = True
    factory.create_connector = classmethod(create_connector)
