# SPDX-License-Identifier: Apache-2.0
"""mx_kv_mode.py — get_kv_quant_mode admission for MX KV dtypes (P6.47
leg B, vllm_gaudi/xpu/patches/).

Exec-hook target: ``vllm.v1.kv_cache_interface``. ``KVQuantMode`` is a
closed IntEnum in the wheel; the MX lane maps its dtypes onto the
element-class-analogue TURBOQUANT members — a bounded alias: the mode's
only dtype-specific consumer is ``is_turboquant`` gating
``customize_spec`` (which MxAttentionBackend overrides wholesale), and
format identity itself travels on the dtype STRING through
``MxAttentionImpl.__init__`` and the platform-stamped
``MxAttentionBackend.active_kv_dtype``. Everything the mode conveys for
an MX cache — packed uint8 slot, TQ-class page plumbing — is true.

  mxfp8 -> TURBOQUANT_K8V4   (8-bit key element class)
  mxfp4 -> TURBOQUANT_4BIT_NC (4-bit key element class)

Idempotent via the installed-marker attribute.
"""

from __future__ import annotations

import functools

# Element-class analogues (kept in lockstep with MX_TQ_INIT_ALIAS in
# mx_kv/engine.py — the same element-width pairing).
_MX_TQ_ALIAS = {"mxfp8": "turboquant_k8v4", "mxfp4": "turboquant_4bit_nc"}


def _install(mod) -> None:
    orig = getattr(mod, "get_kv_quant_mode", None)
    if orig is None:
        raise AttributeError(
            "mx_kv_mode: target module has no get_kv_quant_mode "
            "(wheel drift — re-survey before re-pointing this hook)"
        )
    if getattr(orig, "_arc_mx_alias", False):
        return

    @functools.wraps(orig)
    def get_kv_quant_mode(kv_cache_dtype):
        if isinstance(kv_cache_dtype, str):
            kv_cache_dtype = _MX_TQ_ALIAS.get(kv_cache_dtype, kv_cache_dtype)
        return orig(kv_cache_dtype)

    get_kv_quant_mode._arc_mx_alias = True
    mod.get_kv_quant_mode = get_kv_quant_mode
