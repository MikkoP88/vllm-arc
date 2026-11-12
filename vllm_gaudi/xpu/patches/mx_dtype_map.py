# SPDX-License-Identifier: Apache-2.0
"""mx_dtype_map.py — STR_DTYPE_TO_TORCH_DTYPE admission for MX KV dtypes
(P6.47 leg B, vllm_gaudi/xpu/patches/).

Exec-hook target: ``vllm.utils.torch_utils``. The engine maps
``cache_config.cache_dtype`` to the allocation dtype through this table
when building attention specs (platforms/interface.py per-token page
arithmetic); MX KV caches are byte-packed uint8 pools — the same class
as every other packed dtype in the table (``fp8``, ``int4_per_token_head``,
``fp8_per_token_head`` all map to torch.uint8/int8 carriers).

Idempotent: re-assignment of the same keys is a no-op.
"""

from __future__ import annotations

import torch


def _install(mod) -> None:
    table = getattr(mod, "STR_DTYPE_TO_TORCH_DTYPE", None)
    if table is None:
        raise AttributeError(
            "mx_dtype_map: target module has no STR_DTYPE_TO_TORCH_DTYPE "
            "(wheel drift — re-survey before re-pointing this hook)"
        )
    table["mxfp4"] = torch.uint8
    table["mxfp8"] = torch.uint8
    table["mxk8v4"] = torch.uint8
