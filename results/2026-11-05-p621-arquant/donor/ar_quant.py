# W13-AR: quantized-transport all-reduce for TP2 XPU prefill
# (IEXP_AR_QUANT=1).
#
# Law from the W13D PCIe sweep: on this SYS-topology pair the oneCCL
# all-reduce already runs at the P2P link ceiling (9.5-9.7 GB/s measured,
# ~= the 9.54 GB/s raw copy rate), so no re-wiring of the collective can
# beat it, and oneCCL has no fp8 all-reduce op.  The only byte-cut left
# is precision of the transported payload.
#
# all_gather is a COPY collective -- dtype-agnostic, it moves bytes.  So
# each rank keeps its own partial in exact bf16 and ships the PEER a
# per-row-scaled e4m3 copy of its partial (half the bytes of the bf16
# all-reduce payload); every rank then reconstructs the sum locally:
#
#     out = x_local (exact bf16) + dequant(q_peer)   [fp32 accumulate]
#
# Only the peer's partial is quantized -- half the noise of quantizing
# both sides, and the local contribution (usually the larger, having the
# router/attn locality) stays exact.  Expected error class: e4m3 with
# per-row dynamic scale, ~1e-3..3e-2 relative on the peer term only.
# NOT byte-exact by construction; gate on semantic equivalence.
#
# Sizing gate: prefill-class tensors only (rows*hidden >= threshold
# elements).  Decode/verify ARs (<= 256 rows x hidden) stay on the plain
# collective -- they are latency-bound, and this path must never run
# inside an XPU graph capture (it allocates and calls the process group).
#
# Fail-soft: any error on first use sticks the lane off for the process
# and forwards to the ORIGINAL all-reduce (patch_ar_quant_xpu stashes it
# on vllm.distributed as _iexp_ar_quant_orig before wrapping, so the
# fallback never re-enters the wrapper).

from __future__ import annotations

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_ARMED = os.environ.get("IEXP_AR_QUANT", "0") == "1"
_MIN_ELEMS = int(os.environ.get("IEXP_AR_QUANT_MIN_ELEMS", str(2 * 1024 * 1024)))
_FAILED = False
_LOGGED = False
_ORIG = None

# Per-size gather buffers, cached so a steady prefill chunk rate does
# not churn the allocator (rows are MNBT-stable at 4096 between drains).
_BUFS: dict[tuple[int, int, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _orig_all_reduce(x: torch.Tensor) -> torch.Tensor:
    global _ORIG
    if _ORIG is None:
        import vllm.distributed as _vd
        _ORIG = getattr(_vd, "_iexp_ar_quant_orig", None)
        if _ORIG is None:
            _ORIG = _vd.tensor_model_parallel_all_reduce
    return _ORIG(x)


def _rank_pair() -> tuple[bool, int, int, dist.ProcessGroup | None]:
    try:
        from vllm.distributed import (get_tensor_model_parallel_rank,
                                      get_tensor_model_parallel_world_size,
                                      get_tp_group)
        if get_tensor_model_parallel_world_size() != 2:
            return False, 0, 0, None
        grp = get_tp_group().device_group
        return True, get_tensor_model_parallel_rank(), 1 - get_tensor_model_parallel_rank(), grp
    except Exception:  # noqa: BLE001 - not under TP2, plain AR is correct
        return False, 0, 0, None


def ar_quant_all_reduce(x: torch.Tensor) -> torch.Tensor:
    """Drop-in for tensor_model_parallel_all_reduce on TP2 XPU prefill."""
    global _FAILED, _LOGGED
    if _FAILED or not _ARMED:
        return _orig_all_reduce(x)
    ok, rank, peer, grp = _rank_pair()
    if not ok or x.device.type != "xpu" or x.dtype != torch.bfloat16 \
            or x.numel() < _MIN_ELEMS or torch.xpu.is_current_stream_capturing():
        return _orig_all_reduce(x)
    try:
        rows, hidden = x.reshape(-1, x.shape[-1]).shape
        f = x.reshape(rows, hidden)
        amax = f.abs().float().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = amax / 448.0
        q = (f.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        key = (rows, hidden, x.dtype)
        g = _BUFS.get(key)
        if g is None:
            g = (torch.empty(2, rows, hidden, dtype=torch.uint8,
                             device=x.device),
                 torch.empty(2, rows, dtype=torch.float32, device=x.device))
            _BUFS[key] = g
        g_q, g_s = g
        dist.all_gather_into_tensor(g_q.view(torch.uint8), q.view(torch.uint8),
                                    group=grp)
        dist.all_gather_into_tensor(g_s, scale.reshape(-1), group=grp)
        dq = g_q[peer].view(torch.float8_e4m3fn).float() \
            * g_s[peer].unsqueeze(1)
        out = (f.float() + dq).to(x.dtype)
        if not _LOGGED:
            _LOGGED = True
            logger.warning(
                "ar_quant: quantized-transport all-reduce active "
                "(peer e4m3 per-row, local exact; %dx%d payloads)",
                rows, hidden)
        return out.reshape(x.shape)
    except Exception as e:  # noqa: BLE001 - sticky fallback, never kill
        _FAILED = True
        logger.warning(
            "ar_quant: disabled after failure (%s: %s); plain all-reduce "
            "from now on", type(e).__name__, e)
        return _orig_all_reduce(x)
