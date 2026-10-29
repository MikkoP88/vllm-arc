# arc r43 (2026-10-29): TWO_HS interleaved-pool branch for the torch-baseline
# multi_layer_kv_transfer.
#
# The engine's hybrid FA-XPU attention pool is one stacked tensor per layer,
# (num_blocks, 2=K/V, block_size, kv_heads*head_size), whose PHYSICAL order
# is (num_blocks, block_size, K/V, kvdim) -- per-token K/V interleaving.
# The adapter hands lmcache a contiguous zero-copy 5-D view (NB, BS, 1, TWO,
# kvdim) and the arc detector override (lmc_kvfmt_detector.py) maps it to
# NL_X_NB_BS_NH_TWO_HS. But NEITHER transfer backend can execute that
# format on XPU today:
#   - the native lmcache.xpu_ops multi_layer_kv_transfer only implements
#     NB_NL_TWO_BS_NH_HS / NL_X_TWO_NB_BS_NH_HS / NL_X_NB_TWO_BS_NH_HS
#     (+ MLA) and raises "Unsupported non-MLA EngineKVFormat" -- which is
#     exactly why p40d keeps the four KV-transfer ops on the torch baseline;
#   - the torch fallback (torch_ops.py multi_layer_kv_transfer) has no
#     TWO_HS branch either: the format falls into the generic else-branch,
#     which reinterprets the pool as (2, page_buffer_size, hidden) K/V
#     half-planes. On this interleaved pool that view folds every page P
#     onto page P//2 (slot s -> physical element s*hidden -> page s//1792),
#     storing/loading garbage while remaining self-consistent -- the
#     N2/N4-class warm re-hit integrity break root-caused this leg
#     (r43: warm sha d4735e3a265e16ee != cold 9c1623f0d38e28e9).
#
# This overlay wraps torch_ops.multi_layer_kv_transfer and, for
# NL_X_NB_BS_NH_TWO_HS only, performs the transfer against the pool's true
# physical (NB, BS, TWO, HS) geometry with vectorized advanced indexing:
#   D2H: key_value[:, l, valid, :] = pool[bidx, boff, :, :].T
#   H2D: pool[bidx, boff, :, :] = key_value[:, l, valid, :].T
# byte-true per token (two*HS + hs), one slot per token. All other formats
# pass through to the original implementation unchanged.
#
# Source-override kit file (p41b exec_module pattern); the installed
# lmcache tree is never edited.

from __future__ import annotations

import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection

_TWO_HS = int(EngineKVFormat.NL_X_NB_BS_NH_TWO_HS)
_H2D = int(TransferDirection.H2D)

_ORIG = None
_ANNOUNCED = False


def _arc_multi_layer_kv_transfer(
    key_value,
    key_value_ptrs,
    slot_mapping,
    paged_memory_device,
    page_buffer_size,
    direction,
    engine_kv_format,
    block_size=0,
    head_size=0,
    skip_prefix_n_tokens=0,
    block_stride_elems=0,
):
    global _ANNOUNCED
    try:
        fmt = int(engine_kv_format)
    except Exception:  # noqa: BLE001
        fmt = -1
    if fmt != _TWO_HS or block_size <= 0:
        return _ORIG(
            key_value,
            key_value_ptrs,
            slot_mapping,
            paged_memory_device,
            page_buffer_size,
            direction,
            engine_kv_format,
            block_size,
            head_size,
            skip_prefix_n_tokens,
            block_stride_elems,
        )

    # --- arc NL_X_NB_BS_NH_TWO_HS: per-token K/V interleaved pool ---
    kv_device = key_value.device
    slots_kv = slot_mapping.to(dtype=torch.long).to(kv_device)
    valid_mask_kv = slots_kv >= 0
    if skip_prefix_n_tokens > 0:
        valid_mask_kv[:skip_prefix_n_tokens] = False
    if not bool(valid_mask_kv.any()):
        return
    valid_slots = slots_kv[valid_mask_kv].to(paged_memory_device)

    num_layers = key_value.size(1)
    hidden_size = key_value.size(3)
    num_blocks = page_buffer_size // block_size
    layer_shape = (num_blocks, block_size, 2, hidden_size)
    block_indices = valid_slots // block_size
    block_offsets = valid_slots % block_size
    h2d = int(direction) == _H2D

    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(
            "[r43xfer] TWO_HS interleaved branch live: "
            f"NB={num_blocks} BS={block_size} L={num_layers} "
            f"H={hidden_size} nslots={int(valid_slots.numel())} "
            f"dir={'H2D' if h2d else 'D2H'}",
            flush=True,
        )

    for layer_id in range(num_layers):
        if isinstance(key_value_ptrs, list):
            paged = key_value_ptrs[layer_id]
            if tuple(paged.shape) != layer_shape:
                paged = paged.reshape(layer_shape)
        else:
            ptr = int(key_value_ptrs[layer_id].item())
            paged = _TENSOR_FROM_PTR(
                ptr, layer_shape, key_value.dtype, paged_memory_device
            )
        if h2d:
            src = key_value[:, layer_id, valid_mask_kv, :].transpose(0, 1)
            paged[block_indices, block_offsets, :, :] = src.to(
                paged_memory_device, non_blocking=False
            )
        else:
            gathered = paged[block_indices, block_offsets, :, :].to(
                kv_device, non_blocking=False
            )
            key_value[:, layer_id, valid_mask_kv, :] = gathered.transpose(0, 1)


def _install(mod):
    global _ORIG, _TENSOR_FROM_PTR
    _ORIG = mod.multi_layer_kv_transfer
    _TENSOR_FROM_PTR = getattr(mod, "_tensor_from_ptr")
    mod.multi_layer_kv_transfer = _arc_multi_layer_kv_transfer
