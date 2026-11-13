# SPDX-License-Identifier: Apache-2.0
"""One-shot rig probe (P6.47 leg B2): does the ENGINE-path cache view
(the wheel's forward/do_kv_cache_update pass kv_cache.transpose(1,2)
into the impl) match what mx_store/mx_decode address? Replays both
sides offline on synthetic tensors and compares against the raw-layout
reference (the leg-A-validated semantics)."""

import sys
from pathlib import Path

WORK = Path("/work")
sys.path.insert(0, str(WORK))

import torch  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import slot_bytes  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode import mx_decode  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.triton_mx_store import mx_store  # noqa: E402

torch.manual_seed(0)
DEV = "xpu"

BLOCKS, BS, HK, D = 8, 64, 8, 256
SLOT = slot_bytes(D, "mxfp8")  # 528: K payload + K scales + V payload + V scales
N = 200  # tokens, crossing block boundaries
SCALE = D ** -0.5

k = torch.randn(N, HK, D, device=DEV, dtype=torch.float32)
v = torch.randn(N, HK, D, device=DEV, dtype=torch.float32)
slot_mapping = torch.arange(100, 100 + N, device=DEV, dtype=torch.int64)


def raw_reference() -> torch.Tensor:
    """Leg-A semantics: raw (blocks, block, Hk, slot) cache, blk = s//BS."""
    cache = torch.zeros(BLOCKS, BS, HK, SLOT, dtype=torch.uint8, device=DEV)
    tb = torch.div(slot_mapping, BS, rounding_mode="floor").to(torch.int32)
    tp = (slot_mapping - tb.long() * BS).to(torch.int32)
    fmt = mx_store(k, v, cache, tb, tp)
    # decode the last token: bt covers slots 0..299 -> blocks 0..4
    last = int(slot_mapping[-1])
    bt = torch.div(
        torch.arange(0, ((last // BS) + 1) * BS, device=DEV), BS, rounding_mode="floor"
    ).to(torch.int32)[None, :]
    q = k[-1][:1].unsqueeze(0)  # (1, 1, D): head-0 row of the last token
    out = mx_decode(q, cache, bt, torch.tensor([last + 1], device=DEV, dtype=torch.int32), SCALE)
    del fmt
    return out[0, 0]


def engine_path() -> torch.Tensor:
    """What the impl methods actually receive: transpose(1,2) of the raw
    pool, and _store_kv's shape[1]-based decomposition."""
    cache_raw = torch.zeros(BLOCKS, BS, HK, SLOT, dtype=torch.uint8, device=DEV)
    cache_t = cache_raw.transpose(1, 2)
    print(f"engine-path view: shape={tuple(cache_t.shape)} stride={cache_t.stride()}")
    bs_from_shape1 = cache_t.shape[1]  # what _store_kv reads as block_size
    tb = torch.div(slot_mapping, bs_from_shape1, rounding_mode="floor").to(torch.int32)
    tp = (slot_mapping - tb.long() * bs_from_shape1).to(torch.int32)
    mx_store(k, v, cache_t, tb, tp)
    last = int(slot_mapping[-1])
    bt = torch.div(
        torch.arange(0, ((last // BS) + 1) * BS, device=DEV), BS, rounding_mode="floor"
    ).to(torch.int32)[None, :]
    q = k[-1][:1].unsqueeze(0)
    out = mx_decode(q, cache_t, bt, torch.tensor([last + 1], device=DEV, dtype=torch.int32), SCALE)
    return out[0, 0]


ref = raw_reference().float()
eng = engine_path().float()
dot = torch.dot(ref.reshape(-1), eng.reshape(-1))
nr = torch.linalg.vector_norm(ref)
ne = torch.linalg.vector_norm(eng)
cos = (dot / (nr * ne)).item()
print(f"ref norm={nr.item():.4f} engine norm={ne.item():.4f} cos={cos:.6f}")
print("VERDICT", "SAME-PATH" if nr.item() > 0 and abs(cos - 1.0) < 1e-3 else "DIVERGENT")
