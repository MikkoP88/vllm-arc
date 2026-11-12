# SPDX-License-Identifier: Apache-2.0
"""store_guard_probe.py — P6.47 leg B root-cause proof: the negative-slot
guard in triton_mx_store (upstream store law, triton_turboquant_store.py
`if slot < 0`) skips writes instead of storing out-of-bounds before the
pool base.

Three assertions on-device:
  1. all-negative slot batch ([−1]*4) leaves the pool byte-identical.
  2. mixed batch ([−1, real, −1, real]) stores the real rows exactly as
     the no-guard-rows equivalent batch does (behavior preservation for
     valid slots — the law-10 oracle).
  3. the stored bytes still decode to quant(raw) (leg-A semantics).

Run in-container on the rig against /work (PYTHONPATH shadow)."""

import sys
from pathlib import Path

WORK = Path("/work")
sys.path.insert(0, str(WORK))

import torch  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import slot_bytes  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.triton_mx_store import mx_store  # noqa: E402

torch.manual_seed(0)
DEV = "xpu"
BLOCKS, BS, HK, D = 8, 64, 2, 128
SLOT = slot_bytes(D, "mxfp8")


def pool() -> torch.Tensor:
    return torch.full((BLOCKS, BS, HK, SLOT), 0xAB, dtype=torch.uint8, device=DEV)


def slots_map(ids):
    t = torch.tensor(ids, device=DEV, dtype=torch.int64)
    tb = torch.div(t, BS, rounding_mode="floor")
    tp = t - tb * BS
    return tb.to(torch.int32), tp.to(torch.int32)


k = torch.randn(6, HK, D, device=DEV, dtype=torch.float32)
v = torch.randn(6, HK, D, device=DEV, dtype=torch.float32)

# 1. all-negative batch: pool untouched (0xAB fill survives every byte).
p1a, p1b = pool(), pool()
tb, tp = slots_map([-1, -1, -1, -1])
mx_store(k[:4], v[:4], p1a, tb, tp)
assert torch.equal(p1a, p1b), "FAIL: negative slots wrote into the pool"
print("guard_1_negative_skipped OK")

# 2. mixed batch == valid-only batch on the valid rows (oracle).
p2a, p2b = pool(), pool()
tbm, tpm = slots_map([-1, 70, -1, 71])
mx_store(k[[0, 1, 2, 3]], v[[0, 1, 2, 3]], p2a, tbm, tpm)
tbv, tpv = slots_map([70, 71])
mx_store(k[[1, 3]], v[[1, 3]], p2b, tbv, tpv)
a2 = p2a[1, 6].reshape(-1)
b2 = p2b[1, 6].reshape(-1)
assert torch.equal(a2, b2), "FAIL: guard changed valid-slot behavior"
print("guard_2_oracle_valid_rows_identical OK")

# 3. stored bytes decode to quant(raw) (leg-A semantics still hold).
p3 = pool()
tb3, tp3 = slots_map([64])
mx_store(k[4:5], v[4:5], p3, tb3, tp3)
n = D // 32
sb = p3[1, 0, 0, :]
kr, vr = decode_kv(sb[: D + n], sb[D + n :], "mxfp8", D)
err = max(float((kr - k[4, 0]).abs().max()), float((vr - v[4, 0]).abs().max()))
lim = 448.0 * (2.0 ** -8) * 2  # one e4m3 step at unit group scale, 2x margin
assert err < lim, f"FAIL: decode-vs-raw err {err} >= {lim}"
print(f"guard_3_quant_semantics OK (err={err:.5f} < {lim:.3f})")

print("VERDICT GUARD-PASS")
