#!/usr/bin/env python3
"""r231d_oracle.py -- build25-vs-build28 oracle for the r231d
seqused-tightened split clamp.

The clamp CHANGES the split partition below 8K walked ctx, so the
r231b behavior-preservation gate (plain auto-vs-auto bitwise) does
NOT apply there. The r231d gates instead prove the clamp is EXACTLY
the env-pin semantics at an identical partition:

  G1  build25[ns=C] == build28[auto]  bitwise (C = the clamp value
      each pick derives) -- the clamp selects C and the kernel's
      execution at C is byte-identical to the pin path.
  G2  build25[ns=C] == build28[ns=C]  bitwise (pin path unchanged).
  G3  build25[auto] == build28[auto]  bitwise on capacity-path picks
      (blocks/4 >= 128, i.e. sl >= 8192 -- clamp is a no-op there;
      record-path byte identity).

Replay contract mirrors the r229 scorer cell exactly (banked bt_head
geometry, qB = q.to(bf16), max_seqlen_k = 4114*64, the payload's own
fmt field wins -- the payload-fmt law).

Modes:
  slinfo <payload.pt> [payload2.pt ...]
        per pick: r, the per-row sl range, the clamp value C =
        min(128, max(1, ceil(sl/16)/4)) as a row-UNIFORM value (the
        picks carry per-row seqused; a single ns pin reproduces auto
        only when every row clamps to the same count -- MIXED marks
        the exception), and the capacity-path flag (C == 128)
  save <outdir> auto|ns:<N> <payload.pt> [payload2.pt ...]
        one native call per payload at that split posture; saves
        [:r] cpu output as <outdir>/<basename>; prints the derived
        C per pick so harness logs are self-describing; aborts on a
        MIXED-C pick
  cmp <dirA> <dirB>
        bitwise compare every same-named pair; PASS only if every
        tensor is torch.equal
"""

import os
import sys

import torch

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode

D = 256
ENGINE_BT_WIDTH = 4114
CAPACITY = 128  # the banked engine posture (ARC_MXKV_NSPLIT=128, r217b)
TILE_N = 16  # get<1>(TileShapeQK{}) on the served policy


def clamp_C(blocks: int) -> int:
    return min(CAPACITY, max(1, blocks // 4))


def payload_info(path):
    p = torch.load(path, map_location="cpu", weights_only=False)
    r = int(p["q"].shape[0])
    sl = p["sl"][:r].to(torch.int32).tolist()
    return p, r, sl


def c_set(sl_rows):
    """The clamp values the r231d edit derives for each row (a pick
    carries per-row seqused; a single ns pin reproduces auto only when
    every row clamps to the SAME count)."""
    return sorted({clamp_C(-(-s // TILE_N)) for s in sl_rows})


def rebuild_pool(p, dev, block_stride):
    """Reconstruct the pool view at the banked geometry (r225a law)."""
    shape = tuple(p["kv_shape"])
    stride = tuple(p["kv_stride"])
    off = int(p["kv_off"])
    bt_head = p["bt_head"].to(torch.long)
    win = p["kv_win"].to(torch.uint8)
    storage = torch.zeros(
        off + shape[0] * block_stride, dtype=torch.uint8, device=dev)
    for i, b in enumerate(bt_head.tolist()):
        storage[
            off + b * block_stride: off + (b + 1) * block_stride
        ] = win[i].reshape(-1)
    return torch.as_strided(storage, shape, stride, off), storage


def native_out(path, dev, ns):
    p, r, sl = payload_info(path)
    cs = c_set(sl)
    if len(cs) != 1:
        print(
            f"R231D_ABORT: {os.path.basename(path)} rows clamp to "
            f"MIXED C={cs} -- a single ns cannot reproduce auto")
        sys.exit(1)
    c = cs[0]
    blocks = -(-sl[0] // TILE_N)
    fmt = str(p["fmt"])  # payload fmt wins (r231a5 law)
    block_stride = int(p["kv_stride"][0])
    qB = p["q"].to(torch.bfloat16).to(dev)
    sl_t = p["sl"][:r].to(torch.int32).to(dev)
    scale = float(p["scale"])
    bt_dtype = p["bt"].dtype
    bt_head = p["bt_head"].to(torch.long)
    nbt = int(bt_head.numel())
    bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=bt_dtype, device=dev)
    bt[:, :nbt] = (
        bt_head.unsqueeze(0).repeat(r, 1).to(dev).to(bt_dtype))
    cu_q = torch.arange(r + 1, device=dev, dtype=torch.int32)
    pool, storage = rebuild_pool(p, dev, block_stride)
    kw = {} if ns is None else {"num_splits_kv": ns}
    try:
        out = mx_paged_decode(
            qB, pool, fmt, cu_q, sl_t, bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale,
            **kw,
        )[:r].cpu()
    finally:
        del pool, storage
        torch.xpu.empty_cache()
    print(
        f"R231D_CLAMP {os.path.basename(path)} sl={min(sl)}-{max(sl)} "
        f"blocks={blocks} C={c} "
        f"capacity={'yes' if c == CAPACITY else 'no'} "
        f"ns={'auto' if ns is None else ns}")
    return out


def slinfo(paths):
    for path in paths:
        _, r, sl = payload_info(path)
        cs = c_set(sl)
        c_field = str(cs[0]) if len(cs) == 1 else f"MIXED{cs}"
        print(
            f"R231D_INFO {os.path.basename(path)} r={r} "
            f"sl={min(sl)}-{max(sl)} C={c_field} "
            f"capacity={'yes' if cs == [CAPACITY] else 'no'}")


def save(outdir, posture, paths):
    os.makedirs(outdir, exist_ok=True)
    ns = None
    if posture != "auto":
        if not posture.startswith("ns:"):
            print(__doc__)
            sys.exit(2)
        ns = int(posture[3:])
    for path in paths:
        out = native_out(path, torch.device("xpu"), ns)
        name = os.path.basename(path)
        torch.save(out, os.path.join(outdir, name))
        print(f"R231D_SAVED {name} shape={tuple(out.shape)}")
    return ns


def cmp(dira, dirb):
    names = sorted(
        n for n in os.listdir(dira) if not n.startswith("."))
    if not names:
        print("R231D_CMP FAIL (no saved outputs)")
        return 1
    bad = 0
    for n in names:
        pa = os.path.join(dira, n)
        pb = os.path.join(dirb, n)
        if not os.path.exists(pb):
            print(f"R231D_PAIR {n} MISSING_B")
            bad += 1
            continue
        a = torch.load(pa, map_location="cpu", weights_only=False)
        b = torch.load(pb, map_location="cpu", weights_only=False)
        if torch.equal(a, b):
            print(f"R231D_PAIR {n} BITWISE_EQUAL")
        else:
            d = (a.float() - b.float()).abs()
            n_mism = int((a != b).sum())
            print(
                f"R231D_PAIR {n} DIFF n={n_mism}/{a.numel()} "
                f"dmax={float(d.max()):.3e}")
            bad += 1
    print(f"R231D_ORACLE {'PASS' if bad == 0 else 'FAIL'} "
          f"pairs={len(names)} bad={bad}")
    return 0 if bad == 0 else 1


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    if sys.argv[1] == "slinfo":
        slinfo(sys.argv[2:])
        return 0
    if sys.argv[1] == "save":
        if len(sys.argv) < 4:
            print(__doc__)
            return 2
        save(sys.argv[2], sys.argv[3], sys.argv[4:])
        return 0
    if sys.argv[1] == "cmp":
        return cmp(sys.argv[2], sys.argv[3])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
