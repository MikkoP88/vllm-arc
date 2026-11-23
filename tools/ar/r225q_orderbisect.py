#!/usr/bin/env python3
"""r225q_orderbisect.py — bisect the gate3/gate4-vs-replica harness delta.

gate4 (verbatim r225d_gate.sh re-run on the same build24 wheel + banked
payloads) reproduced gate3's FAIL byte-identically: native-vs-oracle
7.812e-3 / 5.078e-2 / 5.859e-3 (fp64 arbitration 7.685e-3 / 5.158e-2 /
5.999e-3). Every replica harness (r225o page matrix, r225p order probe,
r225l split ladder) replays the SAME payload-geometry CLEAN at the bf16
grid floor (2447.30: dmax 1.556e-2, mism 1/3072, max_ulp 0.54). Both
sides are deterministic -> a real harness delta, not transient
pollution.

Structural deltas between r225a and the replicas, all now candidates:
  1. PAYLOAD ORDER: the gate ran [2447.18, 2447.30, 2547.18] in ONE
     process; 2447.30 was the SECOND replay (fresh allocator only saw
     2447.18 first). r225a's finally does del + torch.xpu.empty_cache()
     per payload, so cross-payload dirt reaches the next replay through
     DRIVER-level page recycling, not the caching allocator.
  2. IN-WINDOW STATE: within one replay_one the sequence is loads ->
     bt -> pool -> TRITON oracle on the same pool/bt/q16 instances ->
     native op. The replicas called the native op alone on a fresh
     pool (r225p's "triton-first" cell rebuilt a different pool tensor
     for the native call -- documented flaw).
  3. BANKED fmt STRING: r225a passes p["fmt"] verbatim; the replicas
     hardcode "mxk8v4". If the banked string differs at all the
     replicas ran a different format lane.
  4. VACUOUS POISON: r225p's poison freed +1.0-filled blocks and THEN
     called torch.xpu.empty_cache() -- releasing the poisoned blocks to
     the driver before the op allocated. The op's temp_out (6.3 MB
     fp32) / exp_sums / max_logits (24 KB fp32 each) are at::empty --
     if any read touches a slot the kernel left unwritten, recycled
     contents become the output's error. That probe measured fresh
     pages, proving nothing.

Cells (one fresh process each; fp64 scoring runs strictly AFTER the
native call so the pre-native sequence is byte-r225a):
  solo_verbatim          one payload, r225a replay_one verbatim
                        (must reproduce 5.158e-2 or the script itself
                        still differs from the gate)
  pre18_verbatim         2447.18 verbatim, then 2447.30 verbatim (the
                        gate's structure minus 2547.18)
  solo_notriton          verbatim minus the mx_decode call
  solo_bare              minus triton AND the banked-output device loads
  solo_fmtfix            verbatim but fmt="mxk8v4" hardcoded
  solo_notriton_poison   notriton + freed-cached fills (+1.0f) of the
                        op's exact internal sizes, NO empty_cache
  solo_notriton_poisonLSE  only the two 24 KB fills (LSE-sized dirt)
  solo_notriton_poisonTMP  only the 6.3 MB fill (temp_out-sized dirt)

Reads: C1 errs + C3/C4 clean -> in-window triton/loads delta; C1 clean
+ C2 errs -> cross-payload driver recycling; C5 flips -> banked fmt;
C6/C7/C8 err -> recycled temp bytes are the vector (LSE-sized vs
TMP-sized names the buffer).

Usage: r225q_orderbisect.py <cell> <30.pt> [18.pt]
"""

import hashlib
import os
import sys

import torch

sys.path.insert(0, "/tmp")
from r225j_iso import dmax, grid_stats, model_row  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.triton_mx_decode import mx_decode  # noqa: E402
from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

BLOCK_STRIDE = 51200  # 64 tokens * 2 kv-heads * 400 B slot (observed pool)
ENGINE_BT_WIDTH = 4114
D = 256

CELLS = (
    "solo_verbatim",
    "pre18_verbatim",
    "solo_notriton",
    "solo_bare",
    "solo_fmtfix",
    "solo_notriton_poison",
    "solo_notriton_poisonLSE",
    "solo_notriton_poisonTMP",
)

TMP_BYTES = 4 * 12 * 256 * 128 * 4  # temp_out: fp32 [q,S,Hq*D] at S=128
LSE_BYTES = 4 * 12 * 128 * 4  # exp_sums == max_logits: fp32 [q,Hq*S]


def rebuild_pool(p, dev):
    """Reconstruct the pool view with the recorded geometry (r225a law)."""
    shape = tuple(p["kv_shape"])
    stride = tuple(p["kv_stride"])
    off = int(p["kv_off"])
    bt_head = p["bt_head"].to(torch.long)
    win = p["kv_win"].to(torch.uint8)
    storage = torch.zeros(off + shape[0] * BLOCK_STRIDE, dtype=torch.uint8, device=dev)
    for i, b in enumerate(bt_head.tolist()):
        storage[off + b * BLOCK_STRIDE : off + (b + 1) * BLOCK_STRIDE] = win[i].reshape(-1)
    return torch.as_strided(storage, shape, stride, off), storage


def digest(t) -> str:
    return hashlib.md5(t.contiguous().view(torch.uint16).cpu().numpy().tobytes()).hexdigest()[:12]


def replay(path, dev, *, triton=True, loads=True, fmt=None, poison=()):
    """r225a replay_one VERBATIM plus cell toggles. Nothing runs before
    the native call that r225a itself did not do."""
    p = torch.load(path, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    q16 = p["q"].to(dev)
    sl = p["sl"][:r].to(dev)
    out_triton = p["out_triton"][:r].to(dev) if loads else None
    out_native = p["out_native"][:r].to(dev) if loads else None
    scale = float(p["scale"])
    use_fmt = fmt if fmt is not None else p["fmt"]
    bt_dtype = p["bt"].dtype  # the engine's table dtype, preserved
    bt_head = p["bt_head"].to(torch.long)
    nbt = int(bt_head.numel())
    bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=bt_dtype, device=dev)
    bt[:, :nbt] = bt_head.unsqueeze(0).repeat(r, 1).to(dev).to(bt_dtype)
    sl32 = sl.to(torch.int32)
    cu_q = torch.arange(r + 1, device=dev, dtype=torch.int32)

    print(
        f"== {os.path.basename(path)} fmt={use_fmt!r} q={p['q'].dtype} "
        f"sl={p['sl'].dtype} bt={bt_dtype} sl4={sl[:4].tolist()} "
        f"scale={scale}"
    )
    pool, storage = rebuild_pool(p, dev)
    try:
        t_rep = mx_decode(q16, pool, bt, sl, scale) if triton else None
        for nbytes in poison:  # freed-but-CACHED dirty blocks, NO flush
            filler = torch.full((nbytes // 4,), 1.0, dtype=torch.float32, device=dev)
            del filler
        n_dev = mx_paged_decode(
            q16.to(torch.bfloat16),
            pool,
            use_fmt,
            cu_q,
            sl32,
            bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale,
            num_splits_kv=0,
        )
        if t_rep is not None and loads:
            print(f"   REPLAY triton-vs-banked dmax={dmax(t_rep, out_triton):.3e}")
            print(f"   REPLAY native-vs-banked dmax={dmax(n_dev[:r], out_native):.3e}")
            print(f"   REPLAY native-vs-oracle dmax={dmax(n_dev[:r], t_rep):.3e}")
        n_rep = n_dev[:r].cpu()
        return p, n_rep
    finally:
        del pool, storage
        torch.xpu.empty_cache()


def score(tag, p, n_out):
    """fp64 identity-walk arbitration (r225j law), after the fact."""
    off = _mx_off(D, "mxk8v4")
    r = n_out.shape[0]
    q = p["q"][:r].to(torch.bfloat16)
    K, V = decode_kv(p["kv_win"][..., : off["v"]], p["kv_win"][..., off["v"] :], "mxk8v4", D)
    worst = (0, 0.0)
    dm = 0.0
    for i in range(r):
        model = model_row(q[i].double(), K.double(), V.double(), int(p["sl"][i]), float(p["scale"]))
        mism, mul = grid_stats(n_out[i], model)
        worst = (max(worst[0], mism), max(worst[1], mul))
        dm = max(dm, dmax(n_out[i], model))
    print(f"R225Q cell={tag} dmax={dm:.3e} mism={worst[0]}/{12 * D} max_ulp={worst[1]:.2f} md5={digest(n_out)}")


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in CELLS:
        print(__doc__)
        print(f"usage: {sys.argv[0]} <{'|'.join(CELLS)}> <30.pt> [18.pt]")
        return 2
    cell, p30 = sys.argv[1], sys.argv[2]
    dev = torch.device("xpu")

    if cell == "pre18_verbatim":
        if len(sys.argv) < 4:
            print("pre18_verbatim needs the 2447.18 payload as argv[3]")
            return 2
        p18, n18 = replay(sys.argv[3], dev)
        score("pre18:" + cell, p18, n18)

    p, n = replay(
        p30,
        dev,
        triton=cell
        not in (
            "solo_notriton",
            "solo_bare",
            "solo_notriton_poison",
            "solo_notriton_poisonLSE",
            "solo_notriton_poisonTMP",
        ),
        loads=cell != "solo_bare",
        fmt="mxk8v4" if cell == "solo_fmtfix" else None,
        poison={
            "solo_notriton_poison": (TMP_BYTES, LSE_BYTES, LSE_BYTES),
            "solo_notriton_poisonLSE": (LSE_BYTES, LSE_BYTES),
            "solo_notriton_poisonTMP": (TMP_BYTES,),
        }.get(cell, ()),
    )
    score(cell, p, n)
    print("R225Q_ORDERBISECT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
