# SPDX-License-Identifier: Apache-2.0
"""dump_pair_deep.py — P6.47 leg B: pairing validation + decoded-vector
analysis for the spec-vs-eager pool differential.

Three probes on the banked dump directories:
  1. internal: the two spec mid-verify payloads (mql==2) vs each other
     at prompt positions — identical means both are the same (target)
     pool; ~1000 B means different impls (one is the draft's pool).
  2. decoded cosine: eager p1-prefill K/V vs spec mid-verify K/V at
     prompt positions — cos ~ 1 = same content different bytes (a
     transform/scale defect); cos ~ 0 = different content.
  3. norms + the first 8 payload bytes per position (fingerprint)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, "/work")
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402


def load(d):
    return [
        torch.load(f, map_location="cpu", weights_only=False)
        for f in sorted(Path(d).glob("*.pt"))
    ]


S = load(sys.argv[1])
E = load(sys.argv[2])
ver = [p for p in S if p["mql"] == 2]
print(f"spec verify payloads: {len(ver)}")
if len(ver) >= 2:
    a, b = ver[0], ver[-1]
    bta, btb = a["bt"][0].tolist(), b["bt"][0].tolist()
    n = sum(int((a["blocks"][bta[p // 64], p % 64] != b["blocks"][btb[p // 64], p % 64]).sum()) for p in range(14))
    print(f"internal verify-vs-verify prompt-bytes differing: {n} of {14 * 1056}")
    print(f"  a: sl={int(a['seq_lens'][0])} bt={bta[:3]} impl={a['impl']}")
    print(f"  b: sl={int(b['seq_lens'][0])} bt={btb[:3]} impl={b['impl']}")

s = max(ver, key=lambda p: int(p["seq_lens"][0]))
e = next(p for p in E if p["mql"] == 14 and int(p["seq_lens"][0]) == 14)
bs, be = s["blocks"], e["blocks"]
bts, bte = s["bt"][0].tolist(), e["bt"][0].tolist()
d_ = 256
ng = 8


def kv_at(blocks, bt, p, head):
    sb = blocks[bt[p // 64], p % 64, head]  # (slot,)
    return decode_kv(sb[: d_ + ng], sb[d_ + ng :], "mxfp8", d_)


for p in [0, 3, 7, 13]:
    for head in [0]:
        ks, vs = kv_at(bs, bts, p, head)
        ke, ve = kv_at(be, bte, p, head)
        ck = torch.nn.functional.cosine_similarity(ks, ke, dim=0).item()
        cv = torch.nn.functional.cosine_similarity(vs, ve, dim=0).item()
        print(
            f"pos {p:2d} h{head}: |K_spec|={ks.norm():7.3f} |K_eager|={ke.norm():7.3f} "
            f"cosK={ck:+.4f} | |V_spec|={vs.norm():7.3f} |V_eager|={ve.norm():7.3f} cosV={cv:+.4f}"
        )
print("fingerprint spec[0:8]:", bs[bts[0], 0, 0, :8].tolist())
print("fingerprint eager[0:8]:", be[bte[0], 0, 0, :8].tolist())
