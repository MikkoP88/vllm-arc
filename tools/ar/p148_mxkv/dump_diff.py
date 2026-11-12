# SPDX-License-Identifier: Apache-2.0
"""dump_diff.py — P6.47 leg B: spec-vs-eager KV-pool ground-truth
differential, SEQUENCE-ALIGNED. Each side is a DIRECTORY of multi-shot
ARC_MXKV_DUMP payloads; the comparator picks the prompt-1-like payload
per side (a live p1 forward: modest seq_lens, non-dummy shapes — the
async scheduler interleaves idle-dummy forwards that eat sentinels),
then compares the sequence bytes position-by-position through each
boot's OWN block table. Committed tokens are identical until the text
diverged (~pos 40 for p1), so LARGE byte differences before that are
wrong stores; small counts (<= ~25 B/pos) are cross-boot numerics
variance (batch-shape-dependent reductions flip a few quant bytes)."""

import sys
from pathlib import Path

import torch


def load_dir(d: Path) -> list[dict]:
    out = []
    for f in sorted(d.glob("*.pt")):
        try:
            out.append(torch.load(f, map_location="cpu", weights_only=False))
        except Exception as exc:  # noqa: BLE001 — skip unreadable shards
            print(f"  (skip {f.name}: {exc})")
    return out


def pick_p1(payloads: list[dict]) -> dict | None:
    # p1 = ~14-token prompt + <= 64 outputs: a live forward has
    # 16 <= sl <= 120 and msl < 1000 (dummy/idle shapes run 1000+).
    best = None
    for p in payloads:
        sl = int(p["seq_lens"][0])
        msl = int(p["msl"])
        deeper = best is None or sl > int(best["seq_lens"][0])
        if 16 <= sl <= 120 and msl < 1000 and deeper:
            best = p  # deepest captured state
    return best


spec_dir, eager_dir = Path(sys.argv[1]), Path(sys.argv[2])
A_all, B_all = load_dir(spec_dir), load_dir(eager_dir)
print(f"spec payloads: {len(A_all)}, eager payloads: {len(B_all)}")
for tag, allp in (("spec", A_all), ("eager", B_all)):
    for i, p in enumerate(allp):
        print(f"  {tag}[{i}] sl={int(p['seq_lens'][0])} mql={p['mql']} msl={p['msl']}")
A, B = pick_p1(A_all), pick_p1(B_all)
if A is None or B is None:
    print("VERDICT NO-P1-PAYLOAD")
    sys.exit(1)

ba, bb = A["blocks"], B["blocks"]  # (256, 64, Hk, slot) uint8
bta, btb = A["bt"][0].tolist(), B["bt"][0].tolist()
print(f"\nspec p1:  sl={int(A['seq_lens'][0])} bt0={bta[:6]} smap={A['smap'][-4:].tolist()}")
print(f"eager p1: sl={int(B['seq_lens'][0])} bt0={btb[:6]} smap={B['smap'][-4:].tolist()}")

N_POS = min(int(A["seq_lens"][0]), int(B["seq_lens"][0]), 96)
slot = ba.shape[-1]
d_ = 256  # D=256 for qwen3.8 (slot 528 => mxfp8: 256+8+256+8)
ng = 8
bounds = {"K": (0, d_), "Ks": (d_, d_ + ng), "V": (d_ + ng, 2 * d_ + ng), "Vs": (2 * d_ + ng, slot)}


def seq_bytes(blocks: torch.Tensor, bt: list[int]) -> torch.Tensor:
    return torch.stack([blocks[bt[p // 64], p % 64] for p in range(N_POS)])


sa, sb = seq_bytes(ba, bta), seq_bytes(bb, btb)
diff = sa != sb
per_pos = diff.sum(dim=(1, 2))
nz = (per_pos > 0).nonzero().flatten().tolist()
big = [p for p in nz if per_pos[p].item() > 64]  # >64 B/pos = beyond variance
print(f"\npositions compared: {N_POS} (spec sl={int(A['seq_lens'][0])}, eager sl={int(B['seq_lens'][0])})")
print(f"differing: {len(nz)} (small <=64B: {len(nz) - len(big)}, LARGE >64B: {len(big)})")
print(f"first 10 differing: {nz[:10]}")
print(f"LARGE positions: {big[:24]}")
for p in (big[:12] if big else nz[:12]):
    row = diff[p]
    parts = [f"{n}:{int(row[:, lo:hi].sum())}" for n, (lo, hi) in bounds.items() if int(row[:, lo:hi].sum())]
    print(f"  pos {p:3d}: {per_pos[p].item():4d} B ({', '.join(parts)})")
print("VERDICT", "SEQUENCE-LARGE-DIFF" if big else ("SEQUENCE-VARIANCE-ONLY" if nz else "SEQUENCE-IDENTICAL"))
