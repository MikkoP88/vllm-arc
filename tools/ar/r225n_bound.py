#!/usr/bin/env python3
"""r225n_bound.py — boundary-defect hypothesis test (r225 follow-up).

The CPU split contract (r225m L1) sits at the bf16 grid floor while the
wheel's auto/nsplit2 outputs carry ~5e-2 — a real kernel defect. Static
review cleared scheduler, partition, layouts, walk, masking, epilogue
and combine; the surviving hypothesis class is a WALK defect at the
split boundary (dropped / duplicated / shifted tiles), which hides at
~one-tile mass in the flat bulk — larger at the recent-token tail —
the observed class.

This probe computes, at S=2, the fp64 output under boundary hypotheses
and scores each against the MEASURED wheel output (r225d_base.pt):

  H0      exact partition (control; = full softmax, row-length aware)
  dropL   last tile of split 0 omitted
  dropF   first tile of split 1 omitted
  dropB   both boundary tiles omitted
  dupL    last tile of split 0 walked twice (in split 1 too)
  shift   split 1 starts one tile EARLY (boundary tile in both)

All row sets are built PER ROW from that row's seqlen (payload rows
carry sl 5462..5465 — one global partition silently truncates tails
that carry 0.1-0.23 of output mass; dbg-proven).

A matching hypothesis collapses dmax(hyp, wheel) to the wheel's own
output-grid class (~1.5e-2); non-matching stay at the defect class
(~5e-2+). Usage: r225n_bound.py <payload.pt> <wheel_base.pt>
"""

import sys

import torch

from r225m_splitsim import (
    D,
    TILE,
    dmax,
    logits_for_payload,
)

GROUP = 6


def softmax_rows(logits, v, rows):
    """fp64 softmax attention over the GIVEN row set (ordered)."""
    if not rows:
        return torch.zeros(v.shape[1], dtype=torch.float64)
    idx = torch.tensor(rows, dtype=torch.long)
    s = logits[idx]
    return torch.softmax(s, -1) @ v[idx]


def row_sets(n):
    """Hypothesis row sets for one row of seqlen n at S=2."""
    tiles = (n + TILE - 1) // TILE
    nbps = (tiles + 1) // 2

    def tile_rows(t):
        return list(range(t * TILE, min((t + 1) * TILE, n)))

    full = list(range(n))
    b_last0 = tile_rows(nbps - 1)   # last tile of split 0
    b_first1 = tile_rows(nbps)      # first tile of split 1
    return (
        ("H0", full),
        ("dropL", [x for x in full if x not in b_last0]),
        ("dropF", [x for x in full if x not in b_first1]),
        ("dropB", [x for x in full
                   if x not in b_last0 and x not in b_first1]),
        ("dupL", full + b_last0),
        ("shift", [x for x in full if x not in b_last0] + b_last0),
    )


def main() -> int:
    payload_path, base_path = sys.argv[1], sys.argv[2]
    stem = payload_path.split("/")[-1][:-3]

    p = torch.load(payload_path, map_location="cpu", weights_only=False)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    r = p["q"].shape[0]
    heads = p["q"].shape[1]
    per_row = logits_for_payload(p)
    wb = base[stem + ".pt"][:r].double()

    o_ref = torch.empty(r, heads, D, dtype=torch.float64)
    for i in range(r):
        for h in range(heads):
            s, v = per_row[i][h]
            o_ref[i, h] = torch.softmax(s, -1) @ v
    print(f"== {stem}  sl={p['sl'].tolist()}  wheel-vs-ref dmax="
          f"{dmax(wb, o_ref):.3e}")

    outs = {name: torch.empty(r, heads, D, dtype=torch.float64)
            for name, _ in row_sets(int(p["sl"][0]))}
    for i in range(r):
        sets = dict(row_sets(int(p["sl"][i])))
        for name, rows in sets.items():
            for h in range(heads):
                s, v = per_row[i][h]
                outs[name][i, h] = softmax_rows(s, v, rows)

    for name, rows in row_sets(int(p["sl"][0])):
        out_b = outs[name].to(torch.bfloat16).double()
        print(f"   {name:<6} rows={len(rows):5d}  vs-ref="
              f"{dmax(out_b, o_ref):.3e}  vs-wheel="
              f"{dmax(out_b, wb):.3e}")
    print("R225N_BOUND_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
