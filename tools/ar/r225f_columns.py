#!/usr/bin/env python3
"""r225f_columns.py — column-structure test of the post-fix residual.

Hypothesis under test: the remaining native-lane error (10-20x the
Triton lane, 2^-8-class tail on 3.5-5.4% of elements) is a WRONG GROUP
SCALE MAPPING on the V walk — the KV-cache MX quant groups run along
the head dim D, so a mis-mapped group scale multiplies V(j, x) by a
fixed factor per OUTPUT COLUMN x, making O_native(h, x) / O_ref(h, x)
a consistent per-column factor (power-of-two-ish), not noise.

If instead the per-column ratio medians hover at 1.0 with noisy spread,
the residual is per-element accumulation noise and the mapping is
clean.

Usage: r225f_columns.py <bank_dir> <out_dir> <stem> [stem2 ...]
"""

import sys

import torch


def main() -> int:
    bank_dir, out_dir = sys.argv[1], sys.argv[2]
    stems = sys.argv[3:]
    if not stems:
        print("R225F_COLUMNS_VERDICT: FAIL (no stems)")
        return 1

    base = torch.load(f"{out_dir}/r225d_base.pt", map_location="cpu",
                      weights_only=False)
    sys.path.insert(0, "/tmp")
    from r225b_arbitrate import ref_from_payload  # noqa: E402

    for stem in stems:
        key = stem + ".pt"
        p = torch.load(f"{bank_dir}/{key}", map_location="cpu",
                       weights_only=False)
        r = p["q"].shape[0]
        o_ref = ref_from_payload(p)
        o_nat = base[key][:r].double()

        err = (o_nat - o_ref).abs()
        big = o_ref.abs() > 0.05 * o_ref.abs().amax()
        ratio = torch.where(big, o_nat / o_ref, torch.ones_like(o_ref))

        print(f"== {stem}")
        print(f"   elements with |O_ref| > 5% max: "
              f"{int(big.sum())}/{big.numel()}")
        # per-column median ratio (over rows+heads) for the 8 MX groups
        # (D=256, group=32): a mapping bug shows medians off 1.0 in
        # power-of-two clusters, aligned to 32-element group boundaries
        for g in range(8):
            sl = slice(g * 32, (g + 1) * 32)
            m = big[:, :, sl]
            vals = ratio[:, :, sl][m]
            if vals.numel() == 0:
                continue
            md = float(vals.median())
            # deviation of the column medians inside the group
            per_col = [float(ratio[:, :, x][big[:, :, x]].median())
                       for x in range(g * 32, (g + 1) * 32)
                       if big[:, :, x].any()]
            spread = max(abs(c - md) for c in per_col) if per_col else 0.0
            print(f"   group {g} (cols {g * 32:3d}-{g * 32 + 31:3d}): "
                  f"median ratio = {md:+.5f}  "
                  f"max |col-med deviation| = {spread:.5f}  "
                  f"n={int(vals.numel())}")
        # elementwise err/|O| quantiles for context
        rel = (err / o_ref.abs().clamp_min(1e-30))[big]
        print(f"   err/|O| quantiles (big elems): "
              f"p50={float(rel.median()):.2e} "
              f"p90={float(rel.quantile(0.9)):.2e} "
              f"p99={float(rel.quantile(0.99)):.2e} "
              f"max={float(rel.max()):.2e}")

    print("R225F_COLUMNS_VERDICT: PROFILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
