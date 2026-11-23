#!/usr/bin/env python3
"""r225e_profile.py — structural profile of the POST-FIX residual
(r225d base output vs the float64 reference) on the banked r224a
payloads.

r225d proved the e2m1 subnormal defect gone (base == dbf0 bitwise) but
the native lane still sits 10-20x above the Triton lane's error, well
beyond the bf16 output grid (265-4252 ulp, 3.5-5.4% of elements > 2
ulp).  This script profiles WHERE the error lives to discriminate:

  structural bug  — error concentrated on specific heads, query rows,
                    or position ranges (layout/dispatch class)
  accumulation    — error uniform across heads/rows, scaling with
                    key-count and inversely with |O| (cancellation
                    noise from a low-precision accumulator class)

Prints per-payload: per-head dmax table (heads sorted), per-row dmax,
error vs seqlen correlation, worst-element coordinates, and the
same stats for the banked triton lane as calibration.

Usage: r225e_profile.py <bank_dir> <out_dir> <stem> [stem2 ...]
"""

import sys

import torch


def dmax(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def main() -> int:
    bank_dir, out_dir = sys.argv[1], sys.argv[2]
    stems = sys.argv[3:]
    if not stems:
        print("R225E_PROFILE_VERDICT: FAIL (no stems)")
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
        sl = p["sl"][:r].to(torch.long)
        o_ref = ref_from_payload(p)                 # (r, H, D) float64
        lane = base[key][:r].double()
        tri = p["out_triton"][:r].double()

        print(f"== {stem}  (rows={r}, heads={o_ref.shape[1]})")
        for name, o in (("native", lane), ("triton", tri)):
            err = (o - o_ref).abs()                 # (r, H, D)
            # per-head profile
            ph = err.amax(dim=(0, 2))               # (H,)
            # per-row profile + seqlen
            pr = err.amax(dim=(1, 2))               # (r,)
            # concentration: share of total error mass in worst head
            mass = err.sum(dim=(0, 2))
            share = float(mass.max() / mass.sum())
            # worst element coordinates
            flat = err.view(-1)
            wi = int(flat.argmax())
            row = wi // (o_ref.shape[1] * o_ref.shape[2])
            rem = wi % (o_ref.shape[1] * o_ref.shape[2])
            head = rem // o_ref.shape[2]
            # cancellation metric: error vs |O| magnitude at that elem
            o_mag = o_ref.abs().view(-1)[wi]
            # correlation of per-row error with seqlen
            c = float(torch.corrcoef(torch.stack(
                [pr, sl.double()]))[0, 1]) if r > 2 else float("nan")
            print(f"   {name:<7} dmax={float(flat.max()):.3e}")
            print(f"           per-head dmax: min={float(ph.min()):.2e}"
                  f" med={float(ph.median()):.2e}"
                  f" max={float(ph.max()):.2e}"
                  f" worst-head-mass-share={share:.2f}")
            print(f"           per-row  dmax: min={float(pr.min()):.2e}"
                  f" med={float(pr.median()):.2e}"
                  f" max={float(pr.max()):.2e} corr(seqlen)={c:.2f}")
            print(f"           worst elem: row={row} head={head}"
                  f" |O_ref|={float(o_mag):.3e}"
                  f" err/|O|={float(flat[wi] / max(o_mag, 1e-30)):.2e}")

    print("R225E_PROFILE_VERDICT: PROFILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
