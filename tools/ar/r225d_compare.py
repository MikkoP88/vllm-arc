#!/usr/bin/env python3
"""r225d_compare.py — post-fix identity + residual-class checks for the
r225 wheel gate.

The r225 fix (e2m1 subnormal mantissa drop in mx_dequant.hpp) must do
two things to the replayed mxk8v4 native output on the banked r224a
payloads:

  1. MOVE it off the banked broken output — base(new wheel) vs
     out_native(payload) must be large (~the old error class).  A
     bitwise match means the new wheel did not actually take (stale
     .so in the venv) — the silent-stale-buffer canary law.
  2. Make the direct-bf16 fast path EQUAL the float path — with the
     subnormal fixed, every e2m1 value (0.5/1/1.5/2/3/4/6 x 2^k) is
     exactly representable in bf16, so bit assembly and float->bf16
     conversion must agree: dmax(base, dbf0) ~ 0.

The arbitration then shows base landing at the float-path residual
(8-25x collapse) rather than the Triton lane's own error — this script
classifies that residual: is it the bf16 OUTPUT-rounding floor (each
lane's output is a bf16 tensor; disagreement with the float64 ref below
~1-2 ulp of the output grid is a rounding class, not a math error) or a
remaining structural error?  The banked Triton lane is scored with the
SAME ulp metric as calibration: if triton sits at ~0.5-1 ulp and native
at ~1-2 ulp, both lanes are on their output grid and the residual is
the grid, not the kernel.

Usage: r225d_compare.py <bank_dir> <out_dir> <stem> [stem2 ...]
"""

import sys
from collections import namedtuple

import torch

UlpStats = namedtuple("UlpStats",
                      "max_ulp frac_gt2 ref_p50 ref_max n_tiny tiny_max")


def dmax(a, b) -> float:
    return float((a.double() - b.double()).abs().max())


def ulp_stats(o_lane, o_ref) -> UlpStats:
    """bf16-ulp classification of a lane output vs the float64 ref.

    ulp(x) = 2^(floor(log2|x|) - 7) for bf16 (1+7 precision bits).
    Elements with |ref| below the bf16-tiny threshold are counted and
    their absolute deviation reported (attention outputs are ~1e-2..10,
    so this tail is normally empty).
    """
    ref = o_ref.double()
    diff = (o_lane.double() - ref).abs()
    a = ref.abs()
    tiny = torch.finfo(torch.bfloat16).tiny
    big = a > tiny
    e = torch.floor(torch.log2(a[big].clamp_min(tiny)))
    ulp = torch.pow(2.0, e - 7)
    ratio = diff[big] / ulp
    n_tiny = int(a.numel() - int(big.sum()))
    t_max = float(diff[~big].max()) if n_tiny else 0.0
    return UlpStats(float(ratio.max()), float((ratio > 2.0).float().mean()),
                    float(a.median()), float(a.max()), n_tiny, t_max)


def main() -> int:
    bank_dir, out_dir = sys.argv[1], sys.argv[2]
    stems = sys.argv[3:]
    if not stems:
        print("R225D_COMPARE_VERDICT: FAIL (no stems)")
        return 1

    base = torch.load(f"{out_dir}/r225d_base.pt", map_location="cpu",
                      weights_only=False)
    dbf0 = torch.load(f"{out_dir}/r225d_dbf0.pt", map_location="cpu",
                      weights_only=False)

    # import the arbitration reference (proven r224d math)
    sys.path.insert(0, "/tmp")
    from r225b_arbitrate import ref_from_payload  # noqa: E402

    ok = True
    floor_ok = True
    for stem in stems:
        key = stem + ".pt"
        p = torch.load(f"{bank_dir}/{key}", map_location="cpu",
                       weights_only=False)
        r = p["q"].shape[0]
        old = p["out_native"]
        o_ref = ref_from_payload(p)

        d_fix = dmax(base[key][:r], old[:r])       # must be LARGE
        d_ident = dmax(base[key][:r], dbf0[key][:r])  # must be ~0
        moved = d_fix >= 1e-3
        ident = d_ident <= 1e-6
        ok = ok and moved and ident

        # residual class: ulp of the bf16 output grid, native vs triton
        tn = ulp_stats(p["out_triton"][:r], o_ref)
        nn = ulp_stats(base[key][:r], o_ref)
        # rounding-floor call: native within 2 output ulp everywhere
        floor = nn.max_ulp <= 2.0 and (nn.n_tiny == 0 or
                                       nn.tiny_max <= 1e-30)
        floor_ok = floor_ok and floor

        print(f"== {stem}")
        print(f"   dmax(base_new, banked_broken) = {d_fix:.3e}"
              f"  [{'MOVED' if moved else 'STALE?'}]")
        print(f"   dmax(base_new, dbf0_new)      = {d_ident:.3e}"
              f"  [{'IDENTICAL' if ident else 'DIVERGED'}]")
        print(f"   ref |O| p50/max               = "
              f"{tn.ref_p50:.3e}/{tn.ref_max:.3e}")
        print(f"   triton lane : max_ulp={tn.max_ulp:.2f}  frac>2ulp="
              f"{tn.frac_gt2:.2e}")
        print(f"   native lane : max_ulp={nn.max_ulp:.2f}  frac>2ulp="
              f"{nn.frac_gt2:.2e}  tiny_elems={nn.n_tiny}"
              f" tiny_max={nn.tiny_max:.2e}"
              f"  [{'ROUNDING_FLOOR' if floor else 'STRUCTURAL'}]")

    verdict = "PASS" if (ok and floor_ok) else (
        "PASS_DEFECT_GONE" if ok else "FAIL")
    print(f"R225D_COMPARE_VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
