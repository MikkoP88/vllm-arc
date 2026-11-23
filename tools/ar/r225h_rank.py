#!/usr/bin/env python3
"""r225h_rank.py — rank/identity test of the post-fix residual.

r225g refuted the P-representation hypothesis: the kernel output sits
~equidistant from the exact/pair/triple float64 rungs.  The surviving
error class is a SEQUENCE-EXTENT error — the kernel's attention mixing
in one or more tokens the reference window excludes (or dropping ones
it includes).  Such contamination has a sharp algebraic signature:

  err(q, h, :) = p_extra(h) * V_extra(:)

for every q-head h sharing the contaminated token's kv-head — a RANK-1
block across the 6 GQA heads, whose top singular vector IS a decoded V
row of the payload (identifiable by cosine to ~1.0, with the token
index and implied p weight).

This script SVDs the (6 x 256) error block per (row, kv_head), reports
the rank concentration (sigma_1 / frobenius), and matches the top
right-singular vector against every decoded V row of that kv-head.

If the spectrum is flat and no cosine exceeds noise, the residual is
accumulation-class and extent is clean.

Usage: r225h_rank.py <bank_dir> <out_dir> <stem> [stem2 ...]
"""

import sys

import torch

BLOCK = 64
D = 256
GROUP = 6  # GQA: Hq=12 per worker / Hk=2


def main() -> int:
    bank_dir, out_dir = sys.argv[1], sys.argv[2]
    stems = sys.argv[3:]
    if not stems:
        print("R225H_RANK_VERDICT: FAIL (no stems)")
        return 1

    base = torch.load(f"{out_dir}/r225d_base.pt", map_location="cpu",
                      weights_only=False)
    sys.path.insert(0, "/tmp")
    from r225b_arbitrate import ref_from_payload  # noqa: E402
    from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off  # noqa: E402
    from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402

    for stem in stems:
        key = stem + ".pt"
        p = torch.load(f"{bank_dir}/{key}", map_location="cpu",
                       weights_only=False)
        r = p["q"].shape[0]
        sl = p["sl"][:r].to(torch.long)
        win = p["kv_win"]
        o_ref = ref_from_payload(p)
        err = base[key][:r].double() - o_ref        # (r, 12, 256)

        off = _mx_off(D, "mxk8v4")
        K, V = decode_kv(win[..., : off["v"]], win[..., off["v"]:],
                         "mxk8v4", D)
        V = V.double()                              # (nb, 64, 2, 256)

        print(f"== {stem}")
        hits = []
        for i in range(r):
            nb_i = int((sl[i] + BLOCK - 1) // BLOCK)
            # per kv-head V matrix of this row's full window
            for kv_h in range(2):
                E = err[i, kv_h * GROUP:(kv_h + 1) * GROUP]  # (6, 256)
                fro = float(E.norm())
                if fro < 1e-12:
                    continue
                U, S, Vh = torch.linalg.svd(E, full_matrices=False)
                conc = float(S[0] / S.norm())
                # top right-singular vector vs every decoded V row
                v_top = Vh[0]
                v_rows = V[:nb_i, :, kv_h].reshape(-1, D)
                vn = v_rows / v_rows.norm(dim=1, keepdim=True).clamp_min(1e-30)
                cos = (vn @ (v_top / v_top.norm().clamp_min(1e-30)))
                best = int(cos.argmax())
                bc = float(cos[best])
                tok = best
                print(f"   row {i} kv{kv_h}: |E|={fro:.3e} "
                      f"sigma1-conc={conc:.3f} "
                      f"best-V-cos={bc:+.4f} @ token {tok}")
                if conc > 0.9 and abs(bc) > 0.99:
                    hits.append((i, kv_h, tok, conc, bc))
        if hits:
            print(f"   RANK-1 CONTAMINATION HITS: {hits}")
        else:
            print("   (no rank-1 contamination hit)")

    print("R225H_RANK_VERDICT: PROFILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
