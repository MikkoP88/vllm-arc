# SPDX-License-Identifier: Apache-2.0
"""dump_sametokens.py — P6.47 leg B: the same-tokens store check (trace22).

The corruption onset is ~2-3 generated tokens in, and every cross-lane
comparison past the divergence point is void (different tokens -> different
K/V bytes by design). This check removes that confound: boot A (eager)
PREFILLS prompt P + the SPEC-GENERATED text S' as one request — the
known-correct store path writes K/V for exactly the token sequence the
spec lane EMITTED. Boot B (spec) generated S' itself. Wherever the spec
lane's cache holds the SAME token at the SAME position, its K/V bytes
must match boot A to MX-quant variance (cross-boot same-content pairs
measure ~14 B/position of 1056):

  prompt region    differ >> variance  -> contradiction with trace18
                                           (pool exonerated) — suspect
                                           this check's payload picking
  generated region differ >> variance  -> the verify-time STORE of
                                           accepted rows writes wrong K/V
                                           content (position/RoPE/row
                                           identity defect)
  all ~variance                      -> store exonerated too: the defect
                                           lives downstream of attention
                                           (logits/accept plumbing) or in
                                           the draft model.

Usage: dump_sametokens.py EAGER_DIR SPEC_DIR PROMPT_LEN
"""

import sys
from collections import Counter
from pathlib import Path

import torch

D = 256
NG = 8
VAR_PER_POS = 64  # B/position beyond MX-quant cross-boot variance
BOUNDS = {
    "K": (0, D),
    "Ks": (D, D + NG),
    "V": (D + NG, 2 * D + NG),
    "Vs": (2 * D + NG, None),
}


def load(d):
    out = []
    for f in sorted(Path(d).glob("*.pt")):
        p = torch.load(f, map_location="cpu", weights_only=False)
        pid, n = f.name.split(".")[1:3]
        p["_pid"], p["_n"] = int(pid), int(n)
        out.append(p)
    return out


def row(p, pos):
    bt = p["bt"][0].tolist()
    blk, off = bt[pos // 64], pos % 64
    if blk >= p["blocks"].shape[0]:
        return None
    return p["blocks"][blk, off]


def main():
    e_dir, s_dir, plen = sys.argv[1], sys.argv[2], int(sys.argv[3])
    E = load(e_dir)
    S = load(s_dir)
    # Boot A: the P+S' prefill is the boot's ONLY large-mql forward.
    a = max(E, key=lambda p: int(p["mql"]))
    # Boot B: dominant pid (rank), deepest verify.
    dom = Counter(p["_pid"] for p in S).most_common(1)[0][0]
    ver = [p for p in S if p["_pid"] == dom and int(p["mql"]) in (1, 2)]
    b = max(ver, key=lambda p: int(p["seq_lens"][0]))
    a_sl, b_sl = int(a["seq_lens"][0]), int(b["seq_lens"][0])
    print(f"A: prefill mql={int(a['mql'])} sl0={a_sl} bt={a['bt'][0, :3].tolist()} pid={a['_pid']}")
    print(f"B: verify  mql={int(b['mql'])} sl0={b_sl} bt={b['bt'][0, :3].tolist()} pid={b['_pid']}")
    if a_sl < b_sl:
        print(f"NOTE: A stops at {a_sl}; comparing 0..{a_sl - 1} only")

    sys.path.insert(0, "/work")
    from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402

    regions = {"prompt": [0, 0, 0], "generated": [0, 0, 0]}  # pos, bytes, over
    for pos in range(a_sl):
        ra, rb = row(a, pos), row(b, pos)
        if ra is None or rb is None:
            print(f"pos {pos}: SKIP (beyond capture)")
            continue
        d = ra != rb
        n = int(d.sum())
        parts = {k: int(d[:, lo:hi or d.shape[-1]].sum()) for k, (lo, hi) in BOUNDS.items()}
        ks_a, vs_a = decode_kv(ra[0, : D + NG], ra[0, D + NG :], "mxfp8", D)
        ks_b, vs_b = decode_kv(rb[0, : D + NG], rb[0, D + NG :], "mxfp8", D)
        cos = torch.nn.functional.cosine_similarity
        ck, cv = float(cos(ks_a, ks_b, dim=0)), float(cos(vs_a, vs_b, dim=0))
        reg = "prompt" if pos < plen else "generated"
        regions[reg][0] += 1
        regions[reg][1] += n
        over = n > VAR_PER_POS
        regions[reg][2] += over
        flag = " <-- BEYOND VARIANCE" if over else ""
        print(f"pos {pos:3d} [{reg:9s}]: {n:4d}/1056 B cosK={ck:+.4f} cosV={cv:+.4f}{flag}")
        if pos >= plen and n and not over and (pos - plen) % 8 == 0:
            print(f"          parts={parts}")
    for reg, (npos, nbytes, nover) in regions.items():
        print(f"{reg}: {nover}/{npos} positions beyond {VAR_PER_POS} B variance, {nbytes} B total")
    verdict = "STORE CLEAN (defect downstream of attention/store)" if all(
        r[2] == 0 for r in regions.values()
    ) else "STORE SUSPECT (generated-region K/V content wrong)"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
