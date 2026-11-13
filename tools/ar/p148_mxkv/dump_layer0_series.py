# SPDX-License-Identifier: Apache-2.0
"""dump_layer0_series.py — P6.47 leg B: layer-pinned pool time series.

Input: one or two dump directories from an ARC_MXKV_DUMP=1 boot with the
r152 ord-pinned instrument (payloads carry "ord"; only ord==0 dumps).
Every layer owns a separate pool tensor at the same slot ids, so only
same-ordinal payloads are comparable (r151's 97%-differ "pair" was two
layers — the cross-layer null result that motivated the pin).

With ONE dir: the in-boot series for the pinned layer —
  A = p1 prefill (mql==sl==14), C = earliest verify (mql==2),
  B = deepest verify (mql==2)
  A-vs-C / C-vs-B / A-vs-B at prompt positions 0..13 through p1's bt.
  Bytes at committed prompt positions must never change after prefill:
  A-vs-B LARGE = a mid-generation writer rewrites live prompt K/V (the
  corruption mechanism, seen inside one boot with zero confounds); the
  A-vs-C / C-vs-B split localizes WHEN.

With TWO dirs (spec eager): additionally the valid cross-boot pair
  spec-B vs eager-A (both ord 0) with decoded-vector cosine —
  cos~0 = different content, cos~1 = same content, corrupted transform.
"""

import sys
from collections import Counter
from pathlib import Path

import torch

D = 256
NG = 8
N_POS = 14
BOUNDS = {
    "K": (0, D),
    "Ks": (D, D + NG),
    "V": (D + NG, 2 * D + NG),
    "Vs": (2 * D + NG, None),
}


def load_series(d):
    out = []
    for f in sorted(Path(d).glob("*.pt")):
        p = torch.load(f, map_location="cpu", weights_only=False)
        if "ord" not in p:
            print(f"OLD-FORMAT payload {f.name} (no ord) — rerun with r152 engine")
            sys.exit(2)
        pid, n = f.name.split(".")[1:3]
        p["_pid"], p["_n"] = int(pid), int(n)
        out.append(p)
    return sorted(out, key=lambda p: p["_n"])


def pick(d, dom_pid):
    p1 = [p for p in d if p["_pid"] == dom_pid]
    ver = [p for p in p1 if p["mql"] in (1, 2)]  # spec verifies / eager decodes
    c = min(ver, key=lambda p: int(p["seq_lens"][0]), default=None)
    b = max(ver, key=lambda p: int(p["seq_lens"][0]), default=None)
    # A = the prefill of the SAME request the verifies belong to (same
    # pid — ranks hold different heads of the same K/V — and same first
    # block). trace18: p2's 14/14 prefill at blocks 247+ was mis-picked
    # as p1's A and blew the 208-block capture bound.
    base = int(b["bt"][0, 0]) if b is not None else None
    a = next(
        (
            p
            for p in p1
            if p["mql"] == 14
            and int(p["seq_lens"][0]) == 14
            and base is not None
            and int(p["bt"][0, 0]) == base
        ),
        None,
    )
    return a, c, b


def diff_pair(x, y, label):
    btx, bty = x["bt"][0].tolist(), y["bt"][0].tolist()
    cap = min(x["blocks"].shape[0], y["blocks"].shape[0])
    need = max(max(btx[: (N_POS - 1) // 64 + 1]), max(bty[: (N_POS - 1) // 64 + 1]))
    if need >= cap:
        print(f"{label}: SKIP — block {need} beyond capture {cap}")
        return -1
    tot, parts_tot = 0, Counter()
    rows = []
    for p in range(N_POS):
        d = x["blocks"][btx[p // 64], p % 64] != y["blocks"][bty[p // 64], p % 64]
        n = int(d.sum())
        tot += n
        parts = {k: int(d[:, lo:hi or d.shape[-1]].sum()) for k, (lo, hi) in BOUNDS.items()}
        for k, v in parts.items():
            parts_tot[k] += v
        if n:
            rows.append(f"pos {p:2d}: {n:4d} B")
    verdict = "LARGE" if tot > 64 * N_POS else ("variance" if tot else "IDENTICAL")
    print(f"{label}: {tot} of {N_POS * 1056} B differ -> {verdict}")
    if rows:
        print("  " + "; ".join(rows[:6]) + (" ..." if len(rows) > 6 else ""))
    print(f"  parts: {dict(parts_tot)}")
    return tot


S = load_series(sys.argv[1])
pids = Counter(p["_pid"] for p in S)
dom = pids.most_common(1)[0][0]
print(f"payloads: {len(S)} (pid {dom}: {pids[dom]}) ords: {sorted({p['ord'] for p in S})}")
for p in S:
    if p["_pid"] == dom:
        print(f"  n={p['_n']:3d} mql={p['mql']:5d} sl={int(p['seq_lens'][0]):5d} bt={p['bt'][0, :4].tolist()}")
a, c, b = pick(S, dom)
print(f"\nseries: A={'sl=' + str(int(a['seq_lens'][0])) if a else 'MISSING'} "
      f"C(sl={int(c['seq_lens'][0]) if c else '?'}) B(sl={int(b['seq_lens'][0]) if b else '?'})")
if c is not None and b is not None:
    diff_pair(c, b, "C-vs-B (first-vs-deepest verify — prompt bytes must be STABLE)")
    # Committed GENERATED positions: everything below C's seq len is
    # frozen history by B's time — bytes at 0..sl(C)-1 must be stable
    # AND non-stale (an empty/unwritten row reads as zeros; a forked
    # store would leave the bt block stale).
    n_gen = min(int(c["seq_lens"][0]), 63) + 1
    saved = N_POS
    N_POS = n_gen
    diff_pair(c, b, f"C-vs-B THROUGH sl={n_gen - 1} (prompt+generated committed history)")
    N_POS = saved
    cbt = c["bt"][0].tolist()
    for p in range(14, min(n_gen, 62), 4):
        row = c["blocks"][cbt[p // 64], p % 64]
        print(f"  C pos {p}: maxbyte={int(row.max())} nonzero={int((row != 0).sum())}/1056")
    if a is not None:
        diff_pair(a, c, "A-vs-C (prefill -> first captured verify)")
        diff_pair(a, b, "A-vs-B (prefill -> deepest verify)")

if len(sys.argv) > 2:
    print()
    E = load_series(sys.argv[2])
    edom = Counter(p["_pid"] for p in E).most_common(1)[0][0]
    ep1 = [p for p in E if p["_pid"] == edom]
    ea = next(
        (p for p in ep1 if p["mql"] == 14 and int(p["seq_lens"][0]) == 14),
        None,
    )
    tgt = b or c
    if tgt is None or ea is None:
        print(f"CROSS-BOOT: missing spec verify ({tgt is not None}) or eager prefill ({ea is not None})")
        sys.exit(0)
    print(f"cross-boot: spec verify sl={int(tgt['seq_lens'][0])} vs eager prefill sl={int(ea['seq_lens'][0])}")
    tot = diff_pair(tgt, ea, "spec-verify-vs-eager-prefill")
    sys.path.insert(0, "/work")
    from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402

    bts, bte = tgt["bt"][0].tolist(), ea["bt"][0].tolist()
    for p in [0, 7, 13]:
        ss = tgt["blocks"][bts[p // 64], p % 64, 0]
        se = ea["blocks"][bte[p // 64], p % 64, 0]
        ks, vs = decode_kv(ss[: D + NG], ss[D + NG :], "mxfp8", D)
        ke, ve = decode_kv(se[: D + NG], se[D + NG :], "mxfp8", D)
        cos = torch.nn.functional.cosine_similarity
        print(f"  pos {p:2d}: cosK={cos(ks, ke, dim=0):+.4f} cosV={cos(vs, ve, dim=0):+.4f} "
              f"|K| {ks.norm():.1f}/{ke.norm():.1f} |V| {vs.norm():.1f}/{ve.norm():.1f}")
