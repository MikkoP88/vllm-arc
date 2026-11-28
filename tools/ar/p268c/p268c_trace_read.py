#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p268c_trace_read.py — P6.47 P-D (r268c): correlate the overlay trace
with the discriminator steps.

The r268c overlay prints, per store call, `MXKVTRACE store ... smap=...
smax=...` (the slot ids the engine handed the store) and, per decode
call at B<=4, `MXKVTRACE decx ... bt0=[...] bt1=[...] bts=(...) btc=...
sl=[...] msl=...` (the block-table rows / stride / seq lens the walk
received). r267 proved the kernel math exact for CORRECT metadata, so a
RED step's decx/store values diverging from T0's (the clean reference)
names the wrong metadata VALUE directly.

Sectioning: the disc driver runs T0..T6 sequentially with >=1.5 s sleeps
and one request per step (T6 = one 2-way burst). serve.log interleaves
APIServer `POST /v1/completions|chat/completions` lines with the
engine's MXKVTRACE lines; each POST line opens the next step's window.
Within a window the FIRST store group is the prefill store (smap = the
prompt's slots, one line per layer) and each later 60-line store group
is one decode step (smap = the single new slot). decx lines print per
layer per decode step (B<=4).

Output: per step -- http/red from disc.json, store-group count, the
prefill smap head/tail, the first decode step's decx (bt0/bts/sl/msl),
and the LAST decode step's decx; then a cross-step divergence table
(sl/msl vs ptok+ctok, bt growth, smap-vs-bt consistency: every decode
smap's block MUST appear in the same step's bt0 row).

Usage: p268c_trace_read.py <serve.log> <disc.json> [out.json]
"""

from __future__ import annotations

import json
import re
import sys

POST_RE = re.compile(r"POST /(v1/completions|v1/chat/completions)")
TRACE_RE = re.compile(r"MXKVTRACE (\w+) (.*)")
KV_RE = re.compile(r"(\w+)=([^ ]+)")


# layers per forward pass (50 sliding + 10 global); a store group is one
# pass's stores for ONE pool class only when the two pools print with
# different cache= shapes -- group by (cache shape, contiguous run).
def parse(serve_log: str):
    windows = []  # each: {"posts": n, "store": [kv dicts], "dec": [...], "decx": [...]}
    cur = {"posts": 0, "store": [], "dec": [], "decx": []}
    with open(serve_log, encoding="utf-8", errors="replace") as f:
        for line in f:
            if "MXKVTRACE" in line:
                m = TRACE_RE.search(line)
                if not m:
                    continue
                tag, rest = m.group(1), m.group(2)
                kv = dict(KV_RE.findall(rest))
                if tag in ("store", "dec", "decx"):
                    cur[tag].append(kv)
                continue
            if POST_RE.search(line):
                # a POST line both closes the previous request window and
                # opens the next; the disc sleeps separate steps, so each
                # POST starts a new window.
                windows.append(cur)
                cur = {"posts": 1, "store": [], "dec": [], "decx": []}
                continue
            if "Engine 000" in line and "throughput" in line:
                pass
    windows.append(cur)
    return [w for w in windows if w["store"] or w["decx"] or w["posts"]]


def parse_list(s: str):
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:  # noqa: BLE001 — best effort on trace text
        return None


def group_stores(stores):
    """Group consecutive store lines into passes by cache shape changes.

    Each layer's pool has a fixed shape per class; a pass = 50 lines of
    the slide shape followed by 10 of the glob shape (or interleaved).
    For addressing analysis the GROUP BOUNDARY matters only for decode
    steps (single slot); we detect a new decode step when smap repeats
    the pattern [x] after a different [x] or the count crosses a layer
    multiple. Simplest robust grouping: by distinct smap values in
    order, keeping the first/last cache shape seen for each.
    """
    if not stores:
        return []
    groups = []  # list of dicts {smap, shapes}
    for kv in stores:
        smap = kv.get("smap", "[]")
        shape = kv.get("cache", "?")
        if groups and groups[-1]["smap"] == smap:
            groups[-1]["shapes"].add(shape)
        else:
            groups.append({"smap": smap, "shapes": {shape}, "n": 0})
        groups[-1]["n"] += 1
    return groups


def main() -> int:
    serve_log, disc_json = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    with open(disc_json, encoding="utf-8") as f:
        steps = json.load(f)["steps"]
    wins = parse(serve_log)
    print(f"windows={len(wins)} steps={len(steps)}")
    report = []
    for i, st in enumerate(steps):
        w = wins[i] if i < len(wins) else {"posts": -1, "store": [], "dec": [], "decx": []}
        groups = group_stores(w["store"])
        decx = w["decx"]
        # decx repeats per layer; collapse to distinct (bt0, sl, msl) sets
        sigs = []
        for kv in decx:
            sig = (kv.get("bt0"), kv.get("sl"), kv.get("msl"), kv.get("bts"), kv.get("btc"), kv.get("B"))
            if not sigs or sigs[-1] != sig:
                sigs.append(sig)
        first = sigs[0] if sigs else None
        last = sigs[-1] if sigs else None
        print(f"\n== {st['tag']}: http={st['http']} red={int(st['red'])} ptok={st['ptok']} text={st['text'][:60]!r}")
        print(f"   store_lines={len(w['store'])} groups={len(groups)} decx_lines={len(decx)} distinct_decx={len(sigs)}")
        for g in groups[:3]:
            print(f"   store-group smap={g['smap'][:60]} shapes={sorted(g['shapes'])[:2]} n={g['n']}")
        if len(groups) > 3:
            print(f"   ... ({len(groups) - 3} more groups)")
        if first:
            print(
                f"   decx-first: bt0={first[0]} sl={first[1]} msl={first[2]} bts={first[3]} btc={first[4]} B={first[5]}"
            )
        if last and last != first:
            print(f"   decx-last:  bt0={last[0]} sl={last[1]} msl={last[2]} bts={last[3]} btc={last[4]} B={last[5]}")
        # consistency: each decode store slot's block must be in a decx bt row
        dec_slots = [g for g in groups[1:]] if len(groups) > 1 else []
        bad = []
        for j, g in enumerate(dec_slots[:40]):
            sm = parse_list(g["smap"])
            if sm is None:
                continue
            sig = sigs[j] if j < len(sigs) else None
            bt0 = parse_list(sig[0]) if sig else None
            if sm and bt0 is not None:
                blocks = {s // 64 for s in sm if s >= 0}
                if not blocks.issubset(set(bt0)):
                    bad.append((j, g["smap"][:40], sig[0][:60] if sig else None))
        if bad:
            print(f"   SMAP_NOT_IN_BT at decode steps: {bad[:6]}")
        report.append(
            {
                "tag": st["tag"],
                "http": st["http"],
                "red": st["red"],
                "ptok": st["ptok"],
                "store_groups": len(groups),
                "distinct_decx": len(sigs),
                "decx_first": first,
                "decx_last": last,
                "smap_not_in_bt": bad[:10],
                "prefill_smap": groups[0]["smap"] if groups else None,
            }
        )
    if out_path:
        with open(out_path, "w", newline="\n") as f:
            json.dump(report, f, indent=1)
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
