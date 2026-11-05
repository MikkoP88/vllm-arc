#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""r99def_reconstruct.py — corrected r99 [1] LONG-RETURN summary.

The r99 [1] battery measured everything; only the summary step looked in
the wrong dir (longprobe writes its jsons to CWD == $CT root, the summary
ran `cd $LEG`). This recomputes the pre-registered bar from the surviving
jsons at $CT root — same logic as r99_cert10_gate.sh longreturn_summary()
plus the r98 raw-bytes ghost fence — and writes def/longreturn_summary.txt.
No GPU time consumed.

Usage (rig):  cd /root/build/p624/r99 && python3 /root/build/p624/r99def_reconstruct.py
"""
import json
import sys

CLASSES = ("essay", "code", "multi", "simple")


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001 — dump bytes to convict ghosts
        with open(p, "rb") as f:
            raw = f.read()
        print(f"LOAD-DEBUG {p}: {type(e).__name__}: {e}; "
              f"bytes={len(raw)} head={raw[:60]!r}")
        raise


def main():
    full = load("longprobe_r99def_full.json")
    rehits = [load(f"longprobe_r99defr{c}_rehit.json") for c in (1, 2, 3)]
    fail, t1_miss = [], []
    for cls in CLASSES:
        v = full.get(cls, {})
        hashes = set(v.get("hashes", []))
        garble_all = list(v.get("garble", {}).values()) + [v.get("t1_garble", [])]
        for r in rehits:
            rv = r.get(cls, {})
            hashes |= set(rv.get("hashes", []))
            garble_all += list(rv.get("garble", {}).values())
        flat = [g for g in garble_all if g]
        if len(hashes) != 1:
            fail.append(f"{cls}: cross-cycle hash drift {sorted(hashes)}")
        if flat:
            fail.append(f"{cls}: garble flags {flat}")
        if any(fin != "stop" for fin in v.get("finish", [])):
            fail.append(f"{cls}: truncation finish={v.get('finish')}")
        if not v.get("t1_seed_identical"):
            t1_miss.append(cls)
        print(f"[{cls}] hashes={sorted(hashes)} finish={v.get('finish')} "
              f"chars={v.get('chars')} t1_seed_identical={v.get('t1_seed_identical')} "
              f"garble_empty={not flat}")
    print("T1_SEED_MISS_CLASSES:", t1_miss)
    bar = ("LONGRETURN_BAR_R99DEF: "
           + ("PASS" if not fail else f"FAIL {fail}"))
    print(bar)
    with open("def/longreturn_summary.txt", "w") as f:
        f.write("\n".join(
            [f"[{c}] hashes={sorted(set(full.get(c, {}).get('hashes', [])) | {h for r in rehits for h in r.get(c, {}).get('hashes', [])})}"
             for c in CLASSES]) + "\n")
        f.write("T1_SEED_MISS_CLASSES: " + str(t1_miss) + "\n")
        f.write(bar + "\n")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
