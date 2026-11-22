#!/usr/bin/env python3
"""Table the r212c ladders (eng256p16 / eng256p64 / reg128p16)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def load(tag):
    out = {}
    for f in sorted(HERE.glob(f"p204_{tag}_c*.json")):
        ctx = int(f.stem.split("_c")[-1])
        out[ctx] = json.load(open(f))
    return out


def fmt_cell(cells, name):
    c = cells.get(name)
    if not isinstance(c, dict) or "dev_us" not in c:
        return None, None
    return c["dev_us"], c.get("canary")


def main():
    ladders = [("eng256p16", "D=256 page16 (clean; _16-at-64 proxy)"),
               ("eng256p64", "D=256 page64 (_64 policy -- CORRUPT)"),
               ("reg128p16", "D=128 page16 (certified-cell regression)")]
    for tag, title in ladders:
        data = load(tag)
        print(f"\n== {title} ==")
        for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
            rows = []
            for ctx in sorted(data):
                cells = data[ctx][fmt]
                v2, c2 = fmt_cell(cells, "v2_es")
                v5, c5 = fmt_cell(cells, "v5_es")
                nat, cn = fmt_cell(cells, "native_es")
                if not (v2 and v5 and nat):
                    continue
                lane = v5 if fmt in ("mxfp8",) else min(v2, v5)
                # per-format engine lane: mxfp8 v5; mxfp4/mxk8v4 best-of
                rows.append((ctx, v2, v5, nat, nat / lane))
            if rows:
                print(f" {fmt}: ctx  v2_es  v5_es  native  n/lane")
                for r in rows:
                    print(f"   {r[0]:6d} {r[1]:7.1f} {r[2]:7.1f} {r[3]:7.1f}  {r[4]:.2f}x")
    # canary audit across everything
    bad = []
    for tag, _ in ladders:
        for ctx, d in load(tag).items():
            for fmt, cells in d.items():
                for n, c in cells.items():
                    if isinstance(c, dict) and c.get("canary", 0) > 2e-3:
                        bad.append(f"{tag} c{ctx} {fmt} {n}: {c['canary']:.2e}")
    print("\nCANARY_DIRTY:" if bad else "\nALL_CANARY_CLEAN (<=2e-3)")
    for b in bad:
        print(" ", b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
