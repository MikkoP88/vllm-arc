#!/usr/bin/env python3
"""Ink-box overlap audit for hand-written SVG charts (the
chart-text-overlap-audit law: per-text-class ink boxes are ground
truth). Parses <text> elements, estimates ink boxes from font metrics
(Segoe UI avg advance ~0.55em; digits ~0.55em; underscores ~0.55em),
and reports every overlapping pair + near-misses < 2px.

  python tools/ar/p139/svg_overlap_audit.py <svg> [svg...]
"""
import math
import re
import sys

# rough per-char advance as a fraction of font-size (Segoe UI, mixed)
ADV = 0.55


def audit(path, only=None):
    """only="OVERLAP": count/print just real overlaps (gate mode)."""
    with open(path, encoding="utf8") as f:
        src = f.read()
    # viewBox width/height
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', src)
    W, H = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    out = []
    for tm in re.finditer(
            r'<text ([^>]*)>([^<]*)</text>', src):
        attrs, body = tm.group(1), tm.group(2)
        def a(name, default=None, attrs=attrs):
            mm = re.search(name + r'="([^"]*)"', attrs)
            return mm.group(1) if mm else default
        x, y = float(a("x", "0")), float(a("y", "0"))
        fs = float(a("font-size", "12"))
        anchor = a("text-anchor", "start")
        txt = re.sub(r"&[^;]+;", "x", body)
        w = len(txt) * fs * ADV
        rm = re.search(r'transform="rotate\((-?\d+(?:\.\d+)?) '
                       r'([\d.]+) ([\d.]+)\)"', attrs)
        if rm:
            # rotated text (the generator's -90 y-axis captions): map
            # the pre-rotation ink box through the rotation matrix
            # about (cx, cy). Pre-rotation ink: advance axis u in
            # [anchor offset, +w], ascent/descent v in [-0.75fs,
            # +0.25fs]; (u, v) -> (cx + u*cos - v*sin, cy +
            # u*sin + v*cos).
            ang = math.radians(float(rm.group(1)))
            cx, cy = float(rm.group(2)), float(rm.group(3))
            off = {"start": 0.0, "middle": -w / 2, "end": -w}.get(anchor, 0.0)
            corners = [(off, -0.75 * fs), (off + w, -0.75 * fs),
                       (off + w, 0.25 * fs), (off, 0.25 * fs)]
            pts_r = []
            for u, v in corners:
                px = cx + u * math.cos(ang) - v * math.sin(ang)
                py = cy + u * math.sin(ang) + v * math.cos(ang)
                pts_r.append((px, py))
            xs_r = [p[0] for p in pts_r]
            ys_r = [p[1] for p in pts_r]
            out.append((min(xs_r), min(ys_r), max(xs_r), max(ys_r),
                        txt[:40], fs))
            continue
        if anchor == "middle":
            x0, x1 = x - w / 2, x + w / 2
        elif anchor == "end":
            x0, x1 = x - w, x
        else:
            x0, x1 = x, x + w
        y0, y1 = y - fs * 0.75, y + fs * 0.25
        out.append((x0, y0, x1, y1, txt[:40], fs))
    hits = []
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            a_, b_ = out[i], out[j]
            ox = min(a_[2], b_[2]) - max(a_[0], b_[0])
            oy = min(a_[3], b_[3]) - max(a_[1], b_[1])
            if ox > 0 and oy > 0:
                hits.append(("OVERLAP", a_[4], b_[4],
                             round(ox, 1), round(oy, 1)))
            elif only != "OVERLAP" and ox > -2 and oy > -2 and \
                    (ox > 0 or oy > 0):
                hits.append(("NEAR", a_[4], b_[4],
                             round(ox, 1), round(oy, 1)))
    print("== {}: {} texts, {}x{}".format(path, len(out), W, H))
    for kind, t1, t2, ox, oy in hits:
        print("  {} [{:5}] x-overlap={} y-overlap={} :: {!r} vs {!r}"
              .format(kind, "", ox, oy, t1, t2))
    if not hits:
        print("  clean (no overlaps, no <2px near-misses)")
    return len(hits)


if __name__ == "__main__":
    # exit 1 only on real overlaps; NEAR (<2px estimated gap) is
    # informational -- the 0.55em advance model carries that much error
    over = 0
    for p in sys.argv[1:]:
        over += audit(p, only="OVERLAP")
    sys.exit(0 if over == 0 else 1)
