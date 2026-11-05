#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""r98_reconstruct.py — corrected r98 verdict from the banked artifacts.

The first r98 execution carried two analysis-layer defects (see the RUN
NOTE in r98_quality.sh): the rehit longprobe labels lacked the per-cycle
suffix (three rehit jsons collapsed into one) and garble_check used the
never-matching BRE escapes. The MEASURED data was captured completely —
the per-cycle console outs print the per-class hashes, garble dicts and
identical flags verbatim — so the pre-registered bars are recomputed here
from the outs + surviving jsons instead of re-burning a GPU window.

Usage (rig, cwd=/root/build/p624/r98):  python3 r98_reconstruct.py
Writes: leg{A,B}/longreturn_summary.txt (corrected), verdict.txt (final).
"""
import json
import re
import sys

CLASSES = ("essay", "code", "multi", "simple")


def rd(p):
    with open(p, errors="replace") as f:
        return f.read()


def full_leg(legdir, tag):
    """full json + 3 rehit console outs -> per-class within-leg bar."""
    with open(f"{legdir}/longprobe_{tag}_full.json") as f:
        full = json.load(f)
    cycles = []
    for c in (1, 2, 3):
        txt = rd(f"{legdir}/longprobe_rehit{c}.out")
        cyc = {}
        for cls in CLASSES:
            m = re.search(rf"^\[{cls}\] post_churn_rehit_identical=(\w+) "
                          rf"hashes=\[('?[0-9a-f]+'?(?:, )?){{2}}\] "
                          rf"garble=(\{{[^}}]*\}})", txt, re.M)
            if not m:
                cyc[cls] = None
                continue
            hashes = re.findall(r"'([0-9a-f]+)'", m.group(0))
            garble_txt = m.group(3)
            cyc[cls] = {"identical": m.group(1) == "True",
                        "hashes": hashes,
                        "garble_empty": re.search(r"\[[^\]]", garble_txt) is None}
        verdict = re.search(r"LONGPROBE \S+ rehit: (\w+)", txt)
        cyc["_verdict"] = verdict.group(1) if verdict else "?"
        cycles.append(cyc)
    fail, t1_miss, lines = [], [], []
    for cls in CLASSES:
        v = full.get(cls, {})
        hashes = set(v.get("hashes", []))
        garble_all = [g for g in list(v.get("garble", {}).values())
                      + [v.get("t1_garble", [])] if g]
        cyc_bad = []
        for i, cyc in enumerate(cycles, 1):
            c = cyc.get(cls)
            if c is None:
                cyc_bad.append(f"cycle{i}-unparsed")
                continue
            hashes |= set(c["hashes"])
            if not c["identical"]:
                cyc_bad.append(f"cycle{i}-rehit-pair-drift")
            if not c["garble_empty"]:
                cyc_bad.append(f"cycle{i}-garble")
        if len(hashes) != 1:
            fail.append(f"{cls}: cross-cycle hash drift {sorted(hashes)}")
        if garble_all:
            fail.append(f"{cls}: garble flags {garble_all}")
        if any(fin != "stop" for fin in v.get("finish", [])):
            fail.append(f"{cls}: truncation finish={v.get('finish')}")
        if not v.get("t1_seed_identical"):
            t1_miss.append(cls)
        note = (" cyc:" + ",".join(cyc_bad)) if cyc_bad else ""
        lines.append(f"[{cls}] hashes={sorted(hashes)} finish={v.get('finish')} "
                     f"chars={v.get('chars')} t1_seed_identical="
                     f"{v.get('t1_seed_identical')} garble_empty={not garble_all}{note}")
        fail += [f"{cls}: {b}" for b in cyc_bad]
    hdr = [f"cycle verdicts: {[c.get('_verdict') for c in cycles]}",
           "T1_SEED_MISS_CLASSES: " + str(t1_miss)]
    bar = f"LONGRETURN_BAR_{tag.upper()}: " + ("PASS" if not fail else f"FAIL {fail}")
    txt = "\n".join(lines + hdr + [bar]) + "\n"
    with open(f"{legdir}/longreturn_summary.txt", "w") as f:
        f.write(txt)
    print(txt, end="")
    return not fail


def tps(p):
    return {k: float(v) for k, v in
            re.findall(r"(solo[12]|agg8): tps=([0-9.]+)", rd(p))}


def run7n(p):
    aggs = [float(x) for x in re.findall(r"aggregate=([0-9.]+)", rd(p))]
    solos = [float(x) for x in re.findall(r"solo200 .*tps=([0-9.]+)", rd(p))]
    ok = all("ok_streams=4/4" in l for l in rd(p).splitlines()
             if "aggregate=" in l) and len(aggs) == 2
    return (sum(aggs) / len(aggs) if aggs else 0.0,
            sum(solos) / len(solos) if solos else 0.0, ok)


def spec(p):
    m = re.search(r"SPEC_GATE PASS: pos1=([0-9.]+)", rd(p))
    return float(m.group(1)) if m else None


def main():
    fails = []
    la = full_leg("legA", "r98a")
    lb = full_leg("legB", "r98b")
    ta, tb = tps("legA/tps.out"), tps("legB/tps.out")
    for k in ("solo1", "solo2", "agg8"):
        if k in ta and k in tb:
            r = tb[k] / ta[k] if ta[k] else 0
            line = f"tps {k}: A={ta[k]:.2f} B={tb[k]:.2f} ratio={r:.3f}"
            if not 0.90 <= r <= 1.10:
                fails.append(line + " OUT of [0.90,1.10]")
            print(line)
    for name, fa, fb in (("t0", "legA/run7n_t0.out", "legB/run7n_t0.out"),
                         ("t1.0-xhigh", "legA/run7n_t1.out", "legB/run7n_t1.out")):
        a_agg, a_solo, a_ok = run7n(fa)
        b_agg, b_solo, b_ok = run7n(fb)
        if not (a_ok and b_ok):
            fails.append(f"run7n {name}: stream health bad (A={a_ok} B={b_ok})")
        if a_agg > 0:
            r = b_agg / a_agg
            line = (f"run7n {name} agg: A={a_agg:.2f} B={b_agg:.2f} "
                    f"ratio={r:.3f} (solo A={a_solo:.2f} B={b_solo:.2f})")
            if not 0.90 <= r <= 1.10:
                fails.append(line + " agg OUT of [0.90,1.10]")
            print(line)
    pa, pb = spec("legA/spec_gate.out"), spec("legB/spec_gate.out")
    print(f"spec pos1: A={pa} B={pb}")
    if pb is None:
        fails.append("spec: leg-B SPEC_GATE no PASS line")
    elif pa is not None and abs(pb - pa) > 0.03:
        fails.append(f"spec: pos1 delta {pb - pa:+.4f} exceeds 0.03")
    print("LONGRETURN within-leg: A:" + ("PASS" if la else "FAIL"),
          "B:" + ("PASS" if lb else "FAIL"))
    if not la:
        fails.append("LONG-RETURN leg-A (baseline) drift — rig issue, not fp8")
    if not lb:
        fails.append("LONG-RETURN leg-B within-leg bar")

    def hashes(legdir, tag):
        p = f"{legdir}/longprobe_{tag}_full.json"
        try:
            with open(p) as f:
                raw = f.read()
            d = json.loads(raw)
        except Exception as e:  # noqa: BLE001 — dump bytes to convict ghosts
            with open(p, "rb") as f:
                raw_b = f.read()
            print(f"HASHES-DEBUG {p}: {type(e).__name__}: {e}; "
                  f"bytes={len(raw_b)} head={raw_b[:60]!r}")
            raise
        return {c: set(d.get(c, {}).get("hashes", [])) for c in CLASSES}
    ha, hb = hashes("legA", "r98a"), hashes("legB", "r98b")
    drift = [c for c in CLASSES if ha[c] and hb[c] and ha[c] != hb[c]]
    print("cross-leg t0 hash drift classes:",
          drift if drift else "none (byte-identical t0)")
    if drift:
        print(f"NOTE: cross-leg drift {drift}: expected e4m3 prefill noise "
              "0.023 rel-rms flipping near-tie argmax tokens; coherence "
              "gated by zero-garble + finish-stop (Class-A precedent)")
    errfail = []
    for leg in ("legA", "legB"):
        t = rd(f"{leg}/erraudit.txt")
        if "ERROR=0" not in t or "TRACEBACK=0" not in t:
            errfail.append(leg)
        print(f"erraudit {leg}: {t.strip().replace(chr(10), ' ')}")
    fails += [f"erraudit {e}" for e in errfail]
    verdict = "PASS" if not fails else f"FAIL {fails}"
    print("R98_QUALITY_LADDER:", verdict)
    with open("verdict.txt", "w") as f:
        f.write("R98_QUALITY_LADDER: " + verdict + "\n")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
