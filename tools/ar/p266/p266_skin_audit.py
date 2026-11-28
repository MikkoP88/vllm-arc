#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p266_skin_audit.py — P6.47 (r266/r267 close-outs): the gemma SKIN
assertion over a banked leg tree.

Owner directive (2026-11-28, the r264 close-out bank): gemma is NEVER
validly tested under the qwen skin -- every gemma boot must carry a
gemma template, empty-or-gemma parsers, and the close-out must ASSERT
it on the banked evidence (never trust the runner's env: r264's gate
banked 13 legs that all ran tmpl=__qwen_default__ rp=<qwen3> while
believing they tested gemma).  This tool is that assertion: it reads
each leg's ARC_SERVE_LAUNCHED posture line (engine_config.txt for
r266-family banks, boot.txt for r264-family gate banks) plus
leg_env.txt and enforces the lawful posture per leg class:

  every gemma leg   model=/models/gemma4-31b, served=gemma4-31b,
                     tmpl basename in {chat_template_gemma4.jinja
                     (the P6.28 g3 lawful skin),
                     chat_template_gemma4_official.jinja}, NEVER a
                     qwen template; rp/tp EMPTY unless the leg name
                     contains _g4 (the official-structure leg
                     b3_g4_29), which must carry rp=gemma4 tp=gemma4
                     and the official template.
  resolved-removed  the r272 directive (owner 2026-11-28): where no
   (r272 class)      matching gemma skin can be found it is FULLY
                     REMOVED, never substituted with the qwen default.
                     The r272 serve script (tools/serve_arc_b70.sh
                     family resolution) renders this as an
                     ARC_SERVE_SKIN line: tmpl=<native> rp=<none>
                     tp=<none> family=gemma.  The SKIN line is the
                     RESOLVED posture and OVERRIDES the raw
                     ARC_SERVE_LAUNCHED render (whose rp/tp show the
                     '<qwen3>' unset literals even when the r272
                     resolution omitted the flags); a bare empty tmpl
                     with no SKIN line stays UNPROVEN = FAIL (the
                     existence-only-green-gate law).
  hard FAIL         any tmpl containing qwen, any rp/tp containing
                     qwen, a qwen model/served, or a missing
                     posture line (unproven = failed -- the
                     existence-only-green-gate law).

Usage: python p266_skin_audit.py <bank-dir>... [--out report.md]
Exit 0 = every leg skin-lawful; nonzero = any FAIL (lines name them).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LAWFUL_TMPL = ("chat_template_gemma4.jinja", "chat_template_gemma4_official.jinja")
OFFICIAL_TMPL = "chat_template_gemma4_official.jinja"
REMOVED_TMPL = "<native>"    # the r272 SKIN-line render of the fully-removed template
REMOVED_PARSER = "<none>"    # the r272 SKIN-line render of the fully-removed parsers
GEMMA_MODEL = "/models/gemma4-31b"
GEMMA_SERVED = "gemma4-31b"
_SERVE_LINE = re.compile(r"^ARC_SERVE_LAUNCHED\b")
_SKIN_LINE = re.compile(r"^ARC_SERVE_SKIN\b")
_KV = re.compile(r"(\w+)=(\S*)")


def parse_serve_line(leg: Path) -> dict[str, str]:
    """The posture line lives in engine_config.txt (r266-family banks)
    or boot.txt (r264-family gate banks) -- first file with one wins.
    ARC_SERVE_SKIN (the r272 serve script) is the RESOLVED posture and
    OVERRIDES the raw ARC_SERVE_LAUNCHED render key-by-key: LAUNCHED
    prints the raw env, so an unset ARC_RPARSER shows the '<qwen3>'
    literal there even after the r272 resolution omitted the flag."""
    launched: dict[str, str] = {}
    skin: dict[str, str] = {}
    for name in ("engine_config.txt", "boot.txt"):
        p = leg / name
        if not p.is_file():
            continue
        for line in p.read_text(errors="replace").splitlines():
            if _SKIN_LINE.search(line):
                skin.update(dict(_KV.findall(line)))
            elif _SERVE_LINE.search(line):
                launched.update(dict(_KV.findall(line)))
        if launched or skin:
            break
    cfg = dict(launched)
    cfg.update(skin)
    if skin:
        cfg["skin_resolved"] = "1"
    return cfg


def parse_leg_env(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    env = dict(_KV.findall(text))
    # leg_env.txt is one bracketed [K=V ...] line; also accept KEY: VAL rows
    for m in re.finditer(r"^(\w+):\s*(\S*)$", text, re.M):
        env[m.group(1)] = m.group(2)
    return env


def audit_leg(leg: Path) -> tuple[bool, list[str], dict[str, str]]:
    fails: list[str] = []
    cfg = parse_serve_line(leg)
    if not cfg:
        fails.append("no ARC_SERVE_LAUNCHED posture line (skin UNPROVEN)")
        return False, fails, cfg
    model = cfg.get("model", "")
    served = cfg.get("served", "")
    tmpl = cfg.get("tmpl", "")
    rp = cfg.get("rp", "")
    tp = cfg.get("tp", "")
    official = "_g4" in leg.name
    if model != GEMMA_MODEL:
        fails.append(f"model={model!r} != {GEMMA_MODEL!r}")
    if served != GEMMA_SERVED:
        fails.append(f"served={served!r} != {GEMMA_SERVED!r}")
    if "qwen" in tmpl.lower():
        fails.append(f"tmpl={tmpl!r} (qwen template on gemma -- the violence posture)")
    elif official and tmpl != OFFICIAL_TMPL:
        fails.append(f"official-structure leg has tmpl {tmpl!r} != {OFFICIAL_TMPL!r}")
    elif not tmpl:
        if not cfg.get("skin_resolved"):
            fails.append("tmpl empty with no ARC_SERVE_SKIN line (removed posture UNPROVEN)")
    elif tmpl != REMOVED_TMPL:
        base = Path(tmpl).name
        if base not in LAWFUL_TMPL:
            fails.append(f"tmpl basename {base!r} not a lawful gemma template")
    # tmpl == REMOVED_TMPL on a non-official leg = the r272
    # fully-removed class (proven by the SKIN line) = LAWFUL
    for tag, val in (("rp", rp), ("tp", tp)):
        if "qwen" in val.lower():
            fails.append(f"{tag}={val!r} (qwen parser on gemma)")
        elif official and val != "gemma4":
            fails.append(f"official-structure leg has {tag}={val!r} != gemma4")
        elif not official and val and val != REMOVED_PARSER:
            fails.append(f"{tag}={val!r} non-empty on a no-parser leg")
    env = parse_leg_env(leg / "leg_env.txt") if (leg / "leg_env.txt").is_file() else {}
    for key in ("ARC_TEMPLATE", "ARC_RPARSER", "ARC_TPARSER"):
        v = env.get(key, "")
        if "qwen" in v.lower():
            fails.append(f"{key}={v!r} (qwen skin knob in the leg env)")
    return not fails, fails, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs="+", help="bank dir(s): leg dirs or parents to recurse")
    ap.add_argument("--out", help="write the report here (markdown, bankable)")
    args = ap.parse_args()

    legs: list[Path] = []
    seen: set[Path] = set()
    marks = ("engine_config.txt", "boot.txt")
    for d in args.dirs:
        p = Path(d)
        found = [p] if any((p / m).is_file() for m in marks) else [
            q.parent for m in marks for q in sorted(p.rglob(m))
        ]
        for leg in found:
            if leg not in seen:
                seen.add(leg)
                legs.append(leg)
    if not legs:
        print("p266_skin_audit: NO legs found (nothing to assert)")
        return 1

    lines: list[str] = ["# r266/r267 gemma SKIN audit", ""]
    n_ok = 0
    for leg in legs:
        ok, fails, cfg = audit_leg(leg)
        tag = "SKIN_PASS" if ok else "SKIN_FAIL"
        if ok:
            n_ok += 1
        posture = f"tmpl={cfg.get('tmpl', '?')} rp={cfg.get('rp', '')} tp={cfg.get('tp', '')}"
        print(f"{tag} {leg.name} {posture}")
        lines.append(f"- **{tag}** `{leg.name}`: {posture}")
        for f in fails:
            print(f"    -> {f}")
            lines.append(f"  - FAIL: {f}")
    summary = f"skin: {n_ok}/{len(legs)} legs lawful"
    print(f"p266_skin_audit: {summary}")
    lines += ["", f"**{summary}**", ""]
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="\n") as fh:  # p139 newline law
            fh.write("\n".join(lines))
        print(f"wrote {out}")
    return 0 if n_ok == len(legs) else 1


if __name__ == "__main__":
    sys.exit(main())
