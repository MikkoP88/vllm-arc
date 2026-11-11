#!/usr/bin/env python3
"""validate_docs.py -- full validation pass for every fork-owned doc.

Checks (per doc):
  links      every relative markdown link resolves (file exists; for
             #anchors the heading/text anchor exists in the target)
  tables     consistent column count per table; no empty header cells
  headings   no level jumps > 1; at most one H1 per file
  fences     code fences balanced (``` opened and closed)
  format     no trailing whitespace; no 3+ consecutive blank lines;
             no tabs
  images     image refs resolve to existing files

Scope: fork-owned docs only (upstream-tracked files are pristine by
law). Exit 0 = all docs clean; exit 1 lists every finding.

Usage: python tools/validate_docs.py [--root <repo>] [--all]
       (--all adds upstream-tracked docs/*.md for information only;
       findings there are reported as UPSTREAM advisories, never errors)
"""
import os
import re
import sys

FORK_DOCS = [
    "README.md",
    "CHANGELOG.md",
    "docs/STATUS-arc.md",
    "docs/PERFORMANCE-arc.md",
    "docs/CONFIGURATION-arc.md",
    "docs/TESTING-arc.md",
    "docs/MIGRATION-arc.md",
    "docs/PATCHES-arc.md",
    "docs/envs.md",
    "docs/release_notes.md",
    "docs/history/STATUS_HISTORY.md",
    "docs/history/PERFORMANCE_HISTORY.md",
    "docs/history/CONFIGURATION_HISTORY.md",
    "docs/history/TESTING_HISTORY.md",
    "docs/history/MIGRATION_HISTORY.md",
    "docs/reports/README.md",
    "docs/reports/qwen3.8-27b-fp8-kv-cache-dtype.md",
]

RE_MD = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
RE_IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
RE_TABLE_SEP = re.compile(r"^\|(\s*:?-+:?\s*\|)+$")

errors = []
advisories = []


def slug_anchor(text):
    """GitHub-style heading anchor."""
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s", "-", s)
    return s


def check_doc(root, rel, advisory=False):
    path = os.path.join(root, rel)
    tag = rel
    bucket = advisories if advisory else errors
    if not os.path.isfile(path):
        bucket.append("{}: MISSING FILE".format(tag))
        return
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    dirn = os.path.dirname(path)

    # ---- fences / tables / headings / format (line scans)
    fence = False
    fence_start = 0
    table_cols = None
    table_first = 0
    headings = []          # (level, line_no, text)
    blank_run = 0
    in_table = False
    for i, ln in enumerate(lines, 1):
        stripped = ln.rstrip("\r")
        # trailing whitespace (outside tables where | alignment padding is
        # not used in this repo) and tabs
        if stripped.rstrip() != stripped:
            bucket.append(f"{tag}:{i}: trailing whitespace")
        if "\t" in stripped:
            bucket.append(f"{tag}:{i}: tab character")
        if stripped == "":
            blank_run += 1
            if blank_run >= 3:
                bucket.append(f"{tag}:{i - 1}: {blank_run} consecutive blank lines")
        else:
            blank_run = 0
        if stripped.startswith("```"):
            if not fence:
                fence = True
                fence_start = i
            else:
                fence = False
            table_cols, in_table = None, False
            continue
        if fence:
            continue
        m = RE_HEADING.match(stripped)
        if m:
            headings.append((len(m.group(1)), i, m.group(2).strip()))
        # tables: a line starting and ending with |
        if stripped.startswith("|") and stripped.endswith("|"):
            cols = stripped.count("|") - 1
            if RE_TABLE_SEP.match(stripped):
                in_table = True
                continue
            if not in_table or table_cols is None:
                table_cols = cols
                table_first = i
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if any(c == "" for c in cells):
                    bucket.append(f"{tag}:{i}: empty header cell in table")
            elif cols != table_cols:
                bucket.append(f"{tag}:{i}: table column count {cols} != "
                              f"{table_cols} (table starts line {table_first})")
        else:
            table_cols, in_table = None, False
    if fence:
        bucket.append(f"{tag}:{fence_start}: unterminated code fence")

    # heading structure: one H1 max, no jumps > 1
    h1 = [h for h in headings if h[0] == 1]
    if len(h1) > 1:
        bucket.append(f"{tag}:{h1[1][1]}: {len(h1)} H1 headings (max 1)")
    prev = 0
    for level, i, text in headings:
        if level - prev > 1 and prev != 0:
            bucket.append(f"{tag}:{i}: heading level jump H{prev} -> "
                          f"H{level} ({text[:40]})")
        prev = level

    # ---- links + images
    anchors_by_file = {}
    for i, ln in enumerate(lines, 1):
        if ln.lstrip().startswith("#"):       # headings not links
            pass
        for m in RE_IMG.finditer(ln):
            target = m.group(1)
            if re.match(r"^[a-z]+://", target):
                continue
            tp = os.path.normpath(os.path.join(dirn, target))
            if not os.path.isfile(tp):
                bucket.append(f"{tag}:{i}: image not found: {target}")
        for m in RE_MD.finditer(ln):
            target = m.group(1)
            if re.match(r"^[a-z]+://", target) or target.startswith("mailto:"):
                continue
            file_part, _, anch = target.partition("#")
            if file_part == "":
                tp = path
            else:
                tp = os.path.normpath(os.path.join(dirn, file_part))
                if not (os.path.isfile(tp) or os.path.isdir(tp)):
                    bucket.append(f"{tag}:{i}: link target not found: {target}")
                    continue
                if os.path.isdir(tp):
                    continue          # directory links carry no anchor
            if anch:
                if tp not in anchors_by_file:
                    if os.path.splitext(tp)[1] != ".md":
                        anchors_by_file[tp] = None
                    else:
                        with open(tp, encoding="utf-8") as tf:
                            targets = set()
                            for tl in tf:
                                hm = RE_HEADING.match(tl.strip())
                                if hm:
                                    targets.add(slug_anchor(hm.group(2)))
                            anchors_by_file[tp] = targets
                tset = anchors_by_file[tp]
                if tset is not None and anch not in tset:
                    bucket.append(f"{tag}:{i}: anchor #{anch} not found in "
                                  f"{os.path.relpath(tp, root)}")


def discover_reports(root):
    """Every .md in docs/reports is fork-owned; enumerate dynamically."""
    rd = os.path.join(root, "docs", "reports")
    out = []
    if os.path.isdir(rd):
        for dirpath, _dirs, files in os.walk(rd):
            for fn in files:
                if fn.endswith(".md"):
                    out.append(os.path.relpath(os.path.join(dirpath, fn), root)
                               .replace("\\", "/"))
    return sorted(out)


def main():
    root = "."
    args = sys.argv[1:]
    include_upstream = "--all" in args
    if "--root" in args:
        root = args[args.index("--root") + 1]
    docs = list(FORK_DOCS)
    reports = discover_reports(root)
    for r in reports:
        if r not in docs:
            docs.append(r)
    for rel in docs:
        check_doc(root, rel)
    if include_upstream:
        base = os.path.join(root, "docs")
        for fn in sorted(os.listdir(base)):
            if fn.endswith(".md") and os.path.isfile(os.path.join(base, fn)):
                rel = "docs/" + fn
                if rel not in docs:
                    check_doc(root, rel, advisory=True)
        rel = "AGENTS.md"
        if os.path.isfile(os.path.join(root, rel)):
            check_doc(root, rel, advisory=True)
    for a in advisories:
        print("ADVISO {}".format(a))
    for e in errors:
        print("ERROR  {}".format(e))
    print(f"== validate_docs: {len(errors)} error(s), {len(advisories)} "
          f"advisory(ies) across {len(docs)} fork docs")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
