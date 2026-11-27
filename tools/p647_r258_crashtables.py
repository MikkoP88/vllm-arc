#!/usr/bin/env python3
"""p647_r258_crashtables.py -- the crash block-table differential (r258a).

Four banked DEVICE_LOST crashes carry the full SchedulerOutput dump
(dump_input fires on the fatal path): the three r255 cert29-gate gemma
legs (m8g/m4g/mkg) + the r257 faithful replay. The faithful minimal
(same request bytes, pristine state) ran CLEAN -- so the table
differential is the state evidence.

Parses every banked crash dump, normalizes the 6-group block table,
and prints the structural differential: per-group real-id counts,
zero-pad counts, consecutive-run structure, and the tail blocks (the
last windowed entries = the fresh allocations for the scheduled
118-token tail).

Usage: python p647_r258_crashtables.py <bank-root>
  <bank-root> = the results/ dir (banks resolved by fixed relative
  paths). Stdout is the bankable analysis.
"""
import re
import sys
from pathlib import Path

RE_BLOCK = re.compile(r"block_ids=\((\[.*\])\),num_computed_tokens=(\d+)", re.S)
RE_SCHED = re.compile(r"num_scheduled_tokens=\{[^}]*:\s*(\d+)\}")


def parse_dump(text):
    m = RE_BLOCK.search(text)
    if not m:
        return None
    rows = re.findall(r"\[[0-9,\s]*\]", m.group(1))
    table = [[int(x) for x in re.findall(r"\d+", r)] for r in rows]
    comp = int(m.group(2))
    sched = RE_SCHED.search(text)
    return table, comp, int(sched.group(1)) if sched else None


def row_stats(row):
    zeros = row.count(0)
    real = [x for x in row if x != 0]
    # consecutive runs among the real ids (adjacent values, +1)
    runs = 0
    for a, b in zip(real, real[1:]):
        if b == a + 1:
            runs += 1
    return zeros, real, runs


def describe(label, table, comp, sched):
    print(f"== {label}: computed={comp} scheduled={sched} groups={len(table)}")
    for gi, row in enumerate(table):
        zeros, real, runs = row_stats(row)
        first_real = row.index(real[0]) if real else -1
        print(f"  g{gi}: len={len(row)} zeros={zeros} (pad at 0..{first_real-1})"
              f" real={len(real)} cons_runs={runs}")
        print(f"      real_ids={real}")


def main():
    root = Path(sys.argv[1])
    dumps = [
        ("r255 m8g", root / "2026-11-26-p647-cert29-gate/m8g/serve.log"),
        ("r255 m4g", root / "2026-11-26-p647-cert29-gate/m4g/serve.log"),
        ("r255 mkg", root / "2026-11-26-p647-cert29-gate/mkg/serve.log"),
        ("r257 replay", root / "2026-11-27-p647-r257-faithful-replay/r257/replay/devicelost_context.txt"),
    ]
    parsed = []
    for label, path in dumps:
        text = path.read_text(encoding="utf-8", errors="replace")
        d = parse_dump(text)
        if d is None:
            print(f"== {label}: NO DUMP FOUND at {path}")
            continue
        describe(label, *d)
        parsed.append((label, d))
        print()
    if len(parsed) < 2:
        return
    # cross-crash differential: row-by-row id equality
    print("== cross-crash windowed-row (g0) id equality")
    base = parsed[0][1][0][0]
    for label, (table, _, _) in parsed[1:]:
        other = table[0]
        same = sum(1 for a, b in zip(base, other) if a == b)
        print(f"  {label}: {same}/{len(base)} g0 entries equal to {parsed[0][0]}")
    print()
    print("== tail blocks (the last 2 windowed entries + the full-attn row tail)")
    for label, (table, _, sched) in parsed:
        w = table[0]
        print(f"  {label}: windowed last2={w[-2:]} fullattn_last2={table[-1][-2:]} sched={sched}")


if __name__ == "__main__":
    main()
