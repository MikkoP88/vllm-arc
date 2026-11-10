#!/usr/bin/env python3
"""p136_timeline.py -- assemble the P6.36-A prefill-window timeline from a
p136_prefill.log. Lines: 'E <epoch> <pid> <rank> <TAG> <rest>'.

For each face request (ARR with ntok >= --min-tok, default 1000):
  ARR -> first big WEXE (chunk 1) -> ... chunks ... -> first small WEXE
  (first decode step) with per-chunk rank0/rank1 start/end walls,
  cross-rank EXEC-start skew, inter-chunk host gaps, SCHE/ESTEP cadence.
Prints the window decomposition: how much of the prefill window is device
EXEC wall vs host turnaround vs skew.
Usage: p136_timeline.py <log> [min_tok]
"""
import sys
from collections import defaultdict

PREFILL_TOT_MIN = 64  # decode steps carry <= ~8*(k+1) tokens


def main():
    log = sys.argv[1]
    min_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    arrs = []           # (epoch, pid, req, ntok)
    scheds = []         # (epoch, pid, ms, run, wait, tot)
    wexe = defaultdict(list)  # pid -> list of dicts
    open_by_pid = {}
    esteps = defaultdict(int)

    for ln in open(log, encoding="utf-8", errors="ignore"):
        p = ln.split()
        if len(p) < 5 or p[0] != "E":
            continue
        epoch, pid, rank, tag = float(p[1]), int(p[2]), p[3], p[4]
        rest = p[5:]
        if tag == "ARR":
            req = rest[0].split("=")[-1]
            ntok = int(rest[1].split("=")[-1])
            arrs.append((epoch, pid, req, ntok))
        elif tag == "SCHE":
            d = dict(x.split("=", 1) for x in rest if "=" in x)
            scheds.append((epoch, pid, float(d.get("ms", "nan")),
                           int(d.get("tot", "-1"))))
        elif tag == "WEXE":
            d = dict(x.split("=", 1) for x in rest if "=" in x)
            if d.get("b") is not None or (rest and rest[0] == "b"):
                open_by_pid[pid] = {"b": epoch, "rank": rank,
                                    "tot": int(d.get("tot", "-1"))}
            else:
                o = open_by_pid.pop(pid, None)
                if o is not None:
                    o["e"] = epoch
                    o["wall"] = (epoch - o["b"]) * 1e3
                    o["pid"] = pid
                    wexe[rank].append(o)
        elif tag == "ESTEP":
            esteps[pid] += 1

    print("ARR requests >= %d tok: %d" % (min_tok, len([a for a in arrs if a[3] >= min_tok])))
    for epoch, pid, req, ntok in arrs:
        if ntok >= min_tok:
            print("  ARR epoch=%.6f pid=%d req=%s ntok=%d" % (epoch, pid, req, ntok))

    # per-rank WEXE sessions in arrival order
    all_sessions = []
    for rank, lst in wexe.items():
        lst.sort(key=lambda x: x["b"])
        for s in lst:
            s["rankn"] = rank
            all_sessions.append(s)
    all_sessions.sort(key=lambda x: x["b"])

    print("\nWEXE sessions: %d (rank0=%d rank1=%d other=%d)" % (
        len(all_sessions),
        len([s for s in all_sessions if s["rankn"] == "0"]),
        len([s for s in all_sessions if s["rankn"] == "1"]),
        len([s for s in all_sessions if s["rankn"] not in ("0", "1")])))
    print("  ESTEP counts by pid: %s" % dict(esteps))
    print("  SCHE ticks: %d" % len(scheds))

    # big (prefill) sessions only, paired across ranks by tot+b proximity
    big = [s for s in all_sessions if s["tot"] >= PREFILL_TOT_MIN]
    print("\n== prefill-class WEXE (tot >= %d): %d ==" % (PREFILL_TOT_MIN, len(big)))
    # pair rank0/rank1 by sequence index within rank
    by_rank = {"0": [], "1": []}
    for s in big:
        if s["rankn"] in by_rank:
            by_rank[s["rankn"]].append(s)
    n = min(len(by_rank["0"]), len(by_rank["1"]))
    prev_end = None
    tot_exec = tot_gap = tot_skew = 0.0
    print("  #    tot   r0_wall  r1_wall   skew_ms   gap_from_prev  r0_start(epoch)")
    for i in range(n):
        a, b = by_rank["0"][i], by_rank["1"][i]
        skew = (b["b"] - a["b"]) * 1e3
        gap = (a["b"] - prev_end) * 1e3 if prev_end is not None else float("nan")
        print("  %-3d %6d %8.1f %8.1f %9.1f %14s  %.6f" % (
            i, a["tot"], a["wall"], b["wall"], skew,
            ("%.1f" % gap) if gap == gap else "-", a["b"]))
        tot_exec += max(a["wall"], b["wall"])
        tot_skew += abs(skew)
        if gap == gap:
            tot_gap += gap
        prev_end = max(a["e"], b["e"])
    if n:
        win = (prev_end - by_rank["0"][0]["b"]) * 1e3
        print("\n  window: %.1f ms | exec(sum max-rank): %.1f | inter-chunk gaps: %.1f | abs skew sum: %.1f" % (
            win, tot_exec, tot_gap, tot_skew))
        first = by_rank["0"][0]["b"]
        # small sessions right after the prefill sequence (first decode)
        after = [s for s in all_sessions
                 if s["tot"] < PREFILL_TOT_MIN and s["b"] > prev_end - 0.05][:3]
        for s in after:
            print("  first decode-class WEXE after window: start=+%d ms tot=%d wall=%.1f" % (
                (s["b"] - first) * 1e3, s["tot"], s["wall"]))


if __name__ == "__main__":
    main()
