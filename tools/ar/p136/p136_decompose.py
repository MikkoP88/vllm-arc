#!/usr/bin/env python3
"""p136_decompose.py -- per-event decomposition of a p136_prefill.log.

Lines: 'E <epoch> <pid> <rank> <TAG> <rest>'. Workers carry no RANK env in
this stack, so the two worker pids are the rank proxies; pid with ARR/SCHE/
ESTEP lines is the engine-core (EC) process. SchedulerOutput exposes no
total_scheduled_tokens at these hooks (logs -1), so prefill-class steps are
classified by WEXE wall (>= 100 ms) and adjacency to the big ARR.

Prints, per big request: the ordered cross-process event table from ARR to
first decode-class WEXE, then the window split -- device-exec walls (worker
WEXE), EC post-exec waits (WEXE e -> next SCHE), schedule cost, and the
pre-ARR to first-WEXE latency.
Usage: p136_decompose.py <log> [min_tok]
"""
import sys
from collections import defaultdict

PREFILL_WALL_MIN_MS = 100.0


def parse(log):
    events = []  # (epoch, pid, tag, rest_str)
    for ln in open(log, encoding="utf-8", errors="ignore"):
        p = ln.split()
        if len(p) < 5 or p[0] != "E":
            continue
        events.append((float(p[1]), int(p[2]), p[4], " ".join(p[5:])))
    events.sort(key=lambda x: x[0])
    return events


def kv(rest):
    return dict(x.split("=", 1) for x in rest.split() if "=" in x)


def main():
    log = sys.argv[1]
    min_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    ev = parse(log)

    arrs = [e for e in ev if e[2] == "ARR" and int(kv(e[3]).get("ntok", "0")) >= min_tok]
    if not arrs:
        print("no big ARR")
        return
    arr = arrs[0]
    t0 = arr[0]
    print("big ARR: epoch=%.6f ntok=%s" % (t0, kv(arr[3])["ntok"]))

    # worker pids: those with WEXE lines
    wpids = sorted({e[1] for e in ev if e[2] == "WEXE"})
    ecpid = arr[1]
    print("engine-core pid=%d ; worker pids=%s" % (ecpid, wpids))

    # WEXE sessions per worker pid
    open_s = {}
    sess = defaultdict(list)
    for epoch, pid, tag, rest in ev:
        if tag != "WEXE":
            continue
        if rest.startswith("b "):
            open_s[pid] = (epoch, kv(rest).get("tot", "-1"))
        elif rest.startswith("e ") and pid in open_s:
            b, tot = open_s.pop(pid)
            sess[pid].append({"b": b, "e": epoch, "wall": (epoch - b) * 1e3, "tot": tot})

    # window: ARR .. end of first decode-class session after last prefill-class
    allw = sorted((s for lst in sess.values() for s in lst), key=lambda x: x["b"])
    big = [s for s in allw if s["wall"] >= PREFILL_WALL_MIN_MS and s["b"] < t0 + 60]
    if not big:
        print("no prefill-class WEXE")
        return
    win_end0 = max(s["e"] for s in big)
    dec = [s for s in allw if s["wall"] < PREFILL_WALL_MIN_MS and s["b"] >= win_end0 - 0.05]
    win_end = dec[0]["e"] if dec else win_end0

    print("\n== events in window (epoch_rel_ms  pid  tag  rest) ==")
    for epoch, pid, tag, rest in ev:
        if t0 - 0.05 <= epoch <= win_end + 0.05:
            who = "EC" if pid == ecpid else ("W%d" % (wpids.index(pid) + 1) if pid in wpids else str(pid))
            print("  %+9.1f  %-3s %-5s %s" % ((epoch - t0) * 1e3, who, tag, rest))

    print("\n== prefill-class WEXE (wall >= %.0f ms): %d ==" % (PREFILL_WALL_MIN_MS, len(big)))
    for i, s in enumerate(big):
        print("  chunk%-2d b=%+8.1f e=%+8.1f wall=%7.1f ms" % (
            i + 1, (s["b"] - t0) * 1e3, (s["e"] - t0) * 1e3, s["wall"]))

    # EC waits: WEXE-e(N) -> SCHE(N+1) gaps, and schedule walls
    sches = [(e[0], float(kv(e[3]).get("ms", "nan"))) for e in ev
             if e[2] == "SCHE" and t0 <= e[0] <= win_end]
    exec_sum = max(s["wall"] for s in big) if big else 0  # per-pair exec ~= same
    # sum of max wall per chunk pair (pair by order)
    per_pid = [sorted(sess[p], key=lambda x: x["b"]) for p in wpids if sess[p]]
    pair_exec = 0.0
    if len(per_pid) == 2:
        for s0, s1 in zip(per_pid[0], per_pid[1]):
            pair_exec += max(s0["wall"], s1["wall"])
    ec_wait = 0.0
    prev_we = None
    for s in big:
        if prev_we is not None:
            gap = (s["b"] - prev_we) * 1e3
            if gap > 0:
                ec_wait += gap
        prev_we = max(x["e"] for x in big if x["b"] < s["b"] + 1e-6 and x["e"] <= s["e"] + 5) or s["e"]
    pre_exec0 = (big[0]["b"] - t0) * 1e3
    tail = (win_end - big[-1]["e"]) * 1e3
    print("\n== window split (ARR -> first decode end, %.1f ms) ==" % ((win_end - t0) * 1e3))
    print("  ARR -> first dispatch : %8.1f ms" % pre_exec0)
    print("  exec walls (sum max)  : %8.1f ms" % pair_exec)
    print("  EC inter-chunk waits   : %8.1f ms" % ec_wait)
    print("  last-exec -> 1st decode end: %6.1f ms" % tail)
    if sches:
        print("  schedule walls: n=%d  max=%.1f ms  sum=%.1f ms" % (
            len(sches), max(x[1] for x in sches), sum(x[1] for x in sches)))


if __name__ == "__main__":
    main()
