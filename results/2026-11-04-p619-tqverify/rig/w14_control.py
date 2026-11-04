#!/usr/bin/env python3
"""W14 incumbent control for the cert8 [code] t1-seed REP-LOOP flag.

Same request shape as the longprobe t1 lane (code prompt, temp 1.0,
seed 1234, 768 tok): does the incumbent engine's seeded temp-1.0 lane
also produce window-repetitive code text? Any hit = the repetition is
a temp-1.0 lane property of the model, not a fork defect. Also banks
the texts (the longprobe runner discards them).
"""
import collections
import json
import os
import sys

sys.path.insert(0, "/root/build/p618")
import p618_longprobe as LP  # noqa: E402
from p618_longprobe import PROMPTS, detect_garble, gen  # noqa: E402

LP.MODEL = os.environ.get("ARC_REPCHAR_MODEL", "qwen3.8-flashnext")


def wins(comb):
    w = collections.Counter()
    for off in range(0, max(0, len(comb) - 25), 5):
        w[comb[off:off + 25]] += 1
    if not w:
        return None, 0
    top, c = w.most_common(1)[0]
    return (repr(top), c), sum(1 for k in w.values() if k >= 4)


out = []
for i in range(4):
    r = gen(PROMPTS["code"], 1.0, 768, seed=1234)
    g = detect_garble("code", r["combined"])
    top, n4 = wins(r["combined"])
    with open(f"w14_ctrl_s1234_{i}.txt", "w") as f:
        f.write(r["combined"])
    rec = {"i": i, "seed": 1234, "sha": r["sha"], "garble": g,
           "finish": r["finish"], "chars": r["chars"],
           "top_window": top, "n_windows_ge4": n4}
    out.append(rec)
    print(json.dumps(rec), flush=True)
for s in (999, 777):
    r = gen(PROMPTS["code"], 1.0, 768, seed=s)
    g = detect_garble("code", r["combined"])
    top, n4 = wins(r["combined"])
    with open(f"w14_ctrl_s{s}.txt", "w") as f:
        f.write(r["combined"])
    rec = {"seed": s, "sha": r["sha"], "garble": g,
           "finish": r["finish"], "chars": r["chars"],
           "top_window": top, "n_windows_ge4": n4}
    out.append(rec)
    print(json.dumps(rec), flush=True)
t0 = gen(PROMPTS["code"], 0.0, 1536)
rec = {"t0": True, "sha": t0["sha"],
       "garble": detect_garble("code", t0["combined"]),
       "chars": t0["chars"]}
out.append(rec)
print(json.dumps(rec), flush=True)
with open("w14_control_summary.json", "w") as f:
    json.dump(out, f, indent=1, default=str)
print("W14_CONTROL_DONE", flush=True)
