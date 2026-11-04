#!/usr/bin/env python3
"""p619e t1-seed REP-LOOP re-characterization (cert8 gate annex, 2026-11-04).

The cert8 fp8 long-return gate's ONLY fail: [code] t1_garble=['REP-LOOP']
(seed 1234, temp 1.0, 768 tok). Context the annex must reproduce/classify:
  - corruption lane (t0 cold/rehit/rehit2 + 3 churn cycles) 100% CLEAN,
    byte-family-identical to cert7 (507d389dc258);
  - t1a==t1b within-boot (seeded determinism INTACT) — the gate's
    t1_seed_identical clause PASSED;
  - cert7's same-lane code trajectory was a different, garble-clean hash
    (085255571bb1) — the p618d-established cross-boot RNG-path variance;
  - the runner DISCARDS t1 texts (tooling gap) — this annex banks them.

What it measures (4 rounds, mirroring the p618d annex shape):
  - per round x class: seed-1234 pair identity, full garble flags,
    finish, chars, and for any flagged text the repeated-window detail
    (top window, count, offsets, how many windows cross the >=4 bar) —
    classifies legit code structure vs degenerate attractor;
  - t0 spot check per round (corruption lane stays clean);
  - different-seed controls (999/777 on code+simple) — expect different
    trajectories;
  - all texts banked under OUT/ for the evidence tree.

Against W14 (engine-class control): run with ARC_REPCHAR_MODEL set to the
incumbent's served model id — window-repetition there too = a temp-1.0
lane property, not fork-specific.
"""
import collections
import json
import os
import sys

sys.path.insert(0, "/root/build/p618")
import p618_longprobe as LP  # noqa: E402
from p618_longprobe import PROMPTS, detect_garble, gen  # noqa: E402

if os.environ.get("ARC_REPCHAR_MODEL"):
    LP.MODEL = os.environ["ARC_REPCHAR_MODEL"]

OUT = sys.argv[1] if len(sys.argv) > 1 else "t1seed_repchar"
os.makedirs(OUT, exist_ok=True)
RUNNER_SEED = 1234          # p618_longprobe.run_class default
CTRL = [(999, "code"), (999, "simple"), (777, "code"), (777, "simple")]


def window_detail(comb):
    """Replicate the detector's window rule; report what crossed the bar."""
    wins = collections.Counter()
    where = collections.defaultdict(list)
    for off in range(0, max(0, len(comb) - 25), 5):
        w = comb[off:off + 25]
        wins[w] += 1
        where[w].append(off)
    if not wins:
        return None
    w, c = wins.most_common(1)[0]
    offs = where[w]
    return {
        "top_window": repr(w), "top_count": c, "threshold": 4,
        "n_windows_ge4": sum(1 for k in wins.values() if k >= 4),
        "top_offsets": offs[:12],
        "first_ctx": comb[max(0, offs[0] - 60):offs[0] + 85],
        "last_ctx": comb[max(0, offs[-1] - 60):offs[-1] + 85],
    }


def bank(fn, text):
    with open(f"{OUT}/{fn}", "w") as f:
        f.write(text)


summary = []
for rnd in range(1, 5):
    for cls, content in PROMPTS.items():
        a = gen(content, 1.0, 768, seed=RUNNER_SEED)
        b = gen(content, 1.0, 768, seed=RUNNER_SEED)
        ga = detect_garble(cls, a["combined"])
        gb = detect_garble(cls, b["combined"])
        flags = ga or gb or []
        bank(f"r{rnd}_{cls}_t1.txt", a["combined"])
        if a["sha"] != b["sha"]:
            bank(f"r{rnd}_{cls}_t1_b.txt", b["combined"])
        rec = {"round": rnd, "cls": cls, "seed": RUNNER_SEED,
               "pair_identical": a["sha"] == b["sha"],
               "sha_a": a["sha"], "sha_b": b["sha"], "garble": flags,
               "garble_a": ga, "garble_b": gb,
               "finish": a["finish"], "chars": a["chars"],
               "content_chars": a["content_chars"],
               "window_detail": window_detail(a["combined"]) if flags else None}
        summary.append(rec)
        print(json.dumps(rec), flush=True)
    t0 = gen(PROMPTS["code"], 0.0, 1536)
    g0 = detect_garble("code", t0["combined"])
    rec = {"round": rnd, "cls": "code-t0", "seed": None, "sha_a": t0["sha"],
           "garble": g0, "finish": t0["finish"], "chars": t0["chars"]}
    summary.append(rec)
    print(json.dumps(rec), flush=True)

for s, cls in CTRL:
    r = gen(PROMPTS[cls], 1.0, 768, seed=s)
    g = detect_garble(cls, r["combined"])
    bank(f"ctrl{s}_{cls}_t1.txt", r["combined"])
    rec = {"cls": cls, "seed": s, "sha_a": r["sha"], "garble": g,
           "finish": r["finish"], "chars": r["chars"],
           "window_detail": window_detail(r["combined"]) if g else None}
    summary.append(rec)
    print(json.dumps(rec), flush=True)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=str)
print("T1SEED_REPCHAR_DONE", flush=True)
