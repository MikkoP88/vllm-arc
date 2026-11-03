#!/usr/bin/env python3
"""t1-seed characterization on the live cert7 fp8 boot (post-gate FAIL).

The long-return gate's only miss was [simple] t1_seed_identical=False.
Characterize: 4 rounds x 4 classes of temp1-seed pairs (seed fixed per
class), plus per-round t0 spot checks (corruption faces), plus a
different-seed control (expect a different hash).
"""
import sys

sys.path.insert(0, "/root/build/p618")
from p618_longprobe import PROMPTS, gen, detect_garble  # noqa: E402

SEEDS = {"essay": 1234, "code": 2345, "multi": 3456, "simple": 1234}
for rnd in range(1, 5):
    for cls, content in PROMPTS.items():
        s = SEEDS[cls]
        a = gen(content, 1.0, 768, seed=s)
        b = gen(content, 1.0, 768, seed=s)
        same = a["sha"] == b["sha"]
        ga = detect_garble(cls, a["combined"])
        gb = detect_garble(cls, b["combined"])
        print(f"R{rnd} [{cls}] t1_seed_identical={same} a={a['sha']} "
              f"b={b['sha']} garble={ga or gb or []} "
              f"chars={a['chars']}/{b['chars']}", flush=True)
    t0a = gen(PROMPTS["simple"], 0.0, 1536)
    t0b = gen(PROMPTS["simple"], 0.0, 1536)
    g0 = detect_garble("simple", t0a["combined"])
    print(f"R{rnd} [simple-t0] identical={t0a['sha'] == t0b['sha']} "
          f"hash={t0a['sha']} garble={g0 or []}", flush=True)
ctrl = gen(PROMPTS["simple"], 1.0, 768, seed=999)
base = gen(PROMPTS["simple"], 1.0, 768, seed=1234)
print(f"CTRL [simple] seed999={ctrl['sha']} vs seed1234={base['sha']} "
      f"different={ctrl['sha'] != base['sha']}", flush=True)
print("T1SEED_CHAR_DONE", flush=True)
