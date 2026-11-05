#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""r99c_probe_t1.py — targeted re-draw of the code-class t1 leg.

Reproduces the exact p618_longprobe t1 call (PROMPTS['code'], temperature
1.0, MAXTOK_T1=768, seed=1234, thinking off) via the module itself, and —
if the REP-LOOP flag appears — captures the loop body verbatim (top
repeated 25-char windows + head/tail of the text). No new instrument
logic; fidelity is by import, not by re-implementation.
"""
import importlib.util

spec = importlib.util.spec_from_file_location(
    "lp", "/root/build/p618/p618_longprobe.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

content = lp.PROMPTS["code"]
for draw in (1, 2):
    r = lp.gen(content, 1.0, lp.MAXTOK_T1, seed=1234)
    comb = r["combined"]
    print(f"DRAW{draw}: sha={r['sha']} content_sha={r['sha_content']} "
          f"chars={r['chars']} finish={r['finish']} nothink={r['nothink']} "
          f"dt={r['dt']}")
    print(f"DRAW{draw}: garble={lp.detect_garble('code', comb)}")
    wins = {}
    for off in range(0, max(0, len(comb) - 25), 5):
        w = comb[off:off + 25]
        wins[w] = wins.get(w, 0) + 1
    top = sorted(wins.items(), key=lambda kv: -kv[1])[:6]
    print(f"DRAW{draw}: TOP_WINDOWS={[(w, c) for w, c in top]!r}")
    print(f"DRAW{draw}: TEXT_HEAD_600={comb[:600]!r}")
    print(f"DRAW{draw}: TEXT_TAIL_1200={comb[-1200:]!r}")
