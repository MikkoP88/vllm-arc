#!/usr/bin/env python3
"""p32 golden-logits instrument — greedy-output equivalence gate (S1 leg (c)).

Fixed token-id prompts (same generator law as arc_pybench: seeded random ids
1000-50000, seed 20261027), temperature 0, max_tokens 48, via /v1/completions.
Greedy decode of identical logits = identical token sequence = identical text,
so a sha256 over the generated TEXT is the golden fingerprint (any single-token
divergence changes it; no token-id API instability, no float tolerance).

Usage:
  capture: python3 p32_golden.py --out golden_ref.json
  compare: python3 p32_golden.py --run golden_run.json --ref golden_ref.json
Exit 0 = zero drift; exit 3 = drift (gate FAIL).
"""
import argparse
import hashlib
import json
import random
import sys
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen3.8-27b-fp8"
LENS = [64, 256, 1024, 2048, 4096]
SEED = 20261027


def make_prompts():
    rng = random.Random(SEED)
    return {n: [rng.randint(1000, 50000) for _ in range(n)] for n in LENS}


def gen_text(ids):
    body = json.dumps({
        "model": MODEL,
        "prompt": ids,
        "max_tokens": 48,
        "temperature": 0.0,
        "seed": 1234,
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    return out["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--run")
    ap.add_argument("--ref")
    a = ap.parse_args()
    prompts = make_prompts()
    if a.out:
        cap = {}
        for n, ids in prompts.items():
            txt = gen_text(ids)
            cap[str(n)] = {
                "sha": hashlib.sha256(txt.encode("utf-8", "surrogatepass")
                                      ).hexdigest(),
                "chars": len(txt),
            }
            print(f"[golden] len={n} sha={cap[str(n)]['sha'][:16]} "
                  f"chars={len(txt)}")
        json.dump(cap, open(a.out, "w"), indent=1)
        print(f"GOLDEN_CAPTURED -> {a.out}")
        return 0
    if a.run and a.ref:
        run = json.load(open(a.run))
        ref = json.load(open(a.ref))
        drift = 0
        for k in sorted(ref, key=int):
            if run.get(k, {}).get("sha") != ref[k]["sha"]:
                drift += 1
                print(f"[golden] len={k} DRIFT "
                      f"run={run.get(k, {}).get('sha')} ref={ref[k]['sha']}")
            else:
                print(f"[golden] len={k} match")
        if drift:
            print(f"GOLDEN_FAIL: {drift}/{len(ref)} prompts drifted")
            return 3
        print(f"GOLDEN_PASS: {len(ref)}/{len(ref)} identical (zero drift)")
        return 0
    ap.error("need --out OR --run/--ref")


if __name__ == "__main__":
    sys.exit(main())
