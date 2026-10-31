#!/usr/bin/env python3
# p70_warm_rehit.py — P6.10 r70: cert4-native warm re-hit faces, with
# CACHE-TIER ATTRIBUTION (owner question 2026-10-31: "is 'Warm prefix
# re-hit (8k) 740 ms' LMCache hits or some else hits?").
#
# Two prefix-reuse tiers exist on the certified line and this probe
# separates them per face:
#   tier 1 — vLLM engine-native APC (--enable-prefix-caching, block 64,
#            GPU-resident block table): surfaces as usage.
#            prompt_tokens_details.cached_tokens on the warm request.
#   tier 2 — LMCacheConnectorV1 (kv_both, CPU 100 GiB tier, 1024-tok
#            chunks): serves prefixes no longer GPU-resident; visible in
#            the engine log (LMCache lookup/store lines) and connector
#            metrics — banked by the driver around this probe's window.
#
# Same prompt generator/seeds as p70_solo_ttft.py (face parity). Per face
# (U8k, U16k, U32k): t1 = cold first request (cached must be ~0 — face
# validity), t2 = IDENTICAL re-request seconds later (warm re-hit; cached
# ~ full prompt = tier-1 APC serving; the log window banked by the driver
# shows whether LMCache was consulted at all). Then the warm-hit face
# proper: 3 re-fires of U8k (the history "740 ms" face) to get a stable
# warm-hit number.
# Usage: p70_warm_rehit.py OUT.json
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"
FACES = {"U8k": (8192, 6101), "U16k": (16384, 6153), "U32k": (32768, 6203)}


def long_prompt(seed, n_words):
    rng = [seed]
    def nxt():
        rng[0] = (rng[0] * 1103515245 + 12345) & 0x7fffffff
        return rng[0]
    parts = []
    for i in range(n_words):
        parts.append(f"Note {i}: the measured latency of stage "
                     f"{nxt() % 9973} under load {nxt() % 977} ms "
                     f"was recorded by sensor {nxt() % 8191}.")
    return " ".join(parts)


def tokenize_count(prompt):
    body = {"model": MODEL, "prompt": prompt}
    req = urllib.request.Request(
        BASE + "/tokenize", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=120))
        return len(d.get("tokens", []))
    except Exception as e:  # noqa: BLE001
        print(f"tokenize_FAIL {type(e).__name__}: {e}", flush=True)
        return -1


def sized_prompt(seed, target_tokens):
    n = max(50, int(target_tokens / 27))
    c = tokenize_count(long_prompt(seed, n))
    if c > 0 and abs(c - target_tokens) > 0.15 * target_tokens:
        n = max(50, int(n * target_tokens / c))
        c = tokenize_count(long_prompt(seed, n))
    return long_prompt(seed, n), (c if c > 0 else n * 27)


def sse_ttft(body):
    """One streaming request; returns (ttft_s, usage_dict, wall_s)."""
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    req = urllib.request.Request(
        BASE + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            d = line[6:]
            if d == "[DONE]":
                break
            try:
                j = json.loads(d)
            except ValueError:
                continue
            if ttft is None and j.get("choices") and j["choices"][0].get("text"):
                ttft = time.perf_counter() - t0
            if "usage" in j:
                usage = j["usage"]
    return ttft, usage, time.perf_counter() - t0


def fire(tag, prompt):
    ttft, usage, wall = sse_ttft({
        "model": MODEL, "prompt": prompt, "max_tokens": 32,
        "temperature": 0, "stream": True})
    ptok = usage.get("prompt_tokens", -1)
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    print(f"  {tag}: prompt={ptok} cached={cached} "
          f"TTFT={ttft:.3f}s wall={wall:.2f}s", flush=True)
    return {"tag": tag, "prompt_tokens": ptok, "cached": cached,
            "ttft_s": ttft, "wall_s": wall}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "p70_warm_rehit.json"

    print("=== warmup (first-traffic law: p12b interlock + first graphs)",
          flush=True)
    _, uw, _ = sse_ttft({"model": MODEL, "prompt": "Count from 1 to 1.",
                         "max_tokens": 8, "temperature": 0, "stream": True})
    print(f"warmup done (completion={uw.get('completion_tokens')})", flush=True)
    time.sleep(2)

    rows = []
    for name, (target, seed) in FACES.items():
        prompt, approx = sized_prompt(seed, target)
        time.sleep(1)
        r1 = fire(f"{name}.t1_cold", prompt)
        time.sleep(3)
        r2 = fire(f"{name}.t2_warm", prompt)
        r1["face"], r2["face"] = name, name
        rows += [r1, r2]
        time.sleep(2)

    print("=== warm-hit face (U8k x3 re-fires, the history 740 ms face)",
          flush=True)
    prompt, approx = sized_prompt(*FACES["U8k"])
    time.sleep(1)
    fire("U8k.warmhit_prime", prompt)   # ensure residency, not scored
    for i in (1, 2, 3):
        time.sleep(2)
        r = fire(f"U8k.warmhit_{i}", prompt)
        r["face"] = "U8k_warmhit"
        rows.append(r)

    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("P70_WARM: DONE", flush=True)


if __name__ == "__main__":
    main()
