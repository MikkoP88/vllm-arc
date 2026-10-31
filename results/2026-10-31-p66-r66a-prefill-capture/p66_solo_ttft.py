#!/usr/bin/env python3
# p66_solo_ttft.py — P6.6 leg-1 probe: SOLO prefill TTFT faces (no
# bystanders, no concurrent streams). One sequential streaming request per
# face; measures TTFT (send -> first content chunk), prompt tokens, and the
# prefix-cache hit (must be ~0 — fresh-prefill face; a nonzero cached= means
# seed reuse or engine prefix hit, face invalid).
# Prompt generator verbatim from p65_timeline_v5.py (comparability with the
# r48/r65 faces). Usage: p66_solo_ttft.py OUT.json [FACES] (default U8k,U32k,U64k)
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"
FACES = {"U8k": (8192, 6101), "U32k": (32768, 6203), "U64k": (65536, 6307)}
FACE_ORDER = (os.environ.get("P66_FACES", "U8k,U32k,U64k")).split(",")


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


def sse_ttft(tag, body):
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


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "p66_solo.json"

    print("=== warmup (first-traffic law: p12b interlock + first graphs)",
          flush=True)
    _, uw, _ = sse_ttft("warmup", {
        "model": MODEL, "prompt": "Count from 1 to 1.",
        "max_tokens": 8, "temperature": 0, "stream": True})
    print(f"warmup done (completion={uw.get('completion_tokens')})", flush=True)
    time.sleep(2)

    rows = []
    for name in FACE_ORDER:
        target, seed = FACES[name]
        prompt, approx = sized_prompt(seed, target)
        time.sleep(1)
        ttft, usage, wall = sse_ttft(name, {
            "model": MODEL, "prompt": prompt, "max_tokens": 32,
            "temperature": 0, "stream": True})
        ptok = usage.get("prompt_tokens", approx)
        cached = ((usage.get("prompt_tokens_details") or {})
                  .get("cached_tokens"))
        tps = ptok / ttft if ttft else float("nan")
        print(f"  {name}: prompt={ptok} cached={cached} TTFT={ttft:.2f}s "
              f"wall={wall:.2f}s -> {tps:.0f} tok/s prefill-rate", flush=True)
        rows.append({"face": name, "prompt_tokens": ptok, "cached": cached,
                     "ttft_s": ttft, "wall_s": wall,
                     "prefill_tok_s": tps})
        time.sleep(2)

    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print("P66_SOLO: DONE", flush=True)


if __name__ == "__main__":
    main()
