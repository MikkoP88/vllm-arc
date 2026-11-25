#!/usr/bin/env python3
"""p139_conc.py -- P6.37 concurrency-sweep face (runs INSIDE the arc container).

Fires N concurrent streaming completions with UNIQUE per-slot prompts
(seed = 20261111 ^ crc32(tag) ^ slot -- the face-tag prompt-seed law
applies per REQUEST: concurrent same-ctx requests sharing a prefix
would retrieve from the prefix cache and stop measuring prefill) and
banks per-request metrics from one common submit barrier, so
  ttft(slot)      = queue wait + prefill share (the millstone queue
                    decomposition: at conc 1 queue ~ 0)
  decode_tps(slot)= max_tokens / (total - ttft)   (spec-aware window)
  agg_tps_wall    = conc * max_tokens / max(total) (system throughput)

Usage:
  /opt/venv/bin/python /root/p139_conc.py <ctx> <max_tokens> <conc> <tag> [model] [temperature]
"""
import json
import random
import sys
import threading
import time
import urllib.request
import zlib

from transformers import AutoTokenizer

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = sys.argv[5] if len(sys.argv) > 5 else "qwen3.8-27b-fp8"
TEMP = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
TOKDIR = "/models/target"


def build_prompt(tok, n, tag, slot):
    # n = the SERVER-side prompt-token budget; the caller derives it as
    # ctx - max_tokens - 64 because the API server enforces prompt_len
    # + max_tokens <= max_model_len (measured on the rig, r140/r140b/
    # r140c 2026-11-11: gemma's sentencepiece inflates decode->encode
    # ~1.026x, so a ctx-8-ids face sent 134479 server tokens at 131K
    # and HTTP-400'd, and a trim to max_model_len alone still 400'd --
    # the completion budget counts; qwen BPE round-trips shrink
    # slightly and always fit). Trim the id prefix -- same rng stream
    # -- until the re-encoded length fits the budget; a no-op whenever
    # the round-trip already fits.
    rng = random.Random(20261111 ^ zlib.crc32(tag.encode()) ^ (slot * 7919))
    ids = [rng.randrange(1000, 50000) for _ in range(n - 8)]
    text = tok.decode(ids)
    m = len(tok(text).input_ids)
    while m > n and len(ids) > 16:
        ids = ids[: max(16, len(ids) - (m - n) - 8)]
        text = tok.decode(ids)
        m = len(tok(text).input_ids)
    return text


def main():
    ctx, mt, conc, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    tok = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
    budget = ctx - mt - 64
    prompts = [build_prompt(tok, budget, tag, s) for s in range(conc)]
    results = [None] * conc
    barrier = threading.Barrier(conc + 1)

    def worker(slot):
        body = json.dumps({
            "model": MODEL,
            "prompt": prompts[slot],
            "max_tokens": mt,
            "temperature": TEMP,
            "ignore_eos": True,
            "stream": True,
        }).encode()
        req = urllib.request.Request(URL, data=body,
                                     headers={"Content-Type": "application/json"})
        barrier.wait()
        t0 = time.perf_counter()
        ttft = None
        stamps = []
        err = None
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                for line in r:
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == b"[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except Exception:
                        continue
                    if chunk.get("choices") and chunk["choices"][0].get(
                            "text") is not None:
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = now - t0
                        stamps.append(now - t0)
        except Exception as e:  # noqa: BLE001 -- bank the refusal verbatim
            err = "{}: {}".format(type(e).__name__, str(e)[:200])
        itls = sorted(stamps[i] - stamps[i - 1] for i in range(1, len(stamps)))
        nn = len(itls)

        def pct(p):
            return itls[min(nn - 1, int(nn * p))] * 1e3 if nn else -1.0

        results[slot] = {
            "slot": slot, "error": err,
            "ttft_s": ttft if ttft is not None else -1,
            "total_s": stamps[-1] if stamps else -1,
            "n_stream_chunks": len(stamps),
            "itl_ms_p50": pct(0.50), "itl_ms_p90": pct(0.90),
            "itl_ms_p99": pct(0.99),
        }

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(conc)]
    for t in threads:
        t.start()
    barrier.wait()  # one common submit barrier for all slots
    for t in threads:
        t.join()

    ok = [r for r in results if r and r["error"] is None and r["total_s"] > 0]
    per_user = [mt / (r["total_s"] - r["ttft_s"]) for r in ok
                if r["total_s"] > r["ttft_s"] > 0]
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] > 0]
    wall = max((r["total_s"] for r in ok), default=-1)
    out = {
        "tag": tag, "ctx": ctx, "conc": conc, "max_tokens": mt,
        "model": MODEL, "temperature": TEMP,
        "n_ok": len(ok), "n_fail": conc - len(ok),
        "requests": results,
        "ttft_mean_s": sum(ttfts) / len(ttfts) if ttfts else -1,
        "ttft_max_s": max(ttfts) if ttfts else -1,
        "per_user_tps": [round(x, 2) for x in sorted(per_user)],
        "per_user_mean_tps": sum(per_user) / len(per_user) if per_user else -1,
        "per_user_min_tps": min(per_user) if per_user else -1,
        "agg_tps_wall": conc * mt / wall if wall > 0 else -1,
        "wall_s": wall,
    }
    path = "/root/p139_conc_{}.json".format(tag)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"CONC_DONE {tag} ok={len(ok)}/{conc} agg={out['agg_tps_wall']:.1f} "
          f"per_user_mean={out['per_user_mean_tps']:.1f} "
          f"ttft_mean={out['ttft_mean_s']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
