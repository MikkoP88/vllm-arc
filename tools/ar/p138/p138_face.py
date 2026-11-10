#!/usr/bin/env python3
"""p138_face.py -- P6.37 close-out report face (runs INSIDE the arc container).

p135_face.py with two close-out extensions (plan phase E/G):
  * 5th arg temperature (default 0.0) -- the MTP acceptance sweep needs
    sampling temps 0.0..1.0; the value goes into the request body and the
    output JSON.
  * per-tag prompt seed -- the r138 cells run many same-ctx faces on ONE
    boot (temp sweep @8192); p135's fixed seed would send byte-identical
    prompts so every face after the first hits the LMCache prefix store
    and TTFT stops measuring prefill. Seed = crc32(tag): every face gets
    a unique prompt (no-cache methodology, millstone-style), still fully
    reproducible from the banked tag.

Usage:
  /opt/venv/bin/python /root/p138_face.py <ctx> <max_tokens> <tag> [model] [temperature]
"""
import json
import random
import sys
import time
import urllib.request
import zlib

from transformers import AutoTokenizer

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = sys.argv[4] if len(sys.argv) > 4 else "qwen3.8-27b-fp8"
TEMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
TOKDIR = "/models/target"


def build_prompt(n, tag):
    rng = random.Random(20261110 ^ zlib.crc32(tag.encode()))
    tok = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
    ids = [rng.randrange(1000, 50000) for _ in range(n - 8)]
    return tok.decode(ids)


def main():
    ctx, mt, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    prompt = build_prompt(ctx, tag)
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": mt,
        "temperature": TEMP,
        "ignore_eos": True,
        "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_send = time.time()
    ttft = None
    stamps = []
    with urllib.request.urlopen(req, timeout=1800) as r:
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
    itls = [stamps[i] - stamps[i - 1] for i in range(1, len(stamps))]
    itls.sort()
    n = len(itls)

    def pct(p):
        return itls[min(n - 1, int(n * p))] * 1e3 if n else -1.0

    out = {
        "tag": tag, "ctx": ctx, "max_tokens": mt, "model": MODEL,
        "temperature": TEMP, "n_stream_chunks": len(stamps),
        "t_send_epoch": t_send, "ttft_s": ttft if ttft is not None else -1,
        "total_s": stamps[-1] if stamps else -1,
        "itl_ms_mean": sum(itls) / n * 1e3 if n else -1,
        "itl_ms_p50": pct(0.50), "itl_ms_p90": pct(0.90),
        "itl_ms_p99": pct(0.99), "itl_ms_max": itls[-1] * 1e3 if n else -1,
        "itl_ms_all": [round(x * 1e3, 3) for x in itls],
    }
    path = "/root/p138_face_%s.json" % tag
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("FACE_DONE %s ttft=%.3fs tokens=%d itl_mean=%.2fms p99=%.2fms" %
          (tag, out["ttft_s"], len(stamps), out["itl_ms_mean"],
           out["itl_ms_p99"]))


if __name__ == "__main__":
    sys.exit(main())
