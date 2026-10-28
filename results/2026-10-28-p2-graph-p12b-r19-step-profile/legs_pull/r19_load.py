"""r19_load.py — 8 concurrent 8k-ctx streams, 400 output tokens, streaming;
keeps the engine in steady c8 decode through the profiler window.
Prompts = pybench construction (seeded random ids, seed 20261027) so the
cache posture matches the r18 k8k cells."""
import json
import random
import threading
import time
import urllib.request

from transformers import AutoTokenizer

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen3.8-27b-fp8"
TOK = AutoTokenizer.from_pretrained("/models/target", trust_remote_code=True)
_rng = random.Random(20261027)
ids = [_rng.randint(1000, 50000) for _ in range(8192)]
PROMPT = TOK.decode(ids)


def one(i, out):
    body = {"model": MODEL, "prompt": PROMPT, "max_tokens": 400,
            "temperature": 0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                p = line[5:].strip()
                if p == b"[DONE]":
                    break
                ch = json.loads(p)
                if ch.get("choices") and ch["choices"][0].get("text"):
                    n += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
        out.append((i, 200, ttft, n, time.perf_counter() - t0))
    except Exception as e:  # noqa: BLE001
        out.append((i, type(e).__name__, None, n, time.perf_counter() - t0))


outs = []
ths = [threading.Thread(target=one, args=(i, outs)) for i in range(8)]
t0 = time.perf_counter()
for t in ths:
    t.start()
for t in ths:
    t.join()
wall = time.perf_counter() - t0
for r in sorted(outs):
    print("stream", r, flush=True)
print(f"LOAD_DONE wall={wall:.1f}s", flush=True)
