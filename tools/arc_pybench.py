"""w15_pybench.py -- uniform image-shootout matrix bench for qwen3.8-27b-fp8.

Runs INSIDE each serving container (needs only /opt/venv python + localhost).
One harness for all images (no vllm-bench CLI drift between 0.26 and 0.30).

Per cell (context x concurrency): `conc` threads each stream one completion
(temperature 0, ignore_eos, max_tokens=100). Reports:
  TTFT mean (ms), prefill rate (ctx/TTFT), decode TPOT mean (ms),
  aggregate output tok/s (total tokens / wall), ok streams.
Prompts: exact-length token ids (vocab 1000..50000, seeded) decoded once.
"""
import json
import sys
import threading
import time
import urllib.request

from transformers import AutoTokenizer

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen3.8-27b-fp8"
MAXTOK = 100

TOK = AutoTokenizer.from_pretrained("/models/target", trust_remote_code=True)
import random  # noqa: E402

_rng = random.Random(20261027)
PROMPTS = {}


def prompt_for(n: int) -> str:
    if n not in PROMPTS:
        ids = [_rng.randint(1000, 50000) for _ in range(n)]
        PROMPTS[n] = TOK.decode(ids)
    return PROMPTS[n]


def one_stream(ctx: int, out: list) -> None:
    body = {"model": MODEL, "prompt": prompt_for(ctx), "max_tokens": MAXTOK,
            "temperature": 0, "ignore_eos": True, "stream": True}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    ntok += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
        dt = time.perf_counter() - t0
        out.append((ttft, dt, ntok))
    except Exception as e:  # noqa: BLE001
        print(f"  stream_FAIL {type(e).__name__}: {e}", flush=True)
        out.append((None, -1.0, 0))


def cell(tag: str, ctx: int, conc: int) -> None:
    outs = [[] for _ in range(conc)]
    ths = [threading.Thread(target=one_stream, args=(ctx, o)) for o in outs]
    w0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - w0
    recs = [r for o in outs for r in o]
    good = [r for r in recs if r[0] is not None and r[2] > 0]
    if good:
        ttft = sum(r[0] for r in good) / len(good) * 1000
        tpot = sum((r[1] - r[0]) / max(1, r[2] - 1) for r in good) / len(good) * 1000
        toks = sum(r[2] for r in good)
        agg = toks / wall if wall > 0 else 0.0
        pre = ctx / (ttft / 1000) if ttft > 0 else 0.0
        print(f"CELL {tag} ttft_ms={ttft:.1f} prefill_tps={pre:.1f} "
              f"tpot_ms={tpot:.2f} agg_tps={agg:.2f} "
              f"per_req_tps={1000/tpot if tpot>0 else 0:.2f} "
              f"ok={len(good)}/{conc} wall={wall:.2f}", flush=True)
    else:
        print(f"CELL {tag} FAIL ok=0/{conc}", flush=True)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "m"
    MATRIX = [(1024, [1, 4, 8, 16]), (8192, [1, 4, 8, 16]),
              (16384, [1, 4, 8, 16]), (32768, [1, 4, 8]), (65536, [1, 2, 4])]
    only = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"=== PYMATRIX {tag} start ===", flush=True)
    # warm the exact-length prompts (tokenize/decode once, untimed)
    for ctx, _ in MATRIX:
        prompt_for(ctx)
    print("prompts ready", flush=True)
    for ctx, concs in MATRIX:
        for c in concs:
            name = f"k{ctx//1024}k-c{c}"
            if only and only not in name:
                continue
            cell(name, ctx, c)
    print(f"=== PYMATRIX {tag} DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
