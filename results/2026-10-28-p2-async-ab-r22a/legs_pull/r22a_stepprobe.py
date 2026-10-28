"""r22a_stepprobe.py — direct step-wall instrument, no profiler.
Streaming completions: with MTP k4 each engine step emits ~acc tokens in ONE
SSE chunk, so the inter-chunk gap distribution IS the engine step wall.
Runs c1 and c8 (counting-prompt family = spec-gate acceptance family);
/metrics deltas give the acceptance for the same window."""
import json
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"


def metrics():
    with urllib.request.urlopen(f"{URL}/metrics", timeout=10) as r:
        out = {}
        for ln in r.read().decode().splitlines():
            if ln.startswith("vllm:spec_decode_num_") and " " in ln:
                k, v = ln.rsplit(" ", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
        return out


def stream(n, mt, gaps_all):
    body = {"model": MODEL, "max_tokens": mt, "temperature": 0,
            "stream": True, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Count from 1 to 1000."}]}
    req = urllib.request.Request(
        f"{URL}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    gaps, last, ntok = [], None, 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            d = chunk.get("choices", [{}])[0].get("delta", {})
            txt = d.get("content") or ""
            if txt:
                now = time.perf_counter()
                if last is not None:
                    gaps.append(now - last)
                last = now
                ntok += len(txt.split()) or 1
    gaps_all.extend(gaps)
    return ntok


def run(label, conc, mt=240):
    m0 = metrics()
    gaps_all = []
    ths = [threading.Thread(target=stream, args=(i, mt, gaps_all))
           for i in range(conc)]
    t0 = time.perf_counter()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.perf_counter() - t0
    time.sleep(2)
    m1 = metrics()
    gs = sorted(g for g in gaps_all if g > 0.005)  # drop stream-start skews
    n = len(gs)
    if n:
        p50 = gs[n // 2] * 1e3
        p90 = gs[(n * 9) // 10] * 1e3
        mean = sum(gs) / n * 1e3
    else:
        p50 = p90 = mean = 0.0
    dr = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get(
        "vllm:spec_decode_num_drafts_total", 0)
    ac = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get(
        "vllm:spec_decode_num_accepted_tokens_total", 0)
    dt_ = m1.get("vllm:spec_decode_num_draft_tokens_total", 0) - m0.get(
        "vllm:spec_decode_num_draft_tokens_total", 0)
    print(f"[{label}] conc={conc} wall={wall:.1f}s chunks={len(gaps_all)} "
          f"gap_p50={p50:.1f}ms mean={mean:.1f}ms p90={p90:.1f}ms | "
          f"drafts={int(dr)} draft_tokens={int(dt_)} accepted={int(ac)} "
          f"acc_len={ac / dr if dr else 0:.2f}", flush=True)


print("STEPPROBE_START", flush=True)
run("c1", 1)
run("c8", 8)
run("c1-again", 1)
print("STEPPROBE_DONE", flush=True)
