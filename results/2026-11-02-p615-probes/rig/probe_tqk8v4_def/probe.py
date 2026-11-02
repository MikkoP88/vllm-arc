import json, time, random, threading, urllib.request

URL = "http://127.0.0.1:8000"

def post(body, timeout=600):
    req = urllib.request.Request(URL + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def stream_ttft(body, timeout=600):
    req = urllib.request.Request(URL + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; ntok = 0
    resp = urllib.request.urlopen(req, timeout=timeout)
    usage = {}
    for line in resp:
        if not line.startswith(b"data: "):
            continue
        p = line[6:].strip()
        if p == b"[DONE]":
            break
        d = json.loads(p)
        ch = d.get("choices")
        if ch:
            if (ch[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ntok += 1
        elif "usage" in d:
            usage = d["usage"]
    return ttft, time.perf_counter() - t0, ntok, usage

def tok_count(prompt):
    req = urllib.request.Request(URL + "/v1/tokenize",
        data=json.dumps({"model": "qwen3.8-27b-fp8", "prompt": prompt}).encode(),
        headers={"Content-Type": "application/json"})
    return len(json.load(urllib.request.urlopen(req, timeout=120))["tokens"])

def uniform_prompt(target):
    random.seed(int(time.time() * 1000) & 0xffffffff)
    parts, i = [], 0
    txt = ""
    while True:
        i += 1
        s = (f"Record {i}: the measurement series continues with value "
             f"{i*7%997} and note {i*13%887}; observation {i%64} is logged "
             f"for stage {i*31%4212}. ")
        txt += s
        if i % 64 == 0 and tok_count(txt) >= target:
            return txt
        if i > 200000:
            return txt

# flip/warmup (first-traffic law)
t0 = time.perf_counter()
r = post({"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
          "max_tokens": 832, "messages": [{"role": "user", "content": "Count from 1 to 20."}]})
print(f"FLIP_FIRST: {r['choices'][0]['finish_reason']} {time.perf_counter()-t0:.1f}s")

# solo200 x3 (#1 warmup)
for k in range(3):
    t0 = time.perf_counter()
    r = post({"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
              "max_tokens": 200,
              "messages": [{"role": "user", "content": "Count from 1 to 1000."}]})
    wall = time.perf_counter() - t0
    ct = r["usage"]["completion_tokens"]
    print(f"SOLO200 #{k+1}: {ct} tok / {wall:.2f}s = {ct/wall:.2f} tok/s" if k else
          f"SOLO200 #{k+1} (warmup): {ct} tok / {wall:.2f}s")

# agg4x256 x2 (#1 warmup)
def agg_run(k):
    out = []
    def one():
        t0 = time.perf_counter()
        r = post({"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
                  "max_tokens": 256,
                  "messages": [{"role": "user", "content": "Count from 1 to 1000."}]})
        out.append(r["usage"]["completion_tokens"] / (time.perf_counter() - t0))
    ts = [threading.Thread(target=one) for _ in range(4)]
    t0 = time.perf_counter()
    [t.start() for t in ts]; [t.join() for t in ts]
    wall = time.perf_counter() - t0
    print(f"AGG4x256 #{k+1}: wall={wall:.2f}s sum_per_stream_tps={sum(out):.2f}")
for k in range(2):
    agg_run(k)

# solo U-bars (fresh prompts, prefix hit ~0)
for name, target in (("U32k", 31805), ("U64k", 63784)):
    p = uniform_prompt(target)
    ttft, tot, ntok, usage = stream_ttft({
        "model": "qwen3.8-27b-fp8", "stream": True, "temperature": 0.0,
        "max_tokens": 32, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": p}]})
    pt = usage.get("prompt_tokens", -1)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"UBAR {name}: prompt={pt} cached={cached} TTFT={ttft:.2f}s "
          f"prefill={pt/ttft:.0f} tok/s total={tot:.2f}s")
print("PROBE_DONE")
