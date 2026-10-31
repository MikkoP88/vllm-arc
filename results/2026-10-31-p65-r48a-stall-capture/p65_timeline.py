#!/usr/bin/env python3
"""p65_timeline.py — P6.5 leg r48a capture harness (plan §12 item 7, P6.5 (a)).

One wave of 6 bystander decode streams (greedy, the certified Run-7n
prompt "Count from 1 to 1000.") recorded chunk-by-chunk client-side,
while three long-context events are injected sequentially:

  F1 (t~20s): UNCACHED ~32k-token context, stream, 48 out — full chunked
      prefill face (candidates [ii] chunked-prefill starvation, [iv] snap
      at chunk ends, [v] batch-shape graph events)
  F2 (t~60s): SAME prompt replay (CACHED by F1's store) — prefix-hit path:
      isolates candidate [i] scheduler-side sync lookup walk + booking
      (near-zero prefill; any stall here is NOT prefill starvation)
  F3 (t~85s): UNCACHED ~8k-token context — length-scaling contrast face

Every SSE chunk arrival is timestamped (perf_counter for gaps, wallclock
for serve-log alignment). Bystander inter-chunk gap percentiles are
reported per analysis window; the P6.5 plan gate face is gap-during-
injection <= 2x the pre-injection baseline gap.

Usage: python3 p65_timeline.py [out.json]
"""
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
TOK_URL = "http://127.0.0.1:8000/tokenize"
MODEL = "qwen3.8-27b-fp8"
SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0}
CTK_OFF = {"enable_thinking": False}

N_BYSTANDER = 6
BYSTANDER_MT = 2000          # decode through the whole event timeline (~100 s)
WAVE_LEAD_S = 20.0           # baseline window before F1
F1_AT, F2_AT, F3_AT = 20.0, 60.0, 85.0   # seconds after wave start
LONG_MT = 48

_events = []                 # (kind, key, t_mono, t_wall)
_lock = threading.Lock()


def _ev(kind, key):
    t = time.perf_counter()
    with _lock:
        _events.append((kind, key, t, time.time()))


def sse_call(tag, body, chunks):
    """POST /v1/chat/completions with stream=true; timestamp every chunk."""
    body = {"model": MODEL, "stream": True, "temperature": 0.0,
            **SAMPLING, "chat_template_kwargs": CTK_OFF, **body}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    _ev("send", tag)
    n_tok = 0
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        for line in resp:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            t = time.perf_counter()
            if payload == b"[DONE]":
                with _lock:
                    chunks.append((tag, t, time.time(), 0, "done"))
                _ev("done", tag)
                break
            try:
                d = json.loads(payload)
            except ValueError:
                continue
            ch = d.get("choices")
            if ch:
                delta = (ch[0].get("delta") or {}).get("content") or ""
                n_tok += 1
                with _lock:
                    chunks.append((tag, t, time.time(), len(delta), "chunk"))
                if ch[0].get("finish_reason"):
                    _ev("finish_reason", tag)
            elif "usage" in d:
                with _lock:
                    chunks.append((tag, t, time.time(),
                                   d["usage"].get("completion_tokens", -1),
                                   "usage"))
    except Exception as e:  # noqa: BLE001
        print(f"STREAM_FAIL {tag}: {type(e).__name__}: {e}", flush=True)
        _ev("fail", tag)


def long_prompt(n_words):
    """Synthetic long context; varied so F1/F3 are unique (no cross-hit)."""
    rng = [12345]  # xorshift-ish deterministic
    def nxt():
        rng[0] = (rng[0] * 1103515245 + 12345) & 0x7fffffff
        return rng[0]
    parts = []
    for i in range(n_words):
        parts.append(f"Note {i}: the measured latency of stage "
                     f"{nxt() % 9973} under load {nxt() % 977} ms "
                     f"was recorded by sensor {nxt() % 8191}.")
    return " ".join(parts)


def sized_prompt(target_tokens):
    """Build a synthetic prompt near target_tokens; one linear correction
    pass against the server's own /tokenize (fallback ~27 tok/note)."""
    n = max(50, int(target_tokens / 27))
    c = tokenize_count(long_prompt(n))
    if c > 0 and abs(c - target_tokens) > 0.15 * target_tokens:
        n = max(50, int(n * target_tokens / c))
        c = tokenize_count(long_prompt(n))
    return long_prompt(n), (c if c > 0 else n * 27)


def tokenize_count(prompt):
    body = {"model": MODEL, "prompt": prompt}
    req = urllib.request.Request(
        TOK_URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=120))
        return len(d.get("tokens", []))
    except Exception as e:  # noqa: BLE001
        print(f"tokenize_FAIL {type(e).__name__}: {e}", flush=True)
        return -1


def pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "p65_timeline.json"

    print("=== warmup (first-traffic law; warms p12b interlock + first graphs)", flush=True)
    sse_call("warmup", {"messages": [{"role": "user",
                                      "content": "Count from 1 to 1000."}],
                        "max_tokens": 8}, [])
    time.sleep(2)

    p32, c32 = sized_prompt(32000)
    p8, c8 = sized_prompt(8000)
    print(f"prompt sizes: F1/F2={c32} tok, F3={c8} tok", flush=True)

    chunks = []
    wave_t0 = time.perf_counter()

    def at(sec, fn):
        def run():
            d = sec - (time.perf_counter() - wave_t0)
            if d > 0:
                time.sleep(d)
            fn()
        th = threading.Thread(target=run)
        th.start()
        return th

    timers = [
        at(F1_AT, lambda: sse_call("L1_32k_uncached",
             {"messages": [{"role": "user", "content": p32}],
              "max_tokens": LONG_MT}, chunks)),
        at(F2_AT, lambda: sse_call("L2_32k_cached",
             {"messages": [{"role": "user", "content": p32}],
              "max_tokens": LONG_MT}, chunks)),
        at(F3_AT, lambda: sse_call("L3_8k_uncached",
             {"messages": [{"role": "user", "content": p8}],
              "max_tokens": LONG_MT}, chunks)),
    ]

    bys = [threading.Thread(
        target=sse_call,
        args=(f"byst{i}", {"messages": [{"role": "user",
                                          "content": "Count from 1 to 1000."}],
                           "max_tokens": BYSTANDER_MT}, chunks))
        for i in range(N_BYSTANDER)]
    for t in bys:
        t.start()
    for t in timers:
        t.join()
    for t in bys:
        t.join()

    # ---- bank raw ----
    with open(out, "w") as f:
        json.dump({"wave_t0_wall": time.time() - (time.perf_counter() - wave_t0),
                   "prompt_tokens": {"F1_F2": c32, "F3": c8},
                   "events": _events,
                   "chunks": chunks}, f)
    print(f"raw -> {out} ({len(chunks)} chunks, {len(_events)} events)", flush=True)

    # ---- analyze ----
    def t_of(tag, kind):
        v = [e[2] for e in _events if e[1] == tag and e[0] == kind]
        return v[0] if v else None

    s1, ft1 = t_of("L1_32k_uncached", "send"), t_of("L1_32k_uncached", "done")
    s2, ft2 = t_of("L2_32k_cached", "send"), t_of("L2_32k_cached", "done")
    s3, ft3 = t_of("L3_8k_uncached", "send"), t_of("L3_8k_uncached", "done")
    first_tok = {}
    for tag, t, w, n, kind in chunks:
        if kind == "chunk" and tag not in first_tok:
            first_tok[tag] = t

    print("\n=== long-stream faces (perf_counter, wave-relative)")
    for tag, s, ft in (("L1_32k_uncached", s1, ft1), ("L2_32k_cached", s2, ft2),
                       ("L3_8k_uncached", s3, ft3)):
        if s is None:
            print(f"  {tag}: FAILED — no events")
            continue
        ftk = first_tok.get(tag)
        ttft = (ftk - s) if ftk else float("nan")
        tot = (ft - s) if ft else float("nan")
        print(f"  {tag}: TTFT={ttft:7.2f}s total={tot:7.2f}s "
              f"(start @ wave+{s - wave_t0:6.1f}s)")

    # bystander gap timeline
    bys_tags = {f"byst{i}" for i in range(N_BYSTANDER)}
    gaps = []  # (gap_s, mid_t)
    per = {}
    for tag in sorted(bys_tags):
        ts = sorted(t for (tg, t, w, n, k) in chunks if tg == tag and k == "chunk")
        per[tag] = ts
        for a, b in zip(ts, ts[1:]):
            gaps.append((b - a, (a + b) / 2))
    gaps.sort(key=lambda g: g[1])

    windows = [("W_base", wave_t0 + 2.0, s1 if s1 else wave_t0 + 20),
               ("W_F1_start", s1, (first_tok.get("L1_32k_uncached") or s1 + 20) + 1.0)]
    if s2:
        windows.append(("W_F2_start(cached)", s2,
                        (first_tok.get("L2_32k_cached") or s2 + 5) + 1.0))
    if ft1:
        windows.append(("W_F1_end", ft1 - 1.0, ft1 + 3.0))
    if ft2:
        windows.append(("W_F2_end", ft2 - 1.0, ft2 + 3.0))
    if s3:
        windows.append(("W_F3_start", s3,
                        (first_tok.get("L3_8k_uncached") or s3 + 10) + 1.0))

    base = [g for g, m in gaps if windows[0][1] <= m <= windows[0][2]]
    bp50, bp95 = pct(base, 50), pct(base, 95)
    print(f"\n=== bystander inter-chunk gaps (6 streams, greedy, k3)")
    print(f"  baseline: n={len(base)} p50={bp50*1e3:7.1f}ms p95={bp95*1e3:7.1f}ms")
    print(f"  {'window':18s} {'n':>5s} {'p50 ms':>9s} {'p95 ms':>9s} {'max ms':>9s} "
          f"{'#>2xb95':>7s} {'stall_s':>8s}")
    for name, lo, hi in windows:
        ws = [g for g, m in gaps if lo <= m <= hi]
        if not ws:
            print(f"  {name:18s}     0        —         —         —       —        —")
            continue
        thr = 2 * bp95
        stall = sum(g - bp50 for g in ws if g > thr)
        print(f"  {name:18s} {len(ws):5d} {pct(ws,50)*1e3:9.1f} {pct(ws,95)*1e3:9.1f} "
              f"{max(ws)*1e3:9.1f} {sum(1 for g in ws if g > thr):7d} {stall:8.2f}")

    print("\nP65_CAPTURE: DONE")


if __name__ == "__main__":
    main()
