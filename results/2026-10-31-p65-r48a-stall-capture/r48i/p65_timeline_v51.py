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
BYSTANDER_MT = 2500          # wave 1 covers the early timeline
WAVE2_AT = 60.0              # second bystander wave (r48a bug fix: wave 1
WAVE2_MT = 1500              # drained at ~45 s, leaving late windows empty)
LONG_MT = 48
# faces: (tag, at_s, seed, target_tokens, cached_replay_of)
FACES = [
    ("U32k", 25.0, 1010101, 32000, None),
    ("C32k", 55.0, None, 0, "U32k"),      # same prompt -> cached path
    ("U8k", 70.0, 2020202, 8000, None),
    ("U64k", 85.0, 3030303, 64000, None),
]
import os as _os
_sel = _os.environ.get("P65_FACES")      # e.g. "U32k,C32k" — subset for
if _sel:                                 # short discriminator boots
    _keep = {s.strip() for s in _sel.split(",") if s.strip()}
    FACES = [f for f in FACES if f[0] in _keep]
# v5.1: extend wave-2 so late faces (U64k @+85) run under live bystanders
# — v5's 1500-token wave drained at ~+89, leaving W_U64k windows empty
# (r48g) and measuring the 64k face near-solo.
WAVE2_MT = int(_os.environ.get("P65_W2MT", str(WAVE2_MT)))

_events = []                 # (kind, key, t_mono, t_wall)
_TEXTS = {}                  # v4: first 240 chars per stream (false-hit probe)
_lock = threading.Lock()


def _ev(kind, key):
    t = time.perf_counter()
    with _lock:
        _events.append((kind, key, t, time.time()))


def sse_call(tag, body, chunks):
    """POST /v1/chat/completions with stream=true; timestamp every chunk."""
    body = {"model": MODEL, "stream": True, "stream_options":
            {"include_usage": True}, "temperature": 0.0,
            **SAMPLING, "chat_template_kwargs": CTK_OFF, **body}
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    _ev("send", tag)
    n_tok = 0
    text = []
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
                if len("".join(text)) < 240:
                    text.append(delta)
                with _lock:
                    chunks.append((tag, t, time.time(), len(delta), "chunk"))
                if ch[0].get("finish_reason"):
                    _ev("finish_reason", tag)
            elif "usage" in d:
                u = d["usage"]
                with _lock:
                    chunks.append((tag, t, time.time(),
                                   u.get("completion_tokens", -1),
                                   "usage"))
                _ev("usage", f"{tag}:prompt_tokens={u.get('prompt_tokens')}"
                   f" cached={((u.get('prompt_tokens_details') or {}).get('cached_tokens'))}"
                   f" completion={u.get('completion_tokens')}")
                _TEXTS[tag] = "".join(text)[:240]
    except Exception as e:  # noqa: BLE001
        print(f"STREAM_FAIL {tag}: {type(e).__name__}: {e}", flush=True)
        _ev("fail", tag)


def long_prompt(seed, n_words):
    """Synthetic long context; UNIQUE seed per face (r48a bug: seed-shared
    generators made the 8k face a strict prefix of the 32k face, so it
    measured the cached path, not a fresh prefill)."""
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


def sized_prompt(seed, target_tokens):
    """Build a synthetic prompt near target_tokens; one linear correction
    pass against the server's own /tokenize (fallback ~27 tok/note)."""
    n = max(50, int(target_tokens / 27))
    c = tokenize_count(long_prompt(seed, n))
    if c > 0 and abs(c - target_tokens) > 0.15 * target_tokens:
        n = max(50, int(n * target_tokens / c))
        c = tokenize_count(long_prompt(seed, n))
    return long_prompt(seed, n), (c if c > 0 else n * 27)


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

    p32, c32 = sized_prompt(1010101, 32000)
    p64, c64 = sized_prompt(3030303, 64000)
    p8, c8 = sized_prompt(2020202, 8000)
    print(f"prompt sizes: U32k={c32} U64k={c64} U8k={c8} tok", flush=True)
    prompts = {"U32k": p32, "U64k": p64, "U8k": p8, "C32k": p32}
    counts = {"U32k": c32, "U64k": c64, "U8k": c8, "C32k": c32}

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

    # v5 FIX: content must use the bound default tg. v3/v4 closed over the
    # comprehension's `tag` cell, whose value after the comprehension is the
    # LAST face — so every face sent the last subset entry's prompt and the
    # r48b-r48f "fresh" faces after the first were same-prompt replays
    # (see VERDICT r48f reversal). Server usage lines are the ground truth.
    timers = [
        at(at_s, (lambda tg=tag: sse_call(tg,
             {"messages": [{"role": "user", "content": prompts[tg]}],
              "max_tokens": LONG_MT}, chunks)))
        for tag, at_s, _s, _t, _r in FACES
    ]

    def bystander_wave(prefix, mt):
        return [threading.Thread(
            target=sse_call,
            args=(f"{prefix}{i}", {"messages": [{"role": "user",
                                                 "content": "Count from 1 to 1000."}],
                                  "max_tokens": mt}, chunks))
            for i in range(N_BYSTANDER)]

    bys = bystander_wave("byst", BYSTANDER_MT)
    for t in bys:
        t.start()
    wave2 = bystander_wave("bys2", WAVE2_MT)
    bys2_timer = at(WAVE2_AT, lambda: [t.start() for t in wave2])
    for t in timers:
        t.join()
    bys2_timer.join()
    for t in bys:
        t.join()
    for t in wave2:
        t.join()

    # ---- bank raw ----
    with open(out, "w") as f:
        json.dump({"wave_t0_wall": time.time() - (time.perf_counter() - wave_t0),
                   "prompt_tokens": counts,
                   "events": _events,
                   "texts": _TEXTS,
                   "chunks": chunks}, f)
    print(f"raw -> {out} ({len(chunks)} chunks, {len(_events)} events)", flush=True)
    for tag, txt in _TEXTS.items():
        print(f"  text[{tag}] {len(txt)}ch: {txt[:100]!r}", flush=True)

    # ---- analyze ----
    def t_of(tag, kind):
        v = [e[2] for e in _events if e[1] == tag and e[0] == kind]
        return v[0] if v else None

    first_tok = {}
    for tag, t, w, n, kind in chunks:
        if kind == "chunk" and tag not in first_tok:
            first_tok[tag] = t

    print("\n=== long-stream faces (perf_counter, wave-relative)")
    for tag, _at, _s, _t, _r in FACES:
        s, ft = t_of(tag, "send"), t_of(tag, "done")
        if s is None:
            print(f"  {tag}: FAILED — no events")
            continue
        ftk = first_tok.get(tag)
        ttft = (ftk - s) if ftk else float("nan")
        tot = (ft - s) if ft else float("nan")
        print(f"  {tag}: TTFT={ttft:7.2f}s total={tot:7.2f}s "
              f"(start @ wave+{s - wave_t0:6.1f}s)")

    # bystander gap timeline (both waves)
    bys_tags = {c[0] for c in chunks if c[0].startswith(("byst", "bys2"))}
    gaps = []  # (gap_s, mid_t)
    for tag in sorted(bys_tags):
        ts = sorted(t for (tg, t, w, n, k) in chunks if tg == tag and k == "chunk")
        for a, b in zip(ts, ts[1:]):
            gaps.append((b - a, (a + b) / 2))
    gaps.sort(key=lambda g: g[1])

    first_send = t_of(FACES[0][0], "send") or wave_t0 + FACES[0][1]
    windows = [("W_base", wave_t0 + 2.0, first_send - 0.5)]
    for tag, _at, _s, _t, replay in FACES:
        s, ft = t_of(tag, "send"), t_of(tag, "done")
        if s:
            windows.append((f"W_{tag}_start" + ("(cached)" if replay else ""),
                            s, (first_tok.get(tag) or s + 40) + 1.0))
        if ft:
            windows.append((f"W_{tag}_end", ft - 1.0, ft + 3.0))

    base = [g for g, m in gaps if windows[0][1] <= m <= windows[0][2]]
    bp50, bp95 = pct(base, 50), pct(base, 95)
    print(f"\n=== bystander inter-chunk gaps ({len(bys_tags)} streams, greedy, k3)")
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
