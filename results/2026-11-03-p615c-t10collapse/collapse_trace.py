#!/usr/bin/env python3
"""collapse_trace.py — per-chunk SSE trace + spec-acceptance deltas for
the t1.0-xhigh solo collapse mode (P6.15c, owner ask 2026-11-03:
"when does the collapse happen, why, is a fix possible").

Reproduces the banked t1.0 xhigh request EXACTLY (bench_run7n_t.py
--temperature 1.0 --thinking xhigh): prompt "Count from 1 to 1000.",
temp 1.0, top_p 0.95/top_k 20/min_p 0/pp 0/rp 1.0, chat_template_kwargs
{"enable_thinking": true, "reasoning_effort": "xhigh"} — but STREAMING,
timestamping every SSE chunk (= every engine step) and snapshotting
/metrics spec/draft/accept counters before/after each request.
"""
import json
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
N_REQ = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0}


def spec_metrics():
    try:
        txt = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception as e:  # noqa: BLE001
        return {"ERR": f"{type(e).__name__}"}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        low = line.lower()
        if "draft" in low or "accept" in low:
            m = re.match(r"^(\S+)\s+([0-9.e+]+)$", line)
            if m:
                try:
                    out[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
    return out


def one_request(idx):
    body = {"model": "qwen3.8-27b-fp8", "max_tokens": 200,
            "temperature": 1.0, **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": True,
                                     "reasoning_effort": "xhigh"},
            "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user",
                          "content": "Count from 1 to 1000."}]}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    m0 = spec_metrics()
    t0 = time.perf_counter()
    chunks = []           # (t_rel, dt_since_prev, content_chars, reason_chars)
    prev = t0
    cc = rc = 0
    usage = finish = None
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            now = time.perf_counter()
            try:
                d = json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
            if "usage" in d and d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices")
            if ch:
                if ch[0].get("finish_reason"):
                    finish = ch[0]["finish_reason"]
                delta = ch[0].get("delta", {}) or {}
                cc += len(delta.get("content") or "")
                rc += len((delta.get("reasoning_content") or ""))
            chunks.append((round(now - t0, 4), round(now - prev, 4),
                           cc, rc))
            prev = now
    except Exception as e:  # noqa: BLE001
        print(f"req{idx} STREAM_FAIL {type(e).__name__}: {e}", flush=True)
        return
    wall = time.perf_counter() - t0
    m1 = spec_metrics()
    deltas = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    n = usage.get("completion_tokens") if usage else None
    # cadence profile: mean chunk-dt per 10-chunk band (engine-step time)
    bands = []
    for b in range(0, len(chunks), 10):
        seg = chunks[b:b + 10]
        bands.append(f"{sum(c[1] for c in seg) / len(seg) * 1000:.0f}")
    # first chunk whose dt stays >2x the run's median (collapse onset, if any)
    dts = [c[1] for c in chunks]
    if dts:
        srt = sorted(dts)
        med = srt[len(srt) // 2]
        onset = next((i for i, d in enumerate(dts)
                      if d > 2 * max(med, 0.02) and
                      all(x > 2 * max(med, 0.02)
                          for x in dts[i:i + 5])), None)
    else:
        med = onset = None
    print(f"req{idx}: chunks={len(chunks)} wall={wall:.2f}s "
          f"completion_tok={n} finish={finish} "
          f"content_chars={cc} reasoning_chars={rc}", flush=True)
    print(f"req{idx}: band_dt_ms[10-chunk bands]={','.join(bands)} "
          f"median_dt={med * 1000 if med else -1:.0f}ms "
          f"collapse_onset_chunk={onset}", flush=True)
    dk = sorted(deltas)
    print(f"req{idx}: metric_deltas " + " ".join(
        f"{k.split(':')[-1]}={deltas[k]:.0f}" for k in dk), flush=True)


def main():
    print(f"=== collapse_trace: {N_REQ} streaming solo t1.0-xhigh "
          f"requests, per-chunk dt + acceptance deltas ===", flush=True)
    for i in range(1, N_REQ + 1):
        one_request(i)
        time.sleep(1.0)
    print("COLLAPSE_TRACE_DONE", flush=True)


if __name__ == "__main__":
    main()
