#!/usr/bin/env python3
"""p138_face.py -- P6.37 close-out report face (runs INSIDE the arc container).

p135_face.py with three close-out extensions (plan phase E/G):
  * 5th arg temperature (default 0.0) -- the MTP acceptance sweep needs
    sampling temps 0.0..1.0; the value goes into the request body and the
    output JSON.
  * per-tag prompt seed -- the r138 cells run many same-ctx faces on ONE
    boot (temp sweep @8192); p135's fixed seed would send byte-identical
    prompts so every face after the first hits the LMCache prefix store
    and TTFT stops measuring prefill. Seed = crc32(tag): every face gets
    a unique prompt (no-cache methodology, millstone-style), still fully
    reproducible from the banked tag.
  * 6th arg pass label (P6.47 P-G, 2026-11-27): `cold` (default) or
    `warm`. A warm face re-sends the byte-identical prompt of its cold
    twin (same tag -> same seed -> same prompt) on the same boot, so the
    prompt KV is served from the prefix cache and TTFT measures the
    cached path instead of a full prefill. The request body is the same
    in both passes; only the output file differs: the warm JSON carries
    a `_warm` suffix, so neither twin can mask the other.
  * cache proof (BOTH passes): the /metrics counter deltas around the
    face are banked in the JSON as `cache`. The cold pass thereby PROVES
    its zero cached prompt tokens per face (until now an assumption of
    the prompt law) and the warm pass proves its hit. The counters are
    the engine's own (vLLM 0.30 token-based prefix-cache stats), read on
    the container loopback -- no server-argument change.

Usage:
  /opt/venv/bin/python /root/p138_face.py <ctx> <max_tokens> <tag> [model] [temperature] [pass]
  (P138_BASE / P138_TOKDIR / P138_OUT exist for the offline selftest --
  tools/ar/p139/p139_harness_selftest.py -- and are never set in a leg.)
"""
import json
import os
import random
import sys
import time
import urllib.request
import zlib

from transformers import AutoTokenizer

BASE = os.environ.get("P138_BASE", "http://127.0.0.1:8000")
URL = BASE + "/v1/completions"
METRICS = BASE + "/metrics"
MODEL = sys.argv[4] if len(sys.argv) > 4 else "qwen3.8-27b-fp8"
TEMP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
PASS = sys.argv[6] if len(sys.argv) > 6 else "cold"
TOKDIR = os.environ.get("P138_TOKDIR", "/models/target")
OUTDIR = os.environ.get("P138_OUT", "/root")   # the selftest redirects it

# /metrics counter -> the JSON key it is banked under. All are monotone
# counters; a face banks its before/after delta. prefix_cache_* are the
# scheduler's token-based prefix-cache stats (queries = prompt tokens
# looked up, hits = tokens served from the cache); prompt_tokens_* and
# request_success are the finished-request stats.
COUNTERS = {
    "prefix_cache_queries_total": "prefix_cache_queries",
    "prefix_cache_hits_total": "prefix_cache_hits",
    "prompt_tokens_total": "prompt_tokens",
    "prompt_tokens_cached_total": "prompt_tokens_cached",
    "external_prefix_cache_queries_total": "external_queries",
    "external_prefix_cache_hits_total": "external_hits",
    "request_success_total": "request_success",
    "num_preempted_total": "num_preempted",
}
# The finished-request counters are logged when the API server processes
# the engine's stats outputs, which can land a few seconds after the last
# streamed chunk: the proof waits for them, capped here.
STATS_WAIT_S = 15.0


def build_prompt(n, tag):
    rng = random.Random(20261110 ^ zlib.crc32(tag.encode()))
    tok = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
    ids = [rng.randrange(1000, 50000) for _ in range(n - 8)]
    return tok.decode(ids)


def sample_metrics():
    """One /metrics scrape -> {counter: value summed over its label sets}.
    None when the endpoint does not answer: a dead engine banks None,
    never a silent 0."""
    try:
        raw = urllib.request.urlopen(METRICS, timeout=10).read().decode(
            "utf-8", "replace")
    except Exception:  # noqa: BLE001 -- the proof degrades, never the face
        return None
    out = {}
    for line in raw.splitlines():
        if not line.startswith("vllm:"):
            continue
        name = line[5:].split("{", 1)[0].split(" ", 1)[0].strip()
        if name not in COUNTERS:
            continue
        try:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return out


def cache_proof(before, n_done):
    """The counter deltas around the face plus the flush wait: scrapes
    until request_success_total rose by the face's completed-request
    count (n_done), capped at STATS_WAIT_S. A face with no completed
    request does not wait."""
    t0 = time.perf_counter()
    after = sample_metrics()
    flushed = False
    while True:
        flushed = (before is not None and after is not None and n_done > 0
                   and after.get("request_success_total", 0.0)
                   - before.get("request_success_total", 0.0) >= n_done)
        if (flushed or n_done <= 0 or before is None
                or time.perf_counter() - t0 > STATS_WAIT_S):
            break
        time.sleep(0.25)
        after = sample_metrics()
    proof = {key: None for key in COUNTERS.values()}
    if before is not None and after is not None:
        for src, key in COUNTERS.items():
            delta = after.get(src, 0.0) - before.get(src, 0.0)
            proof[key] = int(delta) if float(delta).is_integer() else delta
    proof["stats_wait_s"] = round(time.perf_counter() - t0, 3)
    proof["stats_flushed"] = flushed
    return proof


def main():
    if PASS not in ("cold", "warm"):
        print("FACE_PASS_INVALID {!r} (cold|warm)".format(PASS))
        return 2
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
    before = sample_metrics()   # the cache-proof baseline (outside the timed window)
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
    out["pass"] = PASS
    out["cache"] = cache_proof(before, 1 if stamps else 0)
    path = os.path.join(
        OUTDIR, "p138_face_{}{}.json".format(tag, "_warm" if PASS == "warm" else ""))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("FACE_DONE {} pass={} ttft={:.3f}s tokens={} itl_mean={:.2f}ms "
          "p99={:.2f}ms cache={}/{} flushed={}".format(
              tag, PASS, out["ttft_s"], len(stamps), out["itl_ms_mean"],
              out["itl_ms_p99"], out["cache"]["prefix_cache_hits"],
              out["cache"]["prefix_cache_queries"],
              out["cache"]["stats_flushed"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
