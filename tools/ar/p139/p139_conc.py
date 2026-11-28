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

P6.47 P-G extensions (2026-11-27):
  * 7th arg pass label: `cold` (default) or `warm`. A warm face re-sends
    the byte-identical prompts of its cold twin (same tag -> same
    per-slot seeds) on the same boot: every slot's prompt KV comes from
    the prefix cache. The warm JSON carries a `_warm` suffix.
  * cache proof (both passes): the /metrics counter deltas around the
    whole group are banked as `cache` (token-based prefix-cache queries
    and hits summed over the N requests; see p138_face.py).

Usage:
  /opt/venv/bin/python /root/p139_conc.py <ctx> <max_tokens> <conc> <tag> [model] [temperature] [pass]
  (P139_BASE / P139_TOKDIR / P139_OUT exist for the offline selftest --
  p139_harness_selftest.py -- and are never set in a leg.)
"""
import json
import os
import random
import sys
import threading
import time
import urllib.request
import zlib

from transformers import AutoTokenizer

BASE = os.environ.get("P139_BASE", "http://127.0.0.1:8000")
URL = BASE + "/v1/completions"
METRICS = BASE + "/metrics"
MODEL = sys.argv[5] if len(sys.argv) > 5 else "qwen3.8-27b-fp8"
TEMP = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
PASS = sys.argv[7] if len(sys.argv) > 7 else "cold"
TOKDIR = os.environ.get("P139_TOKDIR", "/models/target")
OUTDIR = os.environ.get("P139_OUT", "/root")   # the selftest redirects it

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
STATS_WAIT_S = 15.0


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
    """The counter deltas around the group plus the flush wait: scrapes
    until request_success_total rose by the group's completed-request
    count (n_done), capped at STATS_WAIT_S. A group with no completed
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
        print("CONC_PASS_INVALID {!r} (cold|warm)".format(PASS))
        return 2
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
    before = sample_metrics()   # the cache-proof baseline, taken before the barrier opens
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
    out["pass"] = PASS
    out["cache"] = cache_proof(before, len(ok))
    path = os.path.join(
        OUTDIR, "p139_conc_{}{}.json".format(tag, "_warm" if PASS == "warm" else ""))
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"CONC_DONE {tag} pass={PASS} ok={len(ok)}/{conc} agg={out['agg_tps_wall']:.1f} "
          f"per_user_mean={out['per_user_mean_tps']:.1f} "
          f"ttft_mean={out['ttft_mean_s']:.2f} "
          f"cache={out['cache']['prefix_cache_hits']}/{out['cache']['prefix_cache_queries']} "
          f"flushed={out['cache']['stats_flushed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
