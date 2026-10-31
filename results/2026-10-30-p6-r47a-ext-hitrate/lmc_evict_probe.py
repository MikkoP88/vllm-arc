"""lmc_evict_probe.py — r46z: external-retrieval-after-eviction probe.

The serve's external hits total is 0.0 with 181k tokens of queries: every
repeat so far was served by the LOCAL GPU prefix cache before the connector
was asked for anything (connector only counts tokens BEYOND the local hit).
This probe creates the one scenario that discriminates:

  1. STORE  — send unique prompt P (7.2k tokens): computed, attn KV stored
              to the LMCache pool, mamba boundary banked.
  2. EVICT  — push ~64 distinct 7.3k prompts (>417k tokens =
              kv_cache_size_tokens) through, LRU-evicting P's blocks from
              the GPU cache. LMCache pool (100 GiB) keeps P.
  3. RETRIEVE — resend byte-identical P: local must miss -> scheduler asks
              the connector for the full prefix -> joint (attn n mamba)
              lookup. Watch vllm:external_prefix_cache_hits_total delta.

Non-zero delta = external retrieval works end-to-end (the 0.0% line is
local-cache-first semantics). Zero delta = the joint lookup really misses.
"""
import json
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-fp8"
SALT = "evictprobe-fixed-20261030"

BASEP = (
    "You are maintaining a distributed inference cluster log. Summarize "
    "the following operator notes into one sentence at the end.\n\n"
)


def notes(tag):
    return "".join(
        f"shift-note-{i}-{tag}: the dual-arc node ran the certified fp8 "
        f"hybrid posture with decode-only graphs; interleave verified; "
        f"oneccl pidfd exchange held; lmcache local cpu pool at 100 GiB "
        f"virtual; boundary registry capped at 48 with keep 8; mamba "
        f"align mode on; speculative depth 3 acceptance near 52 percent "
        f"at temperature one; no integrity drift across the soak cycles; "
        f"gpu memory utilization 0.8; block size 64; kv dtype fp8 e4m3; "
        f"ssm dtype float16.\n"
        for i in range(60)
    )


PROMPT_P = f"{BASEP}[{SALT}]\n" + notes(SALT)


def call(prompt, tag, max_tokens=8):
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.perf_counter() - t0
    u = r.get("usage", {})
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", None)
    print(f"{tag}: wall={dt:.2f}s prompt={u.get('prompt_tokens')} "
          f"completion={u.get('completion_tokens')} cached={cached}",
          flush=True)
    return dt


def metric(name):
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith(name + "{"):
                return float(line.rsplit(" ", 1)[1])
    return None


q0, h0 = metric("vllm:external_prefix_cache_queries_total"), metric(
    "vllm:external_prefix_cache_hits_total")
print(f"baseline external: queries={q0} hits={h0}", flush=True)

t = call(PROMPT_P, "1-STORE (cold compute + store)", max_tokens=16)
q1, h1 = metric("vllm:external_prefix_cache_queries_total"), metric(
    "vllm:external_prefix_cache_hits_total")
print(f"after STORE: queries={q1} (+{q1 - q0}) hits={h1} (+{h1 - h0})",
      flush=True)
time.sleep(3)

print("2-EVICT: churning 64 distinct ~7.3k prompts (LRU-evict P)", flush=True)
tc = time.perf_counter()
for j in range(64):
    call(f"{BASEP}[{SALT}-churn-{j}]\n" + notes(f"c{j}"), f"  churn[{j}]")
print(f"churn done in {time.perf_counter() - tc:.0f}s", flush=True)
time.sleep(2)
q2, h2 = metric("vllm:external_prefix_cache_queries_total"), metric(
    "vllm:external_prefix_cache_hits_total")
print(f"after EVICT: queries={q2} (+{q2 - q1}) hits={h2} (+{h2 - h1})",
      flush=True)

call(PROMPT_P, "3-RETRIEVE (identical P after eviction)", max_tokens=16)
time.sleep(2)
q3, h3 = metric("vllm:external_prefix_cache_queries_total"), metric(
    "vllm:external_prefix_cache_hits_total")
print(f"after RETRIEVE: queries={q3} (+{q3 - q2}) hits={h3} (+{h3 - h2})",
      flush=True)
verdict = "EXTERNAL_RETRIEVAL_WORKS" if h3 > h2 else "EXTERNAL_RETRIEVAL_MISS"
print(f"VERDICT: {verdict}", flush=True)
