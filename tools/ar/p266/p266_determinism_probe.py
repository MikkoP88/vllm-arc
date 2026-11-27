#!/usr/bin/env python3
"""p266_determinism_probe.py -- cache-state determinism matrix (P6.47 r266).

The r264 gate's mkg leg failed arc_garbage_probe's state_stability check:
two back-to-back temp-0 sends of the sanity prompt gave different text.
That check cannot say WHY the second send differs -- a request's answer
may depend on the prefix-cache state it lands in (a first send computes
the whole prompt, a repeat may serve part of it from the KV cache), on
traffic that ran in between, or on nothing at all (true run-to-run
nondeterminism). This probe separates the three by controlling the cache
state of every send and recording, per send, what the engine reports:

  cached_tokens   usage.prompt_tokens_details.cached_tokens (the engine's
                  own per-request prefix-cache hit count; vLLM reports no
                  details and treats the prompt as cold for a salted one)
  sha / head      sha1 of the answer text and its first characters
  top / margin    the first answer token's top-5 logprobs and the gap
                  between the best two (a flip at a gap of ~0.01 is a
                  near-tie; a flip at a gap of 2 is not)
  steps / min_gap every answer token's gap and runner-up (an answer can
                  fork at any position, not only the first) and the
                  smallest gap of the answer with its position

Every prompt runs one fixed series:
  salted x2      cache_salt unique per send -> both forced cold
  plain x2       no salt -> the cache serves whatever it holds
  churn          4 concurrent unrelated 256-token requests, as the gate
  plain x2       the gate's "first send after churn" shape, twice
  churn, plain x2 -- the same again (cycle reproducibility)
  salted/plain   alternating x2 -- a cold send right after a warm one

A lane is COLD-STABLE when every salted send of a prompt hashes alike,
WARM-STABLE when every plain send does, CHURN-STABLE when the first
plain send after each churn equals the warm hash, and COLD-EQ-WARM when
the cold and warm hashes coincide (the native bf16-KV property). A
series with a failed send or a failed churn batch supports no stability
claim at all: every axis reads False and ok / churn_ok say why.

Usage: ARC_PROBE_MODEL=<served> python3 p266_determinism_probe.py <tag>
Writes ./p266_determinism_<tag>.json and prints one line per send.
Stdlib only; talks to http://127.0.0.1:8000 (rig host python3).
Exit 0 = every send answered 200 (the verdict is in the lines, not the rc).
(P266_BASE / P266_GAP exist for the offline selftest --
p266_probe_selftest.py -- and are never set in a leg.)
"""
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

URL = os.environ.get("P266_BASE", "http://127.0.0.1:8000") + "/v1/chat/completions"
MODEL = os.environ.get("ARC_PROBE_MODEL", "gemma4-31b")
GAP = float(os.environ.get("P266_GAP", "0.5"))   # pause between two sends
# the garbage suite's sanity request, byte for byte (arc_garbage_probe.py)
SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0}
CTK = {"chat_template_kwargs": {"enable_thinking": False}}
TOP = 5
CHURN_TOKENS = 256

PROMPTS = {
    "sanity": {
        "content": "Reply with exactly: HEALTH CHECK OK",
        "max_tokens": 24, "sampling": True,
    },
    "simple": {
        "content": "Write a Python function `merge_intervals(intervals)` that "
                   "merges overlapping ranges and returns them sorted. Explain "
                   "the algorithm in three sentences.",
        "max_tokens": 64, "sampling": False,
    },
    "para": {
        "content": "Explain, in one paragraph, how a hash table resolves a "
                   "collision and why its average lookup stays constant time.",
        "max_tokens": 64, "sampling": False,
    },
    # the gemma legs run block_size 64: a prompt under one block can never
    # hit the prefix cache, so the three short prompts above never see a
    # cached prefix and only this one measures the cold-vs-cached axis
    "long": {
        "content": "Read the following passage and answer the question in "
                   "one sentence.\n\nPassage: A municipal library keeps a "
                   "register of every book it owns. When a reader borrows a "
                   "book, the clerk records the reader's card number, the "
                   "book's catalogue code, and the date of the loan. The "
                   "register is updated again when the book comes back, and "
                   "a shelf marker is placed so that the book can be "
                   "returned to its proper place. If a book is requested by "
                   "several readers, the library holds it at the desk for a "
                   "few days and sends a letter to the first reader on the "
                   "waiting list. Books that are not collected within the "
                   "holding period go to the next reader, and the clerk "
                   "notes the change in the register. At the end of each "
                   "year the librarian counts every entry, compares the "
                   "count with the shelves, and orders replacements for the "
                   "books that were lost.\n\nQuestion: What does the clerk "
                   "record when a reader borrows a book?",
        "max_tokens": 64, "sampling": False,
    },
}

# (kind, salt suffix) -- the fixed series every prompt runs
SERIES = [
    ("salted", "s1"), ("salted", "s2"),
    ("plain", None), ("plain", None),
    ("churn", None), ("plain", None), ("plain", None),
    ("churn", None), ("plain", None), ("plain", None),
    ("salted", "a1"), ("plain", None), ("salted", "a2"), ("plain", None),
]


def post(body, timeout=900):
    """POST and return (status, parsed-or-None). Never raises."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:  # noqa: BLE001 -- bank the status, never raise
            return e.code, None
    except Exception as e:  # noqa: BLE001
        return -1, {"transport_error": f"{type(e).__name__}: {e}"}


def answer_text(resp):
    """content first; the reasoning fields stand in when a reasoning
    parser moves the stream out of content (the garbage suite's law)."""
    msg = (resp or {}).get("choices", [{}])[0].get("message", {})
    return (msg.get("content") or msg.get("reasoning_content")
            or msg.get("reasoning") or "")


def first_token_top(resp):
    """[(token, logprob), ...] of the first answer token, best first."""
    try:
        step = resp["choices"][0]["logprobs"]["content"][0]
        rows = [(str(t["token"]), float(t["logprob"]))
                for t in step.get("top_logprobs", [])]
        return sorted(rows, key=lambda r: -r[1])
    except Exception:  # noqa: BLE001 -- no logprobs is a fact, not a crash
        return []


def token_steps(resp):
    """Per answer token [position, token, gap, runner-up token], best
    minus second best logprob at every step; [] when there are no logprobs."""
    try:
        steps = resp["choices"][0]["logprobs"]["content"]
    except Exception:  # noqa: BLE001
        return []
    out = []
    for i, st in enumerate(steps):
        rows = sorted(((str(t["token"]), float(t["logprob"]))
                       for t in st.get("top_logprobs", [])), key=lambda r: -r[1])
        if len(rows) < 2:
            out.append([i, str(st.get("token")), None, None])
            continue
        out.append([i, str(st.get("token")), round(rows[0][1] - rows[1][1], 4),
                    rows[1][0]])
    return out


def send(name, kind, salt, logprobs=True):
    """One request of the series; returns its record."""
    spec = PROMPTS[name]
    body = {"model": MODEL, "max_tokens": spec["max_tokens"], "temperature": 0,
            "messages": [{"role": "user", "content": spec["content"]}]}
    if spec["sampling"]:
        body.update(SAMPLING)
        body.update(CTK)
    if salt is not None:
        body["cache_salt"] = salt
    if logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = TOP
    t0 = time.perf_counter()
    st, resp = post(body)
    if st == 400 and logprobs:
        # a server that refuses the logprobs fields: retry without them
        rec = send(name, kind, salt, logprobs=False)
        rec["logprobs_refused"] = "{}: {}".format(st, str(resp)[:200])
        return rec
    usage = (resp or {}).get("usage", {})
    details = usage.get("prompt_tokens_details") or {}
    txt = answer_text(resp) if st == 200 else ""
    top = first_token_top(resp) if st == 200 else []
    steps = token_steps(resp) if st == 200 else []
    gaps = [(s[2], s[0]) for s in steps if s[2] is not None]
    rec = {
        "prompt": name, "kind": kind, "salt": salt, "status": st,
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish": ((resp or {}).get("choices", [{}])[0].get("finish_reason")
                   if st == 200 else None),
        "sha": hashlib.sha1(txt.encode()).hexdigest()[:12],
        "chars": len(txt), "head": txt[:48],
        "top": [[t, round(v, 4)] for t, v in top],
        "margin": round(top[0][1] - top[1][1], 4) if len(top) >= 2 else None,
        "steps": steps,
        "min_gap": min(gaps)[0] if gaps else None,
        "min_gap_pos": min(gaps)[1] if gaps else None,
        "dt": round(time.perf_counter() - t0, 2),
        "logprobs": logprobs,
        "usage_keys": sorted(usage) if st == 200 else None,
        "error": None if st == 200 else str(resp)[:200],
    }
    print("P266_SEND {} {:6s} salt={} st={} cached={} ptok={} ctok={} "
          "sha={} margin={} min_gap={}@{} head={!r}".format(
              name, kind, salt, st, rec["cached_tokens"], rec["prompt_tokens"],
              rec["completion_tokens"], rec["sha"], rec["margin"],
              rec["min_gap"], rec["min_gap_pos"], rec["head"]),
          flush=True)
    return rec


def churn():
    """The gate's churn: 4 concurrent unrelated essays, run to completion."""
    out = []

    def one(i):
        body = {"model": MODEL, "stream": False, "temperature": 0.0,
                "max_tokens": CHURN_TOKENS, "seed": 900 + i,
                "messages": [{"role": "user", "content":
                              f"Essay {i}: explain cache coherence in "
                              f"distributed systems, 200 words."}]}
        st, resp = post(body)
        n = (resp or {}).get("usage", {}).get("completion_tokens")
        out.append((st, n))

    ts = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    ok = all(st == 200 and n for st, n in out)
    print("P266_CHURN 4x{} ok={} {}".format(CHURN_TOKENS, ok, out), flush=True)
    return ok


def run_prompt(name, tag):
    recs = []
    after_churn = False
    for kind, suf in SERIES:
        if kind == "churn":
            ok = churn()
            after_churn = True
            recs.append({"prompt": name, "kind": "churn", "ok": ok})
            continue
        salt = f"r266-{tag}-{name}-{suf}" if kind == "salted" else None
        rec = send(name, kind, salt)
        rec["after_churn"] = after_churn
        after_churn = False
        recs.append(rec)
        time.sleep(GAP)
    sends = [r for r in recs if r["kind"] != "churn"]
    salted = [r for r in sends if r["kind"] == "salted"]
    plain = [r for r in sends if r["kind"] == "plain"]
    hs = sorted({r["sha"] for r in salted})
    hp = [r["sha"] for r in plain]
    first_after = [r["sha"] for r in plain if r["after_churn"]]
    warm = hp[0] if hp else None
    # a series with a failed send or a failed churn batch supports no
    # stability claim at all (failed sends hash as the empty string)
    ok = all(r["status"] == 200 for r in sends)
    churn_ok = all(r["ok"] for r in recs if r["kind"] == "churn")
    summ = {
        "prompt": name,
        "prompt_tokens": sends[0]["prompt_tokens"] if sends else None,
        "cold_hashes": hs,
        "warm_hashes": hp,
        "first_after_churn": first_after,
        "salted_cached_tokens": [r["cached_tokens"] for r in salted],
        "plain_cached_tokens": [r["cached_tokens"] for r in plain],
        # (kind, salt, margin, sha) per send, in series order: a flip between
        # two sends is read against the first-token gap they were decided at
        "margins": [[r["kind"], r["salt"], r["margin"], r["sha"]] for r in sends],
        "cold_stable": ok and len(hs) == 1,
        "warm_stable": ok and len(set(hp)) == 1,
        "churn_stable": (ok and churn_ok and bool(first_after)
                         and all(h == warm for h in first_after)),
        "cold_eq_warm": ok and len(hs) == 1 and bool(hp) and all(h == hs[0] for h in hp),
        "ok": ok,
        "churn_ok": churn_ok,
    }
    # the prompt's size against the served block size (the gemma legs run
    # block_size 64): under one block a cached prefix is impossible
    ptok = summ["prompt_tokens"]
    summ["blocks64"] = ptok // 64 if ptok else 0
    print("P266_SERIES {} ptok={} blocks64={} cold={} warm={} first_after_churn={} "
          "cached(salted)={} cached(plain)={}".format(
              name, ptok, summ["blocks64"], hs, hp, first_after,
              summ["salted_cached_tokens"], summ["plain_cached_tokens"]), flush=True)
    print("P266_SERIES {} cold_stable={} warm_stable={} churn_stable={} "
          "cold_eq_warm={} ok={} churn_ok={}".format(
              name, summ["cold_stable"], summ["warm_stable"],
              summ["churn_stable"], summ["cold_eq_warm"], ok, churn_ok), flush=True)
    return summ, recs


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "x"
    out = {"tag": tag, "model": MODEL, "ts": time.strftime("%F %T"),
           "series": [list(s) for s in SERIES], "prompts": {}, "sends": []}
    for name in PROMPTS:
        summ, recs = run_prompt(name, tag)
        out["prompts"][name] = summ
        out["sends"].extend(recs)
    n = len(PROMPTS)
    agg = {
        "cold_stable": sum(s["cold_stable"] for s in out["prompts"].values()),
        "warm_stable": sum(s["warm_stable"] for s in out["prompts"].values()),
        "churn_stable": sum(s["churn_stable"] for s in out["prompts"].values()),
        "cold_eq_warm": sum(s["cold_eq_warm"] for s in out["prompts"].values()),
        "all_200": all(s["ok"] for s in out["prompts"].values()),
    }
    salted_cold = all(
        (c in (0, None)) for s in out["prompts"].values()
        for c in s["salted_cached_tokens"])
    print("P266_DETERMINISM {}: cold_stable={}/{} warm_stable={}/{} "
          "churn_stable={}/{} cold_eq_warm={}/{} all_200={} "
          "salt_forces_cold={}".format(
              tag, agg["cold_stable"], n, agg["warm_stable"], n,
              agg["churn_stable"], n, agg["cold_eq_warm"], n, agg["all_200"],
              salted_cold), flush=True)
    out["verdict"] = dict(agg, n=n, salt_forces_cold=salted_cold)
    with open(f"p266_determinism_{tag}.json", "w", newline="\n") as f:
        json.dump(out, f, indent=1)
    return 0 if agg["all_200"] else 1


if __name__ == "__main__":
    sys.exit(main())
