#!/usr/bin/env python3
"""p266_probe_selftest.py -- offline selftest of the cache-state
determinism probe tools/ar/p266/p266_determinism_probe.py (local: no GPU,
no vLLM).

A mock engine on the loopback implements just what the probe reads:
  POST /v1/chat/completions
      a toy 16-token-block prefix cache with the engine's shape: a plain
      request reports usage.prompt_tokens_details.cached_tokens for every
      leading full block it has seen before; a salted request (cache_salt)
      reports no details object and stores nothing a plain request hits
      (words stand in for tokens)
      every answer is five tokens with top logprobs: the first token is
      decided at a gap of 3.2, the other four at 1.9
  the answer text is a function of the mock's MODE:
      faithful    the prompt alone decides it (a healthy lane)
      cache_flip  a request with a cache hit answers differently from a
                  cold one (a cold-vs-warm numeric fork)
      churn_flip  the first plain request after an essay batch answers
                  differently (the r264 mkg shape: r1 differs, r2 is back)
      no_logprobs faithful text, HTTP 400 whenever the request asks for
                  logprobs (the probe must drop the fields and go on)

Proves, per mode, that the probe's verdict lines say exactly what the
mock did: a healthy lane is stable on all four axes and the salted sends
report no cached tokens; the short sanity prompt reports no cached token
while the probe's long prompt (many blocks) hits every full block from
its second send on; a cold/warm fork is charged to cold_eq_warm (and to
warm/churn stability) and not to cold stability; the first-after-churn
flip is charged to churn stability; a refused logprobs request is
retried and banked as such; a dead endpoint exits 1 with the JSON still
written.

Usage: python tools/ar/p266/p266_probe_selftest.py
Exit 0 = every case holds.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "p266_determinism_probe.py")
BLOCK = 16         # the mock's block (the rig's gemma legs run 64)
MODES = ("faithful", "cache_flip", "churn_flip", "no_logprobs")
N_PROMPTS = 4
N_SALTED = 4      # per prompt, from the probe's SERIES
N_PLAIN = 8
N_CHURN = 2


def probe_words(name):
    """The probe's own prompt, counted the way the mock counts tokens."""
    spec = importlib.util.spec_from_file_location("p266_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return len(mod.PROMPTS[name]["content"].split())


N_LONG = probe_words("long")


class Engine:
    def __init__(self, mode):
        self.lock = threading.Lock()
        self.mode = mode
        self.blocks = set()          # full plain blocks seen (leading-prefix trie)
        self.after_churn = False     # an essay batch finished since the last non-essay request
        self.n_req = 0

    def answer(self, body):
        """Returns (status, response-dict)."""
        content = body["messages"][0]["content"]
        words = content.split()
        n_tok = max(1, len(words))
        first_after_churn = False
        with self.lock:
            self.n_req += 1
            if content.startswith("Essay "):
                self.after_churn = True
            else:
                first_after_churn = self.after_churn
                self.after_churn = False
            salted = body.get("cache_salt") is not None
            hits = 0
            if not salted:
                full = n_tok // BLOCK * BLOCK
                for i in range(0, full, BLOCK):
                    if tuple(words[i:i + BLOCK]) in self.blocks:
                        hits += BLOCK
                    else:
                        break
                for i in range(0, full, BLOCK):
                    self.blocks.add(tuple(words[i:i + BLOCK]))
        if self.mode == "no_logprobs" and body.get("logprobs"):
            return 400, {"error": {"message": "logprobs are disabled on this mock",
                                   "type": "BadRequestError"}}
        text = "{}:stable".format(content[:12])
        if self.mode == "cache_flip":
            text = "{}:{}".format(content[:12], "warm" if hits else "cold")
        elif self.mode == "churn_flip" and first_after_churn:
            text = "{}:post-churn".format(content[:12])
        usage = {"prompt_tokens": n_tok, "completion_tokens": 5,
                 "total_tokens": n_tok + 5}
        if not salted:
            usage["prompt_tokens_details"] = {"cached_tokens": hits}
        # five answer tokens: the first is decided at a gap of 3.2, the
        # other four at 1.9 (the smallest gap of an answer is not always
        # at its first token)
        steps = [{"token": "A", "logprob": -0.05, "top_logprobs": [
            {"token": "A", "logprob": -0.05},
            {"token": "B", "logprob": -3.25},
            {"token": "C", "logprob": -6.0}]}]
        steps += [{"token": "B", "logprob": -0.10, "top_logprobs": [
            {"token": "B", "logprob": -0.10},
            {"token": "D", "logprob": -2.00}]} for _ in range(4)]
        choice = {"message": {"role": "assistant", "content": text},
                  "finish_reason": "stop", "logprobs": None}
        if body.get("logprobs"):
            choice["logprobs"] = {"content": steps}
        return 200, {"choices": [choice], "usage": usage}


class Handler(BaseHTTPRequestHandler):
    engine = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        code, resp = self.engine.answer(body)
        payload = json.dumps(resp).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class Mock:
    def __init__(self, mode):
        self.engine = Engine(mode)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Handler.engine = self.engine
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:{}".format(self.server.server_address[1])

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def run_probe(base, outdir, tag):
    env = dict(os.environ)
    env["P266_BASE"] = base
    env["P266_GAP"] = "0"
    env["ARC_PROBE_MODEL"] = "mock"
    env.pop("PYTHONSTARTUP", None)
    p = subprocess.run([sys.executable, PROBE, tag], cwd=outdir, env=env,
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    fails = []
    n_cases = 0

    def check(label, cond, detail=""):
        nonlocal n_cases
        n_cases += 1
        if not cond:
            fails.append("{}: {}".format(label, detail))
            print("  FAIL {} -- {}".format(label, detail))
        else:
            print("  ok   {}".format(label))

    tmp = tempfile.mkdtemp(prefix="p266_selftest_")

    print("[1] faithful engine: stable on every axis")
    mock = Mock("faithful")
    rc, log = run_probe(mock.base, tmp, "t1")
    path = os.path.join(tmp, "p266_determinism_t1.json")
    check("rc", rc == 0, log[-600:])
    check("json written", os.path.exists(path), path)
    d = load(path)
    v = d["verdict"]
    check("cold_stable on every prompt", v["cold_stable"] == N_PROMPTS, v)
    check("warm_stable on every prompt", v["warm_stable"] == N_PROMPTS, v)
    check("churn_stable on every prompt", v["churn_stable"] == N_PROMPTS, v)
    check("cold_eq_warm on every prompt", v["cold_eq_warm"] == N_PROMPTS, v)
    check("all_200", v["all_200"] is True, v)
    check("salt forces cold", v["salt_forces_cold"] is True, v)
    sends = [r for r in d["sends"] if r["kind"] != "churn"]
    churns = [r for r in d["sends"] if r["kind"] == "churn"]
    check("series size: 12 sends x every prompt", len(sends) == 12 * N_PROMPTS, len(sends))
    check("series size: 2 churns x every prompt",
          len(churns) == N_CHURN * N_PROMPTS and all(c["ok"] for c in churns), churns)
    check("salted sends: no cached-token details",
          all(r["cached_tokens"] is None for r in sends if r["kind"] == "salted"),
          [r["cached_tokens"] for r in sends if r["kind"] == "salted"])
    s = d["prompts"]["sanity"]
    check("sanity (6 tokens, below one block): every plain send reports 0",
          s["plain_cached_tokens"] == [0] * N_PLAIN, s["plain_cached_tokens"])
    p = d["prompts"]["simple"]
    check("simple: first plain send is a miss, every later one hits one block",
          p["plain_cached_tokens"] == [0] + [BLOCK] * (N_PLAIN - 1), p["plain_cached_tokens"])
    full = N_LONG // BLOCK * BLOCK
    p = d["prompts"]["long"]
    check("long: the prompt spans many blocks (the only prompt a 64-token-block "
          "engine can cache)", N_LONG >= 128 and p["prompt_tokens"] == N_LONG,
          (N_LONG, p["prompt_tokens"]))
    check("long: first plain send is a miss, every later one hits every full block",
          p["plain_cached_tokens"] == [0] + [full] * (N_PLAIN - 1),
          (full, p["plain_cached_tokens"]))
    check("blocks64 read from the reported prompt tokens",
          [d["prompts"][n]["blocks64"] for n in d["prompts"]]
          == [probe_words(n) // 64 for n in d["prompts"]],
          [(n, d["prompts"][n]["blocks64"]) for n in d["prompts"]])
    check("cached tokens never exceed the prompt",
          all(c is None or c <= r["prompt_tokens"] for r in sends
              for c in [r["cached_tokens"]]), "cached > prompt")
    check("first-token margins banked (3.2 on the mock)",
          all(r["margin"] == 3.2 and len(r["top"]) == 3 for r in sends),
          [(r["kind"], r["margin"]) for r in sends if r["margin"] != 3.2][:3])
    check("every step's gap banked: five steps, the smallest (1.9) at position 1",
          all(len(r["steps"]) == 5 and r["min_gap"] == 1.9 and r["min_gap_pos"] == 1
              and r["steps"][0][2] == 3.2 and r["steps"][1] == [1, "B", 1.9, "D"]
              for r in sends),
          [(r["steps"], r["min_gap"], r["min_gap_pos"]) for r in sends
           if r["min_gap"] != 1.9][:1])
    check("two first-after-churn sends flagged per prompt",
          all(sum(1 for r in sends if r["prompt"] == n and r["after_churn"]) == N_CHURN
              for n in d["prompts"]), "after_churn flags")
    check("per-send lines printed",
          log.count("P266_SEND ") == 12 * N_PROMPTS and log.count("P266_CHURN ") == N_CHURN * N_PROMPTS,
          (log.count("P266_SEND "), log.count("P266_CHURN ")))
    check("verdict line printed",
          "P266_DETERMINISM t1: cold_stable={}/{}".format(N_PROMPTS, N_PROMPTS) in log,
          log[-300:])
    mock.stop()

    print("[2] cache_flip engine: a warm request answers differently from a cold one")
    mock = Mock("cache_flip")
    rc, log = run_probe(mock.base, tmp, "t2")
    d = load(os.path.join(tmp, "p266_determinism_t2.json"))
    v = d["verdict"]
    check("rc", rc == 0, log[-600:])
    check("cold stays stable", v["cold_stable"] == N_PROMPTS, v)
    check("cold_eq_warm fails on the three cacheable prompts", v["cold_eq_warm"] == 1, v)
    check("warm stability fails (first plain send is still cold)", v["warm_stable"] == 1, v)
    check("churn stability fails (first send after churn is warm)", v["churn_stable"] == 1, v)
    check("the sanity prompt cannot hit the cache: fully stable",
          all(d["prompts"]["sanity"][k] for k in
              ("cold_stable", "warm_stable", "churn_stable", "cold_eq_warm")),
          d["prompts"]["sanity"])
    check("simple: cold hash != warm hash",
          d["prompts"]["simple"]["cold_hashes"][0] != d["prompts"]["simple"]["warm_hashes"][-1],
          d["prompts"]["simple"])
    check("long: cold hash != warm hash",
          d["prompts"]["long"]["cold_hashes"][0] != d["prompts"]["long"]["warm_hashes"][-1],
          d["prompts"]["long"])
    mock.stop()

    print("[3] churn_flip engine: the first plain send after an essay batch differs")
    mock = Mock("churn_flip")
    rc, log = run_probe(mock.base, tmp, "t3")
    d = load(os.path.join(tmp, "p266_determinism_t3.json"))
    v = d["verdict"]
    check("rc", rc == 0, log[-600:])
    check("cold stays stable", v["cold_stable"] == N_PROMPTS, v)
    check("churn_stable on no prompt", v["churn_stable"] == 0, v)
    check("warm_stable on no prompt", v["warm_stable"] == 0, v)
    check("cold_eq_warm on no prompt", v["cold_eq_warm"] == 0, v)
    check("ok and churn_ok hold on every prompt (the flip is a finding, not a failure)",
          all(p["ok"] and p["churn_ok"] for p in d["prompts"].values()),
          [(p["ok"], p["churn_ok"]) for p in d["prompts"].values()])
    # plain sends in series order: the 3rd and 5th are the first after a churn
    hp = d["prompts"]["simple"]["warm_hashes"]
    check("simple: warm series is stable / stable / post-churn / stable / post-churn / ...",
          hp[0] == hp[1] == hp[3] == hp[5] == hp[6] == hp[7] and hp[2] == hp[4] != hp[0], hp)
    heads = {}
    for r in d["sends"]:
        if r["kind"] != "churn":
            heads.setdefault(r["prompt"], []).append(r["head"])
    # all sends in series order: the 5th and 7th are the first after a churn
    check("every first-after-churn head carries the post-churn tag, none else",
          all(h.endswith("post-churn") == (i in (4, 6))
              for hs in heads.values() for i, h in enumerate(hs)),
          heads.get("simple"))
    check("cached tokens identical to the faithful run (the flip is not a cache effect)",
          d["prompts"]["simple"]["plain_cached_tokens"] == [0] + [BLOCK] * (N_PLAIN - 1),
          d["prompts"]["simple"]["plain_cached_tokens"])
    mock.stop()

    print("[4] no_logprobs engine: the probe drops the fields and goes on")
    mock = Mock("no_logprobs")
    rc, log = run_probe(mock.base, tmp, "t4")
    d = load(os.path.join(tmp, "p266_determinism_t4.json"))
    sends = [r for r in d["sends"] if r["kind"] != "churn"]
    check("rc", rc == 0, log[-600:])
    check("every send answered 200", all(r["status"] == 200 for r in sends),
          [r["status"] for r in sends if r["status"] != 200][:3])
    check("every send was retried without logprobs",
          all(r["logprobs"] is False and str(r["logprobs_refused"]).startswith("400")
              for r in sends),
          [(r["logprobs"], r.get("logprobs_refused")) for r in sends][:3])
    check("the refusal is banked apart from the send's own error field",
          all(r["error"] is None and "logprobs are disabled" in r["logprobs_refused"]
              for r in sends),
          [(r["error"], r["logprobs_refused"]) for r in sends][:1])
    check("no margins, no tops, no steps",
          all(r["margin"] is None and r["top"] == [] and r["steps"] == []
              and r["min_gap"] is None and r["min_gap_pos"] is None for r in sends), "")
    check("verdict unaffected (faithful text)", d["verdict"]["cold_eq_warm"] == N_PROMPTS, d["verdict"])
    mock.stop()

    print("[5] dead endpoint: rc 1, the JSON is still written")
    dead = Mock("faithful")
    port = dead.server.server_address[1]
    dead.stop()
    time.sleep(0.3)
    rc, log = run_probe("http://127.0.0.1:{}".format(port), tmp, "t5")
    check("rc 1", rc == 1, "rc={} log={}".format(rc, log[-300:]))
    check("json written", os.path.exists(os.path.join(tmp, "p266_determinism_t5.json")), tmp)
    d = load(os.path.join(tmp, "p266_determinism_t5.json"))
    check("all_200 False", d["verdict"]["all_200"] is False, d["verdict"])
    check("each prompt's ok False", all(not p["ok"] for p in d["prompts"].values()), "")
    check("no stability claim from failed sends",
          all(d["verdict"][k] == 0 for k in ("cold_stable", "warm_stable", "churn_stable",
                                             "cold_eq_warm")), d["verdict"])
    check("the dead endpoint is reported on the send line",
          "st=-1" in log and "st=200" not in log, log[-300:])

    if fails:
        print("P266_SELFTEST FAIL n_cases={} fails={}".format(n_cases, len(fails)))
        return 1
    print("P266_SELFTEST PASS n_cases={}".format(n_cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
