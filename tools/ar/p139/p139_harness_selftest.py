#!/usr/bin/env python3
"""p139_harness_selftest.py -- offline selftest of the report harnesses
tools/ar/p138/p138_face.py and tools/ar/p139/p139_conc.py (local: no GPU,
no vLLM, no transformers).

A mock engine on the loopback implements just what the harnesses read:
  POST /v1/completions   streaming SSE chunks; a toy 64-token-block prefix
                         cache: every full block of a prompt seen before is
                         a hit, and a fully cached prompt recomputes its
                         last block (the vLLM rule)
  GET  /metrics          the vLLM 0.30 counter names the cache proof reads;
                         the finished-request counters flush 1.5 s after a
                         request completes (the engine's stats lag), or
                         never in no-flush mode
A fake `transformers` module stands in for the tokenizer (ids <-> words).

Proves, per case: the cold request keeps its file name and legacy fields
and gains the cache proof with ZERO hits; the warm twin re-sends the
byte-identical prompt, lands in the `_warm` file and banks the expected
hit count; an invalid pass label exits 2 without a file; the flush wait
honors the stats lag and times out cleanly when the counters never
flush; a partially failed group waits only for its completed requests.

Usage: python tools/ar/p139/p139_harness_selftest.py
Exit 0 = every case holds.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
P138 = os.path.join(HERE, "..", "p138", "p138_face.py")
P139 = os.path.join(HERE, "p139_conc.py")
BLOCK = 64
FLUSH_DELAY = 1.5

FAKE_TRANSFORMERS = '''
import types


class _Tok:
    def decode(self, ids):
        return " ".join("tok%d" % i for i in ids)

    def __call__(self, text):
        return types.SimpleNamespace(input_ids=[int(w[3:]) for w in text.split()])


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path, trust_remote_code=False):
        return _Tok()
'''

FACE_KEYS = {"tag", "ctx", "max_tokens", "model", "temperature", "n_stream_chunks",
             "t_send_epoch", "ttft_s", "total_s", "itl_ms_mean", "itl_ms_p50",
             "itl_ms_p90", "itl_ms_p99", "itl_ms_max", "itl_ms_all"}
CONC_KEYS = {"tag", "ctx", "conc", "max_tokens", "model", "temperature", "n_ok",
             "n_fail", "requests", "ttft_mean_s", "ttft_max_s", "per_user_tps",
             "per_user_mean_tps", "per_user_min_tps", "agg_tps_wall", "wall_s"}
COUNTERS = ["prefix_cache_queries_total", "prefix_cache_hits_total", "prompt_tokens_total",
            "prompt_tokens_cached_total", "external_prefix_cache_queries_total",
            "external_prefix_cache_hits_total", "request_success_total", "num_preempted_total"]


class Engine:
    def __init__(self, flush=True, fail_after=None):
        self.lock = threading.Lock()
        self.blocks = set()
        self.m = {name: 0.0 for name in COUNTERS}
        self.flush = flush
        self.fail_after = fail_after      # None, or the ordinal of the request to refuse
        self.n_req = 0
        self.prompts = []                 # every prompt text, in arrival order

    def admit(self, prompt):
        """The toy prefix cache: returns (tokens queried, tokens hit)."""
        words = prompt.split()
        with self.lock:
            self.n_req += 1
            self.prompts.append(prompt)
            refuse = self.fail_after is not None and self.n_req == self.fail_after
            if refuse:
                return None
            full = len(words) // BLOCK * BLOCK
            hits = 0
            for i in range(0, full, BLOCK):
                if tuple(words[i:i + BLOCK]) in self.blocks:
                    hits += BLOCK
            if full > 0 and hits == full and len(words) % BLOCK == 0:
                hits -= BLOCK            # a fully cached prompt recomputes its last block
            for i in range(0, full, BLOCK):
                self.blocks.add(tuple(words[i:i + BLOCK]))
            self.m["prefix_cache_queries_total"] += len(words)
            self.m["prefix_cache_hits_total"] += hits
            return len(words), hits

    def complete(self, n_prompt, hits):
        """The finished-request counters, flushed after the stats lag."""
        def _flush():
            with self.lock:
                self.m["request_success_total"] += 1
                self.m["prompt_tokens_total"] += n_prompt
                self.m["prompt_tokens_cached_total"] += hits
        if self.flush:
            threading.Timer(FLUSH_DELAY, _flush).start()

    def metrics_text(self):
        with self.lock:
            vals = dict(self.m)
        out = []
        for name in COUNTERS:
            if name == "request_success_total":
                # two label series: the proof must sum over them
                out.append('vllm:request_success_total{engine="0",finished_reason="length",'
                           'model_name="mock"} ' + str(int(vals[name])))
                out.append('vllm:request_success_total{engine="0",finished_reason="stop",'
                           'model_name="mock"} 0')
            else:
                out.append('vllm:{}{{engine="0",model_name="mock"}} {}'.format(
                    name, vals[name]))
        out.append("vllm:some_gauge{engine=\"0\"} 3.0")   # a non-counter line: must be ignored
        return "\n".join(out) + "\n"


class Handler(BaseHTTPRequestHandler):
    engine = None

    def log_message(self, *a):
        pass

    def _reply(self, code, body=b"", ctype="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self):
        if self.path == "/metrics":
            self._reply(200, self.engine.metrics_text().encode())
        elif self.path == "/health":
            self._reply(200, b"ok")
        else:
            self._reply(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        admitted = self.engine.admit(body["prompt"])
        if admitted is None:              # the refused slot: the engine closes the socket
            self.close_connection = True
            return
        n_prompt, hits = admitted
        time.sleep(0.05 + 0.0002 * (n_prompt - hits))   # TTFT ~ the uncached tokens
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for _i in range(int(body["max_tokens"])):
            self.wfile.write(b'data: {"choices":[{"text":"x"}]}\n\n')
            self.wfile.flush()
            time.sleep(0.004)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.engine.complete(n_prompt, hits)


class Mock:
    def __init__(self, **kw):
        self.engine = Engine(**kw)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        Handler.engine = self.engine
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:{}".format(self.server.server_address[1])

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def run(script, base, outdir, fake_path, args):
    env = dict(os.environ)
    env["PYTHONPATH"] = fake_path
    pre = "P138" if script is P138 else "P139"
    env[pre + "_BASE"] = base
    env[pre + "_OUT"] = outdir
    env[pre + "_TOKDIR"] = "/nonexistent-must-not-be-opened"
    env.pop("PYTHONSTARTUP", None)
    p = subprocess.run([sys.executable, script] + [str(a) for a in args],
                       env=env, capture_output=True, text=True, timeout=120)
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

    tmp = tempfile.mkdtemp(prefix="p139_selftest_")
    fake = os.path.join(tmp, "fake")
    out = os.path.join(tmp, "out")
    os.makedirs(fake)
    os.makedirs(out)
    with open(os.path.join(fake, "transformers.py"), "w", encoding="utf-8") as fh:
        fh.write(FAKE_TRANSFORMERS)

    # ---------------- flush mode: the engine's stats land 1.5 s late
    mock = Mock()
    base = mock.base

    print("[1] p138 solo face, default pass (6 argv): cold by default")
    rc, log = run(P138, base, out, fake, [200, 20, "cellA_c200_t0.0", "mock", "0.0"])
    d = load(os.path.join(out, "p138_face_cellA_c200_t0.0.json"))
    check("rc", rc == 0, log)
    check("legacy fields intact", set(d) >= FACE_KEYS, str(sorted(set(d))))
    check("pass default", d["pass"] == "cold", d["pass"])
    c = d["cache"]
    check("cold: zero hits", c["prefix_cache_hits"] == 0, c)
    check("cold: queries = prompt tokens", c["prefix_cache_queries"] == 192, c)
    check("cold: finished-request proof", c["request_success"] == 1
          and c["prompt_tokens"] == 192 and c["prompt_tokens_cached"] == 0, c)
    check("flush wait honored the lag", c["stats_flushed"] and c["stats_wait_s"] >= FLUSH_DELAY - 0.3, c)
    check("no external store on the mock", c["external_queries"] == 0 and c["external_hits"] == 0, c)
    cold_ttft = d["ttft_s"]

    print("[2] p138 warm twin (same tag): same prompt, _warm file, the hit")
    rc, log = run(P138, base, out, fake, [200, 20, "cellA_c200_t0.0", "mock", "0.0", "warm"])
    d = load(os.path.join(out, "p138_face_cellA_c200_t0.0_warm.json"))
    check("rc", rc == 0, log)
    check("pass warm", d["pass"] == "warm", d["pass"])
    c = d["cache"]
    check("warm: fully cached prompt recomputes one block",
          c["prefix_cache_queries"] == 192 and c["prefix_cache_hits"] == 128, c)
    check("warm: finished-request proof", c["prompt_tokens_cached"] == 128, c)
    check("warm TTFT below cold (uncached tokens only)", 0 < d["ttft_s"] < cold_ttft,
          "{} vs {}".format(d["ttft_s"], cold_ttft))
    check("same prompt bytes", mock.engine.prompts[0] == mock.engine.prompts[1])

    print("[3] p138 partial-block prompt (242 tokens: 3 full blocks + 50)")
    rc, log = run(P138, base, out, fake, [250, 20, "cellB_c250_t0.0", "mock", "0.0", "cold"])
    d = load(os.path.join(out, "p138_face_cellB_c250_t0.0.json"))
    check("rc", rc == 0, log)
    check("cold: zero hits", d["cache"]["prefix_cache_hits"] == 0, d["cache"])
    check("cold: queries = prompt tokens", d["cache"]["prefix_cache_queries"] == 242, d["cache"])
    rc, log = run(P138, base, out, fake, [250, 20, "cellB_c250_t0.0", "mock", "0.0", "warm"])
    d = load(os.path.join(out, "p138_face_cellB_c250_t0.0_warm.json"))
    check("rc", rc == 0, log)
    check("warm: hits stop at the full blocks",
          d["cache"]["prefix_cache_hits"] == 192 and d["cache"]["prefix_cache_queries"] == 242,
          d["cache"])

    print("[4] p138 invalid pass label: exit 2, no file")
    rc, log = run(P138, base, out, fake, [200, 20, "cellX_c200_t0.0", "mock", "0.0", "hot"])
    check("rc 2", rc == 2, "rc={} log={}".format(rc, log))
    check("FACE_PASS_INVALID printed", "FACE_PASS_INVALID" in log, log)
    check("no file written", not os.path.exists(os.path.join(out, "p138_face_cellX_c200_t0.0.json")))

    print("[5] p139 concurrency group, cold (n=3): zero hits, per-slot unique prompts")
    rc, log = run(P139, base, out, fake, [300, 20, 3, "cellC_c300_n3", "mock", "0.0"])
    d = load(os.path.join(out, "p139_conc_cellC_c300_n3.json"))
    check("rc", rc == 0, log)
    check("legacy fields intact", set(d) >= CONC_KEYS, str(sorted(set(d))))
    check("pass default", d["pass"] == "cold", d["pass"])
    check("n_ok 3", d["n_ok"] == 3 and d["n_fail"] == 0, (d["n_ok"], d["n_fail"]))
    c = d["cache"]
    check("cold: zero hits over the group", c["prefix_cache_hits"] == 0, c)
    # per slot: the budget ctx - mt - 64 = 216 minus the harness's own
    # 8-token margin -> 208 prompt tokens
    check("queries = 3 x (ctx - mt - 64 - 8)", c["prefix_cache_queries"] == 3 * 208, c)
    check("finished-request proof", c["request_success"] == 3, c)
    trio = mock.engine.prompts[-3:]
    check("per-slot prompts unique", len(set(trio)) == 3)
    cold_mean = d["ttft_mean_s"]

    print("[6] p139 warm twin: the same three prompts, every slot hits")
    rc, log = run(P139, base, out, fake, [300, 20, 3, "cellC_c300_n3", "mock", "0.0", "warm"])
    d = load(os.path.join(out, "p139_conc_cellC_c300_n3_warm.json"))
    check("rc", rc == 0, log)
    check("n_ok 3", d["n_ok"] == 3 and d["n_fail"] == 0, (d["n_ok"], d["n_fail"]))
    check("warm: 3 x 192 hit (the 24-token partial tails stay uncached)",
          d["cache"]["prefix_cache_hits"] == 3 * 192, d["cache"])
    check("warm TTFT mean below cold", 0 < d["ttft_mean_s"] < cold_mean,
          "{} vs {}".format(d["ttft_mean_s"], cold_mean))
    check("same prompts as the cold group", sorted(mock.engine.prompts[-3:]) == sorted(trio))
    mock.stop()

    print("[7] no-flush mode: the proof times out cleanly, prefix-cache stats still banked")
    mock = Mock(flush=False)
    rc, log = run(P138, mock.base, out, fake, [200, 20, "cellD_c200_t0.0", "mock", "0.0"])
    d = load(os.path.join(out, "p138_face_cellD_c200_t0.0.json"))
    check("cold rc", rc == 0, log)
    check("cold not flushed", d["cache"]["stats_flushed"] is False, d["cache"])
    check("cold wait capped near STATS_WAIT_S", 14.0 <= d["cache"]["stats_wait_s"] <= 17.0, d["cache"])
    check("cold zero hits", d["cache"]["prefix_cache_hits"] == 0, d["cache"])
    rc, log = run(P138, mock.base, out, fake, [200, 20, "cellD_c200_t0.0", "mock", "0.0", "warm"])
    d = load(os.path.join(out, "p138_face_cellD_c200_t0.0_warm.json"))
    check("warm rc", rc == 0, log)
    check("warm not flushed", d["cache"]["stats_flushed"] is False, d["cache"])
    check("warm wait capped near STATS_WAIT_S", 14.0 <= d["cache"]["stats_wait_s"] <= 17.0, d["cache"])
    check("prefix-cache stats still banked", d["cache"]["prefix_cache_hits"] == 128, d["cache"])
    check("finished-request deltas banked as 0 (not yet flushed)", d["cache"]["request_success"] == 0,
          d["cache"])
    mock.stop()

    print("[8] partially failed group: the proof waits for the completed requests only")
    mock = Mock(fail_after=2)
    rc, log = run(P139, mock.base, out, fake, [300, 20, 3, "cellE_c300_n3", "mock", "0.0"])
    d = load(os.path.join(out, "p139_conc_cellE_c300_n3.json"))
    check("rc", rc == 0, log)
    check("n_ok 2 n_fail 1", d["n_ok"] == 2 and d["n_fail"] == 1, (d["n_ok"], d["n_fail"]))
    check("proof covers the two completed requests",
          d["cache"]["request_success"] == 2 and d["cache"]["stats_flushed"], d["cache"])
    check("the refused slot never reached the cache", d["cache"]["prefix_cache_queries"] == 2 * 208,
          d["cache"])
    mock.stop()

    print("[9] dead endpoint: p139 banks the per-slot errors, the proof is None-valued")
    dead = Mock()
    port = dead.server.server_address[1]
    dead.stop()
    rc, log = run(P139, "http://127.0.0.1:{}".format(port), out, fake,
                  [300, 20, 2, "cellF_c300_n2", "mock", "0.0"])
    d = load(os.path.join(out, "p139_conc_cellF_c300_n2.json"))
    check("rc", rc == 0, log)
    check("n_ok 0 n_fail 2", d["n_ok"] == 0 and d["n_fail"] == 2, (d["n_ok"], d["n_fail"]))
    check("no wait on a failed group", not d["cache"]["stats_flushed"] and d["cache"]["stats_wait_s"] < 5,
          d["cache"])
    check("counter deltas None", d["cache"]["prefix_cache_hits"] is None, d["cache"])

    if fails:
        print("HARNESS_SELFTEST FAIL n_cases={} fails={}".format(n_cases, len(fails)))
        return 1
    print("HARNESS_SELFTEST PASS n_cases={}".format(n_cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
