#!/usr/bin/env python3
"""p140_probe.py -- r140c API-boundary probe (runs INSIDE the arc
container). Pinpoints the exact request-length rule behind the gm4x
131K HTTP 400s: builds a budget-sized prompt through the same
rng/tokenizer path as p139_conc.py, POSTs it with a completion budget,
and banks {budget, max_tokens, client_len, status, body} -- a 400 body
states the server's own length math verbatim.

Usage:
  /opt/venv/bin/python /root/p140_probe.py <budget> <max_tokens> <tag>
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zlib

from transformers import AutoTokenizer

URL = "http://127.0.0.1:8000/v1/completions"
TOKDIR = "/models/target"


def build_prompt(tok, n, tag, slot):
    rng = random.Random(20261111 ^ zlib.crc32(tag.encode()) ^ (slot * 7919))
    ids = [rng.randrange(1000, 50000) for _ in range(n - 8)]
    text = tok.decode(ids)
    m = len(tok(text).input_ids)
    while m > n and len(ids) > 16:
        ids = ids[: max(16, len(ids) - (m - n) - 8)]
        text = tok.decode(ids)
        m = len(tok(text).input_ids)
    return text, m


def main():
    budget, mt, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    tok = AutoTokenizer.from_pretrained(TOKDIR, trust_remote_code=True)
    text, client_len = build_prompt(tok, budget, tag, 0)
    body = json.dumps({"model": "gemma4-31b", "prompt": text,
                       "max_tokens": mt, "temperature": 0.0,
                       "ignore_eos": True, "stream": False}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    status, snip = 0, ""
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            status = r.status
            snip = r.read(300).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        status = e.code
        snip = e.read(1200).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 -- bank the refusal verbatim
        status = -1
        snip = "{}: {}".format(type(e).__name__, str(e)[:400])
    out = {"tag": tag, "budget": budget, "max_tokens": mt,
           "client_len": client_len, "status": status,
           "elapsed_s": round(time.time() - t0, 2), "body": snip}
    with open("/root/p140_probe_{}.json".format(tag), "w") as f:
        json.dump(out, f, indent=1)
    print("PROBE_DONE {} status={} client_len={} elapsed={:.1f}".format(
        tag, status, client_len, out["elapsed_s"]))


if __name__ == "__main__":
    sys.exit(main())
