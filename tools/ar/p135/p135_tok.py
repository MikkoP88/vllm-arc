#!/usr/bin/env python3
"""p135_tok.py -- P6.35 TTFT lane: standalone tokenizer/detokenizer cost for
the face prompts (same seeded-random-token law as p135_face). Runs INSIDE
the container. The u8k face showed TTFT 4.55 s with only ~1.05 s of engine
(prefill exec + first decode); this bench attributes the rest.
"""
import json
import random
import time

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("/models/target", trust_remote_code=True)


def build_text(n):
    rng = random.Random(20261110)
    ids = [rng.randrange(1000, 50000) for _ in range(n - 8)]
    return tok.decode(ids)


out = {}
for n in (2048, 8192, 32768):
    text = build_text(n)
    t0 = time.perf_counter()
    enc = tok(text)["input_ids"]
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    tok(text)
    warm = time.perf_counter() - t0
    # detokenize: full 600-token decode + incremental streaming (1 + a few
    # tokens per chunk, ~150 chunks)
    rng = random.Random(7)
    gen = [rng.randrange(1000, 50000) for _ in range(600)]
    t0 = time.perf_counter()
    tok.decode(gen)
    full = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(596):
        tok.decode(gen[i:i + 4])
    incr = time.perf_counter() - t0
    out[n] = {"enc_cold_s": round(cold, 3), "enc_warm_s": round(warm, 3),
              "n_enc": len(enc), "dec_full600_s": round(full, 3),
              "dec_incr596x4_s": round(incr, 3)}
    print(n, out[n])

with open("/root/p135_tok.json", "w") as f:
    json.dump(out, f, indent=1)
print("P135TOK_DONE")
