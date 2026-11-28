#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p268c_disc.py — P6.47 P-D (r268c): the trace discriminator set.

Runs on the r268c boot (e2/c3 posture + the r268c_overlay whose engine.py
hardcodes _TRACE=True — the ARC_MXKV_TRACE dead-letter fix). The overlay
prints, per store call, the slot ids the engine handed the store (smap)
and, per decode call at B<=4, the block-table rows/stride/seq_lens it
hands the walk (decx). r267 proved the kernel math exact for CORRECT
metadata, and r268b proved the corruption request-conditional -- so the
decx/smap values on a RED request vs a CLEAN request NAME the wrong
metadata directly.

The set is SEQUENTIAL (T0..T5; T6 is one 2-way burst) so every request's
trace section is bounded by the API-server POST lines in serve.log:

  T0  control raw "The capital of France is" (mt=8) -- the r268 CLEAN
      class: the trace reference for CORRECT addressing
  T1  raw FILLER x12 (ptok=122, mt=16) -- the A_r12 sequential RED
      (total 138 crosses the 2->3 page boundary mid-decode)
  T2  gate-battery chat (temp=0, mt=64, no logprobs) -- the 'Paris our
      our' finite-wrong class
  T3  card chat full params (mt=96) -- the 400-nan class
  T4  raw the probe's own g4 'health' prompt (ntok=20, NON-crossing at
      mt=96: 20+96=116 < 128) with the probe's exact raw body -- the
      templated non-crossing RED that kills the page-boundary law
  T5  raw the probe's own g3 'capital_france' prompt (ntok=40, crossing
      at mt=96) -- the templated crossing RED
  T6  the C1 2-way burst (FILLER x3/x4 + tail, mt=16) -- the concurrency
      RED pair

Prompts for T4/T5 are read byte-exact from the r268b bank's matrix json
(formats.<fmt>.<pid>.prompt) -- the same strings the 216/216-RED probe
sent. Exit 0 = set complete (red or not -- the JSON is the evidence).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
FILLER = "The quick brown fox jumps over the lazy dog. "
CONTROL = "The capital of France is"
MATRIX = "/root/build/p647/r268b_bisect/out/matrix_r268b_bisect.json"

REC: list[dict] = []
LOCK = threading.Lock()


def post(path: str, body: dict, timeout: int = 180):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time() - t0
    except urllib.error.HTTPError as e:
        raw = e.read().decode() or "{}"
        try:
            return e.code, json.loads(raw), time.time() - t0
        except json.JSONDecodeError:
            return e.code, {"raw": raw[:200]}, time.time() - t0
    except Exception as e:  # noqa: BLE001 — bank every failure mode
        return -1, {"raw": f"{type(e).__name__}: {e}"[:200]}, time.time() - t0


def one(tag: str, path: str, body: dict) -> dict:
    http, j, wall = post(path, body)
    txt, ptok, lp_nan, lp_n, err = "", None, None, 0, ""
    if http == 200:
        ch = (j.get("choices") or [{}])[0]
        txt = ch.get("text") or (ch.get("message") or {}).get("content") or ""
        ptok = (j.get("usage") or {}).get("prompt_tokens")
        lp = ch.get("logprobs") or {}
        vals = [t for t in (lp.get("token_logprobs") or []) if isinstance(t, float)]
        lp_n = len(vals)
        lp_nan = sum(1 for v in vals if v != v)
    else:
        err = str(j.get("error", j))[:160]
    red = (http != 200) or bool(lp_nan) or (" our our" in txt)
    rec = dict(
        tag=tag,
        http=http,
        ptok=ptok,
        lp_nan=lp_nan,
        lp_n=lp_n,
        wall=round(wall, 2),
        red=bool(red),
        text=txt[:140],
        err=err,
    )
    with LOCK:
        REC.append(rec)
    flag = "RED " if red else "ok  "
    print(
        f"[{flag}] {tag}: http={http} ptok={ptok} lp_nan={lp_nan}/{lp_n} "
        f"wall={wall:.1f} text={txt[:70]!r} err={err[:80]!r}",
        flush=True,
    )
    return rec


def raw_body(prompt: str, max_tokens: int = 16, extra: dict | None = None) -> dict:
    b = {"model": "gemma4-31b", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "logprobs": 1}
    if extra:
        b.update(extra)
    return b


def chat_body(content: str, extra: dict | None = None) -> dict:
    b = {"model": "gemma4-31b", "messages": [{"role": "user", "content": content}]}
    if extra:
        b.update(extra)
    return b


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else "disc.json"
    print(f"[p268c] disc start {time.strftime('%H:%M:%S')}", flush=True)

    fmts = {}
    try:
        with open(MATRIX, encoding="utf-8") as f:
            fmts = json.load(f).get("formats", {})
    except Exception as e:  # noqa: BLE001 — T4/T5 need the banked prompts
        print(f"[p268c] WARN matrix json unreadable ({e}) -- T4/T5 skipped", flush=True)
    probe_raw_extra = {
        "add_special_tokens": False,
        "skip_special_tokens": False,
        "logprobs": 1,
    }

    one("T0_control_raw", "/v1/completions", raw_body(CONTROL, max_tokens=8))
    time.sleep(1.5)
    one("T1_len122_raw", "/v1/completions", raw_body(FILLER * 12, max_tokens=16))
    time.sleep(1.5)
    one(
        "T2_gate_chat",
        "/v1/chat/completions",
        chat_body(
            "What is the capital of France? Answer with just the city name.",
            {"max_tokens": 64, "skip_special_tokens": False, "temperature": 0},
        ),
    )
    time.sleep(1.5)
    one(
        "T3_card_chat",
        "/v1/chat/completions",
        chat_body(
            "What is the capital of France?",
            {
                "max_tokens": 96,
                "skip_special_tokens": False,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "seed": 101,
                "stop_token_ids": [106, 50],
                "logprobs": True,
                "top_logprobs": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
    )
    time.sleep(1.5)
    if "g4" in fmts and "health" in fmts.get("g4", {}):
        one(
            "T4_g4short_raw",
            "/v1/completions",
            raw_body(fmts["g4"]["health"]["prompt"], max_tokens=96, extra=probe_raw_extra),
        )
        time.sleep(1.5)
    if "g3" in fmts and "capital_france" in fmts.get("g3", {}):
        one(
            "T5_g3cap_raw",
            "/v1/completions",
            raw_body(fmts["g3"]["capital_france"]["prompt"], max_tokens=96, extra=probe_raw_extra),
        )
        time.sleep(1.5)

    # T6: the C1 2-way burst (both threads, 50 ms stagger)
    def run(sub: str, prompt: str) -> None:
        one(f"T6_two_way/{sub}", "/v1/completions", raw_body(prompt, max_tokens=16))

    ths = []
    for i in range(2):
        t = threading.Thread(target=run, args=(f"r{i}", f"{FILLER * (i + 3)} Now the answer is:"))
        t.start()
        ths.append(t)
        time.sleep(0.05)
    for t in ths:
        t.join()

    reds = [r["tag"] for r in REC if r["red"]]
    verdict = f"DISC_RED={len(reds)}/{len(REC)}: {reds}" if reds else "DISC_CLEAN (unexpected -- compare trace anyway)"
    print(f"\n[p268c] VERDICT: {verdict}", flush=True)
    with open(out, "w", newline="\n") as f:
        json.dump({"verdict": verdict, "steps": REC}, f, indent=1)
    print(f"[p268c] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
