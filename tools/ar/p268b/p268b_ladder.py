#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p268b_ladder.py — P6.47 P-D (r268b): the request-bisection ladder.

THE r268 ANOMALY this ladder bisects: the r268 trace boot (e2 posture:
cert29, gemma g3 skin, kv mxfp8 + native, ARC_GRAPH=0, serve args
byte-identical to e2) was CLEAN on 3 small sequential raw requests
("The capital of France is" ptok=6 -> " a city of romance, art, and",
finite logprobs) while e2 — the SAME posture — was RED from its very
FIRST request (all 324 probe requests 400-nan; the no-logprobs gate
battery 'Paris our our our' at ptok=40). The kernels/store math are
proven exact offline at the engine's true lanes (r267 canary PASS), so
the corruption is request-CONDITIONAL engine wiring: some attribute of
the probe's requests flips decode onto the broken path.

Axes bisected (each step tagged, JSON-banked):
  A  prompt LENGTH: raw filler ladder ptok ~6..~120 (temp=0,
     max_tokens=16, logprobs=1 — the free nan detector)
  B  the EXACT e2 request shapes, sequential: the gate-battery chat
     request (ptok=40 RED on e2) and the card chat request (the e2
     first 400: full sampling params + seed + stop_token_ids +
     logprobs + chat_template_kwargs)
  C  CONCURRENCY (the probe runs --workers 8; r268 was sequential):
     2-way and 8-way simultaneous raw + chat bursts
  D  poisoning recheck: re-run the A1 control after everything above —
     does a previously-clean request STAY clean on this boot?
  E  determinism: re-run the first RED step (if any) twice

RED = http!=200 OR any nan in token_logprobs OR ' our our' in the
text. The FIRST red tag names the axis; the runner banks ladder.json.
Exit 0 = ladder complete (red or not — the JSON is the evidence).
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
    txt, ptok, lp_nan, lp_n = "", None, None, 0
    if http == 200:
        ch = (j.get("choices") or [{}])[0]
        txt = ch.get("text") or ch.get("content") or ""
        ptok = (j.get("usage") or {}).get("prompt_tokens")
        lp = ch.get("logprobs") or {}
        vals = [t for t in (lp.get("token_logprobs") or []) if isinstance(t, float)]
        lp_n = len(vals)
        lp_nan = sum(1 for v in vals if v != v)
    red = (http != 200) or bool(lp_nan) or (" our our" in txt)
    rec = dict(
        tag=tag, http=http, ptok=ptok, lp_n=lp_n, lp_nan=lp_nan, wall=round(wall, 2), red=bool(red), text=txt[:140]
    )
    with LOCK:
        REC.append(rec)
    flag = "RED " if red else "ok  "
    print(
        f"[{flag}] {tag}: http={http} ptok={ptok} lp_nan={lp_nan}/{lp_n} wall={wall:.1f} text={txt[:70]!r}", flush=True
    )
    return rec


def burst(tag: str, bodies: list[tuple[str, str, dict]]) -> list[dict]:
    """Fire (subtag, path, body) triples simultaneously; join."""
    out: list[dict] = []
    ths = []

    def run(sub: str, path: str, body: dict) -> None:
        r = one(f"{tag}/{sub}", path, body)
        with LOCK:
            out.append(r)

    for sub, path, body in bodies:
        t = threading.Thread(target=run, args=(sub, path, body))
        t.start()
        ths.append(t)
        time.sleep(0.05)  # stagger admission: all land in one scheduler
    for t in ths:
        t.join()
    return out


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
    out = sys.argv[1] if len(sys.argv) > 1 else "ladder.json"
    print(f"[p268b] ladder start {time.strftime('%H:%M:%S')}", flush=True)

    # -- A: prompt-length ladder (raw, temp=0, logprobs=1) ------------
    one("A1_control_ptok6", "/v1/completions", raw_body(CONTROL))
    for reps in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        one(f"A_len_reps{reps}", "/v1/completions", raw_body(FILLER * reps))

    # -- B: the exact e2 request shapes, sequential -------------------
    one(
        "B1_gate_battery_chat",
        "/v1/chat/completions",
        chat_body(
            "What is the capital of France? Answer with just the city name.",
            {"max_tokens": 64, "skip_special_tokens": False, "temperature": 0},
        ),
    )
    one(
        "B2_chat_card_full_params",
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

    # -- C: concurrency (the probe's --workers 8) ---------------------
    burst(
        "C1_two_way",
        [(f"r{i}", "/v1/completions", raw_body(f"{FILLER * (i + 3)} Now the answer is:")) for i in range(2)],
    )
    burst(
        "C2_eight_way_raw",
        [(f"r{i}", "/v1/completions", raw_body(f"{FILLER * (i + 3)} Now the answer is:")) for i in range(8)],
    )
    burst(
        "C3_eight_way_chat",
        [
            (
                f"c{i}",
                "/v1/chat/completions",
                chat_body(
                    f"Question number {i}: what is the capital of France? Answer with just the city name.",
                    {
                        "max_tokens": 64,
                        "skip_special_tokens": False,
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 64,
                        "seed": 101,
                        "stop_token_ids": [106, 50],
                        "logprobs": True,
                        "top_logprobs": 1,
                    },
                ),
            )
            for i in range(8)
        ],
    )

    # -- D: poisoning recheck ----------------------------------------
    one("D_control_repeat", "/v1/completions", raw_body(CONTROL))

    # -- E: determinism — re-run the first RED step twice -------------
    first_red = next((r["tag"] for r in REC if r["red"]), None)
    if first_red:
        # re-issue the same body as the first red tag
        for i in (1, 2):
            if first_red.startswith("A"):
                one(
                    f"E_redrepeat{i}",
                    "/v1/completions",
                    raw_body(CONTROL if first_red == "A1_control_ptok6" else FILLER * int(first_red.split("reps")[1])),
                )
            elif first_red.startswith("B1"):
                one(
                    f"E_redrepeat{i}",
                    "/v1/chat/completions",
                    chat_body(
                        "What is the capital of France? Answer with just the city name.",
                        {"max_tokens": 64, "skip_special_tokens": False, "temperature": 0},
                    ),
                )
            elif first_red.startswith("B2"):
                one(
                    f"E_redrepeat{i}",
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
            else:  # a C burst tag — re-run the 8-way chat burst
                burst(
                    f"E_redrepeat{i}",
                    [
                        (
                            f"c{j}",
                            "/v1/chat/completions",
                            chat_body(
                                f"Question number {j}: what is the capital of France? Answer with just the city name.",
                                {
                                    "max_tokens": 64,
                                    "skip_special_tokens": False,
                                    "temperature": 1.0,
                                    "top_p": 0.95,
                                    "top_k": 64,
                                    "seed": 101,
                                    "stop_token_ids": [106, 50],
                                    "logprobs": True,
                                    "top_logprobs": 1,
                                },
                            ),
                        )
                        for j in range(8)
                    ],
                )
    else:
        print("[p268b] E skipped (no red step to repeat)", flush=True)

    reds = [r["tag"] for r in REC if r["red"]]
    verdict = f"FIRST_RED={reds[0]}" if reds else "LADDER_CLEAN (no red step — see the probe phase)"
    print(f"\n[p268b] VERDICT: {verdict}", flush=True)
    print(f"[p268b] red_steps={len(reds)}/{len(REC)}: {reds}", flush=True)
    with open(out, "w", newline="\n") as f:
        json.dump({"verdict": verdict, "steps": REC}, f, indent=1)
    print(f"[p268b] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
