#!/usr/bin/env python3
"""r256 -- minimal repro + tail bisect of the r255 gemma MX DEVICE_LOST.

The r255 open defect: the p647 longprobe `multi` face, SECOND sample
(an A/A prefix-cache-hit resend), crashed all three gemma MX legs with
the byte-identical signature -- prompt_token_ids_len=3446,
num_computed_tokens=3328 (26 full page-128 blocks prefix-cached),
num_scheduled_tokens=118 (the tail-prefill chunk) ->
UR_RESULT_ERROR_DEVICE_LOST; dmesg: a ccs (compute) engine reset on
BOTH GPUs 13-18 s before the python surface. The r255 devcoredumps
were auto-deleted by the retention window before capture -- the r256
driver's watcher grabs them within 2 s of the reset landing.

Count oracle history (both run1 defects are fixed here): a LOCAL
AutoTokenizer is unusable twice over -- (a) in-container /models/
holds ONLY the remap mount target (the serve script maps ARC_MODEL's
host dir to /models/target; the host name never appears), and (b) the
model dir carries NO embedded chat_template ("tokenizer.chat_template
is not set" -- the serve script passes the template override to the
engine), so a local tokenizer can never reproduce the engine's prompt
stream. The oracle is now the server's own POST /tokenize with the
messages payload: it applies the serve's template resolution --
byte-for-byte the chat-completion preprocessing. A CALIB line at
start compares the oracle count against a live completion's
usage.prompt_tokens: MATCH = every rung count is engine-true.

Bisect design: hold the cached prefix CONSTANT (26 blocks = 3,328
tok) and vary the TAIL: 24, 48 (the r255 essay/code known-safe tail
sizes), 63, 64, 65 (the page-64 multi-block boundary), 96, 118 (the
crash tail). Rung 1's fresh send is a cold full prefill (its resend is
the first prefix-hit); every LATER rung's fresh send is itself a
prefix-hit on the shared 3,328-tok prefix (the truncations share the
same beginning -- the first 3328 tokens are identical because each
cut sits above token 3352), so each later rung directly tests its
tail as a prefix-hit prefill. The r255 fault fired inside the
tail-PREFILL step (~4 s in), so the generation length is irrelevant;
max_tokens=16 keeps each rung under a minute while seeding the prefix
cache exactly the way the r255 probe did (the resend hits blocks
0..N-2 and recomputes only the tail: the fresh sequence's last full
block carries gen bytes and cannot match the resent prompt).

The multi face is rebuilt byte-identically (build_multi copied
verbatim from p647_longprobe.py); the content is truncated under
oracle guidance to land each tail target exactly. A target at or
above the full face's length uses the UNCUT text (phase A: the exact
r255 request, byte-identical). Requests replicate the r255 probe's
fallback path (plain chat body, no chat_template_kwargs -- gemma's
template 400s on the kwarg and the r255 probe fell back to the plain
body).

The --control N mode runs the content-independence face:
English-only text truncated to the same 3,446-token length (same
26-block prefix + 118-tail geometry, no multilingual tokens).

Runs INSIDE the container: docker exec -i <name> /opt/venv/bin/python3
- < this file [--rungs 24,48,63,64,65,96,118] [--control 3446].
Stdlib only. Talks to 127.0.0.1:8000 in-container.
"""
import argparse
import json
import os
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
TOKURL = "http://127.0.0.1:8000/tokenize"
MODEL = os.environ.get("ARC_PROBE_MODEL", "gemma4-31b")
CACHED = 3328  # 26 full page-128 blocks -- the r255 crash prefix


def build_multi(n_words_target):
    # verbatim from /root/build/p647/p647_longprobe.py (the r255 face)
    lines = [f"Moniste XRK7-CHARLIE — kolme kieltä, yksi johtopäätös. "
             f"まず日本語の要約：以下の文章はキャッシュの一貫性を検証するためのものです。"]
    i = 0
    words = 50
    while words < n_words_target:
        lines.append(
            f"{i}. Suomenkielinen kappale {i}: ajastin mitaa vastausaikaa, "
            f" ja välimuisti palauttaa arvot {i} kerralla. 次の行：整合性 "
            f"(せいせいせい) を保つため、鍵は {i} 番のトークン列から作ります. "
            f"Row {i} (EN): the harness hashes each reply and compares cold "
            f"vs warm — any drift is a defect, not noise. ✅ marker {i}.")
        words += 50
        i += 1
    lines.append("Loppu. Kirjoita yhteenveto englanniksi.")
    return "\n".join(lines)


def build_english(n_words_target):
    # the content-independence control: same builder SHAPE, English-only
    # (over-provisioned ~2x so the oracle-guided cut lands at 3446)
    lines = ["Document XRK7-DELTA — English-only control face, r256."]
    i = 0
    words = 50
    while words < n_words_target:
        lines.append(
            f"{i}. English paragraph {i}: the timer measures response "
            f"latency, and the cache returns values {i} at a time. "
            f"Row {i}: the harness hashes each reply and compares cold "
            f"against warm — any drift is a defect, not noise. "
            f"Marker {i} recorded.")
        words += 50
        i += 1
    lines.append("End. Write a summary in English.")
    return "\n".join(lines)


def n_tokens(text):
    # the ENGINE's own count: /tokenize with the messages payload runs
    # the serve's chat-template resolution (identical to the completion
    # preprocessing -- the CALIB line proves the equality at runtime)
    req = urllib.request.Request(
        TOKURL,
        data=json.dumps({"model": MODEL,
                         "messages": [{"role": "user", "content": text}]}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    return int(r.get("count", len(r.get("tokens", []))))


def truncate_to(text, target_n):
    # the char cut is a monotone step function over token count; every
    # integer N in range is achieved by some cut -- binary search it
    lo, hi = 0, len(text)
    for _ in range(48):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        n = n_tokens(text[:mid])
        if n == target_n:
            return text[:mid]
        if n < target_n:
            lo = mid
        else:
            hi = mid
    return text[:lo]


def post(body, timeout):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        msg = r["choices"][0]["message"]
        return {"ok": True, "dt": round(time.perf_counter() - t0, 1),
                "prompt_tok": r.get("usage", {}).get("prompt_tokens"),
                "finish": r["choices"][0].get("finish_reason"),
                "head": ((msg.get("content") or "")
                        + (msg.get("reasoning") or ""))[:60]}
    except Exception as e:
        return {"ok": False, "dt": round(time.perf_counter() - t0, 1),
                "err": repr(e)[:160]}


def health():
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=4)
        return True
    except Exception:
        return False


def calibrate():
    # oracle-vs-engine self-check: same messages through /tokenize and
    # through a live chat completion. MATCH = every rung's n is the
    # engine's prompt_token_ids_len. MISMATCH = a uniform template
    # offset -- the rung records carry the engine truth (fresh
    # prompt_tok) for offline correction.
    text = "Calibration sentence for the r256 probe: one, two, three."
    n_tok = n_tokens(text)
    r = post({"model": MODEL, "stream": False, "temperature": 0.0,
              "max_tokens": 1,
              "messages": [{"role": "user", "content": text}]}, 120)
    n_eng = r.get("prompt_tok") if r.get("ok") else None
    verdict = "MATCH" if n_tok == n_eng else \
        "MISMATCH (uniform template offset -- read fresh.prompt_tok per rung)"
    print(f"CALIB tokenize={n_tok} engine_prompt_tokens={n_eng} {verdict}",
          flush=True)


def rung(full_text, full_n, tail, label):
    # a rung = fresh send (seeds the prefix cache) + the A/A resend
    # (the prefix-hit tail prefill under test). alive=False = the
    # engine died on this rung (the DEVICE_LOST repro).
    target_n = CACHED + tail
    if target_n >= full_n:
        text = full_text  # the uncut face -- phase A's exact r255 request
    else:
        text = truncate_to(full_text, target_n)
    body = {"model": MODEL, "stream": False, "temperature": 0.0,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": text}]}
    fresh = post(body, 300)
    resend = post(body, 300)
    alive = health()
    rec = {"face": label, "tail": tail, "n": fresh.get("prompt_tok"),
           "fresh": fresh, "resend": resend, "alive": alive}
    print("RUNG " + json.dumps(rec), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="24,48,63,64,65,96,118",
                    help="comma list of tail targets; the ladder stops at "
                         "the first rung that leaves the engine dead")
    ap.add_argument("--control", default="",
                    help="run the English-content control at this TOTAL "
                         "token length (e.g. 3446) instead of the ladder")
    args = ap.parse_args()
    calibrate()
    if args.control:
        full = build_english(5000)
        full_n = n_tokens(full)
        print("FULL_ENGLISH_N", full_n, flush=True)
        rung(full, full_n, int(args.control) - CACHED, "english-control")
        print("R256_PROBE_DONE", flush=True)
        return
    full = build_multi(1650)
    full_n = n_tokens(full)
    print("FULL_MULTI_N", full_n,
          "(r255 crash prompt = 3446; a constant offset here shifts the "
          "tails uniformly -- the per-rung n records the actual)",
          flush=True)
    dead = False
    for tail in [int(x) for x in args.rungs.split(",")]:
        rec = rung(full, full_n, tail, "multi")
        if not rec["alive"]:
            print("LADDER_DEAD_AT_TAIL", tail, flush=True)
            dead = True
            break
    if not dead:
        print("LADDER_SURVIVED_ALL_RUNGS", flush=True)
    print("R256_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
