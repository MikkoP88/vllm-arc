#!/usr/bin/env python3
"""r257 -- the FAITHFUL minimal repro of the r255 gemma MX DEVICE_LOST.

The r256 leg proved the repro gap: the r255 crash request rode the
chat template's OFF branch (chat_template_kwargs {"enable_thinking":
False}; the qwen38_high template serves on gemma -- GEMMA_NAT sets no
ARC_TEMPLATE), while every r256 request rode the ON branch (no kwarg,
+47 template tokens). Same length, different bytes: the r255 tail
window was never tested.

This probe sends the r255 request BYTE-EXACT (modulo max_tokens, which
does not touch the prompt stream -- the crash fired in the resend's
tail-PREFILL step, before any decode): the UNCUT build_multi(1650)
text + the enable_thinking kwarg. The RUNG line's fresh prompt_tok IS
the OFF-branch engine count -- 3446 confirms the 47-token delta theory
and the byte-exactness of the replay.

The CALIB line banks the branch delta itself: the same short sentence
through /tokenize (ON branch) and a live kwarg completion (OFF branch).

Runs INSIDE the container: docker exec -i <name> /opt/venv/bin/python3
- < this file. Stdlib only. Talks to 127.0.0.1:8000 in-container.
"""
import json
import os
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
TOKURL = "http://127.0.0.1:8000/tokenize"
MODEL = os.environ.get("ARC_PROBE_MODEL", "gemma4-31b")


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


def n_tokens(text):
    # the ON-branch oracle (r256's engine-true count: CALIB MATCH x3)
    req = urllib.request.Request(
        TOKURL,
        data=json.dumps({"model": MODEL,
                         "messages": [{"role": "user", "content": text}]}).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=60))
    return int(r.get("count", len(r.get("tokens", []))))


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


def kwarg_body(text, max_tokens):
    # the r255 request construction: the enable_thinking kwarg rides the
    # qwen template's OFF branch (accepted by the serve -- no 400, so
    # the longprobe's fallback never fired on the gemma legs)
    return {"model": MODEL, "stream": False, "temperature": 0.0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}


def main():
    text = "Calibration sentence for the r257 faithful probe: one, two, three."
    n_on = n_tokens(text)
    r = post(kwarg_body(text, 1), 120)
    n_off = r.get("prompt_tok") if r.get("ok") else None
    delta = None if n_off is None else n_on - n_off
    print(f"CALIB on_branch_tokenize={n_on} off_branch_engine={n_off} "
          f"delta={delta}", flush=True)
    full = build_multi(1650)  # UNCUT -- the r255 bytes
    n_on_full = n_tokens(full)
    print(f"FULL_MULTI_ON_BRANCH_N {n_on_full} (r256 measured; the OFF-branch"
          f" count lands in the RUNG line)", flush=True)
    body = kwarg_body(full, 16)
    fresh = post(body, 300)
    resend = post(body, 300)
    alive = health()
    rec = {"face": "multi-uncut-kwarg", "n": fresh.get("prompt_tok"),
           "fresh": fresh, "resend": resend, "alive": alive}
    print("RUNG " + json.dumps(rec), flush=True)
    print("R257_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
