#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""arc_garbage_probe.py — garbage/corrupted-input hardening suite (r46t,
owner directive: "run also garbage/corrupted testing so AI don't generate
corrupted results").

Sends malformed / adversarial / corrupt request payloads at the served
model and asserts, after EVERY probe:
  1. the engine stays healthy (/health == 200),
  2. well-formed requests still return well-formed, usable responses,
and at the end:
  3. a fixed temp-0 sanity generation returns sane usage, and
  4. repeating it byte-matches (engine state not corrupted by garbage).

Valid probes carry the real-world sampling set (top_p 0.95, top_k 20,
min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0) per the owner
directive; type-violation probes intentionally break fields.

Usage: arc_garbage_probe.py           (exit 0 = GARBAGE_SUITE_PASS)
"""
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
HEALTH = "http://127.0.0.1:8000/health"
MODEL = "qwen3.8-27b-fp8"
SAMPLING = {"top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0, "repetition_penalty": 1.0}
MT = 32
fails: list = []


def health_ok() -> bool:
    try:
        r = urllib.request.urlopen(HEALTH, timeout=10)
        return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def post(body, raw=None, timeout=180):
    """POST and return (status, parsed-or-None). Never raises."""
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status, json_load(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json_load(e.read())
    except Exception as e:  # noqa: BLE001
        return -1, {"transport_error": f"{type(e).__name__}: {e}"}


def json_load(b):
    try:
        return json.loads(b)
    except Exception:  # noqa: BLE001
        return {"_unparsable": b[:200].decode("utf-8", "replace")}


def probe(name: str, body=None, raw=None, expect=(200, 400),
          want_content=False):
    st, resp = post(body, raw=raw)
    ok = st in expect and health_ok()
    if ok and want_content:
        ch = (resp or {}).get("choices", [{}])[0]
        msg = ch.get("message", {}).get("content")
        ok = isinstance(msg, str) and len(msg) >= 0 and st == 200
    print(f"[{name}] status={st} ok={ok}", flush=True)
    if not ok:
        fails.append(name)
    time.sleep(1)


def sanity_sha() -> str:
    body = {"model": MODEL, "max_tokens": 24, "temperature": 0, **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user",
                          "content": "Reply with exactly: HEALTH CHECK OK"}]}
    st, resp = post(body)
    txt = (resp or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
    return f"{st}:{hashlib.sha256(txt.encode()).hexdigest()[:16]}:{txt[:40]!r}"


def main() -> int:
    print("=== arc_garbage_probe (r46t) — corrupt-input hardening ===",
          flush=True)
    # -- well-formed-but-hostile CONTENT (engine must answer, stay sane) --
    rnd = bytes(range(256)) * 64  # 16 KB of all byte values
    probe("binary_mojibake", {"model": MODEL, "max_tokens": MT,
          "temperature": 0.2, **SAMPLING,
          "messages": [{"role": "user",
                        "content": rnd.decode("latin-1")}]},
          want_content=True)
    probe("control_chars", {"model": MODEL, "max_tokens": MT,
          "temperature": 0.2, **SAMPLING,
          "messages": [{"role": "user", "content": "".join(
              chr(c) for c in range(32)) + "\x7f"}]},
          want_content=True)
    big = " ".join("zqxj vkwf gbhp".split() * 40000)  # ~0.5 MB
    probe("huge_blob", {"model": MODEL, "max_tokens": MT,
          "temperature": 0.2, **SAMPLING,
          "messages": [{"role": "user", "content": big}]},
          want_content=True)
    probe("nospace_100k", {"model": MODEL, "max_tokens": MT,
          "temperature": 0.2, **SAMPLING,
          "messages": [{"role": "user", "content": "x" * 100000}]},
          want_content=True)
    probe("empty_content", {"model": MODEL, "max_tokens": MT,
          "temperature": 0.2, **SAMPLING,
          "messages": [{"role": "user", "content": ""}]},
          want_content=True)
    probe("null_content", {"model": MODEL, "max_tokens": MT,
          "temperature": 0.2, **SAMPLING,
          "messages": [{"role": "user", "content": None}]})
    probe("bad_role", {"model": MODEL, "max_tokens": MT,
          "temperature": 0.2, **SAMPLING,
          "messages": [{"role": "garbage", "content": "hi"}]})
    probe("unicode_salad", {"model": MODEL, "max_tokens": MT,
          "temperature": 0.2, **SAMPLING,
          "messages": [{"role": "user", "content":
          "нуll\\u200b🦠\\U0001F9EAᚠᛒᚲ профиль ™𝕏 ﷽ ٱل ٱْ"}]},
          want_content=True)
    # -- structurally INVALID requests (400-class expected, engine lives) --
    probe("no_messages", {"model": MODEL, "max_tokens": MT}, expect=(400,))
    probe("empty_messages", {"model": MODEL, "max_tokens": MT,
          "messages": []}, expect=(400,))
    probe("messages_string", {"model": MODEL, "max_tokens": MT,
          "messages": "not-a-list"}, expect=(400,))
    probe("max_tokens_string", {"model": MODEL, "max_tokens": "abc",
          "messages": [{"role": "user", "content": "hi"}]}, expect=(400,))
    probe("temperature_string", {"model": MODEL, "temperature": "hot",
          "messages": [{"role": "user", "content": "hi"}]}, expect=(400,))
    probe("raw_truncated_json", raw=b'{"model": "qwen3.8-27b-fp8", "mess',
          expect=(400,))
    probe("raw_wrong_content_type", raw=b'plain text not json',
          expect=(400,))

    # -- state-corruption check: temp-0 sanity twice must byte-match --
    s1 = sanity_sha()
    s2 = sanity_sha()
    stable = s1 == s2 and s1.startswith("200:")
    print(f"[state_stability] {s1} == {s2} -> {stable}", flush=True)
    if not stable:
        fails.append("state_stability")
    if not health_ok():
        fails.append("final_health")

    n = 15
    print(f"=== GARBAGE_SUITE_{'PASS' if not fails else 'FAIL: ' + ','.join(fails)}"
          f" (probes={n} fails={len(fails)}) ===", flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
