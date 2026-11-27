#!/usr/bin/env python3
"""p266_it_probe.py -- the gemma-4-31B-it serving-skin probe battery (P6.47 r266 v2).

Every gemma leg from 2026-11-06 to the r264 gate served the PRE-TRAINED
checkpoint google/gemma-4-31B (the rig's /models/gemma4-31b) through a chat
template that was never its own (P6.28: hand-written gemma-3 turn markers
that are not in the gemma-4 vocabulary; 2026-11-08..r264: qwen's template
and parsers). The instruction-tuned google/gemma-4-31B-it is the checkpoint
the chat template, the reasoning parser and the tool parser are defined for
(model card "Getting Started"; vLLM recipes "Gemma 4 Usage Guide"). This
probe exercises that surface end to end on the chat endpoint:

  QA      seven prompts, thinking off (the template's default: the request
          names no chat_template_kwargs), temperature 0 -- the answer must
          be in message.content and end with finish_reason "stop" (the
          checkpoint's generation_config stops on <|turn>|, <tool_response>
          and <eos>)
  THINK   three questions with thinking on, twice through
          chat_template_kwargs {"enable_thinking": true} and once through
          OpenAI reasoning_effort (every value but "none" turns thinking on
          for a template that declares the variable) -- message.reasoning
          carries the thought block, message.content the answer
  TOOL    a get_weather tool, thinking off and on: the model must call the
          tool with the asked city, then answer from the tool response
          (the assistant turn is replayed WITHOUT its reasoning, the
          model-card multi-turn rule)
  SAMPLED the model card's sampling configuration (temperature 1.0,
          top_p 0.95, top_k 64) on the five factual prompts, two seeds
  STREAM thinking on, stream=true: reasoning then content in the deltas,
          finish_reason "stop", a final [DONE]

The train question's right answer is 6:00 pm (a 60 km gap closing at 30
km/h); the P6.28-era "4:40 pm" the old battery enshrined was wrong.

Usage: ARC_PROBE_MODEL=<served> python3 p266_it_probe.py <tag>
Writes ./p266_it_probe_<tag>.json and prints one line per check.
Stdlib only; talks to http://127.0.0.1:8000 (rig host python3).
Exit 0 = IT_PROBE_PASS (every gate holds).
(P266_BASE exists for the offline selftest -- p266_it_probe_selftest.py --
and is never set in a leg.)
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("P266_BASE", "http://127.0.0.1:8000")
URL = BASE + "/v1/chat/completions"
MODEL = os.environ.get("ARC_PROBE_MODEL", "gemma4-31b-it")
TIMEOUT = 900

# ---- the questions and what a correct answer looks like ----
GASES = ("hydrogen", "helium", "nitrogen", "oxygen", "fluorine", "neon",
         "chlorine", "argon", "krypton", "xenon", "radon")
TRAIN_Q = ("If a train leaves at 3pm traveling 60 km/h and another at 4pm "
           "traveling 90 km/h the same direction, when does the second catch "
           "the first? Answer with just the time.")
BAT_Q = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more "
         "than the ball. How much does the ball cost? Answer with just the "
         "amount.")
SHIRT_Q = ("A shirt costs $40 after a 20% discount. What was the original "
           "price? Answer with just the amount.")

# 6:00 pm / 6 pm / 6.00 p.m. / 18:00 / 18h -- never the "6" of "60 km/h"
SIX_PM = re.compile(
    r"(?i)(?<![\d.:])(?:6(?![\d.])\s*(?::\s*00|\.00)?\s*"
    r"(?:pm|p\.m\.|in the evening)"
    r"|18(?![\d.])\s*(?::\s*00)?\s*(?:h\b|hrs?\b|hours?\b|o'?clock)?)")
NICKEL = re.compile(r"(?i)(?<![\d.])(?:0\.05|5\s*cents?|a nickel|5¢)")
FIFTY = re.compile(r"(?i)(?<![\d.])(?:\$?\s*50(?![\d.])|fifty)")
PARIS = re.compile(r"(?i)\bparis\b")
FIB = re.compile(r"(?i)(def |fib)")
SEVEN = re.compile(r"[Ss]even|SEVEN")
REVERSE = re.compile(r"s\[::-1\]")
SANITY = re.compile(r"HEALTH CHECK OK")


def gases(text):
    return len({g for g in GASES if re.search(r"(?i)\b" + g + r"\b", text)})


def ok_gases(text):
    return gases(text) >= 2


QA = [
    # id, prompt, max_tokens, matcher, matcher name
    ("P1", "What is the capital of France? Answer with just the city name.",
     32, PARIS.search, "Paris"),
    ("P2", "Write a Python function that returns the n-th Fibonacci number "
           "using iteration.", 256, FIB.search, "def/fib"),
    ("P3", "Name three chemical elements that are gases at room temperature.",
     64, ok_gases, "gases>=2"),
    ("P4", TRAIN_Q, 128, SIX_PM.search, "6:00 pm"),
    ("P5", "Say the number seven in words, then stop.", 24, SEVEN.search, "seven"),
    ("P6", "Write a Python one-liner that reverses a string s. Output only "
           "the code.", 64, REVERSE.search, "s[::-1]"),
    ("P7", "Reply with exactly: HEALTH CHECK OK", 24, SANITY.search, "HEALTH CHECK OK"),
]
SAMPLED_IDS = ("P1", "P3", "P4", "P5", "P7")
CARD_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}   # the model card
SEEDS = (11, 23)
THINK_QS = [
    # id, question, max_tokens, matcher, how thinking is asked
    ("t1", TRAIN_Q, 1024, SIX_PM.search, {"chat_template_kwargs": {"enable_thinking": True}}),
    ("t2", BAT_Q, 1024, NICKEL.search, {"chat_template_kwargs": {"enable_thinking": True}}),
    ("t3", SHIRT_Q, 1024, FIFTY.search, {"reasoning_effort": "high"}),
]
TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}
TOOL_Q = "What's the weather in Paris right now? Use the tool."
TOOL_RESULT = json.dumps({"temp_c": 17, "condition": "cloudy"})
SEVENTEEN = re.compile(r"(?<![\d.])17(?![\d.])")


def post(body, timeout=TIMEOUT):
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


def fields(msg):
    """(content, reasoning): the response field names of this server
    generation, the deprecated alias included."""
    content = msg.get("content")
    reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "")
    return content, reasoning or ""


def chat(messages, **kw):
    body = {"model": MODEL, "messages": messages}
    body.update(kw)
    st, resp = post(body)
    ch = ((resp or {}).get("choices") or [{}])[0]
    return st, ch, resp


def show(text, n=110):
    return repr((text or "")[:n])


class Battery:
    def __init__(self):
        self.fails = []
        self.rec = {"model": MODEL, "qa": [], "think": [], "tool": [],
                    "sampled": [], "stream": None, "default_sampling": None}

    def gate(self, ok, name, line):
        print(line, flush=True)
        if not ok:
            self.fails.append(name)

    # -- QA: thinking off, the request names no template kwargs --
    def qa(self):
        n_ok = n_stop = 0
        for pid, q, mt, match, label in QA:
            st, ch, _ = chat([{"role": "user", "content": q}],
                             max_tokens=mt, temperature=0)
            content, reasoning = fields(ch.get("message", {}))
            finish = ch.get("finish_reason")
            ok = st == 200 and isinstance(content, str) and bool(match(content or ""))
            stop = ok and finish == "stop"
            n_ok += ok
            n_stop += stop
            self.rec["qa"].append({"id": pid, "status": st, "finish": finish,
                                   "content": content, "reasoning": reasoning,
                                   "ok": bool(ok), "stop": bool(stop)})
            why = "" if ok else f" -- want {label}, content={show(content)}"
            if not ok and reasoning and not content:
                why += f" [the text is in the reasoning field: {show(reasoning, 60)}]"
            self.gate(ok, f"qa_{pid.lower()}",
                      f"QA {pid} {'ok' if ok else 'FAIL'} finish={finish} "
                      f"chars={len(content or '')}{why} {show(content, 70)}")
        self.gate(n_ok == len(QA), "qa_count", f"QA_SUMMARY ok={n_ok}/{len(QA)}")
        self.gate(n_stop == len(QA), "qa_stop", f"QA_STOP stop={n_stop}/{len(QA)}")

    # -- THINK: the thought block in reasoning, the answer in content --
    def think(self):
        n_ok = 0
        for tid, q, mt, match, how in THINK_QS:
            st, ch, _ = chat([{"role": "user", "content": q}],
                             max_tokens=mt, temperature=0, **how)
            content, reasoning = fields(ch.get("message", {}))
            finish = ch.get("finish_reason")
            ok = (st == 200 and bool(reasoning.strip()) and isinstance(content, str)
                  and bool(match(content or "")))
            n_ok += ok
            self.rec["think"].append({"id": tid, "how": how, "status": st, "finish": finish,
                                      "reasoning": reasoning, "content": content,
                                      "ok": bool(ok)})
            self.gate(ok, f"think_{tid}",
                      f"THINK {tid} {'ok' if ok else 'FAIL'} finish={finish} "
                      f"reasoning_chars={len(reasoning)} how={sorted(how)} {show(content, 70)}")
        self.gate(n_ok == len(THINK_QS), "think_count",
                  f"THINK_SUMMARY ok={n_ok}/{len(THINK_QS)}")

    # -- TOOL: call, then answer from the tool response --
    def tool_round(self, rid, thinking):
        how = {"chat_template_kwargs": {"enable_thinking": True}} if thinking else {}
        user = {"role": "user", "content": TOOL_Q}
        out = {"id": rid, "thinking": thinking}
        st, ch, _ = chat([user], tools=[TOOL], tool_choice="auto",
                         max_tokens=512, temperature=0, **how)
        msg = ch.get("message", {}) or {}
        calls = msg.get("tool_calls") or []
        name = args = None
        if calls:
            fn = calls[0].get("function", {})
            name = fn.get("name")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:  # noqa: BLE001
                args = {"_raw": fn.get("arguments")}
        _, reasoning = fields(msg)
        ok1 = (st == 200 and name == "get_weather"
               and bool(args) and PARIS.search(json.dumps(args)) is not None
               and (not thinking or bool(reasoning.strip())))
        out.update({"status": st, "tool_calls": calls, "reasoning": reasoning,
                    "r1_ok": bool(ok1)})
        self.gate(ok1, f"tool_{rid}_call",
                  f"TOOL {rid} call {'ok' if ok1 else 'FAIL'} name={name} args={args} "
                  f"reasoning_chars={len(reasoning)}")
        # the assistant turn goes back WITHOUT its reasoning (model-card
        # multi-turn rule); the tool response carries the id
        assistant = {"role": "assistant", "content": msg.get("content")}
        if calls:
            assistant["tool_calls"] = calls
        tool_msg = {"role": "tool", "tool_call_id": calls[0].get("id") if calls else "call_x",
                    "content": TOOL_RESULT}
        st2, ch2, _ = chat([user, assistant, tool_msg], tools=[TOOL],
                           max_tokens=512, temperature=0, **how)
        content, reasoning2 = fields(ch2.get("message", {}))
        ok2 = st2 == 200 and isinstance(content, str) and bool(SEVENTEEN.search(content or ""))
        out.update({"status2": st2, "content": content, "reasoning2": reasoning2,
                    "r2_ok": bool(ok2)})
        self.gate(ok2, f"tool_{rid}_answer",
                  f"TOOL {rid} answer {'ok' if ok2 else 'FAIL'} status={st2} "
                  f"reasoning_chars={len(reasoning2)} {show(content, 90)}")
        return out

    def tool(self):
        self.rec["tool"].append(self.tool_round("off", False))
        self.rec["tool"].append(self.tool_round("on", True))
        n_ok = sum(r["r1_ok"] + r["r2_ok"] for r in self.rec["tool"])
        self.gate(n_ok == 4, "tool_count", f"TOOL_SUMMARY ok={n_ok}/4")

    # -- SAMPLED: the model card's configuration, two seeds --
    def sampled(self):
        n_ok = 0
        spec = {pid: (q, mt, match, label) for pid, q, mt, match, label in QA}
        for pid in SAMPLED_IDS:
            q, mt, match, _ = spec[pid]
            for seed in SEEDS:
                st, ch, _ = chat([{"role": "user", "content": q}],
                                 max_tokens=128, seed=seed, **CARD_SAMPLING)
                content, _ = fields(ch.get("message", {}))
                ok = st == 200 and isinstance(content, str) and bool(match(content or ""))
                n_ok += ok
                self.rec["sampled"].append({"id": pid, "seed": seed, "status": st,
                                            "finish": ch.get("finish_reason"),
                                            "content": content, "ok": bool(ok)})
                print(f"SAMPLED {pid} seed={seed} {'ok' if ok else 'MISS'} "
                      f"{show(content, 70)}", flush=True)
        # sampling at temperature 1.0 is allowed one miss in ten; the
        # answers are banked either way
        total = len(SAMPLED_IDS) * len(SEEDS)
        self.gate(n_ok >= total - 1, "sampled_count",
                  f"SAMPLED_SUMMARY ok={n_ok}/{total}")

    # -- STREAM: reasoning then content in the deltas, [DONE] last --
    def stream(self):
        body = {"model": MODEL, "stream": True, "max_tokens": 1024, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": True},
                "messages": [{"role": "user", "content": TRAIN_Q}]}
        req = urllib.request.Request(
            URL, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        reasoning = ""
        content = ""
        finish = None
        done = False
        chunks = 0
        err = None
        try:
            r = urllib.request.urlopen(req, timeout=TIMEOUT)
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    done = True
                    break
                chunks += 1
                choice = (json.loads(payload).get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                reasoning += delta.get("reasoning") or delta.get("reasoning_content") or ""
                content += delta.get("content") or ""
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        ok = (err is None and bool(reasoning.strip()) and bool(SIX_PM.search(content))
              and finish == "stop" and done)
        self.rec["stream"] = {"reasoning": reasoning, "content": content,
                              "finish": finish, "done": done, "chunks": chunks,
                              "error": err, "ok": bool(ok)}
        self.gate(ok, "stream",
                  f"STREAM {'ok' if ok else 'FAIL'} reasoning_chars={len(reasoning)} "
                  f"content_chars={len(content)} finish={finish} done={done} "
                  f"chunks={chunks} err={err} {show(content, 70)}")

    # -- the checkpoint's own sampling defaults: one request that names
    # no sampling field (banked: the server does not echo what it applied) --
    def default_sampling(self):
        q = dict((pid, q) for pid, q, _, _, _ in QA)["P1"]
        st, ch, resp = chat([{"role": "user", "content": q}], max_tokens=32, seed=5)
        content, _ = fields(ch.get("message", {}))
        self.rec["default_sampling"] = {"status": st, "content": content,
                                        "usage": (resp or {}).get("usage")}
        print(f"DEFAULT_SAMPLING status={st} {show(content, 70)}", flush=True)


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "x"
    print(f"=== p266_it_probe model={MODEL} tag={tag} base={BASE} ===", flush=True)
    b = Battery()
    t0 = time.perf_counter()
    b.default_sampling()
    b.qa()
    b.think()
    b.tool()
    b.sampled()
    b.stream()
    b.rec["fails"] = b.fails
    b.rec["seconds"] = round(time.perf_counter() - t0, 1)
    with open(f"p266_it_probe_{tag}.json", "w", newline="\n") as f:
        json.dump(b.rec, f, indent=1)
    verdict = "PASS" if not b.fails else "FAIL: " + ",".join(b.fails)
    print(f"=== IT_PROBE_{verdict} ({b.rec['seconds']}s) ===", flush=True)
    return 0 if not b.fails else 1


if __name__ == "__main__":
    sys.exit(main())
