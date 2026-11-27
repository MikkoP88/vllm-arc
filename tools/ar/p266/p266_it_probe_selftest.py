#!/usr/bin/env python3
"""p266_it_probe_selftest.py -- offline selftest of the gemma-4-31B-it
serving-skin probe tools/ar/p266/p266_it_probe.py (local: no GPU, no vLLM).

A mock engine on the loopback implements the chat endpoint surface the
probe reads and answers in one of four MODES:
  sane            an instruction-tuned model under its own chat template:
                  the answer in message.content, the thought block in
                  message.reasoning when thinking is on, tool calls in
                  message.tool_calls, SSE deltas reasoning-then-content
  base            the pre-trained-checkpoint shape the r264 gemma legs
                  banked: run-on text past the turn end (finish "length"),
                  the wrong train answer, thinking requests answered
                  without any reasoning split, no tool calls
  reasoning_only  a parser/template defect: with thinking off the answer
                  lands in message.reasoning and content stays None
  dead            nothing listens (the probe must fail every gate, write
                  its JSON and never raise)

Proves that the probe's verdict says exactly what the engine did, and that
the probe ASKS for the supported surface: no template kwargs and
temperature 0 on the QA sends, thinking through chat_template_kwargs and
through OpenAI reasoning_effort, the assistant tool turn replayed without
its reasoning, the model card's sampling configuration with a seed.

Usage: python tools/ar/p266/p266_it_probe_selftest.py
Exit 0 = every case holds.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "p266_it_probe.py")

ANSWER = {
    "capital of France": "Paris",
    "Fibonacci": "```python\ndef fib(n):\n    a, b = 0, 1\n"
                 "    for _ in range(n):\n        a, b = b, a + b\n    return a\n```",
    "gases at room temperature": "Hydrogen, helium and nitrogen.",
    "when does the second catch": "6:00 pm",
    "number seven in words": "Seven",
    "reverses a string": "s[::-1]",
    "HEALTH CHECK OK": "HEALTH CHECK OK",
    "bat and a ball": "$0.05",
    "original price": "$50",
}
THOUGHT = {
    "when does the second catch": "The first train is 60 km ahead at 4pm. "
                                  "The gap closes at 30 km/h, so 2 hours: 6pm.",
    "bat and a ball": "ball + (ball + 1.00) = 1.10, so the ball is 0.05.",
    "original price": "0.8 times the price is 40, so the price is 50.",
}
TOOL_THOUGHT = "The user wants live weather, so I have to call get_weather."
ANSWER_FROM_TOOL = "It's 17 °C and cloudy in Paris right now."
TOOL_ANSWER_THOUGHT = "The tool reports 17 °C and cloud."


def question(content):
    for key in ANSWER:
        if key in content:
            return key
    return None


def wants_thinking(body):
    ctk = body.get("chat_template_kwargs") or {}
    if "enable_thinking" in ctk:
        return bool(ctk["enable_thinking"])
    effort = body.get("reasoning_effort")
    return effort is not None and effort != "none"


class Engine:
    def __init__(self, mode):
        self.lock = threading.Lock()
        self.mode = mode
        self.requests = []      # every request body, for the asks-what-it-should checks

    def answer(self, body):
        """Returns (status, message, finish_reason)."""
        msgs = body["messages"]
        last = msgs[-1]
        thinking = wants_thinking(body)
        key = question(last.get("content") or "")
        with self.lock:
            self.requests.append(body)
        if body.get("tools"):
            if last.get("role") == "tool":
                # the base shape ignores the tool response (no temperature)
                if self.mode == "base":
                    return 200, {"role": "assistant", "content": "The weather in Paris is sunny.",
                                 "reasoning": ""}, "stop"
                content, reasoning = ANSWER_FROM_TOOL, TOOL_ANSWER_THOUGHT if thinking else ""
                return 200, {"role": "assistant", "content": content, "reasoning": reasoning}, "stop"
            if self.mode == "base":
                return 200, {"role": "assistant", "content": "The weather in Paris is sunny.",
                             "reasoning": ""}, "stop"
            call = {"id": "chatcmpl-tool-mock", "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": json.dumps({"city": "Paris"})}}
            reasoning = TOOL_THOUGHT if thinking else ""
            if self.mode == "reasoning_only" and not thinking:
                # the answer-in-reasoning defect: the call is announced, never made
                return 200, {"role": "assistant", "content": None,
                             "reasoning": "I'll call get_weather for Paris."}, "stop"
            return 200, {"role": "assistant", "content": None, "reasoning": reasoning,
                         "tool_calls": [call]}, "stop"
        text = ANSWER.get(key, "I don't know.")
        reasoning = THOUGHT.get(key, "Let me think.") if thinking else ""
        finish = "stop"
        if self.mode == "base":
            reasoning = ""
            if key == "when does the second catch":
                text = "4:40 pm"
            elif key == "capital of France":
                text = "Paris<|turn>user\nWhat is the capital of Germany?<turn|>model\nBerlin"
                finish = "length"
        elif self.mode == "reasoning_only" and not thinking:
            return 200, {"role": "assistant", "content": None, "reasoning": text}, finish
        return 200, {"role": "assistant", "content": text, "reasoning": reasoning}, finish

    def chunks(self, body):
        """SSE data payloads for a streaming request (thinking is on)."""
        _, msg, finish = self.answer(body)
        out = []
        for i, part in enumerate((msg["reasoning"][:20], msg["reasoning"][20:])):
            if part:
                out.append({"choices": [{"index": 0, "delta": {"role": "assistant" if i == 0 else None,
                                                               "reasoning": part},
                                         "finish_reason": None}]})
        content = msg["content"] or ""
        for part in (content[:4], content[4:]):
            if part:
                out.append({"choices": [{"index": 0, "delta": {"content": part},
                                         "finish_reason": None}]})
        out.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        return out


class Handler(BaseHTTPRequestHandler):
    engine = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in self.engine.chunks(body):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        status, msg, finish = self.engine.answer(body)
        resp = {"choices": [{"message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}}
        payload = json.dumps(resp).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


N_CASES = 0
N_FAILS = 0


def check(cond, label, detail=""):
    global N_CASES, N_FAILS
    N_CASES += 1
    if cond:
        print(f"  ok   {label}")
    else:
        N_FAILS += 1
        print(f"  FAIL {label} -- {detail}")


def run_probe(port, mode, workdir):
    env = dict(os.environ, P266_BASE=f"http://127.0.0.1:{port}",
               ARC_PROBE_MODEL="mock-model")
    p = subprocess.run([sys.executable, PROBE, mode], cwd=workdir, env=env,
                       capture_output=True, text=True, timeout=300)
    jf = os.path.join(workdir, f"p266_it_probe_{mode}.json")
    rec = None
    if os.path.exists(jf):
        with open(jf, encoding="utf-8") as f:
            rec = json.load(f)
    return p, rec


def line(out, prefix):
    for ln in out.splitlines():
        if ln.startswith(prefix):
            return ln
    return ""


def expected_fails(rec, names):
    return sorted(rec["fails"]) == sorted(names)


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="p266_it_probe_selftest_")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    # -- sane: every gate holds; the probe asked for the supported surface --
    print("== sane")
    eng = Engine("sane")
    Handler.engine = eng
    p, rec = run_probe(port, "sane", workdir)
    check(p.returncode == 0 and "IT_PROBE_PASS" in p.stdout, "exit 0 + IT_PROBE_PASS",
          f"rc={p.returncode} {p.stdout[-300:]}")
    check("Traceback" not in p.stderr, "no traceback (sane)", p.stderr[-300:])
    for want in ("QA_SUMMARY ok=7/7", "QA_STOP stop=7/7", "THINK_SUMMARY ok=3/3",
                 "TOOL_SUMMARY ok=4/4", "SAMPLED_SUMMARY ok=10/10"):
        check(line(p.stdout, want.split()[0]).endswith(want) or want in p.stdout,
              f"line '{want}'")
    check(line(p.stdout, "STREAM ").startswith("STREAM ok"), "STREAM ok", line(p.stdout, "STREAM "))
    qa_reqs = [r for r in eng.requests if not r.get("tools") and not r.get("stream")
               and r.get("temperature") == 0 and "chat_template_kwargs" not in r
               and "reasoning_effort" not in r]
    check(len(qa_reqs) >= 7, "the QA sends name no template kwargs", str(len(qa_reqs)))
    t3 = [r for r in eng.requests if r.get("reasoning_effort") == "high"]
    check(len(t3) == 1 and "chat_template_kwargs" not in t3[0],
          "one thinking send rides OpenAI reasoning_effort")
    sampled = [r for r in eng.requests if r.get("seed") in (11, 23)]
    check(len(sampled) == 10 and all(
        r.get("temperature") == 1.0 and r.get("top_p") == 0.95 and r.get("top_k") == 64
        for r in sampled), "the sampled sends carry the model card's configuration",
        str(len(sampled)))
    tool_r2 = [r for r in eng.requests if r.get("tools")
               and r["messages"][-1]["role"] == "tool"]
    check(len(tool_r2) == 2 and all(
        m["messages"][1].get("tool_calls") and "reasoning" not in m["messages"][1]
        and m["messages"][2]["tool_call_id"] == "chatcmpl-tool-mock" for m in tool_r2),
        "the tool turn is replayed without its reasoning")
    check(rec is not None and all(r["reasoning"] for r in rec["think"]),
          "every thinking answer carries a thought block")
    check(rec is not None and rec["stream"]["done"] and rec["stream"]["finish"] == "stop"
          and rec["stream"]["reasoning"] and "6:00 pm" in rec["stream"]["content"],
          "the stream record: reasoning then content, stop, [DONE]")

    # -- base: the pre-trained shape fails exactly the skin-dependent gates --
    print("== base")
    Handler.engine = Engine("base")
    p, rec = run_probe(port, "base", workdir)
    check(p.returncode == 1 and "IT_PROBE_FAIL" in p.stdout, "exit 1 + IT_PROBE_FAIL")
    # (sampled_count fails too: the wrong train answer at both seeds is 8 of 10)
    base_fails = ["qa_p4", "qa_count", "qa_stop",
                  "think_t1", "think_t2", "think_t3", "think_count",
                  "tool_off_call", "tool_off_answer", "tool_on_call", "tool_on_answer",
                  "tool_count", "sampled_count", "stream"]
    check(rec is not None and expected_fails(rec, base_fails),
          "fails exactly the skin-dependent gates", str(rec and rec["fails"]))
    check("4:40 pm" in p.stdout, "the wrong train answer is shown as the failure")

    # -- reasoning_only: the answer in the wrong field fails QA, not THINK --
    print("== reasoning_only")
    Handler.engine = Engine("reasoning_only")
    p, rec = run_probe(port, "reasoning_only", workdir)
    check(p.returncode == 1, "exit 1")
    # (the thinking-off tool round fails at its CALL; its answer gate reads a
    # tool response the mock answers sanely whichever mode is running)
    ro_fails = [f"qa_p{i}" for i in range(1, 8)] + ["qa_count", "qa_stop",
                "tool_off_call", "tool_count", "sampled_count"]
    check(rec is not None and expected_fails(rec, ro_fails),
          "fails QA + the thinking-off tool round + sampled; THINK passes",
          str(rec and rec["fails"]))
    check("THINK_SUMMARY ok=3/3" in p.stdout, "THINK still passes")
    check("[the text is in the reasoning field" in p.stdout,
          "the QA line says where the text went")

    # -- dead: every gate fails, the JSON is still written, nothing raises --
    print("== dead")
    # stop serving AND close the listening socket: a socket left listening
    # would accept connections it never answers (a hang, not a dead engine)
    srv.shutdown()
    srv.server_close()
    p, rec = run_probe(port, "dead", workdir)
    check(p.returncode == 1 and "IT_PROBE_FAIL" in p.stdout, "exit 1 + IT_PROBE_FAIL")
    check("Traceback" not in p.stderr, "no traceback (dead)", p.stderr[-300:])
    check(rec is not None and "stream" in rec["fails"] and "qa_p1" in rec["fails"],
          "the JSON is written with every gate failed", str(rec and rec["fails"]))

    print(f"{'P266_IT_PROBE_SELFTEST PASS' if N_FAILS == 0 else 'P266_IT_PROBE_SELFTEST FAIL'} "
          f"n_cases={N_CASES} fails={N_FAILS}")
    return 0 if N_FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
