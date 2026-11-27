#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p266_matrix_probe_selftest.py -- offline selftest of p266_matrix_probe.py.

No engine and no real tokenizer: a toy tokenizer renders three template
shapes and a scripted transport answers the three endpoints. Checks the
grading on banked r264 output shapes, the request bodies the probe sends,
the prompt-token oracle and the exit code on each failure path.

  python p266_matrix_probe_selftest.py     (exit 0 = PASS)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p266_matrix_probe", HERE / "p266_matrix_probe.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

GOOD = {"capital_france": "Paris", "fibonacci": "def fib(n):\n    a, b = 0, 1\n    return a",
        "gases": "Hydrogen, helium and nitrogen.", "seven": "seven", "health": "HEALTH CHECK OK",
        "train": "6:00 pm", "mul": "391", "capital_norway": "Oslo", "multiname": "Teemu"}
QWEN_GARBAGE = {
    "capital_france": "<|im_start>user\nWhat is the capital of France?\n<|im_end>\n"
                      "<|im_start|>user\nWhat is the capital of the USA our our our our our",
    "fibonacci": "1 our our our our our our our our",
    "gases": "< our our our our our our our our",
    "health": "</ our our our our our our our our",
}
NL_GARBAGE = "\n\n\n\n\n\n\n\n\n\n\n\n"

fails: list[str] = []
n_cases = 0


def check(name, cond):
    global n_cases
    n_cases += 1
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        fails.append(name)


class ToyTok:
    """Renders by a marker word inside the template text; ids = ordinals."""

    def apply_chat_template(self, messages, chat_template=None, tokenize=False, add_generation_prompt=True,
                            **kw):
        if chat_template and "G4" in chat_template:
            s = "<bos><|think|>\n" if kw.get("enable_thinking") else "<bos>"
            s += "".join(f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in messages)
            return s + "<|turn>model\n<|channel>thought\n<channel|>"
        s = "<bos>" + "".join(f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>\n" for m in messages)
        return s + "<start_of_turn>model\n"

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(c) % 250 for c in text]
        if add_special_tokens:
            ids = [2] + ids
        return {"input_ids": ids}


STATE = {"mode": "good", "served": "g3", "bodies": []}


def fake_post(path, body, timeout=300):
    STATE["bodies"].append((path, body))
    if STATE["mode"] == "dead":
        return 500, {"error": "EngineDead"}, 0.01
    tok = ToyTok()
    q = (body.get("messages") or [{}])[-1].get("content", "")
    pid = next((p for p, t, _ in M.SINGLE if t == q), "multiname" if "What is my name" in q else None)
    if body.get("max_tokens") == 24:  # the garbage probe's sanity request
        pid = None
    if path == "/tokenize":
        kw = body.get("chat_template_kwargs") or {}
        tokkw = dict(kw)
        if STATE["mode"] == "oracle_off":
            tokkw["enable_thinking"] = not tokkw.get("enable_thinking", False)  # invert: force a drift
        served_tmpl = {"g3": "P628 template", "g4": "G4 template"}[STATE["served"]]
        s = tok.apply_chat_template(body["messages"], chat_template=served_tmpl, tokenize=False, **tokkw)
        return 200, {"tokens": tok(s)["input_ids"], "count": len(s)}, 0.01
    if path == "/v1/chat/completions":
        if pid is None:  # the sanity prompt
            STATE["sanity_n"] = STATE.get("sanity_n", 0) + 1
            if STATE["mode"] == "good":
                txt = "HEALTH CHECK OK"
            elif STATE["mode"] == "newlines":
                txt = NL_GARBAGE
            else:  # the banked mkg shape: the two garbage attractors alternate
                txt = "</ our our our our our" if STATE["sanity_n"] % 2 else NL_GARBAGE
        elif STATE["mode"] == "good":
            txt = GOOD[pid] + ("<|channel>thought\n<channel|>" if False else "")
        elif STATE["mode"] == "newlines":
            txt = NL_GARBAGE
        else:
            txt = QWEN_GARBAGE.get(pid, NL_GARBAGE)
        if STATE["mode"] == "good" and pid == "multiname":
            txt = "Teemu"
        finish = "stop" if txt in GOOD.values() or txt == "HEALTH CHECK OK" else "length"
        msg = {"role": "assistant", "content": txt}
        if STATE["mode"] == "thought" and pid is not None:  # a gemma4 reasoning parser split
            msg = {"role": "assistant", "content": GOOD[pid], "reasoning_content": "thinking"}
        ch = {"message": msg, "finish_reason": finish}
        if body.get("logprobs"):  # chat shape: logprobs.content[{token, logprob}]
            ch["logprobs"] = {"content": [{"token": w, "logprob": round(-0.1 * i, 3)}
                                          for i, w in enumerate(txt.split())]}
        return 200, {"choices": [ch],
                     "usage": {"prompt_tokens": 20, "completion_tokens": len(txt.split())}}, 0.01
    if path == "/v1/completions":
        prompt = body["prompt"]
        ids = tok(prompt, add_special_tokens=body.get("add_special_tokens", False))["input_ids"]
        n = len(ids) + (1 if STATE["mode"] == "ptok_drift" else 0)
        if STATE["mode"] == "dead":
            return 500, {"error": "EngineDead"}, 0.01
        if body.get("prompt_logprobs"):
            pls = [None] + [{i: {"logprob": -0.25, "rank": 1}} for i in ids[1:]]
            top = [{7: {"logprob": -0.1, "rank": 1}}]
            return 200, {"choices": [{"text": "x", "prompt_logprobs": pls,
                                      "logprobs": {"top_logprobs": top}}],
                         "usage": {"prompt_tokens": n, "completion_tokens": 1}}, 0.01
        alien = "<|im_start|>" in prompt
        # the multi-turn prompt also contains the capital_norway question in turn 1: the LAST
        # user turn decides the answer
        rpid = ("multiname" if "What is my name" in prompt
                else next((p for p, t, _ in M.SINGLE if t in prompt), None))
        txt = ("<|im_start|>user\nmore\n<|im_end|>\n our our our our our" if alien
               else GOOD[rpid] + "<turn|>\n<|turn>user\nnext")
        ch = {"text": txt, "finish_reason": "stop" if not alien else "length"}
        if body.get("logprobs"):  # completion shape: tokens[] + token_logprobs[]
            ch["logprobs"] = {"tokens": txt.split()[:2], "token_logprobs": [-0.05, -0.2]}
        return 200, {"choices": [ch],
                     "usage": {"prompt_tokens": n, "completion_tokens": len(txt.split())}}, 0.01
    return 404, {"error": "no route " + path}, 0.01


def run_main(mode, served="g3", extra=()):
    STATE.update(mode=mode, served=served, bodies=[])
    M.post = fake_post
    M.load_tokenizer = lambda p: ToyTok()
    d = tempfile.mkdtemp(prefix="p266_selftest_")
    tmpl = {}
    for name, text in (("g3", "P628 template"), ("g4", "G4 template")):
        p = os.path.join(d, name + ".jinja")
        Path(p).write_text(text, encoding="utf-8", newline="\n")
        tmpl[name] = p
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = M.main(["--model", "m", "--tokenizer", "/x", "--out", d, "--tag", "t", "--served", served,
                     "--g3", tmpl["g3"], "--g4", tmpl["g4"], "--workers", "2", *extra])
    doc = json.loads(Path(d, "matrix_t.json").read_text(encoding="utf-8"))
    return rc, doc, buf.getvalue(), STATE["bodies"]


def main() -> int:
    print("== grading (banked r264 shapes and good answers)")
    g = M.grade("capital_france", M.question_of("capital_france"), QWEN_GARBAGE["capital_france"])
    check("r264 m8g capital shape: not ok, REPLOOP + NEXTTURN",
          g["ok"] is False and "REPLOOP" in g["cls"] and "NEXTTURN" in g["cls"])
    g = M.grade("health", M.question_of("health"), "</ our our our our our our our our")
    check("r264 sanity '</ our our..': REPLOOP", "REPLOOP" in g["cls"] and g["ok"] is False)
    g = M.grade("health", M.question_of("health"), NL_GARBAGE)
    check("r264 sanity newlines: EMPTY", "EMPTY" in g["cls"] and g["ok"] is False)
    g = M.grade("capital_france", M.question_of("capital_france"),
                "\nParis<|im_end|>\n<|im_start|>assistant\n<think>\nThe question asks for the capital of "
                "France, which is Paris.\n\nParis<|im_end|>\n")
    check("r264 g0 capital: ok (Paris before the boundary) + NEXTTURN", g["ok"] is True and "NEXTTURN" in g["cls"])
    g = M.grade("capital_france", M.question_of("capital_france"),
                "<|channel>thought\nThe capital of France is Paris.<channel|>Paris<turn|>")
    check("g4 thought channel: ok, answer is Paris", g["ok"] is True and g["answer"] == "Paris")
    g = M.grade("capital_france", M.question_of("capital_france"), "<think>\nblah\n</think>\n\nParis")
    check("qwen think block: ok", g["ok"] is True and g["answer"] == "Paris")
    g = M.grade("capital_france", M.question_of("capital_france"),
                "What is the capital of France? Answer with just the city name.")
    check("prompt echo is flagged ECHO and not ok", "ECHO" in g["cls"] and g["ok"] is False)
    for text, want in (("The second train catches the first at 6:00 pm.", True), ("4:40 pm", False),
                       ("at 6 pm", True), ("18:00", True), ("16:00", False), ("7:00 pm", False),
                       ("It catches up at 6:00pm.", True), ("around 6 in the evening", True)):
        check(f"train {text!r} -> {want}", M.matches("train", text) is want)
    for text, want in (("391", True), ("1391", False), ("The answer is 391.", True), ("3910", False)):
        check(f"mul {text!r} -> {want}", M.matches("mul", text) is want)
    for text, want in (("Hydrogen, helium and nitrogen.", True), ("Iron, copper, gold.", False),
                       ("oxygen", False), ("Neon and argon are gases.", True)):
        check(f"gases {text!r} -> {want}", M.matches("gases", text) is want)
    check("seven lowercase-insensitive", M.matches("seven", "Seven") is True)
    check("health exact phrase", M.matches("health", "HEALTH CHECK OK") is True
          and M.matches("health", "HEALTH CHECK") is False)
    check("fibonacci needs a def line", M.matches("fibonacci", GOOD["fibonacci"]) is True
          and M.matches("fibonacci", "You can use a loop.") is False)
    check("multiname", M.matches("multiname", "Your name is Teemu.") is True
          and M.matches("multiname", "I do not know.") is False)
    check("classes: EMPTY on whitespace only", M.classes(" \n\t") == ["EMPTY"])
    check("classes: REPLOOP on a char loop", "REPLOOP" in M.classes("aaaaaab"))
    check("classes: clean answer has none", M.classes("Paris<turn|>") == [])
    check("pl_stats mean", M.pl_stats({"choices": [{"prompt_logprobs": [None] + [
        {i: {"logprob": -0.5, "rank": 1}} for i in range(4)]}]} , [9, 0, 1, 2, 3])["nll_mean"] == 0.5)
    check("top_of: legacy float map (the b2 run-1 crash shape)",
          M.top_of({"top_logprobs": [{7: -0.1, 3: -2.0}]}) == [(-0.1, 7), (-2.0, 3)])
    check("top_of: object map + empty -> None",
          M.top_of({"top_logprobs": [{7: {"logprob": -0.1, "rank": 1}}]}) == [(-0.1, 7)]
          and M.top_of({}) is None)

    g = M.gen_logprobs({"logprobs": {"content": [{"token": "Par", "logprob": -0.5},
                                                  {"token": "is", "logprob": None}]}})
    check("gen_logprobs: chat shape, None logprob kept", g == [("Par", -0.5), ("is", None)])
    g = M.gen_logprobs({"logprobs": {"tokens": ["a", "b"], "token_logprobs": [-1.0, -2.0]}})
    check("gen_logprobs: completion shape", g == [("a", -1.0), ("b", -2.0)])
    check("gen_logprobs: none -> []", M.gen_logprobs({}) == [] and M.gen_logprobs(None) == [])
    check("glp_stats: empty -> None fields", M.glp_stats([]) == {"glp_first": None, "glp_mean": None})
    st = M.glp_stats([("a", -1.0), ("b", -3.0)])
    check("glp_stats: first + mean", st == {"glp_first": -1.0, "glp_mean": -2.0})

    print("== end-to-end good mode (served qwen)")
    rc, doc, out, bodies = run_main("good")
    check("rc == 0", rc == 0)
    check("oracle match recorded", doc["oracle"]["match"] is True)
    check("P266_MATRIX_DONE line", "P266_MATRIX_DONE tag=t" in out and "rc=0" in out)
    check("three gemma formats rendered (no qwen shape anywhere)", sorted(doc["formats"]) == ["g3", "g4", "g4think"])
    check("g4think carries <|think|>", "<|think|>" in doc["formats"]["g4think"]["capital_france"]["prompt"])
    check("chat matrix size = 2 kw x 16 requests x 9 prompts", len(doc["chat"]) == 2 * 16 * 9)
    check("raw matrix size = 3 fmt x 8 requests x 9 prompts", len(doc["raw"]) == 3 * 8 * 9)
    chat_bodies = [b for p, b in bodies if p == "/v1/chat/completions" and b.get("max_tokens") == M.MAXTOK]
    check("chat bodies carry stop_token_ids [106, 50]", all(b.get("stop_token_ids") == [106, 50] for b in chat_bodies))
    check("chat bodies carry skip_special_tokens False",
          all(b.get("skip_special_tokens") is False for b in chat_bodies))
    check("chat bodies ask logprobs + top_logprobs 1", all(
        b.get("logprobs") is True and b.get("top_logprobs") == 1 for b in chat_bodies))
    check("greedy body = temperature only (no seed/top_p)", any(
        b.get("temperature") == 0 and "seed" not in b and "top_p" not in b for b in chat_bodies))
    card = [b for b in chat_bodies if b.get("temperature") == 1.0 and b.get("top_k") == 64]
    check("card bodies = 1.0 / 0.95 / 64 + seed", len(card) == 2 * 3 * 9 and all(
        b["top_p"] == 0.95 and b["seed"] in M.SEEDS for b in card))
    t1 = [b for b in chat_bodies if b.get("temperature") == 1.0 and "top_k" not in b]
    check("t1only bodies carry no top_p/top_k", len(t1) == 2 * 2 * 9 and all("top_p" not in b for b in t1))
    nothink = [b for b in chat_bodies if (b.get("chat_template_kwargs") or {}).get("enable_thinking") is False]
    check("nothink kwarg on half the chat bodies", len(nothink) == 16 * 9)
    raw_bodies = [b for p, b in bodies if p == "/v1/completions" and not b.get("prompt_logprobs")]
    check("raw bodies use add_special_tokens False", all(b.get("add_special_tokens") is False for b in raw_bodies))
    check("raw bodies never carry stop tokens", all("stop_token_ids" not in b for b in raw_bodies))
    check("raw bodies ask logprobs 1", all(b.get("logprobs") == 1 for b in raw_bodies))
    answered_chat = [r for r in doc["chat"] if r["http"] == 200]
    check("chat records carry per-token glp + stats", all(
        r["glp_first"] == 0.0 and r["glp_mean"] == round(-0.1 * (len(r["glp"]) - 1) / 2, 3)
        and len(r["glp"]) == len(r["text"].split()) for r in answered_chat))
    answered_raw = [r for r in doc["raw"] if r["http"] == 200]
    check("raw records carry per-token glp + stats", all(
        r["glp_first"] == -0.05 and r["glp_mean"] == -0.125 and len(r["glp"]) == 2 for r in answered_raw))
    check("summary lines carry lp1= and lp_gen= (12 chat + 15 raw + 1 gate line)", sum(
        1 for ln in out.splitlines() if " lp1=" in ln and " lp_gen=" in ln) == 2 * 6 + 3 * 5 + 1)
    check("raw prompt-token oracle all true", all(r.get("ptok_ok") for r in doc["raw"] if r.get("http") == 200))
    check("raw cells are all gemma formats and grade ok",
          all(r["fmt"] in ("g3", "g4", "g4think") for r in doc["raw"])
          and all(r["ok"] for r in doc["raw"] if r["http"] == 200))
    check("gate section: battery ok + sanity stable",
          len(doc["gate"]) == 5 and all(r["ok"] for r in doc["gate"]))
    gate_bodies = [b for p, b in bodies if p == "/v1/chat/completions" and b.get("max_tokens") == 64]
    sanity_bodies = [b for p, b in bodies if p == "/v1/chat/completions" and b.get("max_tokens") == 24]
    check("gate bodies carry no logprobs (verbatim requests)", len(sanity_bodies) == 2 and not any(
        "logprobs" in b for b in gate_bodies + sanity_bodies))
    check("gate battery bodies = r264 bytes (max_tokens 64, temperature 0, no extras)", len(gate_bodies) == 3
          and all(set(b) == {"model", "messages", "max_tokens", "skip_special_tokens", "temperature"}
                  for b in gate_bodies))
    nll = doc["nll"]
    check("nll para stats + 24 prompt stats (3 fmt x 8 single prompts)",
          nll["para"]["nll"]["nll_mean"] == 0.25 and len(nll["prompts"]) == 24
          and all(r["ptok_ok"] for r in nll["prompts"].values()))
    nll_lines = [ln for ln in out.splitlines() if ln.startswith("NLL ") and not ln.startswith("NLL para")]
    check("NLL summary lines present", "NLL para nll_mean=0.25" in out and len(nll_lines) == 24)
    check("summary: chat card plain ok=9", "chat kw=plain card: n=27 ok=27" in out)
    s = [ln for ln in out.splitlines() if ln.startswith("chat kw=plain t0:")]
    check("summary: greedy cells stable (distinct_texts=9)", len(s) == 1 and "distinct_texts=9" in s[0])

    print("== cell selectors")
    rc, doc, out, _ = run_main("good", extra=["--chat-cells", "t0", "--raw-cells", "t0,card"])
    check("chat-cells t0 -> 2 kw x 3 x 9", rc == 0 and len(doc["chat"]) == 2 * 3 * 9
          and {r["cell"] for r in doc["chat"]} == {"t0"})
    check("raw-cells t0,card -> 3 fmt x (3+2) x 9", len(doc["raw"]) == 3 * 5 * 9
          and {r["cell"] for r in doc["raw"]} == {"t0", "card"})
    try:
        run_main("good", extra=["--chat-cells", "t0,nope"])
        raised = False
    except SystemExit as e:
        raised = "nope" in str(e)
    check("unknown cell name stops the probe", raised)

    print("== served-template refusal (no substitution is lawful)")
    import contextlib
    import io
    import tempfile
    d_ref = tempfile.mkdtemp(prefix="p266_refusal_")
    g4_ref = os.path.join(d_ref, "g4.jinja")
    Path(g4_ref).write_text("G4 template", encoding="utf-8", newline="\n")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            M.main(["--model", "m", "--tokenizer", "/x", "--out", d_ref, "--tag", "t",
                    "--served", "g3", "--g4", g4_ref, "--workers", "2"])
        refused = False
    except SystemExit as e:
        refused = "P266_REFUSE_SERVED_TEMPLATE_ABSENT" in str(e)
    check("probe refuses when the served gemma template is absent", refused)

    print("== served g4 adds the think kwarg variant")
    rc, doc, out, _ = run_main("good", served="g4")
    check("rc == 0 and oracle via g4 kwargs", rc == 0 and doc["oracle"]["match"] is True)
    check("chat matrix size = 3 kw x 16 x 9", len(doc["chat"]) == 3 * 16 * 9)

    print("== failure paths")
    rc, doc, out, _ = run_main("ptok_drift")
    check("ptok drift -> rc 1 + ptok_mismatch>0", rc == 1 and "ptok_mismatch=" in out
          and "ptok_mismatch=0" not in out)
    check("ptok drift does not change grading counts", any(r["ok"] for r in doc["raw"]))
    rc, doc, out, _ = run_main("oracle_off", served="g4")  # only g4 carries think kwargs to drift
    check("oracle mismatch -> rc 1, oracle None/False", rc == 1 and doc["oracle"]["match"] is not True)
    rc, doc, out, _ = run_main("dead")
    check("dead engine -> rc 1, every record carries http 500, no exception",
          rc == 1 and all(r["http"] == 500 for r in doc["chat"] + doc["raw"] + doc["gate"])
          and "http_err=" in out)
    check("dead engine: nll records carry err", all("err" in r for r in doc["nll"]["prompts"].values()))
    rc, doc, out, _ = run_main("garbage")
    check("banked-shape garbage mode: sanity flip is visible (two different garbage texts)",
          rc == 0 and doc["gate"][3]["text"] != doc["gate"][4]["text"]
          and doc["gate"][3]["ok"] is False and doc["gate"][4]["ok"] is False
          and [c for c in doc["gate"][3]["cls"] if c in ("REPLOOP", "EMPTY")] == ["REPLOOP"]
          and [c for c in doc["gate"][4]["cls"] if c in ("REPLOOP", "EMPTY")] == ["EMPTY"])
    gmode = [r for r in doc["chat"] if r["cell"] == "t0" and r["kw"] == "plain"]
    check("banked-shape garbage mode: every matrix greedy chat cell is not ok",
          all(r["ok"] is False for r in gmode))
    rc, doc, out, _ = run_main("thought")
    check("reasoning parser split: content graded, reasoning recorded",
          rc == 0 and all(r["ok"] and r["reasoning"] == "thinking" for r in doc["chat"]
                         if r["cell"] == "t0" and r["pid"] != "health"))
    print(f"P266_MATRIX_PROBE_SELFTEST {'PASS' if not fails else 'FAIL'} n_cases={n_cases}"
          f" fails={len(fails)}")
    for f in fails:
        print("  FAILED:", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
