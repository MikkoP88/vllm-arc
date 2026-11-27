#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p266_matrix_probe.py -- gemma-4 request-parameter matrix (r266).

Measures on ONE engine boot which REQUEST-side parameters decide the answer
quality of google/gemma-4-31B. Three sections, all recorded with the full
text, finish reason and token counts -- nothing here judges the model:

  chat  the OpenAI chat endpoint with the template this boot serves, over a
        sampling grid (greedy / 0.3 / 0.7 / the model card's 1.0 + top_p
        0.95 + top_k 64 / temperature alone), with and without
        enable_thinking; greedy is repeated solo and as a concurrent batch.
  raw   /v1/completions on pre-rendered prompts, so several prompt formats
        (the served qwen skin, the P6.28 hand-written template, the official
        google/gemma-4-31B-it template) compare on the same weights, dtype
        and kernels.
  nll   teacher-forced logprob of every prompt token plus a template-free
        paragraph: engine-numerics numbers for the transformers reference
        (p266_cpu_reference.py), which replays the same token sequences.

Every chat and raw request also asks for the logprob of each generated
token (recorded as glp): a token the model emits with logprob near 0 is a
confident choice, one emitted at -3 is a low-confidence pick -- this
separates "the model confidently answers something else" from "the
distribution is flat". The gate section sends the gate's requests verbatim
(no logprobs).

Runs INSIDE the engine container: it needs the checkpoint tokenizer to
render the prompts and to count the prompt tokens the engine saw.

  docker cp p266_matrix_probe.py $CNAME:/root/
  docker exec $CNAME /opt/venv/bin/python /root/p266_matrix_probe.py \
      --model gemma4-31b --tokenizer /models/target --out /root/p266_out \
      --tag <leg> --served qwen --qwen <file> --g3 <file> --g4 <file>

Exit 0 when every section ran and the prompt-token oracle held on every
answered raw/NLL request (the engine tokenized each prompt exactly as this
probe did); 1 otherwise. Content is NOT gated here.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("P266_BASE", "http://127.0.0.1:8000")
WORKERS = [6]  # request pool width (main() sets it from --workers)
# <turn|> and <|tool_response>: the turn-ending tokens the official
# checkpoint's generation_config lists (eos 1, 106, 50). The base
# checkpoint's generation_config lists only eos 1, so a request must name
# them for a generation to stop at the end of the model's turn.
TURN_STOPS = [106, 50]
SEEDS = (101, 202, 303)
MAXTOK = 96

GASES = ("hydrogen", "helium", "nitrogen", "oxygen", "fluorine", "neon", "chlorine", "argon",
         "krypton", "xenon", "radon")
TRAIN_RE = r"(?<![\d.])(6[:.]00|6)\s*(pm|p\.m\.|in the evening)|(?<!\d)18[:.]00|six( o'?clock)? pm"

# The first three are the gate's text-battery prompts verbatim
# (tools/ar/r264_cert29_gate.sh text_battery); "health" is the garbage
# probe's sanity prompt; "train" is the P6.28 G1 probe (3pm/4pm: the gap is
# 60 km at 4pm, closed at 30 km/h -> 6:00 pm).
SINGLE = [
    ("capital_france", "What is the capital of France? Answer with just the city name.", r"\bparis\b"),
    ("fibonacci", "Write a Python function that returns the n-th Fibonacci number using iteration.",
     r"\bdef\b[^\n]*\([^\n]*\)[^\n]*:"),
    ("gases", "Name three chemical elements that are gases at room temperature.", None),
    ("seven", "Say the number seven in words, then stop.", r"\bseven\b"),
    ("health", "Reply with exactly: HEALTH CHECK OK", r"health check ok"),
    ("train", "If a train leaves at 3pm traveling 60 km/h and another at 4pm traveling 90 km/h "
              "the same direction, when does the second catch the first? Answer with just the time.",
     TRAIN_RE),
    ("mul", "What is 17 times 23? Answer with just the number.", r"(?<!\d)391(?!\d)"),
    ("capital_norway", "What is the capital of Norway? Answer with just the city name.", r"\boslo\b"),
]
MULTI = [
    {"role": "user", "content": "My name is Teemu. What is the capital of Norway? "
                                "Answer with just the city name."},
    {"role": "assistant", "content": "Oslo."},
    {"role": "user", "content": "What is my name? Answer with just the name."},
]
MULTI_ID, MULTI_RE = "multiname", r"\bteemu\b"
GATE_IDS = ("capital_france", "fibonacci", "gases")

PARA = (
    "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions. It extends "
    "from the Arctic Ocean in the north to the Southern Ocean in the south, and is bounded "
    "by Asia and Australia in the west and the Americas in the east. Its average depth is "
    "about 4,000 metres, and its deepest point, the Challenger Deep in the Mariana Trench, "
    "reaches roughly 10,900 metres. The ocean covers more than 60 million square miles, "
    "nearly a third of the planet's surface, and it holds more than half of the world's "
    "ocean water. Ocean currents move warm water from the equator toward the poles and "
    "cold water back toward the equator, and these currents strongly influence the climate "
    "of the coastal regions along their path."
)

BOUNDARY_RE = re.compile(
    r"<turn\|>|<\|turn>|<\|channel>|<channel\|>|<\|im_start\|>|<\|im_end\|>|</?think>"
    r"|<start_of_turn>|<end_of_turn>|<\|tool_call>|<tool_call\|>|<\|tool_response>")
THOUGHT_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.S)
THINK_RE = re.compile(r"\s*<think>.*?</think>\s*", re.S)
REPLOOP_WORD_RE = re.compile(r"(\S+)(?:\s+\1){3,}")
REPLOOP_CHAR_RE = re.compile(r"(.{1,3})\1{5,}")
NEXTTURN_RE = re.compile(r"<\|im_start\|>(user|assistant)|<\|turn>user|<start_of_turn>(user|model)")


class Cell:
    """One sampling point: name, request sampling fields, repeats."""

    def __init__(self, name, n, concurrent=False, **samp):
        self.name, self.n, self.concurrent, self.samp = name, n, concurrent, samp

    def body(self, rep):
        b = dict(self.samp)
        if "temperature" in self.samp and self.samp["temperature"] > 0:
            b["seed"] = SEEDS[rep % len(SEEDS)]
        return b


CELLS = [
    Cell("t0", 3, temperature=0.0),
    Cell("t0b4", 4, concurrent=True, temperature=0.0),
    Cell("t03", 2, temperature=0.3, top_p=0.95, top_k=64),
    Cell("t07", 2, temperature=0.7, top_p=0.95, top_k=64),
    Cell("card", 3, temperature=1.0, top_p=0.95, top_k=64),  # google/gemma-4-31B model card
    Cell("t1only", 2, temperature=1.0),  # top_p/top_k = the checkpoint generation_config defaults
]
RAW_REPEATS = {"t0": 3, "t03": 1, "t07": 1, "card": 2, "t1only": 1}


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read()), time.time() - t0
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read()[:400].decode("utf-8", "replace")}, time.time() - t0


def answer_part(text):
    """The model's first answer: drop a leading thought/think block, cut at the first
    turn/channel boundary (a fabricated next turn is not part of the answer)."""
    t = text or ""
    if t.startswith("<|channel>thought"):
        m = THOUGHT_RE.match(t)
        if m:
            t = t[m.end():]
    m = THINK_RE.match(t)
    if m:
        t = t[m.end():]
    cut = BOUNDARY_RE.search(t)
    if cut:
        t = t[:cut.start()]
    return t.strip()


def gen_logprobs(ch):
    """[(token, logprob)] of the generated tokens as the engine reports them; [] when
    the response carries none."""
    lp = (ch or {}).get("logprobs") or {}
    if lp.get("content"):
        return [(d.get("token"), None if d.get("logprob") is None else round(d["logprob"], 3))
                for d in lp["content"]]
    toks, lps = lp.get("tokens"), lp.get("token_logprobs")
    if toks and lps:
        return [(t, None if x is None else round(x, 3)) for t, x in zip(toks, lps)]
    return []


def glp_stats(g):
    if not g:
        return {"glp_first": None, "glp_mean": None}
    vals = [x for _, x in g if x is not None]
    return {"glp_first": g[0][1], "glp_mean": round(sum(vals) / len(vals), 3) if vals else None}


def classes(text):
    cls = []
    if not (text or "").strip():
        cls.append("EMPTY")
    if REPLOOP_WORD_RE.search(text or "") or REPLOOP_CHAR_RE.search(text or ""):
        cls.append("REPLOOP")
    if NEXTTURN_RE.search(text or ""):
        cls.append("NEXTTURN")
    return cls


def matches(pid, answer):
    low = answer.lower()
    if pid == "gases":
        return sum(1 for g in GASES if re.search(r"\b" + g + r"\b", low)) >= 2
    exp = dict((p[0], p[2]) for p in SINGLE)[pid] if pid != MULTI_ID else MULTI_RE
    return re.search(exp, low) is not None


def grade(pid, question, text):
    ans = answer_part(text)
    echo = bool(question) and " ".join(re.findall(r"\w+", question.lower())) in " ".join(
        re.findall(r"\w+", (text or "").lower()))
    return {"ok": matches(pid, ans) if ans else False, "cls": classes(text) + (["ECHO"] if echo else []),
            "answer": ans[:240]}


def messages_for(pid):
    return MULTI if pid == MULTI_ID else [{"role": "user", "content": dict((p[0], p[1]) for p in SINGLE)[pid]}]


def load_tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path)


def render_formats(tok, specs, pids):
    """specs: {fmt: (template_file_or_None, kwargs)} -> {fmt: {pid: {prompt, ids, ntok}}}."""
    out = {}
    for fmt, (path, kw) in specs.items():
        tmpl = Path(path).read_text(encoding="utf-8") if path else None
        out[fmt] = {}
        for pid in pids:
            s = tok.apply_chat_template(messages_for(pid), chat_template=tmpl, tokenize=False,
                                        add_generation_prompt=True, **kw)
            ids = tok(s, add_special_tokens=False)["input_ids"]
            out[fmt][pid] = {"prompt": s, "ids": ids, "ntok": len(ids)}
    return out


def engine_prompt_ids(model, msgs, kw):
    body = {"model": model, "messages": msgs, "add_special_tokens": False}
    if kw:
        body["chat_template_kwargs"] = kw
    st, resp, _ = post("/tokenize", body)
    return (resp or {}).get("tokens") if st == 200 else None


def do_chat(model, msgs, samp, kw, maxtok=MAXTOK, lp=True, **extra):
    body = {"model": model, "messages": msgs, "max_tokens": maxtok, "skip_special_tokens": False,
            **samp, **extra}
    if lp:
        body.update(logprobs=True, top_logprobs=1)
    if kw:
        body["chat_template_kwargs"] = kw
    st, resp, wall = post("/v1/chat/completions", body)
    rec = {"http": st, "wall": round(wall, 2), "req": {k: v for k, v in body.items() if k != "messages"}}
    if st != 200:
        rec["err"] = str(resp)[:300]
        return rec
    ch = resp["choices"][0]
    msg = ch.get("message") or {}
    content = msg.get("content")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    text = content or reasoning or ""
    rec.update(text=text, content=content, reasoning=reasoning, finish=ch.get("finish_reason"),
               ptok=resp["usage"]["prompt_tokens"], ctok=resp["usage"]["completion_tokens"])
    rec["glp"] = gen_logprobs(ch)
    rec.update(glp_stats(rec["glp"]))
    return rec


def do_raw(model, prompt, samp, **extra):
    body = {"model": model, "prompt": prompt, "max_tokens": MAXTOK, "add_special_tokens": False,
            "skip_special_tokens": False, "logprobs": 1, **samp, **extra}
    st, resp, wall = post("/v1/completions", body)
    rec = {"http": st, "wall": round(wall, 2), "req": {k: v for k, v in body.items() if k != "prompt"}}
    if st != 200:
        rec["err"] = str(resp)[:300]
        return rec
    ch = resp["choices"][0]
    rec.update(text=ch["text"], finish=ch.get("finish_reason"), ptok=resp["usage"]["prompt_tokens"],
               ctok=resp["usage"]["completion_tokens"])
    rec["glp"] = gen_logprobs(ch)
    rec.update(glp_stats(rec["glp"]))
    return rec


def run_pool(jobs, workers):
    """jobs: list of (key, fn) -> records keyed by job."""
    out = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn): key for key, fn in jobs}
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result()
    return out


def question_of(pid):
    return messages_for(pid)[-1]["content"]


def section_chat(model, pids, kw_variants, cells=None):
    recs, jobs = [], []
    cells = [c for c in CELLS if c.name in cells] if cells else CELLS
    for kwname, kw in kw_variants:
        for cell in cells:
            for pid in pids:
                for rep in range(cell.n):
                    jobs.append(((kwname, cell.name, pid, rep),
                                 lambda c=cell, p=pid, r=rep, k=kw: do_chat(
                                     model, messages_for(p), c.body(r), k, stop_token_ids=TURN_STOPS)))
    # greedy solo repeats run sequentially (the stability condition is "same engine, one request
    # in flight"); everything else through the pool.
    res = {}
    for key, fn in [j for j in jobs if j[0][1] == "t0"]:
        res[key] = fn()
        time.sleep(0.2)
    res.update(run_pool([j for j in jobs if j[0][1] != "t0"], WORKERS[0]))
    for key in sorted(res):
        kwname, cname, pid, rep = key
        rec = res[key]
        rec.update(kw=kwname, cell=cname, pid=pid, rep=rep)
        if rec.get("http") == 200:
            rec.update(grade(pid, question_of(pid), rec["text"]))
        recs.append(rec)
    return recs


def section_raw(model, formats, pids, cells=None):
    recs, jobs = [], []
    for fmt in formats:
        for cname, n in RAW_REPEATS.items():
            if cells and cname not in cells:
                continue
            cell = next(c for c in CELLS if c.name == cname)
            for pid in pids:
                for rep in range(n):
                    jobs.append(((fmt, cname, pid, rep),
                                 lambda c=cell, f=fmt, p=pid, r=rep: do_raw(
                                     model, formats[f][p]["prompt"], c.body(r))))
    res = run_pool(jobs, WORKERS[0])
    for key in sorted(res):
        fmt, cname, pid, rep = key
        rec = res[key]
        rec.update(fmt=fmt, cell=cname, pid=pid, rep=rep)
        if rec.get("http") == 200:
            rec.update(grade(pid, question_of(pid), rec["text"]))
            rec["ptok_ok"] = rec["ptok"] == formats[fmt][pid]["ntok"]
        recs.append(rec)
    return recs


def section_nll(model, formats, pids, para_ids):
    out = {"para": {}, "prompts": {}}
    st, resp, _ = post("/v1/completions", {"model": model, "prompt": PARA, "max_tokens": 1,
                                           "add_special_tokens": True, "prompt_logprobs": 1,
                                           "logprobs": 5, "temperature": 0})
    out["para"]["http"] = st
    if st == 200:
        out["para"]["nll"] = pl_stats(resp, None)
        out["para"]["ptok_ok"] = resp["usage"]["prompt_tokens"] == len(para_ids)
        out["para"]["first_top"] = top_of(resp["choices"][0].get("logprobs"))
    else:
        out["para"]["err"] = str(resp)[:300]
    for fmt in formats:
        for pid in pids:
            if pid == MULTI_ID:
                continue
            ids = formats[fmt][pid]["ids"]
            st, resp, _ = post("/v1/completions", {
                "model": model, "prompt": formats[fmt][pid]["prompt"], "max_tokens": 1,
                "add_special_tokens": False, "prompt_logprobs": 1, "logprobs": 5, "temperature": 0})
            rec = {"http": st}
            if st == 200:
                rec["nll"] = pl_stats(resp, ids)
                rec["ptok_ok"] = resp["usage"]["prompt_tokens"] == len(ids)
                rec["first_top"] = top_of(resp["choices"][0].get("logprobs"))
                rec["greedy_first"] = resp["choices"][0]["text"]
            else:
                rec["err"] = str(resp)[:300]
            out["prompts"][f"{fmt}/{pid}"] = rec
    return out


def pl_stats(resp, ids):
    """Per-token logprobs of the prompt (positions >= 1): mean nll, tail mean (last 8), top-1 agreement."""
    pls = resp["choices"][0].get("prompt_logprobs") or []
    lps, agree, n = [], 0, 0
    for i, d in enumerate(pls):
        if not d:
            continue
        tid = ids[i] if ids else max(d, key=lambda t: d[t].get("rank", 1 << 30))
        lp = d.get(tid)
        if lp is None:
            continue
        lps.append(lp["logprob"])
        agree += int(min(d, key=lambda t: d[t].get("rank", 1 << 30)) == tid)
        n += 1
    if not lps:
        return {"n": 0}
    tail = lps[-8:]
    return {"n": n, "nll_mean": round(-sum(lps) / len(lps), 4), "nll_tail8": round(-sum(tail) / len(tail), 4),
            "top1_agree": round(agree / len(lps), 3), "tok_nll": [round(-x, 3) for x in lps]}


def top_of(logprobs):
    if not logprobs or not logprobs.get("top_logprobs"):
        return None
    d = logprobs["top_logprobs"][0]
    return sorted(((round(v["logprob"], 3), k) for k, v in d.items()), reverse=True)[:5]


def section_gate(model):
    """The r264 gate requests verbatim: text battery (think on, 64 tokens) and the garbage
    probe's sanity request (nothink, 24 tokens), the latter twice."""
    recs = []
    for pid in GATE_IDS:
        recs.append(dict(do_chat(model, messages_for(pid), {"temperature": 0}, {}, maxtok=64, lp=False),
                         pid=pid, kind="battery"))
    samp = {"top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}
    for i in range(2):
        r = do_chat(model, messages_for("health"),
                    {"temperature": 0, **samp}, {"enable_thinking": False}, maxtok=24, lp=False)
        r.update(pid="health", kind=f"sanity{i}")
        recs.append(r)
    for r in recs:
        r["text"] = r.get("content") or r.get("reasoning") or r.get("text") or ""
        r.update(grade(r["pid"], question_of(r["pid"]), r["text"]))
    return recs


def summarize(chat, raw, gate):
    lines = []

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return f"{sum(vals) / len(vals):.2f}" if vals else "-"

    def line(tag, recs):
        okr = [r for r in recs if r.get("http") == 200]
        cnt = lambda c: sum(1 for r in okr if c in r.get("cls", []))  # noqa: E731
        distinct = len({r.get("text") for r in okr})
        lines.append(f"{tag}: n={len(recs)} ok={sum(1 for r in okr if r.get('ok'))} "
                     f"http_err={len(recs) - len(okr)} reploop={cnt('REPLOOP')} empty={cnt('EMPTY')} "
                     f"nextturn={cnt('NEXTTURN')} echo={cnt('ECHO')} distinct_texts={distinct} "
                     f"lp1={mean([r.get('glp_first') for r in okr])} "
                     f"lp_gen={mean([r.get('glp_mean') for r in okr])}")
    for kw in sorted({r["kw"] for r in chat}):
        for cname in [c.name for c in CELLS]:
            line(f"chat kw={kw} {cname}", [r for r in chat if r["kw"] == kw and r["cell"] == cname])
    for fmt in sorted({r["fmt"] for r in raw}):
        for cname in RAW_REPEATS:
            rs = [r for r in raw if r["fmt"] == fmt and r["cell"] == cname]
            line(f"raw fmt={fmt} {cname}", rs)
            if rs:
                lines.append(f"    ptok_ok={sum(1 for r in rs if r.get('ptok_ok'))}/{len(rs)}")
    line("gate264 battery+sanity", gate)
    for r in gate:
        lines.append(f"    [gate {r['kind']}] finish={r.get('finish')} {r['text'][:90]!r}")
    return lines


def per_prompt_lines(recs, keyfields):
    out = []
    for key in sorted({tuple(r[k] for k in keyfields) for r in recs}):
        rs = [r for r in recs if tuple(r[k] for k in keyfields) == key]
        parts = []
        for r in rs:
            if r.get("http") != 200:
                parts.append(f"{r['pid']}:HTTP{r['http']}")
            else:
                parts.append(f"{r['pid']}:{'ok' if r.get('ok') else 'NO'}[{r.get('finish', '?')[:1]}"
                             f"{'R' if 'REPLOOP' in r.get('cls', []) else ''}]"
                             f"{(r.get('answer') or r.get('text') or '')[:38]!r}")
        out.append("  " + " ".join(str(k) for k in key) + " :: " + " | ".join(parts))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--served", required=True, help="format name equal to the served template")
    ap.add_argument("--qwen")
    ap.add_argument("--g3")
    ap.add_argument("--g4")
    ap.add_argument("--cells", default="chat,raw,nll,gate")
    ap.add_argument("--chat-cells", default="", help="comma list of chat sampling cells (default all)")
    ap.add_argument("--raw-cells", default="", help="comma list of raw sampling cells (default all)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    WORKERS[0] = max(1, a.workers)
    t_start = time.time()
    tok = load_tokenizer(a.tokenizer)
    specs = {}
    if a.qwen:
        specs["qwen"] = (a.qwen, {})
        specs["qwen_nothink"] = (a.qwen, {"enable_thinking": False})
    if a.g3:
        specs["g3"] = (a.g3, {})
    if a.g4:
        specs["g4"] = (a.g4, {"enable_thinking": False})
        specs["g4think"] = (a.g4, {"enable_thinking": True})
    pids = [p[0] for p in SINGLE] + [MULTI_ID]
    formats = render_formats(tok, specs, pids)
    para_ids = tok(PARA, add_special_tokens=True)["input_ids"]
    eng = engine_prompt_ids(a.model, messages_for("capital_france"),
                            specs.get(a.served, (None, {}))[1])
    mine = formats[a.served]["capital_france"]["ids"] if a.served in formats else None
    oracle = {"served": a.served, "engine_ids": eng, "probe_ids": mine,
              "match": eng == mine if eng is not None and mine is not None else None}
    print(f"PROMPT_ORACLE served={a.served} engine_ntok={len(eng) if eng else None} "
          f"probe_ntok={len(mine) if mine else None} match={oracle['match']}", flush=True)
    for fmt in formats:
        p = formats[fmt]["capital_france"]
        print(f"FORMAT {fmt} ntok={p['ntok']} prompt={p['prompt'][:230]!r}", flush=True)
    cells = [c.strip() for c in a.cells.split(",") if c.strip()]
    ccells = pick(a.chat_cells, [c.name for c in CELLS])
    rcells = pick(a.raw_cells, list(RAW_REPEATS))
    chat = section_chat(a.model, pids, kw_variants_for(a), ccells) if "chat" in cells else []
    print(f"SECTION chat done {time.time() - t_start:.0f}s", flush=True)
    raw = section_raw(a.model, formats, pids, rcells) if "raw" in cells and formats else []
    print(f"SECTION raw done {time.time() - t_start:.0f}s", flush=True)
    nll = section_nll(a.model, formats, [p[0] for p in SINGLE], para_ids) if "nll" in cells else {}
    gate = section_gate(a.model) if "gate" in cells else []
    lines = summarize(chat, raw, gate)
    print("\n".join(lines), flush=True)
    print("-- per cell:", flush=True)
    print("\n".join(per_prompt_lines(chat, ("kw", "cell"))), flush=True)
    print("\n".join(per_prompt_lines(raw, ("fmt", "cell"))), flush=True)
    if nll.get("para", {}).get("nll"):
        print(f"NLL para nll_mean={nll['para']['nll']['nll_mean']} n={nll['para']['nll']['n']} "
              f"ptok_ok={nll['para'].get('ptok_ok')}", flush=True)
    for k in sorted(nll.get("prompts", {})):
        r = nll["prompts"][k]
        if r.get("nll", {}).get("n"):
            print(f"NLL {k} nll_mean={r['nll']['nll_mean']} tail8={r['nll']['nll_tail8']} "
                  f"top1={r['nll']['top1_agree']} first_top={r['first_top']}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    doc = {"meta": {"tag": a.tag, "model": a.model, "tokenizer": a.tokenizer, "served": a.served,
                    "specs": {k: list(v) for k, v in specs.items()}, "args": vars(a),
                    "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "turn_stops": TURN_STOPS, "maxtok": MAXTOK, "seeds": SEEDS},
           "oracle": oracle, "formats": formats, "para_ids": para_ids,
           "chat": chat, "raw": raw, "nll": nll, "gate": gate, "summary": lines}
    path = os.path.join(a.out, f"matrix_{a.tag}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False)
    answered = [r for r in raw if r.get("http") == 200]
    bad = [r for r in answered if not r.get("ptok_ok")]
    para_bad = bool(nll.get("para", {}).get("nll")) and not nll["para"].get("ptok_ok")
    rc = 0 if oracle["match"] and not bad and not para_bad else 1
    print(f"P266_MATRIX_DONE tag={a.tag} chat={len(chat)} raw={len(raw)} ptok_mismatch={len(bad)} "
          f"para_ptok_ok={nll['para'].get('ptok_ok') if nll else None} oracle={oracle['match']} "
          f"wall={time.time() - t_start:.0f}s json={path} rc={rc}", flush=True)
    return rc


def pick(spec, allowed):
    """No spec = every cell; a comma list = those cells (an unknown name stops the probe)."""
    if not spec:
        return None
    names = [c.strip() for c in spec.split(",") if c.strip()]
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise SystemExit(f"unknown cell(s) {unknown}; allowed: {allowed}")
    return names


def kw_variants_for(a):
    kw = [("plain", {}), ("nothink", {"enable_thinking": False})]
    if a.served in ("g4", "g4think"):
        kw.append(("think", {"enable_thinking": True}))
    return kw


if __name__ == "__main__":
    sys.exit(main())
