#!/usr/bin/env python3
"""p618 longprobe v4 — thinking OFF for signal (P6.18b; v4 P6.47).

v1 flaw: max_tokens=768 consumed by the thinking channel -> EMPTY
content -> vacuous CLEAN. v2 raised the budget to 3072 + hashed
reasoning_content — STILL vacuous on TQ faces (the response carries no
reasoning field and thinking ate 3072 too: chars=[0,0,0]). v3 turns
thinking OFF via chat_template_kwargs (Qwen3 family kwarg) so answers
are direct; reasoning/content capture kept as fallback signal.

v4 (P6.47 cert25y): the REP-LOOP flag alone can't disposition gemma
faces — gemma-4 ignores enable_thinking, so its generations are
reasoning-channel, and the synthetic essay repeats a 25-char window per
paragraph (sane quoting trips the >=4 flag). Flags unchanged (verdicts
stay comparable); banks add per-phase top-window counts, the
content/reasoning char split, and 2K raw heads per phase.

Usage: p647_longprobe.py <tag> <mode>
  mode = full  (cold+rehit+rehit2 + temp1 x2, all classes)
         rehit (post-churn rehits only)
Writes ./longprobe_<tag>_<mode>.json + .txt next to cwd.
Stdlib only; talks to http://127.0.0.1:8000 (rig host python3).
"""
import hashlib
import os
import json
import re
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = os.environ.get("ARC_PROBE_MODEL", "qwen3.8-27b-fp8")
MAXTOK_T0 = 1536   # v3: direct answers — 1536 is ample
MAXTOK_T1 = 768

MARKERS = {
    "essay": "XRK7-ALPHA",
    "code": "XRK7-BRAVO",
    "multi": "XRK7-CHARLIE",
}


def para(i, topic_words):
    a = topic_words[i % len(topic_words)]
    b = topic_words[(i * 7 + 3) % len(topic_words)]
    c = topic_words[(i * 13 + 5) % len(topic_words)]
    return (f"Section {i}. The {a} of the {b} follows a measured cadence: "
            f"first the {c} is established, then the {a} is re-examined "
            f"under the constraint that the {b} remains fixed. Historical "
            f"note {i}: engineers treating the {c} as a free variable "
            f"observed the {a} drift, so the {b} was pinned by contract "
            f"in revision {100 + i}.")


def build_essay(n_words_target):
    tw = ["pipeline", "scheduler", "cache", "allocator", "kernel",
          "tensor", "layout", "engine", "queue", "buffer"]
    out = [f"Study document {MARKERS['essay']}. Summarize the argument "
           f"below, then write a 400-word conclusion."]
    i = 0
    words = 60
    while words < n_words_target:
        out.append(para(i, tw))
        words += 55
        i += 1
    out.append("End of document. Now write the conclusion.")
    return "\n".join(out)


def build_code(n_words_target):
    fns = []
    i = 0
    words = 40
    out = [f"Reference codebase {MARKERS['code']} (do not run; summarize "
           f"the API and then write an improved `merge_chunks` in Python)."]
    while words < n_words_target:
        fns.append(
            f"def helper_{i}(rows, stride={i}):\n"
            f"    # invariant: len(rows) % stride == {i}\n"
            f"    acc = []\n"
            f"    for r in rows[::{max(1, i % 5)}]:\n"
            f"        acc.append((r['id'] * {i + 1}) % 977)\n"
            f"    return sorted(set(acc))\n")
        words += 45
        i += 1
    out.extend(fns)
    out.append("End of codebase. Now write the improved function.")
    return "\n".join(out)


def build_multi(n_words_target):
    lines = [f"Moniste {MARKERS['multi']} — kolme kieltä, yksi johtopäätös. "
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


def build_simple():
    return ("Write a Python function `merge_intervals(intervals)` that "
            "merges overlapping ranges and returns them sorted. Explain "
            "the algorithm in three sentences.")


PROMPTS = {
    "essay": build_essay(1650),     # ~2.2k tok: 1 full 1664 page + tail
    "code": build_code(3300),       # ~4.4k tok: 2 full pages + tail
    "multi": build_multi(1650),     # ~2.3k tok multilingual
    "simple": build_simple(),       # ordinary chat traffic (78 tok)
}


def _post(body):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def gen(content, temperature, max_tokens, seed=None):
    body = {"model": MODEL, "stream": False, "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": content}]}
    if seed is not None:
        body["seed"] = seed
    t0 = time.perf_counter()
    nothink = True
    try:
        r = _post(body)
    except urllib.error.HTTPError as e:
        if e.code != 400:
            raise
        # template lacks the kwarg — fall back to a large budget and
        # rely on the reasoning/content capture (v2 mode)
        body.pop("chat_template_kwargs", None)
        body["max_tokens"] = max(max_tokens, 6144)
        r = _post(body)
        nothink = False
    dt = time.perf_counter() - t0
    ch = r["choices"][0]
    msg = ch["message"]
    txt = msg.get("content") or ""
    rsn = msg.get("reasoning_content") or msg.get("reasoning") or ""
    comb = (rsn + "\x00<<<SPLIT>>>\x00" + txt) if (rsn or txt) else ""
    return {
        "sha": hashlib.sha1(comb.encode()).hexdigest()[:12],
        "sha_reason": hashlib.sha1(rsn.encode()).hexdigest()[:12],
        "sha_content": hashlib.sha1(txt.encode()).hexdigest()[:12],
        "chars": len(comb), "content_chars": len(txt),
        "finish": ch.get("finish_reason"), "nothink": nothink,
        "prompt_tok": r.get("usage", {}).get("prompt_tokens"),
        "dt": round(dt, 1), "combined": comb,
    }


def window_stats(comb, top=3):
    """v4: the REP-LOOP flag alone can't distinguish pathological loops
    (one window hundreds of times) from sane quoting of a repetitive
    document (the synthetic essay repeats 'follows a measured cadence'
    -- exactly 25 chars -- in every paragraph; thinking-mode reasoning
    that quotes 4+ paragraphs trips the >=4 flag legitimately). Bank
    the top window counts so the disposition is evidence-based."""
    wins = {}
    for off in range(0, max(0, len(comb) - 25), 5):
        w = comb[off:off + 25]
        wins[w] = wins.get(w, 0) + 1
    topw = sorted(wins.items(), key=lambda kv: -kv[1])[:top]
    return {
        "distinct_windows": len(wins),
        "top_windows": [
            {"w": w.replace("\n", "\\n"), "n": n,
             "span_pct": round(100.0 * n * 25 / max(1, len(comb)), 1)}
            for w, n in topw
        ],
    }


def detect_garble(name, comb):
    flags = []
    if not comb.strip():
        flags.append("EMPTY")
    wins = {}
    for off in range(0, max(0, len(comb) - 25), 5):
        w = comb[off:off + 25]
        wins[w] = wins.get(w, 0) + 1
    if wins and max(wins.values()) >= 4:
        flags.append("REP-LOOP")
    ok = re.compile(
        r"[A-Za-z0-9 \n\r\t.,:;!?'\"()\[\]{}<>=+\-*/_#@$%&~`|^\\—–…“”‘’"
        r"À-ɏЀ-ӿ぀-ヿ一-鿿　-〿"
        r"✅❌ÄäÖöåÅ]")
    bad = sum(1 for c in comb if not ok.match(c))
    if len(comb) and bad / len(comb) > 0.05:
        flags.append(f"JUNK({bad * 100 // max(1, len(comb))}%)")
    for cls, mk in MARKERS.items():
        if cls != name and mk in comb:
            flags.append(f"BLEED:{cls}")
    return flags


def run_class(name, content, results, temp1_seed=1234):
    runs = []
    for phase in ("cold", "rehit", "rehit2"):
        r = gen(content, 0.0, MAXTOK_T0)
        r["phase"] = phase
        r["garble"] = detect_garble(name, r["combined"])
        runs.append(r)
    ident = (runs[0]["sha"] == runs[1]["sha"]
             and runs[1]["sha"] == runs[2]["sha"])
    t1a = gen(content, 1.0, MAXTOK_T1, seed=temp1_seed)
    t1b = gen(content, 1.0, MAXTOK_T1, seed=temp1_seed)
    t1_same = t1a["sha"] == t1b["sha"]
    results[name] = {
        "t0_identical": ident,
        "hashes": [r["sha"] for r in runs],
        "content_hashes": [r["sha_content"] for r in runs],
        "garble": {r["phase"]: r["garble"] for r in runs},
        "finish": [r["finish"] for r in runs],
        "prompt_tok": runs[0]["prompt_tok"],
        "chars": [r["chars"] for r in runs],
        "content_chars": [r["content_chars"] for r in runs],
        "nothink": [r["nothink"] for r in runs],
        "win_stats": {r["phase"]: window_stats(r["combined"]) for r in runs},
        "raw_cold_head": runs[0]["combined"][:2048],
        "raw_rehit_head": runs[1]["combined"][:2048],
        "t1_seed_identical": t1_same,
        "t1_hashes": [t1a["sha"], t1b["sha"]],
        "t1_garble": detect_garble(name, t1a["combined"]),
        "dt_cold": runs[0]["dt"],
    }
    print(f"[{name}] t0_identical={ident} hashes="
          f"{results[name]['hashes']} garble={results[name]['garble']} "
          f"t1_seed_identical={t1_same} prompt_tok={results[name]['prompt_tok']} "
          f"chars={results[name]['chars']} "
          f"content_chars={results[name]['content_chars']} "
          f"topwin={results[name]['win_stats']['cold']['top_windows'][0]}",
          flush=True)


def run_rehit_only(name, content, results):
    r1 = gen(content, 0.0, MAXTOK_T0)
    r2 = gen(content, 0.0, MAXTOK_T0)
    same = r1["sha"] == r2["sha"]
    results[name] = {
        "post_churn_rehit_identical": same,
        "hashes": [r1["sha"], r2["sha"]],
        "garble": {"r1": detect_garble(name, r1["combined"]),
                   "r2": detect_garble(name, r2["combined"])},
        "win_stats": {"r1": window_stats(r1["combined"]),
                      "r2": window_stats(r2["combined"])},
        "chars": [r1["chars"], r2["chars"]],
        "content_chars": [r1["content_chars"], r2["content_chars"]],
        "raw_r1_head": r1["combined"][:2048],
        "prompt_tok": r1["prompt_tok"],
    }
    print(f"[{name}] post_churn_rehit_identical={same} "
          f"hashes={results[name]['hashes']} garble={results[name]['garble']}",
          flush=True)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "x"
    mode = sys.argv[2] if len(sys.argv) > 2 else "full"
    results = {"tag": tag, "mode": mode, "probe": "v4",
               "ts": time.strftime("%F %T")}
    for name in list(PROMPTS):
        try:
            if mode == "full":
                run_class(name, PROMPTS[name], results)
            else:
                run_rehit_only(name, PROMPTS[name], results)
        except Exception as e:  # noqa: BLE001 — bank the failure, continue
            results[name] = {"error": repr(e)}
            print(f"[{name}] ERROR {e!r}", flush=True)
    ok = all(
        (v.get("t0_identical") or v.get("post_churn_rehit_identical"))
        and not any(v.get("garble", {}).get("cold") or [])
        and v.get("t1_seed_identical", True)
        for k, v in results.items() if k in PROMPTS)
    results["VERDICT"] = "CLEAN" if ok else "CORRUPTION_SIGNAL"
    print(f"LONGPROBE {tag} {mode}: {results['VERDICT']}", flush=True)
    fn = f"longprobe_{tag}_{mode}"
    with open(fn + ".json", "w") as f:
        json.dump(results, f, indent=1, default=str)
    with open(fn + ".txt", "w") as f:
        for name in PROMPTS:
            v = results.get(name, {})
            f.write(f"===== {name} =====\n{json.dumps(v, indent=1, default=str)}\n\n")


if __name__ == "__main__":
    main()
