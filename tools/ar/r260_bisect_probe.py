#!/usr/bin/env python3
"""r260_bisect_probe.py -- the suite-face bisect probe (P6.47 r260).

r259 closed the mode question: the suite crashes EAGER (ARC_GRAPH=0)
with the same DEVICE_LOST class — the engine + suite state is the
necessary pair, graphs exonerated. The r257 minimal pair (multi alone,
pristine boot) ran CLEAN, so some PRIOR-STATE PREFIX of the suite is
required. This probe runs a caller-selected SEQUENCE of faces, each in
the ACTUAL p647_longprobe v3 class semantics (cold -> rehit -> rehit2
at temp 0 / max_tokens 1536, then t1a/t1b at temp 1 seed 1234 / 768 —
byte-identical prompts, the enable_thinking OFF kwarg, the 400
fallback), and STOPS at the first failed generation (the engine is
dead; further faces would be vacuous).

Usage: r260_bisect_probe.py <tag> <face> [<face> ...]
  faces: essay code multi simple (order = the requested sequence;
  each prior face's full 5-gen class runs before the next)
Stdlib only; talks to http://127.0.0.1:8000 (rig host python3).
Prints R260_GEN per generation (the crash ORDINAL is legible in the
bank) and ends R260_DONE crashed=<bool>.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = os.environ.get("ARC_PROBE_MODEL", "gemma4-31b")
MAXTOK_T0 = 1536
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
    "essay": build_essay(1650),     # ~2.2k tok
    "code": build_code(3300),       # ~5.4k tok (r259: 5433)
    "multi": build_multi(1650),     # 3446 tok (the crash geometry)
    "simple": build_simple(),       # 78 tok
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
        "chars": len(comb),
        "finish": ch.get("finish_reason"), "nothink": nothink,
        "prompt_tok": r.get("usage", {}).get("prompt_tokens"),
        "dt": round(dt, 1), "combined": comb,
    }


PHASES = ("cold", "rehit", "rehit2", "t1a", "t1b")


def run_face(name, results):
    """One face's full class; returns False when a generation failed
    (the engine is dead — the sequence stops)."""
    content = PROMPTS[name]
    shas = []
    for phase in PHASES:
        try:
            r = (gen(content, 1.0, MAXTOK_T1, seed=1234)
                 if phase.startswith("t1")
                 else gen(content, 0.0, MAXTOK_T0))
        except Exception as e:  # noqa: BLE001 -- the crash IS the signal
            results[name] = {"error": f"phase={phase} {e!r}"}
            print(f"R260_GEN face={name} phase={phase} ERROR {e!r}",
                  flush=True)
            print(f"[{name}] ERROR at phase={phase}: {e!r}", flush=True)
            return False
        shas.append(r["sha"])
        print(f"R260_GEN face={name} phase={phase} sha={r['sha']} "
              f"chars={r['chars']} finish={r['finish']} "
              f"prompt_tok={r['prompt_tok']} dt={r['dt']}", flush=True)
    results[name] = {"shas": shas}
    print(f"[{name}] class_complete shas={shas}", flush=True)
    return True


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "x"
    faces = sys.argv[2:] or ["essay", "multi"]
    results = {"tag": tag, "faces": faces, "ts": time.strftime("%F %T")}
    crashed = False
    for name in faces:
        if name not in PROMPTS:
            print(f"R260_UNKNOWN_FACE {name}", flush=True)
            sys.exit(2)
        if not run_face(name, results):
            crashed = True
            break
    print(f"R260_DONE crashed={crashed} faces={'+'.join(faces)} "
          f"-> longprobe_seq_{tag}.json", flush=True)
    fn = f"longprobe_seq_{tag}"
    with open(fn + ".json", "w", newline="\n") as f:
        json.dump(results, f, indent=1, default=str)
    with open(fn + ".txt", "w", newline="\n") as f:
        for name in faces:
            v = results.get(name, {})
            f.write(f"===== {name} =====\n{json.dumps(v, indent=1, default=str)}\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
