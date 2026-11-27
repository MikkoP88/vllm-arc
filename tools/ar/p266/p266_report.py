"""r266 fact-table generator: banked matrix_*.json + leg evidence -> the
per-leg fact block and the cross-leg discriminator table (analysis close-out).

Usage: python p266_report.py BANK_DIR [BANK_DIR ...] [--out FILE]

Each BANK_DIR is one r266 leg (b2_g3_29, c3_m8g_29, ...). Every input is
optional except the matrix json -- a boot-failed leg still reports its boot
evidence. Output is plain text, one fact block per leg, then the cross-leg
table the verdict cites.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

LEG_ORDER = ["b2", "c3", "c4", "c5", "w1", "w2", "b3", "b5", "b4", "c1", "c2"]


def leg_key(name: str):
    base = name.split("_")[0]
    return LEG_ORDER.index(base) if base in LEG_ORDER else len(LEG_ORDER)


def parse_env(line: str) -> dict:
    m = re.search(r"env=\[(.*)\]", line)
    return dict(x.split("=", 1) for x in m.group(1).split() if "=" in x) if m else {}


def kv_of(env: dict, engine_cfg: str) -> str:
    if "ARC_KVDTYPE" in env:
        return env["ARC_KVDTYPE"] + ("+native" if env.get("ARC_MXKV_NATIVE") == "1" else "")
    m = re.search(r"kv(?:_cache_dtype|dtype)['\"]?:\s*['\"]?([\w.]+)", engine_cfg)
    if m:
        return m.group(1)
    return "fp8_e4m3(default)"


def weights_of(env: dict) -> str:
    q = env.get("ARC_QUANT", "fp8")
    if q == "fp8":
        return "fp8"
    mode = env.get("ARC_MXFP4_MODE", "")
    return f"{q}{(' ' + mode) if mode else ''}"


def skin_of(env: dict) -> str:
    t = env.get("ARC_TEMPLATE", "")
    rp = env.get("ARC_RPARSER", "<unset>")
    tp = env.get("ARC_TPARSER", "<unset>")
    if t.endswith("chat_template_gemma4.jinja"):
        s = "g3(P6.28)"
    elif t.endswith("chat_template_gemma4_official.jinja"):
        s = "g4(official-it)"
    elif t.endswith("chat_template_qwen38_high.jinja"):
        s = "QWEN"
    else:
        s = t or "<unset>"
    return f"{s} rp={rp!r} tp={tp!r}"


def load_leg(bank: Path) -> dict:
    leg = {"name": bank.name, "env": {}, "image": "-", "boot": "-", "engine": {}, "facts": {}}
    envf = bank / "leg_env.txt"
    if envf.exists():
        line = envf.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        leg["env"] = parse_env(line)
        m = re.search(r"image=(\S+)", line)
        leg["image"] = m.group(1).split(":")[-1] if m else "-"
        leg["boot"] = line
    cfg = ""
    cfgf = bank / "engine_config.txt"
    if cfgf.exists():
        cfg = cfgf.read_text(encoding="utf-8", errors="replace")
    for pat, key in [
        (r"Casting torch\.bfloat16 to torch\.float16", "cast_f16"),
        (r"Using MX KV attention backend \(dtype=(\S+?),", "mx_backend"),
        (r"MXKV native call", "mx_native"),
        (r"UR_RESULT_ERROR_DEVICE_LOST", "device_lost"),
        (r"Quantized (\d+) layers", "quant_layers"),
    ]:
        m = re.search(pat, cfg)
        if m:
            leg["engine"][key] = m.groups() or True
    leg["engine"]["cfg_text"] = cfg
    leg["engine"]["kv"] = kv_of(leg["env"], cfg)
    leg["engine"]["weights"] = weights_of(leg["env"])
    leg["engine"]["skin"] = skin_of(leg["env"])
    mk = bank / "markers.txt"
    if mk.exists():
        for ln in mk.read_text(encoding="utf-8", errors="replace").splitlines():
            if ln.startswith("post_") and "=" in ln:
                k, v = ln.split("=", 1)
                leg["engine"][k] = v
    errs = bank / "erraudit.txt"
    if errs.exists():
        for ln in errs.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                leg["engine"][f"err_{k.strip()}"] = v.strip()
    return leg


def probe_facts(leg: dict, doc: dict) -> dict:
    f: dict = {}
    f["served"] = doc["meta"]["served"]
    f["oracle"] = doc["oracle"]["match"]
    chat = [r for r in doc["chat"] if r.get("http") == 200]
    for cell in ("t0", "card"):
        rs = [r for r in chat if r["cell"] == cell]
        if not rs:
            continue
        f[cell] = {
            "n": len(rs),
            "ok": sum(1 for r in rs if r.get("ok")),
            "reploop": sum(1 for r in rs if "REPLOOP" in r.get("cls", [])),
            "empty": sum(1 for r in rs if "EMPTY" in r.get("cls", [])),
            "echo": sum(1 for r in rs if "ECHO" in r.get("cls", [])),
            "lp1": [r.get("glp_first") for r in rs],
            "lp_gen": [r.get("glp_mean") for r in rs],
        }
    served = f["served"]
    raw_t0 = [r for r in doc["raw"] if r.get("cell") == "t0" and r.get("fmt") == served]
    if raw_t0:
        f["raw_served_t0"] = {
            "n": len(raw_t0),
            "ok": sum(1 for r in raw_t0 if r.get("ok")),
            "ptok_ok": sum(1 for r in raw_t0 if r.get("ptok_ok")),
        }
    ptok_bad = [r for r in doc["raw"] if r.get("http") == 200 and not r.get("ptok_ok", True)]
    f["ptok_mismatch"] = len(ptok_bad)
    p = doc.get("nll", {}).get("prompts", {})
    served_nll = [v["nll"]["nll_mean"] for k, v in p.items()
                  if k in ("capital_france", "fibonacci", "gases") and v.get("nll", {}).get("n")]
    f["nll_mean3"] = round(sum(served_nll) / len(served_nll), 3) if served_nll else None
    fmt_nll = {}
    for k, v in p.items():
        if v.get("nll", {}).get("n"):
            fmt_nll[k] = v["nll"]["nll_mean"]
    f["nll_by_prompt"] = fmt_nll
    f["gate"] = [(g.get("kind"), g.get("http"), (g.get("text") or "")[:70])
                 for g in doc.get("gate", [])]
    return f


def mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def block(leg: dict) -> str:
    out = [f"## {leg['name']}", "```"]
    out.append(leg["boot"] if leg["boot"] != "-" else "(no leg_env.txt)")
    e = leg["engine"]
    if e:
        out.append(f"skin: {e['skin']}")
        out.append(f"weights: {e['weights']}  kv: {e['kv']}")
        bits = [f"{k}={e[k]}" for k in sorted(e) if k not in ("cfg_text", "skin", "weights", "kv")]
        if bits:
            out.append("engine: " + " ".join(bits))
    f = leg["facts"]
    if not f:
        out.append("(no matrix json)")
        out.append("```")
        return "\n".join(out)
    out.append(f"probe: served={f['served']} oracle_match={f['oracle']} ptok_mismatch={f['ptok_mismatch']}")
    for cell in ("t0", "card"):
        c = f.get(cell)
        if c:
            out.append(f"chat {cell}: ok={c['ok']}/{c['n']} reploop={c['reploop']} "
                       f"empty={c['empty']} echo={c['echo']} "
                       f"lp1={mean(c['lp1'])} lp_gen={mean(c['lp_gen'])}")
    r = f.get("raw_served_t0")
    if r:
        out.append(f"raw served-fmt t0: ok={r['ok']}/{r['n']} ptok_ok={r['ptok_ok']}/{r['n']}")
    if f.get("nll_mean3") is not None:
        out.append(f"nll_mean(g3 prompts): {f['nll_mean3']}  per-prompt: {f['nll_by_prompt']}")
    for kind, http, text in f["gate"]:
        out.append(f"gate {kind}: http={http} {text!r}")
    out.append("```")
    return "\n".join(out)


def table_row(leg: dict) -> str:
    e, f = leg["engine"], leg["facts"]
    if not f:
        return (f"| {leg['name']} | {leg['image']} | {e.get('skin', '-')} | {e.get('weights', '-')} | "
                f"{e.get('kv', '-')} | BOOT/PROBE FAIL | - | - |")
    t0 = f.get("t0", {})
    return (f"| {leg['name']} | {leg['image']} | {e.get('skin', '-')} | {e.get('weights', '-')} | "
            f"{e.get('kv', '-')} | {f['oracle']}/{'ok' if not f['ptok_mismatch'] else 'PTOK'} | "
            f"{t0.get('ok', 0)}/{t0.get('n', 0)} | rp{t0.get('reploop', 0)} e{t0.get('empty', 0)} | "
            f"{mean(t0.get('lp1', []))} | {f.get('nll_mean3')} |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("banks", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    legs = []
    for bank in a.banks:
        if not bank.is_dir():
            print(f"SKIP {bank} (not a dir)", file=sys.stderr)
            continue
        leg = load_leg(bank)
        jss = sorted(bank.glob("matrix_*.json"))
        if jss:
            try:
                leg["facts"] = probe_facts(leg, json.loads(jss[0].read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError) as ex:
                leg["facts"]["error"] = f"{type(ex).__name__}: {ex}"
        legs.append(leg)
    legs.sort(key=lambda L: leg_key(L["name"]))
    parts = ["# r266 gemma issue-analysis fact table", ""]
    parts += [block(L) for L in legs]
    parts += ["", "## cross-leg discriminators", "",
              "| leg | image | skin | weights | kv | oracle/ptok | chat_t0 ok | "
              "t0 flags | lp1 | nll |", "|---|---|---|---|---|---|---|---|---|---|"]
    parts += [table_row(L) for L in legs]
    txt = "\n".join(parts) + "\n"
    if a.out:
        a.out.write_text(txt, encoding="utf-8", newline="\n")
        print(f"WROTE {a.out} ({len(legs)} legs)")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
