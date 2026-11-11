#!/usr/bin/env python3
"""p139_report.py -- P6.37B/P6.39 report generator v3 (local, post-pull).

Reads the r138 bank (solo ctx sweeps + temperature sweeps + KV-dtype cells)
AND the r139 bank (concurrency sweeps {1,2,3,4,8} per context, TurboQuant
conc cells, gemma KV-dtype cells) plus the r140 gemma-128K annex bank and
regenerates the full docs/reports/ tree in the world-class format (per the
millstoneai inference-benchmark layout):

  per-config reports: identity card, executive summary + key findings,
    use-case guidance (thresholds + capacity verdicts), system throughput
    (peak points + tokens/hour), per-user speed with UX bands, TTFT with
    UX bands, capacity analysis (per use case + KV pool), technical
    analysis (queue decomposition, prefill, ITL extremes, scaling
    efficiency), temperature analysis, copyable configuration, charts
  KV-cache dtype report: qwen + gemma + TurboQuant, pools, capacity
    cliffs, conc sweeps, refusal boundaries, gemma 128K annex
  methodology report: harness definitions, metric formulas, UX bands,
    use-case thresholds, honest-limits statement
  index README: cross-config tables incl. concurrency, tokens/hour and
    use-case capacity matrix + overlay charts

Derived metrics (client-side, concurrency aware):
  per-user tok/s = max_tokens / (total_s - ttft_s)   per request
                   (numerator law: rate-accounting-spec-sse-chunks)
  aggregate tok/s = conc * max_tokens / wall_s       (wall includes
                    the staggered start + slowest user tail)
  solo decode tok/s = max_tokens / (total_s - ttft_s) (r138 faces)
  prefill tok/s = ctx / ttft_s                       (queue ~0 at c=1)
  added wait(n) = ttft_mean(n) - ttft_mean(1)        (derived queue+
                  prefill-share under the barrier burst)
  scaling eff(n) = agg(n) / (n * agg(1))

Usage: p139_report.py <bank138> <bank139> [bank140] <docs_out> [date]
       (bank140 optional: the r140 gemma-128K annex bank)
"""
import glob
import json
import math
import os
import sys
import contextlib

QCTX = [1024, 8192, 32768, 65536, 131072]
GCTX = [1024, 8192, 32768]
GCTX128 = [1024, 8192, 32768, 131072]
TEMPS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
CONCS = [1, 2, 3, 4, 8]
QCFG = '{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'

QF8 = "ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1"
QMX = ("ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b "
       "ARC_QUANTCFG=" + QCFG + " ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0")
GM = ("ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 "
      "ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0")
GM128 = ("ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 "
         "ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=131072 ARC_FP8=0")

# cell -> report identity. env = the exact words the gate passed (parity
# with tools/ar/r139_conc.sh / r140_gemma128k.sh; the banked posture.txt
# echo is the runtime evidence for each). kvdt = --kv-cache-dtype.
CONFIGS = [
    dict(cell="qf3", slug="qwen3.8-27b-fp8-k3", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="FP8 (fp8 checkpoint, compressed-tensors)", plan="qwen3", mtp="MTP k3",
         env=QF8 + " ARC_SPEC=3", kvdt="fp8_e4m3", ctxs=QCTX, temps=True,
         blurb="the certified incumbent posture"),
    dict(cell="qf0", slug="qwen3.8-27b-fp8-k0", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="FP8 (fp8 checkpoint, compressed-tensors)", plan="qwen0", mtp="no MTP",
         env=QF8 + " ARC_SPEC=0", kvdt="fp8_e4m3", ctxs=QCTX, temps=False,
         blurb="the MTP-off ablation of the incumbent"),
    dict(cell="q83", slug="qwen3.8-27b-mxfp8-k3", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="MXFP8 (online, bf16 base checkpoint)", plan="qwen3", mtp="MTP k3",
         env=QMX + " ARC_QUANT=mxfp8 ARC_SPEC=3", kvdt="fp8_e4m3", ctxs=QCTX, temps=True,
         blurb="the online-MXFP8 lane"),
    dict(cell="q80", slug="qwen3.8-27b-mxfp8-k0", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="MXFP8 (online, bf16 base checkpoint)", plan="qwen0", mtp="no MTP",
         env=QMX + " ARC_QUANT=mxfp8 ARC_SPEC=0", kvdt="fp8_e4m3", ctxs=QCTX, temps=False,
         blurb="the MTP-off ablation of the MXFP8 lane"),
    dict(cell="q43", slug="qwen3.8-27b-mxfp4-k3", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="MXFP4 W4A16 (online, bf16 base checkpoint)", plan="qwen3", mtp="MTP k3",
         env=QMX + " ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=3", kvdt="fp8_e4m3",
         ctxs=QCTX, temps=True,
         blurb="the decode champion posture (P6.35-V winner)"),
    dict(cell="q40", slug="qwen3.8-27b-mxfp4-k0", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="MXFP4 W4A16 (online, bf16 base checkpoint)", plan="qwen0", mtp="no MTP",
         env=QMX + " ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=0", kvdt="fp8_e4m3",
         ctxs=QCTX, temps=False,
         blurb="the MTP-off ablation of the MXFP4 lane"),
    dict(cell="gf8", slug="gemma-4-31b-fp8", model="Gemma-4-31B", org="Google",
         quant="FP8 (online, bf16 base checkpoint)", plan="gemma", mtp="no MTP",
         env=GM + " ARC_QUANT=fp8", kvdt="fp8_e4m3", ctxs=GCTX, temps=False,
         blurb="the gemma online-FP8 capability lane"),
    dict(cell="gm8", slug="gemma-4-31b-mxfp8", model="Gemma-4-31B", org="Google",
         quant="MXFP8 (online, bf16 base checkpoint)", plan="gemma", mtp="no MTP",
         env=GM + " ARC_QUANT=mxfp8", kvdt="fp8_e4m3", ctxs=GCTX, temps=False,
         blurb="the gemma MXFP8 lane"),
    dict(cell="gm4", slug="gemma-4-31b-mxfp4", model="Gemma-4-31B", org="Google",
         quant="MXFP4 W4A16 (online, bf16 base checkpoint)", plan="gemma", mtp="no MTP",
         env=GM + " ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16", kvdt="fp8_e4m3", ctxs=GCTX,
         temps=False, blurb="the gemma MXFP4 lane"),
    dict(cell="gm4x", slug="gemma-4-31b-mxfp4-128k", model="Gemma-4-31B", org="Google",
         quant="MXFP4 W4A16 (online, bf16 base checkpoint)", plan="gemma", mtp="no MTP",
         env=GM128 + " ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16", kvdt="fp8_e4m3",
         ctxs=GCTX128, temps=False,
         blurb="the gemma 128K annex - MXFP4 weight savings buy the KV "
               "headroom for max_model_len 131072"),
    dict(cell="gm8x", slug="gemma-4-31b-mxfp8-128k", model="Gemma-4-31B", org="Google",
         quant="MXFP8 (online, bf16 base checkpoint)", plan="gemma", mtp="no MTP",
         env=GM128 + " ARC_QUANT=mxfp8", kvdt="fp8_e4m3", ctxs=GCTX128, temps=False,
         blurb="the 128K-maxlen control for the gemma annex (8-bit weights: "
               "the KV headroom question)"),
]

# r139 TurboQuant cells: QF8 posture + spec k3 + the TQ kv dtype.
TQ_PRESETS = [
    dict(cell="tq_k8v4", dtype="turboquant_k8v4", poolx="1.98x",
         note="4-bit K/V, 8-level scales - the balanced preset"),
    dict(cell="tq_4bit", dtype="turboquant_4bit_nc", poolx="2.86x",
         note="cleanest identity record of the family"),
    dict(cell="tq_k3v4", dtype="turboquant_k3v4_nc", poolx="3.22x",
         note="k3 scales + 4-bit V"),
    dict(cell="tq_3bit", dtype="turboquant_3bit_nc", poolx="3.68x",
         note="max capacity, 3-bit precision floor"),
]

# r138 KV-dtype cells (qwen); kve5m2/kvauto are refusal banks.
KVDTS_Q = [("fp8_e4m3", "qf3", "the certified incumbent"),
           ("auto", "kvauto131",
            "fp16 KV (2 bytes/elem); refuses at max_model_len 262144 "
            "(8.73 > 6.78 GiB), measured at 131072"),
           ("fp8", "kvfp8", ""),
           ("fp8_e5m2", "kve5m2",
            "refuses to boot - FA2 has no e5m2 KV path on Xe2")]
# r139 gemma KV-dtype cells.
KVDTS_G = [("fp8_e4m3", "gm8", "the certified gemma incumbent"),
           ("auto", "gkv_auto16",
            "fp16 KV; refuses at max_model_len 40960 (the certified gemma "
            "posture), servable at 16384"),
           ]

# banked pool facts where the cell has no parsed posture pool (P6.37 G
# close-out analysis, results/2026-11-10-p637-reports/kvauto131 etc.)
POOL_FACTS = {
    "kvauto131": (198735, "P6.37 G: exactly half the fp8 pool (fp16 law)"),
    "kvfp8": (397470, "P6.37 G: ~= fp8_e4m3 (boot-day free-memory variance)"),
    "kvauto": (None, "refusal: needs 8.73 GiB vs 6.78 GiB available"),
    "kve5m2": (None, "refusal: no e5m2 KV dequant path in esimd FA2 on Xe2"),
    "gkv_auto": (None, "refusal expected at maxlen 40960 (needs ~28K-token pool)"),
}

OVERHEAD = 600 + 64   # max_tokens + scheduler slack; matches the r139 gate

# Use-case guidance (the millstone-style deployment mapping). Thresholds
# are the band edges of the UX classification already documented in the
# reports: TTFT <= 10 s is the top of the "slow" band (still acceptable),
# 15 tok/s is the "acceptable" streaming floor; batch document work holds
# the speed bar only (TTFT tolerated).
USE_CASES = [
    dict(key="chat8", name="Interactive chat", ctx=8192,
         ttft_bar=10.0, speed_bar=15.0,
         desc="Conversational assistant sessions: system prompt plus a few "
              "turns fit 8K. Both first-token wait and streaming speed "
              "matter."),
    dict(key="assist32", name="Long-context assistant (RAG)", ctx=32768,
         ttft_bar=10.0, speed_bar=15.0,
         desc="Retrieval-augmented / large-prompt assistants at 32K "
              "context."),
    dict(key="docmax", name="Document processing (batch)", ctx=None,
         ttft_bar=None, speed_bar=15.0,
         desc="Offline summarization / extraction at the largest measured "
              "context. Turnaround matters, not first-token wait: the bar "
              "is streaming speed (and total throughput)."),
]

# ---------------------------------------------------------------- loading


class Bank:
    """One bank dir (r138 / r139 / r140) with existence-aware lookups."""

    def __init__(self, path):
        self.path = path
        self.ok = bool(path) and os.path.isdir(path)

    def file(self, *parts):
        p = os.path.join(self.path, *parts) if self.ok else None
        return p if p and os.path.isfile(p) else None


def load_face(b138, cell, tag):
    p = b138.file(cell, "p138_face_{}.json".format(tag))
    if not p:
        return None
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    if d.get("ttft_s", -1) <= 0 or d.get("total_s", -1) <= 0:
        return None
    dw = d["total_s"] - d["ttft_s"]
    d["decode_tps"] = d["max_tokens"] / dw if dw > 0 else 0.0
    d["prefill_tps"] = d["ctx"] / d["ttft_s"]
    d["tok_per_step"] = (d["max_tokens"] / d["n_stream_chunks"]
                         if d.get("n_stream_chunks") else 0.0)
    d["decode_window_s"] = dw
    return d


def ctx_face(b138, cell, c):
    return load_face(b138, cell, f"{cell}_c{c}_t0.0")


def temp_face(b138, cell, t):
    return load_face(b138, cell, "{}_s8192_t{}".format(cell, t))


def load_conc(banks, cell, ctx, conc, temp="0.0"):
    """Load one concurrency face JSON from whichever bank has it."""
    suf = "" if temp in ("0.0", 0.0) else "_t{}".format(temp)
    for b in banks:
        p = b.file(cell, f"p139_conc_{cell}_c{ctx}_n{conc}{suf}.json")
        if p:
            return json.load(open(p, encoding="utf-8"))
    return None


def conc_ctxs(banks, cell):
    """Contexts that have at least one concurrency face for the cell."""
    pre = "p139_conc_{}_c".format(cell)
    xs = set()
    for b in banks:
        for p in glob.glob(os.path.join(b.path or "\\\\", cell, pre + "*_n*.json")) \
                if b.ok else []:
            base = os.path.basename(p)
            if not base.startswith(pre):
                continue
            with contextlib.suppress(ValueError):
                xs.add(int(base[len(pre):].split("_")[0]))
    return sorted(xs)


def conc_concs(banks, cell, ctx, temp="0.0"):
    """Concurrency levels measured for (cell, ctx) - t0.0 only."""
    pre = f"p139_conc_{cell}_c{ctx}_n"
    out = set()
    for b in banks:
        for p in glob.glob(os.path.join(b.path or "\\\\", cell, pre + "*.json")) \
                if b.ok else []:
            base = os.path.basename(p)
            if not base.startswith(pre):
                continue
            rest = base[len(pre):]           # "8.json" | "1_t1.0.json"
            if "_" in rest:               # t-suffixed face: wrong temperature
                continue
            with contextlib.suppress(ValueError):
                out.add(int(rest.split(".")[0]))
    return sorted(out)


def itl_agg(d):
    """Aggregate per-request ITL percentiles: p50 = median of the
    per-request p50s, p90/p99 = max (worst user) - conservative."""
    rs = [r for r in d.get("requests", [])
          if r.get("itl_ms_p50") is not None and not r.get("error")]
    if not rs:
        return None
    p50s = sorted(r["itl_ms_p50"] for r in rs)
    med = p50s[len(p50s) // 2]
    return dict(p50=med, p90=max(r["itl_ms_p90"] for r in rs),
                p99=max(r["itl_ms_p99"] for r in rs))


def solo_face(banks, cell, c):
    """Normalized solo face at context c: r138 face if banked, else the
    r139/r140 conc-1 face (same cold-prefill measurement). Returns None
    when neither exists. Fields: ttft_s, decode_tps, prefill_tps,
    tok_per_step, itl_ms_{p50,p90,p99}, src ('r138'|'r139')."""
    f = ctx_face(banks[0], cell, c)
    if f:
        f["src"] = "r138"
        return f
    for b in banks[1:]:
        d = load_conc(banks[1:], cell, c, 1) if b.ok else None
        if d is not None and _conc_face_ok(d):
            ia = itl_agg(d) or {}
            tps = d["per_user_mean_tps"]
            chunks = [r["n_stream_chunks"] for r in d["requests"]
                      if not r.get("error") and r.get("n_stream_chunks")]
            return dict(
                ctx=c, ttft_s=d["ttft_mean_s"], total_s=d["wall_s"],
                decode_tps=tps, prefill_tps=(c / d["ttft_mean_s"]
                                             if d["ttft_mean_s"] > 0 else 0.0),
                tok_per_step=(d["max_tokens"] * d["n_ok"] / sum(chunks)
                              if chunks else 0.0),
                itl_ms_p50=ia.get("p50", float("nan")),
                itl_ms_p90=ia.get("p90", float("nan")),
                itl_ms_p99=ia.get("p99", float("nan")),
                src="r139")
    return None


def pool_of(banks, cell):
    """KV pool tokens from the cell's banked posture echo, else POOL_FACTS."""
    for b in banks:
        p = b.file(cell, "posture.txt")
        if not p:
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("kv_pool_tokens="):
                    v = int(line.split("=", 1)[1].strip())
                    if v > 0:
                        return v, "r139 posture"
                if "POOL_PARSE_FAIL" in line or "BOOT_FAIL" in line:
                    break
    if cell in POOL_FACTS:
        return POOL_FACTS[cell]
    return None, ""


def outcome(banks, cell):
    for b in banks:
        p = b.file(cell, "outcome.txt")
        if p:
            return open(p, encoding="utf-8").read().strip()
    return "MISSING"


# ------------------------------------------------------------------ UX

def speed_class(v):
    if v is None:
        return "-"
    if v < 15:
        return "slow"
    if v < 25:
        return "acceptable"
    if v < 50:
        return "fast"
    return "**very fast**"


def ttft_class(v):
    if v is None:
        return "-"
    if v < 0.5:
        return "**instant**"
    if v < 2:
        return "responsive"
    if v < 5:
        return "noticeable"
    if v < 10:
        return "slow"
    return "frustrating"


def ttft_fmt(v):
    return "-" if v is None else ("{:.2f}".format(v) if v < 10 else "{:.1f}".format(v))


SPEED_BANDS = [("< 15 tok/s", "Slow - may be slower than reading speed"),
               ("15-25 tok/s", "Acceptable - keeps pace with reading"),
               ("25-50 tok/s", "Fast - exceeds reading speed"),
               (">= 50 tok/s", "Very fast - text appears nearly instantly")]
TTFT_BANDS = [("< 0.5 s", "Feels instant"),
              ("0.5-2 s", "Feels responsive"),
              ("2-5 s", "Noticeable but still acceptable"),
              ("5-10 s", "Feels slow; acceptable at higher context"),
              ("> 10 s", "Can be frustrating; users may retry or abandon")]


# ------------------------------------------------------------------ SVG

PALETTE = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c",
           "#0891b2", "#65a30d", "#be185d"]
MARKERS = ["circle", "square", "triangle", "diamond"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_tick(v):
    if v >= 1048576 and v % 1048576 == 0:
        return f"{v // 1048576}M"
    if v >= 1024 and v % 1024 == 0:
        return f"{v // 1024}K"
    return str(int(v))


def _fmt_comma(v):
    return "{:,.0f}".format(v)


def _fmt_hourly(tps):
    """Tokens/hour at human precision (3 significant digits)."""
    v = tps * 3600
    for unit, div in (("M", 1e6), ("K", 1e3)):
        if v >= div:
            return "{:.2f} {} tokens/hour".format(v / div, unit)
    return f"{v} tokens/hour"


def _niceTicks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** int(math.floor(math.log10(raw)))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    t0 = int(lo // step) * step
    ts = []
    t = t0
    while t <= hi + step * 0.01:
        if t >= lo * 0.999:
            ts.append(t)
        t += step
    return ts


def chart(path, title, series, xlabel, ylabel, xcat=None, yfmt="{:.1f}",
          y_from_zero=True, hlines=(), ideal=False, label_all=True,
          ylog=False):
    """SVG line chart. series = [(label, [(x, y)|None, ...])].
    xcat = category labels -> x = index. hlines = [(y, tag)] dashed
    class thresholds (right-margin tags). ideal = dashed linear
    reference per series anchored at its first point (throughput
    scaling charts). label_all = value label on every point (default);
    labels are auto-staggered so close neighbours never overlap."""
    # legend rows (bottom, greedy wrap): extend canvas as needed
    _PW = 880 - 84 - 64
    _legw = [16 + 7.2 * min(len(lab), 30) + 20 for lab, _ in series]
    _lrows, _lx = 1, 0
    for w in _legw:
        if _lx + w > _PW:
            _lrows += 1
            _lx = 0
        _lx += w
    W, H, ML, MR, MT, MB = 880, 478 + 22 * _lrows, 84, 64, 52, 64
    PW, PH = W - ML - MR, H - MT - MB
    pts = [p for _, ps in series for p in ps if p[1] is not None]
    if not pts:
        return
    xs = sorted({p[0] for p in pts})
    ys = [p[1] for p in pts]
    if xcat is not None:
        def X(v):
            return ML + (v + 0.5) / len(xcat) * PW
    elif len(xs) > 1 and xs[-1] / max(xs[0], 1) > 8:
        l0, l1 = math.log2(xs[0]), math.log2(xs[-1])

        def X(v):
            return ML + (math.log2(v) - l0) / (l1 - l0) * PW
    else:
        x0, x1 = xs[0], xs[-1]

        def X(v):
            return ML + (v - x0) / (x1 - x0) * PW

    ylo = 0.0 if y_from_zero else min(ys) * 0.92
    yhi = max(max(ys), max([h for h, _ in hlines] or [0])) * 1.14
    if yhi <= 0:
        yhi = 1.0

    def Y(v):
        return MT + PH - (v - ylo) / (yhi - ylo) * PH

    def ylin(v):   # tick positions (linear even on log-y)
        return v

    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
           'viewBox="0 0 {} {}" font-family="Segoe UI, Helvetica, Arial, '
           'sans-serif">'.format(W, H, W, H)]
    out.append('<rect width="{}" height="{}" fill="#ffffff"/>'.format(W, H))
    out.append('<text x="{}" y="26" font-size="17" font-weight="600" '
               'fill="#101828">{}</text>'.format(ML, esc(title)))
    # y grid + ticks
    for t in _niceTicks(ylo, yhi, 5):
        yy = Y(ylin(t))
        if yy < MT - 2 or yy > MT + PH + 2:
            continue
        out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" '
                   'stroke="#f0f0f0" stroke-width="1"/>'.format(ML, yy,
                                                                ML + PW, yy))
        out.append('<text x="{}" y="{:.1f}" font-size="11" fill="#667085" '
                   'text-anchor="end">{}</text>'.format(ML - 8, yy + 4,
                                                        yfmt.format(t)))
    # x ticks
    if xcat is not None:
        for i, lab in enumerate(xcat):
            xx = X(i)
            out.append('<line x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}" '
                       'stroke="#f0f0f0" stroke-width="1"/>'.format(xx, MT, xx,
                                                                    MT + PH))
            out.append('<text x="{:.1f}" y="{}" font-size="11" fill="#667085" '
                       'text-anchor="middle">{}</text>'.format(xx, MT + PH + 18,
                                                               esc(lab)))
    else:
        for xv in xs:
            xx = X(xv)
            out.append('<line x1="{:.1f}" y1="{}" x2="{:.1f}" y2="{}" '
                       'stroke="#f0f0f0" stroke-width="1"/>'.format(xx, MT, xx,
                                                                    MT + PH))
            out.append('<text x="{:.1f}" y="{}" font-size="11" fill="#667085" '
                       'text-anchor="middle">{}</text>'.format(xx, MT + PH + 18,
                                                               _fmt_tick(xv)))
    out.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#d0d5dd" '
               'stroke-width="1"/>'.format(ML, MT + PH, ML + PW, MT + PH))
    out.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#d0d5dd" '
               'stroke-width="1"/>'.format(ML, MT, ML, MT + PH))
    # class threshold hlines (dashed, tagged at the right margin)
    for hv, htag in hlines:
        if not (ylo < hv < yhi):
            continue
        yy = Y(hv)
        out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" '
                   'stroke="#98a2b3" stroke-width="1" stroke-dasharray="5 4"/>'
                   .format(ML, yy, ML + PW, yy))
        out.append('<text x="{}" y="{:.1f}" font-size="9.5" fill="#667085" '
                   'font-style="italic">{}</text>'
                   .format(ML + PW + 4, yy + 3, esc(htag)))
    # ideal-linear dashed references (anchored per series first point)
    if ideal:
        for si, (_lab, ps) in enumerate(series):
            good = [(xv, yv) for xv, yv in ps if yv is not None]
            if len(good) < 2:
                continue
            (x0v, y0v), (x1v, _) = good[0], good[-1]
            col = PALETTE[si % len(PALETTE)]
            out.append('<line x1="{:.1f}" y1="{:.1f}" x2="{:.1f}" y2="{:.1f}" '
                       'stroke="{}" stroke-width="1.4" stroke-dasharray="3 5" '
                       'opacity="0.5"/>'.format(X(x0v), Y(y0v), X(x1v), Y(y0v * x1v / max(x0v, 1e-9)), col))
    # series
    cand = []                     # (xx, yy, value_text, colour) — placed after
    for si, (lab, ps) in enumerate(series):
        col = PALETTE[si % len(PALETTE)]
        mk = MARKERS[si % len(MARKERS)]
        good = [(xv, yv) for xv, yv in ps if yv is not None]
        if len(good) >= 2:
            pl = " ".join("{:.1f},{:.1f}".format(X(xv), Y(yv)) for xv, yv in good)
            out.append('<polyline points="{}" fill="none" stroke="{}" '
                       'stroke-width="2.4"/>'.format(pl, col))
        for pi, (xv, yv) in enumerate(good):
            xx, yy = X(xv), Y(yv)
            if mk == "circle":
                out.append('<circle cx="{:.1f}" cy="{:.1f}" r="4.4" fill="{}"/>'.format(xx, yy, col))
            elif mk == "square":
                out.append('<rect x="{:.1f}" y="{:.1f}" width="9" height="9" '
                           'fill="{}"/>'.format(xx - 4.5, yy - 4.5, col))
            elif mk == "triangle":
                out.append('<polygon points="{:.1f},{:.1f} {:.1f},{:.1f} '
                           '{:.1f},{:.1f}" fill="{}"/>'.format(
                               xx, yy - 5.5, xx - 5.5, yy + 4.5,
                               xx + 5.5, yy + 4.5, col))
            else:
                out.append('<polygon points="{:.1f},{:.1f} {:.1f},{:.1f} '
                           '{:.1f},{:.1f} {:.1f},{:.1f}" fill="{}"/>'.format(
                               xx, yy - 5.5, xx + 5.5, yy, xx, yy + 5.5,
                               xx - 5.5, yy, col))
            if label_all or pi == len(good) - 1:
                cand.append((xx, yy, yfmt.format(yv), col))
    # value labels: every point, staggered so close neighbours never
    # overlap (greedy left-to-right over an alternating above/below
    # offset ladder; zero-overlap offset preferred, least-overlap
    # fallback for pathological clusters)
    DYES = (-9, 16, -21, 27, -33, 38, -45, 49, -57, 60, -69, 71,
            -81, 82, -93, 94, -105, 105, -117, 117)
    boxes = []
    for xx, yy, text, col in sorted(cand, key=lambda c: (c[0], c[1])):
        w = 6.1 * len(text) + 2
        scored = []
        for rank, dy in enumerate(DYES):
            x1, y1 = xx - w / 2.0, yy + dy - 10
            x2, y2 = xx + w / 2.0, yy + dy
            if y1 < 32 or y2 > MT + PH + 6:
                continue          # title band, x-tick band, or off-canvas
            ov = 0.0
            for b in boxes:
                ox = min(x2, b[2]) - max(x1, b[0])
                oy = min(y2, b[3]) - max(y1, b[1])
                if ox > 0 and oy > 0:
                    ov += ox * oy
            scored.append((ov, rank, dy))
        if not scored:            # extreme edge point: clamp above
            scored = [(0.0, 0, -9)]
        _ov, _rank, dy = min(scored)
        boxes.append((xx - w / 2.0, yy + dy - 10, xx + w / 2.0, yy + dy))
        out.append('<text x="{:.1f}" y="{:.1f}" font-size="10" fill="{}" '
                   'text-anchor="middle" font-weight="600">'
                   '{}</text>'.format(xx, yy + dy, col, text))
    # legend (bottom, greedy wrap - same packing as the height calc)
    lx, ly = ML, MT + PH + 58
    for si, (lab, _ps) in enumerate(series):
        col = PALETTE[si % len(PALETTE)]
        w = 16 + 7.2 * min(len(lab), 30) + 20
        if lx + w > ML + PW:
            lx, ly = ML, ly + 22
        out.append('<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" '
                   'stroke-width="3"/>'.format(int(lx), int(ly),
                                               int(lx + 16), int(ly), col))
        out.append('<text x="{}" y="{}" font-size="11.5" fill="#1d2939">'
                   '{}</text>'.format(int(lx + 20), int(ly + 4),
                                      esc(lab[:30])))
        lx += w
    out.append('<text x="{}" y="{}" font-size="12.5" fill="#475467" '
               'text-anchor="middle">{}</text>'.format(ML + PW // 2,
                                                       MT + PH + 34,
                                                       esc(xlabel)))
    out.append('<text x="16" y="{}" font-size="12.5" fill="#475467" '
               'text-anchor="middle" transform="rotate(-90 16 {})">{}</text>'
               .format(MT + PH // 2, MT + PH // 2, esc(ylabel)))
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def bars(path, title, groups, slabels, values, ylabel, yfmt="{:.0f}",
         hlines=()):
    """Grouped-bar SVG chart. groups = [group label]; slabels = series
    labels; values[g][s] = value or None."""
    W, H, ML, MR, MT, MB = 880, 470, 84, 64, 52, 76
    PW, PH = W - ML - MR, H - MT - MB
    vals = [v for row in values for v in row if v is not None]
    if not vals:
        return
    yhi = max(max(vals), max([h for h, _ in hlines] or [0])) * 1.16
    ylo = 0.0

    def Y(v):
        return MT + PH - (v - ylo) / (yhi - ylo) * PH

    gw = PW / max(len(groups), 1)
    bw = min(34.0, (gw * 0.72) / max(len(slabels), 1))
    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" '
           'viewBox="0 0 {} {}" font-family="Segoe UI, Helvetica, Arial, '
           'sans-serif">'.format(W, H, W, H)]
    out.append('<rect width="{}" height="{}" fill="#ffffff"/>'.format(W, H))
    out.append('<text x="{}" y="26" font-size="17" font-weight="600" '
               'fill="#101828">{}</text>'.format(ML, esc(title)))
    for t in _niceTicks(ylo, yhi, 5):
        yy = Y(t)
        out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" '
                   'stroke="#f0f0f0" stroke-width="1"/>'.format(ML, yy,
                                                                ML + PW, yy))
        out.append('<text x="{}" y="{:.1f}" font-size="11" fill="#667085" '
                   'text-anchor="end">{}</text>'.format(ML - 8, yy + 4,
                                                        yfmt.format(t)))
    for hv, htag in hlines:
        if not (ylo < hv < yhi):
            continue
        yy = Y(hv)
        out.append('<line x1="{}" y1="{:.1f}" x2="{}" y2="{:.1f}" '
                   'stroke="#98a2b3" stroke-width="1" stroke-dasharray="5 4"/>'
                   .format(ML, yy, ML + PW, yy))
        out.append('<text x="{}" y="{:.1f}" font-size="9.5" fill="#667085" '
                   'font-style="italic">{}</text>'
                   .format(ML + PW + 4, yy + 3, esc(htag)))
    for gi, glab in enumerate(groups):
        gx = ML + gi * gw
        for si, v in enumerate(values[gi]):
            if v is None:
                continue
            col = PALETTE[si % len(PALETTE)]
            bx = gx + gw / 2 - bw * len(slabels) / 2 + si * bw
            by = Y(v)
            out.append('<rect x="{:.1f}" y="{:.1f}" width="{:.1f}" height="{:.1f}" '
                       'fill="{}" rx="2"/>'.format(bx + 2, by, bw - 4, MT + PH - by, col))
            out.append('<text x="{:.1f}" y="{:.1f}" font-size="10" fill="{}" '
                       'text-anchor="middle" font-weight="600">'
                       '{}</text>'.format(bx + bw / 2, by - 6, col,
                                          yfmt.format(v)))
        out.append('<text x="{:.1f}" y="{}" font-size="11.5" fill="#475467" '
                   'text-anchor="middle">{}</text>'.format(gx + gw / 2,
                                                           MT + PH + 18,
                                                           esc(glab)))
    lx = ML
    for si, slab in enumerate(slabels):
        col = PALETTE[si % len(PALETTE)]
        out.append('<rect x="{}" y="{}" width="12" height="12" fill="{}" '
                   'rx="2"/>'.format(int(lx), H - MB + 22, col))
        out.append('<text x="{}" y="{}" font-size="11.5" fill="#1d2939">'
                   '{}</text>'.format(int(lx + 17), H - MB + 32, esc(slab)))
        lx += 30 + 7.2 * len(slab) + 14
    out.append('<text x="{}" y="{}" font-size="12.5" fill="#475467" '
               'text-anchor="middle">{}</text>'.format(ML + PW // 2, H - 12,
                                                       esc(ylabel)))
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


# ------------------------------------------------------------ md helpers

def ctxrow(c):
    return f"{c} ({c // 1024}K)"


def md_table(header, rows):
    out = ["| " + " | ".join(str(x) for x in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def img(name, caption):
    return ('![{}](charts/{}.svg)\n\n<i>{}</i>'.format(caption, name, caption))


SPEED_HL = [(15, "slow"), (25, "acceptable"), (50, "fast")]
TTFT_HL = [(0.5, "instant"), (2, "responsive"), (5, "noticeable"), (10, "slow")]


def drop_missing_imgs(lines, out):
    """Filter image blocks whose SVG was never produced (missing cells)."""
    keep = []
    for i, ln in enumerate(lines):
        if ln.startswith("![") and "](charts/" in ln:
            fn = ln.split("(charts/", 1)[1].split(")", 1)[0]
            if not os.path.isfile(os.path.join(out, "charts", fn)):
                if i + 1 < len(lines) and lines[i + 1] == "":
                    continue   # drop the image line; its blank line follows
                continue
        keep.append(ln)
    return keep


# ------------------------------------------------------ capacity logic

def _conc_face_ok(d):
    return d is not None and d.get("n_fail", 1) == 0 and d.get("n_ok", 0) > 0


def uc_metrics(cd, uc):
    """Per-concurrency checks for one use case at its context.
    cd = {n: json} (green faces only). Returns (ns, ok_rows) where
    ok_rows[n] = (ttft_ok, speed_ok)."""
    ns = sorted(n for n, d in cd.items() if _conc_face_ok(d))
    ok = {}
    for n in ns:
        d = cd[n]
        t_ok = (uc["ttft_bar"] is None or d["ttft_max_s"] < uc["ttft_bar"])
        s_ok = d["per_user_mean_tps"] >= uc["speed_bar"]
        ok[n] = (t_ok, s_ok)
    return ns, ok


def uc_verdict(cd, uc):
    """Capacity verdict string for one use case."""
    ns, ok = uc_metrics(cd, uc)
    if not ns:
        return None, "not measured at this context"
    hold = [n for n in ns if all(ok[n])]
    if not hold:
        return 0, "thresholds exceeded at concurrency 1 - not recommended"
    cap = max(hold)
    nxt = min((n for n in ns if n > cap), default=None)
    if nxt is None:
        return cap, None            # every measured point holds
    t_ok, s_ok = ok[nxt]
    if s_ok or t_ok:
        # exactly one bar breaks at nxt: linear estimate between cap, nxt
        d0, d1 = cd[cap], cd[nxt]
        if not s_ok:
            v0, v1 = d0["per_user_mean_tps"], d1["per_user_mean_tps"]
            bar = uc["speed_bar"]
        else:
            v0, v1 = d0["ttft_max_s"], d1["ttft_max_s"]
            bar = uc["ttft_bar"]
        if v0 != v1:
            est = cap + (v0 - bar) / (v0 - v1) * (nxt - cap)
            est = max(cap, min(nxt, est))
            whole = int(est)
            return cap, ("~{} concurrent requests (estimated between the "
                         "measured {} and {} points)".format(whole, cap, nxt))
    return cap, ("{} concurrent requests (measured; the next point, "
                 "concurrency {}, exceeds the bars)".format(cap, nxt))


def uc_annotated_row(cd, uc, n, field, bar, below_bad=True):
    """One metric row for the capacity table with threshold annotations."""
    cells = []
    ns, _ok = uc_metrics(cd, uc)
    for nn in ns:
        if nn != n:
            continue
        d = cd[nn]
        v = d["ttft_max_s"] if field == "ttft" else d["per_user_mean_tps"]
        if bar is None:
            cells.append(ttft_fmt(v) if field == "ttft" else "{:.1f}".format(v))
        else:
            bad = (v >= bar) if below_bad else (v < bar)
            mark = " *(threshold exceeded)*" if field == "ttft" and bad else \
                   (" *(below threshold)*" if field != "ttft" and bad else "")
            val = ttft_fmt(v) if field == "ttft" else "{:.1f}".format(v)
            cells.append(val + mark)
    return cells


def requests_stats(faces, tfaces, conc, t1ok):
    """(ok, total) across every banked face of this config. Solo faces
    count only where banked (a green face = one successful request);
    concurrency faces count every launched request."""
    ok = tot = 0
    for f in list(faces.values()) + list(tfaces.values()):
        if f:
            ok += 1
            tot += 1
    for cd in conc.values():
        for d in cd.values():
            if d is not None:
                ok += d.get("n_ok", 0)
                tot += d.get("n_ok", 0) + d.get("n_fail", 0)
    for d in t1ok.values():
        ok += d.get("n_ok", 0)
        tot += d.get("n_ok", 0) + d.get("n_fail", 0)
    return ok, tot


# ------------------------------------------------------- config rendering

def config_block(cfg, date):
    env = cfg["env"]
    # dedupe repeated words (ARC_MXFP4_MODE rides both the lane knob and
    # the env chain) preserving first-occurrence order
    seen, words_l = set(), []
    for w in env.split():
        if w not in seen:
            seen.add(w)
            words_l.append(w)
    env = " ".join(words_l)
    words = dict(w.split("=", 1) for w in env.split())
    spec = words.get("ARC_SPEC", "0")
    gmu = words.get("ARC_GMU", "0.8")
    maxlen = words.get("ARC_MAXLEN", "262144")
    quant = words.get("ARC_QUANT", "")
    is_gemma = "Gemma" in cfg["model"]
    kvdt = cfg.get("kvdt", "fp8_e4m3")
    lmc = words.get("ARC_LMC", "1") == "1"
    spec_flag = ("--speculative-config "
                 "'{{\"method\":\"mtp\",\"num_speculative_tokens\":{},"
                 "\"draft_sample_method\":\"probabilistic\"}}'".format(spec)
                 if spec != "0" else "")
    quant_flag = ("--quantization {}".format(quant)) if quant else ""
    qcfg_flag = ("--quantization-config '{}'".format(QCFG)) if "ARC_QUANTCFG" in words else ""
    kv_flag = ("ARC_KVDTYPE={} ".format(kvdt)) if kvdt != "fp8_e4m3" else ""
    if kvdt != "fp8_e4m3":
        maxlen = words.get("ARC_MAXLEN", maxlen)
    lines = []
    lines.append("### Validated invocation (what the gate ran, byte-exact)\n")
    lines.append("```bash")
    lines.append("# image vllm-arc:v0.30.0-b70-cert23 ; {} ; TP2 on 2x Arc Pro B70".format(date))
    lines.append("ARC_IMAGE=vllm-arc:v0.30.0-b70-cert23 ARC_NAME=arc-b70 \\")
    lines.append(("  {}{} \\".format(kv_flag, env)).replace("  \\ ", " \\")
                 if kv_flag else "  {} \\".format(env))
    lines.append("  bash serve_arc_b70.sh     # tools/serve_arc_b70.sh in this repo")
    lines.append("```")
    lines.append("\nThe wrapper is the audited source of truth (banked posture echo: "
                 "`{}/posture.txt`). What it expands to:\n".format(cfg["cell"]))
    lines.append("```bash")
    lines.append("# 1) container boot layer  [serve_arc_b70.sh:199-210]")
    lines.append("docker run -d --name arc-b70 \\")
    lines.append("  --privileged --network host --ipc host \\")
    lines.append("  --shm-size 16g \\              # close-out F: 64 GB-RAM hosts")
    lines.append("  -v {}:/models/target:ro \\".format(words.get("ARC_MODEL", "/models/<model>")))
    if is_gemma:
        lines.append("  -v <chat-template.json>:/root/chat_template.json:ro \\")
    lines.append("  -v ./neo_cache:/root/.neo_cache -v ./triton_cache:/root/.triton_cache \\")
    lines.append("  -v ./inductor_cache:/root/.inductor_cache \\")
    lines.append("  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \\")
    lines.append("  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert23 -c 'sleep infinity'")
    lines.append("")
    lines.append("# 2) engine layer (docker exec)  [serve_arc_b70.sh:327-375]")
    lines.append("export PYTHONHASHSEED=0 "
                 "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS={} "
                 "ZE_AFFINITY_MASK=0,1".format(words.get("ARC_RPC", "300")))
    lines.append("export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1")
    lines.append("export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072")
    lines.append("export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 "
                 "CCL_SYCL_ALLGATHERV_TMP_BUF=0 "
                 "CCL_SYCL_ALLREDUCE_TMP_BUF=0")
    lines.append("export ARC_TRACE_FLIP=1 VLLM_XPU_ENABLE_XPU_GRAPH=1   # decode-only XPU graphs")
    if "ARC_FP8" in words and words["ARC_FP8"] == "1":
        lines.append("export CCL_SYCL_ALLREDUCE_FP8=1                    # cert10 AR overlay")
    if "ARC_MXFP4_MODE" in words:
        lines.append("export ARC_MXFP4_MODE=w4a16                        # weights-only MXFP4 lane")
    if lmc:
        lines.append("export LMCACHE_USE_EXPERIMENTAL_API=True LMCACHE_MAX_LOCAL_CPU_SIZE=16 LMCACHE_LOCAL_CPU=True")
        lines.append("export LMCACHE_EVICT_POLICY=lr_u LMCACHE_CHUNK_SIZE=1024 LMCACHE_SAVE_DECODE_CACHE=True")
        lines.append("export LMCACHE_MIN_RETRIEVE_TOKENS=1024 LMCACHE_LOG_LEVEL=INFO LMCACHE_DISABLE_PINNED_MEMORY=1")
        lines.append("export LMCACHE_TRACK_USAGE=false                   # close-out F: 16 GB CPU tier")
    lines.append("exec vllm serve --model /models/target \\")
    lines.append("  --served-model-name {} --tensor-parallel-size 2 \\".format(words.get("ARC_SERVED", "<model>")))
    lines.append("  --gpu-memory-utilization {} --max-model-len {} \\".format(gmu, maxlen))
    lines.append("  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \\")
    lines.append("  --dtype float16 {} {} \\".format(quant_flag, qcfg_flag))
    lines.append("  --kv-cache-dtype {} \\".format(kvdt))
    lines.append("  --mamba-ssm-cache-dtype float16 --mamba-cache-mode=align \\")
    lines.append("  --enable-prefix-caching --trust-remote-code \\")
    if is_gemma:
        lines.append("  --chat-template /root/chat_template.json \\   # gemma: no qwen parsers")
    else:
        lines.append("  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \\")
    lines.append("  --no-disable-hybrid-kv-cache-manager \\")
    if lmc:
        lines.append("  --kv-transfer-config '{\"kv_connector\":\"LMCacheConnectorV1\","
                     "\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":"
                     "{\"use_native\":true,\"lmcache.extra_config\":{\"mamba_pool_ratio\":2}},"
                     "\"engine_id\":\"arc-p40\"}' \\")
    lines.append("  --compilation-config '{\"mode\":\"none\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\","
                 "\"cudagraph_capture_sizes\":[1..320]}' \\   # 64-size discipline, see wrapper")
    if spec_flag:
        lines.append("  {} \\".format(spec_flag))
    lines.append("  --port 8000")
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------- per-config v3

def per_config_report(banks, out, cfg, date):
    b138 = banks[0]
    b139 = banks[1:]
    cell = cfg["cell"]
    ch = os.path.join(out, "charts")
    slug = cfg["slug"]
    oc = outcome(banks, cell)
    # solo faces: r138 bank first, r139 n1 faces fill gaps (annex cells)
    faces = {c: solo_face(banks, cell, c) for c in cfg["ctxs"]}
    tfaces = {t: temp_face(b138, cell, t) for t in TEMPS} if cfg["temps"] else {}
    cctxs = conc_ctxs(b139, cell)
    conc = {c: {n: load_conc(b139, cell, c, n) for n in conc_concs(b139, cell, c)}
            for c in cctxs}
    t1 = {c: load_conc(b139, cell, c, 1, "1.0") for c in cctxs}
    t1ok = {c: d for c, d in t1.items() if _conc_face_ok(d)}
    pool, pool_src = pool_of(banks, cell)
    good = {c: f for c, f in faces.items() if f}
    qshort = cfg["quant"].split(" (")[0]
    annex = cell.endswith("x")          # r140 annex cells

    has_data = bool(good or any(_conc_face_ok(d) for cd in conc.values()
                                for d in cd.values()))

    L = []
    L.append("# {} on 2x Intel Arc Pro B70 — {}, {}{}".format(cfg["model"], qshort, cfg["mtp"],
              " (128K annex)" if annex else ""))
    L.append("")
    L.append(md_table(["Field", "Value"], [
        ["Organization", cfg["org"]],
        ["Parameters", "27B dense hybrid-GDN (+MTP head)" if "Qwen" in cfg["model"]
         else "31B dense hybrid (two-head-size GDN, no MTP head)"],
        ["Precision", cfg["quant"]],
        ["Accelerators", "2x Intel Arc Pro B70 (BMG-G31, 32 GB each, 64 GB total), TP2"],
        ["Host RAM", "64 GB (close-out F posture: --shm-size 16g, 16 GB LMCache CPU tier)"],
        ["Engine", "vLLM v0.30.0 — vllm-arc fork, image v0.30.0-b70-cert23, XPU graphs"],
        ["KV cache dtype", cfg.get("kvdt", "fp8_e4m3")],
        ["Speculative decoding", cfg["mtp"] + ", probabilistic drafting" if "k3" in cfg["mtp"] else "disabled"],
        ["Max model len", (cfg["env"].split("ARC_MAXLEN=")[1].split()[0]
                           if "ARC_MAXLEN=" in cfg["env"] else "262144")],
        ["KV pool (this boot)", "{:,} tokens ({})".format(pool, pool_src) if pool else "not parsed"],
        ["Date", date],
        ["Cell outcome", oc],
    ]))
    L.append("")

    if not has_data and oc != "FACES_OK":
        L.append("> **Cell did not complete** (`{}`). See the cell directory "
                 "in the bank — a refusal or boot failure is a valid matrix "
                 "outcome and is banked with its serve log.".format(oc))
        L.append("")
    elif not has_data:
        L.append("> **No faces banked for this cell yet** (annex pending or "
                 "skipped). The invocation below is the certified posture "
                 "with the documented deltas.")
        L.append("")

    # ------------------------------------------------ executive summary
    if has_data:
        allctx = sorted(set(list(good.keys()) + list(conc.keys())))
        lastc = max(allctx)
        firstc = min(allctx)
        nok, ntot = requests_stats(faces, tfaces, conc, t1ok)
        L.append("## Executive summary")
        L.append("")
        L.append("Infrastructure decisions require real performance data. This "
                 "report measures user-facing performance — how many "
                 "concurrent requests the configuration supports at a given "
                 "context length before streaming speed or first-token wait "
                 "degrade below the user-experience bands.")
        L.append("")
        L.append("This benchmark evaluates **{}** ({}; {}) running **{}** on "
                 "**2x Intel Arc Pro B70** (32 GB each, TP2) — {}.".format(cfg["model"], cfg["org"],
                  "27B parameters, dense hybrid-GDN" if "Qwen" in cfg["model"]
                  else "31B parameters, dense hybrid",
                  cfg["quant"], cfg["blurb"]))
        L.append("")
        L.append("Test parameters: context lengths {}–{} tokens; concurrency "
                 "1–{}; {} output tokens per request; unique seeded-random "
                 "prompts (cold prefill, no prefix-cache hits); {}; KV cache "
                 "`{}`{}.".format(
                     f"{firstc // 1024}K", f"{lastc // 1024}K",
                     max([n for cd in conc.values() for n in cd] or [1]),
                     600,
                     cfg["mtp"] + " (probabilistic drafting)" if "k3" in cfg["mtp"]
                     else "no speculative decoding",
                     cfg.get("kvdt", "fp8_e4m3"),
                     "; temperature 0.0 with paired 1.0 faces"
                     if t1ok or tfaces else "; temperature 0.0"))
        L.append("")

        # key findings
        kf = []
        if good:
            peakf = max(good.values(), key=lambda f: f["decode_tps"])
            kf.append(["Generation speed, single request",
                       "{:.1f} tok/s ({}K) → {:.1f} tok/s ({}K)".format(good[firstc]["decode_tps"], firstc // 1024,
                        good[lastc]["decode_tps"], lastc // 1024)])
            kf.append(["TTFT, single request",
                       "{} ({}K) → {} ({}K)".format(ttft_fmt(good[firstc]["ttft_s"]), firstc // 1024,
                        ttft_fmt(good[lastc]["ttft_s"]), lastc // 1024)])
            kf.append(["Peak per-user speed (solo)",
                       "**{:.1f} tok/s** @ {}K ctx ({})".format(peakf["decode_tps"], peakf["ctx"] // 1024,
                        speed_class(peakf["decode_tps"]))])
        # peak system throughput across all conc faces
        peaks = [(d["agg_tps_wall"], c, n) for c, cd in conc.items()
                 for n, d in cd.items() if _conc_face_ok(d)]
        if peaks:
            pagg, pc, pn = max(peaks)
            kf.append(["Peak system throughput",
                       "**{:.0f} tok/s** @ {} concurrent, {}K ctx — ≈ {}".format(
                           pagg, pn, pc // 1024, _fmt_hourly(pagg))])
            base = conc.get(pc, {}).get(1)
            if _conc_face_ok(base) and pn > 1:
                kf.append([f"Throughput scaling (1 → {pn}, {pc // 1024}K ctx)",
                           "{:.1f}x ({:.0f}% of ideal linear)".format(pagg / base["agg_tps_wall"],
                            100 * pagg / (pn * base["agg_tps_wall"]))])
        # use-case capacities
        for uc in USE_CASES:
            uctx = uc["ctx"] if uc["ctx"] is not None else lastc
            cd = {n: d for n, d in conc.get(uctx, {}).items()}
            if any(_conc_face_ok(d) for d in cd.values()):
                cap, note = uc_verdict(cd, uc)
                kf.append(["{} capacity ({}K)".format(uc["name"], uctx // 1024),
                           note if note else f"{cap} concurrent requests"])
        kf.append(["Success rate",
                   "**{:.1f}%** across {} requests ({} faces green)".format(
                       100.0 * nok / max(ntot, 1), ntot,
                      sum(1 for f in faces.values() if f)
                      + sum(1 for f in tfaces.values() if f)
                      + sum(1 for cd in conc.values() for d in cd.values()
                            if _conc_face_ok(d)))])
        L.append(md_table(["Metric", "Result"], kf))
        L.append("")
        L.append("> Every concurrent request models one user session holding "
                 "a full generation (prompt prefill + 600 streamed tokens), "
                 "barrier-started together — the worst-case burst. Interactive "
                 "sessions idle between turns, so sustained user counts run "
                 "well below request concurrency. Full definitions: "
                 "[METHODOLOGY.md](METHODOLOGY.md).")
        L.append("")

        # ------------------------------------------------ use case guidance
        L.append("## Use case guidance")
        L.append("")
        L.append("The table maps this configuration to common deployment "
                 "scenarios. Capacity limits are where TTFT or streaming "
                 "speed fall below the accepted thresholds for a comfortable "
                 "user experience (the band edges of the UX classification "
                 "below).")
        L.append("")
        uc_rows = []
        for uc in USE_CASES:
            uctx = uc["ctx"] if uc["ctx"] is not None else lastc
            thresh = []
            if uc["ttft_bar"] is not None:
                thresh.append("TTFT < {:g}s".format(uc["ttft_bar"]))
            thresh.append("speed >= {:g} tok/s".format(uc["speed_bar"]))
            cd = {n: d for n, d in conc.get(uctx, {}).items()}
            if any(_conc_face_ok(d) for d in cd.values()):
                cap, note = uc_verdict(cd, uc)
                verdict = note if note else f"{cap} concurrent requests"
            elif uctx in good:
                f = good[uctx]
                t_ok = uc["ttft_bar"] is None or f["ttft_s"] < uc["ttft_bar"]
                s_ok = f["decode_tps"] >= uc["speed_bar"]
                verdict = ("1 request (solo meets the bars; concurrency not "
                           "measured here)" if t_ok and s_ok else
                           "solo already below the bars - not recommended")
            else:
                verdict = "not measured at %sK" % (uctx // 1024)
            uc_rows.append([uc["name"], "%sK" % (uctx // 1024),
                            " · ".join(thresh), verdict])
        L.append(md_table(["Use case", "Context", "Thresholds", "Capacity"],
                          uc_rows))
        L.append("")
        L.append("The limits are conservative: past them the system keeps "
                 "serving with slower response times that may still be "
                 "acceptable for your workload.")
        L.append("")

        # ------------------------------------------------ system throughput
        if peaks:
            L.append("## System throughput")
            L.append("")
            L.append("Aggregate token generation across all concurrent "
                     "requests (output tokens only; prefill excluded). "
                     "Measured as n × 600 / wall — the wall includes the "
                     "barrier start and the slowest-user tail, so it is the "
                     "honest saturated-burst number.")
            L.append("")
            tp_rows = []
            for label, csel, nsel in [("Peak", None, None)] + [
                    (f"{c // 1024}K ctx, {n} req", c, n)
                    for c in (8192, 32768, 131072) for n in (4, 8)
                    if _conc_face_ok(conc.get(c, {}).get(n))]:
                if label == "Peak":
                    tp_rows.append([f"Peak ({pc // 1024}K, {pn} req)",
                                    "{:.0f} tok/s".format(pagg)])
                else:
                    d = conc[csel][nsel]
                    tp_rows.append([label, "{:.0f} tok/s".format(d["agg_tps_wall"])])
            L.append(md_table(["Condition", "Aggregate throughput"], tp_rows))
            L.append("")
            L.append("At the peak point this configuration produces "
                     "approximately **{}** — the number that "
                     "matters for batch workloads (document processing, "
                     "synthetic-data generation, offline analysis).".format(_fmt_hourly(pagg)))
            L.append("")
            L.append(img("{}_conc_agg".format(slug),
                         "Aggregate throughput vs concurrency (dashed = ideal "
                         "linear) — {}, {}".format(qshort, cfg["mtp"])))
            L.append("")

        # ------------------------------------------------ per-user speed
        L.append("## Per-user generation speed")
        L.append("")
        L.append("The streaming rate each user experiences. As concurrency "
                 "rises, per-user speed falls because the accelerators are "
                 "shared; with MTP, multiple accepted tokens ride one stream "
                 "chunk (spec-aware accounting).")
        L.append("")
        rows = []
        for c in allctx:
            f = faces.get(c)
            c8 = conc.get(c, {}).get(8)
            best = max(((_conc_face_ok(d) and d["per_user_mean_tps"]) or 0
                        for d in conc.get(c, {}).values()), default=0)
            rows.append([
                ctxrow(c),
                ("{:.1f}".format(f["decode_tps"])) if f else "-",
                speed_class(f["decode_tps"]) if f else "-",
                ("{:.1f}".format(best)) if best else "-",
                (("{:.1f}".format(c8["per_user_mean_tps"]))
                 if _conc_face_ok(c8) else "-"),
                speed_class(c8["per_user_mean_tps"]) if _conc_face_ok(c8) else "-",
            ])
        L.append(md_table(["Context", "Solo (tok/s)", "class",
                           "Best @ conc (tok/s)", "@ conc 8 (tok/s)", "class"],
                          rows))
        L.append("")
        L.append("### How fast is this?")
        L.append("")
        L.append(md_table(["Speed", "User experience"], SPEED_BANDS))
        L.append("")
        if good:
            lo = min(good.values(), key=lambda f: f["decode_tps"])
            hi = max(good.values(), key=lambda f: f["decode_tps"])
            worst8 = [(c, d) for c, cd in conc.items()
                      for n, d in cd.items() if n == 8 and _conc_face_ok(d)]
            extra = ""
            if worst8:
                wc, wd = min(worst8, key=lambda x: x[1]["per_user_mean_tps"])
                extra = ("; at concurrency 8 the floor is {:.1f} tok/s "
                         "({}K ctx)".format(wd["per_user_mean_tps"],
                                            wc // 1024))
            L.append("Solo speed spans **{:.1f}–{:.1f} tok/s** ({}K to {}K "
                     "context){}.".format(lo["decode_tps"], hi["decode_tps"],
                                      lo["ctx"] // 1024, hi["ctx"] // 1024,
                                      extra))
            L.append("")
        chart(os.path.join(ch, "{}_decode.svg".format(slug)),
              "Per-user decode speed vs context length (temperature 0.0, solo)",
              [(cfg["mtp"], [(c, faces[c]["decode_tps"]) if faces.get(c) else (c, None)
                             for c in cfg["ctxs"]])],
              "Context length (tokens)", "Decode speed (tok/s)", yfmt="{:.0f}",
              hlines=SPEED_HL, label_all=True)
        L.append(img("{}_decode".format(slug),
                     "Per-user decode speed vs context length (solo) — {}, {}".format(qshort, cfg["mtp"])))
        L.append("")
        chart(os.path.join(ch, "{}_conc_speed.svg".format(slug)),
              "Per-user decode speed vs concurrency (temperature 0.0)",
              [(f"{c // 1024}K ctx",
                [(n, conc[c][n]["per_user_mean_tps"])
                 if _conc_face_ok(conc[c].get(n)) else (n, None)
                 for n in sorted(conc[c])]) for c in cctxs],
              "Concurrent requests", "Per-user mean speed (tok/s)", yfmt="{:.0f}",
              hlines=SPEED_HL, label_all=True)
        L.append(img("{}_conc_speed".format(slug),
                     "Per-user decode speed vs concurrency — {}, {}".format(qshort, cfg["mtp"])))
        L.append("")

        # ------------------------------------------------ TTFT
        L.append("## Time to first token")
        L.append("")
        L.append("Time from request submission to the first streamed token — "
                 "the primary responsiveness metric. Client-side on the rig "
                 "loopback, it decomposes into queue wait (rising with "
                 "concurrency) and prefill (rising with context); the "
                 "technical analysis below separates the two.")
        L.append("")
        rows = []
        for c in allctx:
            f = faces.get(c)
            c8 = conc.get(c, {}).get(8)
            rows.append([
                ctxrow(c),
                ttft_fmt(f["ttft_s"]) if f else "-",
                ttft_class(f["ttft_s"]) if f else "-",
                ttft_fmt(c8["ttft_mean_s"]) if _conc_face_ok(c8) else "-",
                ttft_fmt(c8["ttft_max_s"]) if _conc_face_ok(c8) else "-",
                ttft_class(c8["ttft_max_s"]) if _conc_face_ok(c8) else "-",
            ])
        L.append(md_table(["Context", "Solo (s)", "class",
                           "@ conc 8 mean (s)", "@ conc 8 worst (s)", "class"],
                          rows))
        L.append("")
        L.append("### How responsive is this?")
        L.append("")
        L.append(md_table(["TTFT", "User experience"], TTFT_BANDS))
        L.append("")
        L.append("> **Caching note.** Every face uses a unique seeded-random "
                 "prompt — a deliberate cold prefill, the worst case. The "
                 "certified posture runs prefix caching (LMCache): in "
                 "production, incremental conversations pay TTFT only for "
                 "new tokens, which is dramatically faster than these "
                 "numbers.")
        L.append("")
        chart(os.path.join(ch, "{}_ttft.svg".format(slug)),
              "Time to first token vs context length (temperature 0.0, solo)",
              [(cfg["mtp"], [(c, faces[c]["ttft_s"]) if faces.get(c) else (c, None)
                             for c in cfg["ctxs"]])],
              "Context length (tokens)", "TTFT (s)", yfmt="{:.1f}",
              hlines=TTFT_HL, label_all=True)
        L.append(img("{}_ttft".format(slug),
                     "TTFT vs context length (solo) — {}, {}".format(qshort, cfg["mtp"])))
        L.append("")
        chart(os.path.join(ch, "{}_conc_ttft.svg".format(slug)),
              "TTFT vs concurrency (temperature 0.0; solid = mean, dashed = worst user)",
              [(f"{c // 1024}K mean",
                [(n, conc[c][n]["ttft_mean_s"])
                 if _conc_face_ok(conc[c].get(n)) else (n, None)
                 for n in sorted(conc[c])]) for c in cctxs] +
              [(f"{c // 1024}K worst",
                [(n, conc[c][n]["ttft_max_s"])
                 if _conc_face_ok(conc[c].get(n)) else (n, None)
                 for n in sorted(conc[c])]) for c in cctxs],
              "Concurrent requests", "TTFT (s)", yfmt="{:.1f}")
        L.append(img("{}_conc_ttft".format(slug),
                     "TTFT vs concurrency (mean and worst user) — {}, {}".format(qshort, cfg["mtp"])))
        L.append("")

        # ------------------------------------------------ capacity analysis
        if pool:
            L.append("## Capacity analysis")
            L.append("")
            for uc in USE_CASES:
                uctx = uc["ctx"] if uc["ctx"] is not None else lastc
                cd = {n: d for n, d in conc.get(uctx, {}).items()
                      if _conc_face_ok(d)}
                if not cd:
                    continue
                cap, note = uc_verdict(cd, uc)
                L.append("### {} ({}K context)".format(uc["name"], uctx // 1024))
                L.append("")
                L.append("{} Thresholds: {}.".format(uc["desc"],
                          " · ".join((["TTFT < {:g}s".format(uc["ttft_bar"])]
                                      if uc["ttft_bar"] is not None else [])
                                     + ["speed >= {:g} tok/s".format(uc["speed_bar"])])))
                L.append("")
                ns = sorted(cd)
                trow = ["TTFT worst user (s)"]
                srow = ["Per-user mean speed (tok/s)"]
                for n in ns:
                    d = cd[n]
                    bad_t = uc["ttft_bar"] is not None and \
                        d["ttft_max_s"] >= uc["ttft_bar"]
                    bad_s = d["per_user_mean_tps"] < uc["speed_bar"]
                    trow.append(ttft_fmt(d["ttft_max_s"]) +
                                (" *(exceeded)*" if bad_t else ""))
                    srow.append("{:.1f}".format(d["per_user_mean_tps"]) +
                                (" *(below)*" if bad_s else ""))
                L.append(md_table(["Metric"] + [f"@ {n}" for n in ns],
                                  [trow, srow]))
                L.append("")
                L.append("**Capacity limit: {}**".format(
                    note if note else f"{cap} concurrent requests"))
                L.append("")
            L.append("### KV pool (the whole-pool bound)")
            L.append("")
            L.append("This boot exposed a KV pool of **{} tokens** ({}). A "
                     "session occupies ctx + {} (output + scheduler slack), "
                     "so the pool holds:".format(_fmt_comma(pool), pool_src,
                                                 OVERHEAD))
            L.append("")
            rows = []
            for c in allctx:
                fit = pool // (c + OVERHEAD)
                measured = max([n for n, d in conc.get(c, {}).items()
                                if _conc_face_ok(d)] or [0])
                rows.append([ctxrow(c), _fmt_comma(fit),
                             (f"{measured} (green)") if measured else "-",
                             "yes" if fit >= 8 else
                             ("marginal" if fit >= 4 else "no")])
            L.append(md_table(["Context", "Pool capacity (sessions)",
                               "Measured green @ conc", "c8 fully resident?"],
                              rows))
            L.append("")
            L.append("*Sessions are the whole-pool bound; `--max-num-seqs 64` "
                     "bounds the running batch above it. TurboQuant dtypes "
                     "multiply the pool up to 3.68x: see the "
                     "[KV-cache dtype report](qwen3.8-27b-fp8-kv-cache-dtype.md).*")
            L.append("")

        # ------------------------------------------------ technical analysis
        tech_any = bool(conc)
        if tech_any:
            L.append("## Technical analysis")
            L.append("")
            # queue decomposition
            L.append("### Queue + prefill decomposition (derived)")
            L.append("")
            L.append("At concurrency 1 the queue is empty, so solo TTFT is "
                     "pure prefill. Under the barrier burst the added "
                     "first-token wait — `ttft_mean(n) − ttft_mean(1)` — is "
                     "the queue+prefill-share stagger the Nth user pays. "
                     "Derived client-side; engine-internal queueing is not "
                     "exported by this engine version.")
            L.append("")
            rows = []
            for c in cctxs:
                base = conc.get(c, {}).get(1)
                if not _conc_face_ok(base):
                    continue
                r = [ctxrow(c)]
                for n in (2, 4, 8):
                    d = conc.get(c, {}).get(n)
                    r.append((f'+{d["ttft_mean_s"] - base["ttft_mean_s"]:.1f} s')
                             if _conc_face_ok(d) else "-")
                rows.append(r)
            if rows:
                L.append(md_table(["Context", "added wait @ n2", "@ n4", "@ n8"],
                                  rows))
                L.append("")
                chart(os.path.join(ch, "{}_queue.svg".format(slug)),
                      "Added first-token wait vs concurrency (derived: "
                      "ttft_mean(n) - ttft_mean(1))",
                      [(f"{c // 1024}K ctx",
                        [(n, conc[c][n]["ttft_mean_s"] - conc[c][1]["ttft_mean_s"])
                         if _conc_face_ok(conc[c].get(n)) and
                         _conc_face_ok(conc[c].get(1)) else (n, None)
                         for n in sorted(conc[c])]) for c in cctxs],
                      "Concurrent requests", "Added wait (s)", yfmt="{:.1f}",
                      label_all=True)
                L.append(img("{}_queue".format(slug),
                             "Added first-token wait vs concurrency (derived "
                             "queue+prefill stagger) — {}, {}".format(qshort, cfg["mtp"])))
                L.append("")
            # prefill
            L.append("### Per-user prefill speed")
            L.append("")
            L.append("Rate at which input context is processed before the "
                     "first token. Client-observed (ctx / TTFT at queue ~0); "
                     "the engine-side device rate is higher — the client view "
                     "includes chunk drain and the decode-transition tail.")
            L.append("")
            rows = []
            for c in cctxs:
                cells = [ctxrow(c)]
                for n in (1, 4, 8):
                    d = conc.get(c, {}).get(n)
                    cells.append((f'{c / d["ttft_mean_s"]:.0f}')
                                 if _conc_face_ok(d) and d["ttft_mean_s"] > 0
                                 else "-")
                rows.append(cells)
            if rows:
                L.append(md_table(["Context", "prefill @ n1 (tok/s)",
                                   "@ n4", "@ n8"], rows))
                L.append("")
                chart(os.path.join(ch, "{}_prefill.svg".format(slug)),
                      "Client-observed prefill rate vs context length (temperature 0.0, solo)",
                      [(cfg["mtp"], [(c, faces[c]["prefill_tps"]) if faces.get(c)
                                     else (c, None) for c in cfg["ctxs"]])],
                      "Context length (tokens)", "Prefill rate (tok/s)",
                      yfmt="{:,.0f}", label_all=True)
                L.append(img("{}_prefill".format(slug),
                             "Client-observed prefill rate vs context length "
                             "(solo) — {}, {}".format(qshort, cfg["mtp"])))
                L.append("")
            # ITL
            L.append("### Inter-token latency")
            L.append("")
            itl_pts = []
            for c in cctxs:
                d = conc.get(c, {}).get(1)
                ia = itl_agg(d) if _conc_face_ok(d) else None
                if ia:
                    itl_pts.append((c, ia))
            if itl_pts:
                lo = min(x["p50"] for _, x in itl_pts)
                hi = max(x["p99"] for _, x in itl_pts)
                loc = [c for c, x in itl_pts if x["p50"] == lo][0]
                hic = [c for c, x in itl_pts if x["p99"] == hi][0]
                L.append("Time between consecutive streamed tokens — the "
                         "smoothness of streaming. Solo p50 spans **{:.1f} ms** "
                         "({}K) to {:.1f} ms; the worst observed p99 is "
                         "**{:.0f} ms** ({}K). With MTP one chunk can carry "
                         "multiple accepted tokens, so the perceived stream "
                         "is smoother than the raw chunk gap.".format(lo, loc // 1024,
                            max(x["p50"] for _, x in itl_pts), hi, hic // 1024))
                L.append("")
                chart(os.path.join(ch, "{}_itl.svg".format(slug)),
                      "Inter-token latency vs context length (temperature 0.0, solo)",
                      [("ITL p50", [(c, faces[c]["itl_ms_p50"]) if faces.get(c) else (c, None)
                                    for c in cfg["ctxs"]]),
                       ("ITL p90", [(c, faces[c]["itl_ms_p90"]) if faces.get(c) else (c, None)
                                    for c in cfg["ctxs"]]),
                       ("ITL p99", [(c, faces[c]["itl_ms_p99"]) if faces.get(c) else (c, None)
                                    for c in cfg["ctxs"]])],
                      "Context length (tokens)", "ITL (ms)", yfmt="{:.0f}",
                      label_all=True)
                L.append(img("{}_itl".format(slug),
                             "Inter-token latency vs context length (solo) — "
                             "{}, {}".format(qshort, cfg["mtp"])))
                L.append("")
            # scaling efficiency
            L.append("### Scaling efficiency")
            L.append("")
            L.append("Percentage of ideal linear scaling achieved as "
                     "concurrency rises (100% = n× the concurrency-1 "
                     "throughput with no per-user cost). Real efficiency is "
                     "always lower — the accelerators are shared; high "
                     "efficiency at your target concurrency signals "
                     "headroom.")
            L.append("")
            chart(os.path.join(ch, "{}_conc_eff.svg".format(slug)),
                  "Scaling efficiency vs concurrency (aggregate / (n x aggregate-at-1))",
                  [(f"{c // 1024}K ctx",
                    [(n, 100.0 * conc[c][n]["agg_tps_wall"] /
                      (n * conc[c][1]["agg_tps_wall"]))
                     if _conc_face_ok(conc[c].get(n)) and
                     _conc_face_ok(conc[c].get(1)) and
                     conc[c][1]["agg_tps_wall"] > 0 else (n, None)
                     for n in sorted(conc[c])]) for c in cctxs],
                  "Concurrent requests", "Scaling efficiency (%)", yfmt="{:.0f}",
                  y_from_zero=False, label_all=True)
            L.append(img("{}_conc_eff".format(slug),
                         "Scaling efficiency vs concurrency — {}, {}".format(qshort, cfg["mtp"])))
            L.append("")

        # ------------------------------------------------ temperature
        if cfg["temps"] and tfaces and any(tfaces.values()):
            L.append("## Temperature analysis ({})".format(cfg["mtp"]))
            L.append("")
            L.append("Higher sampling temperature lowers draft acceptance "
                     "(tok/step falls), so decode speed drops as sampling "
                     "sharpens — the certified probabilistic drafting keeps "
                     "the target distribution correct at every temperature.")
            L.append("")
            L.append(md_table(
                ["Temp", "TTFT (s)", "Decode (tok/s)", "Tok/step",
                 "ITL p50 (ms)", "ITL p90 (ms)"],
                [[t,
                  "{:.2f}".format(f["ttft_s"]) if f else "-",
                  "{:.1f}".format(f["decode_tps"]) if f else "-",
                  "{:.2f}".format(f["tok_per_step"]) if f else "-",
                  "{:.1f}".format(f["itl_ms_p50"]) if f else "-",
                  "{:.1f}".format(f["itl_ms_p90"]) if f else "-"]
                 for t, f in ((t, tfaces.get(t)) for t in TEMPS)]))
            L.append("")
            chart(os.path.join(ch, "{}_temp_decode.svg".format(slug)),
                  "Decode speed vs sampling temperature (8K context, solo)",
                  [("decode", [(float(t), tfaces[t]["decode_tps"]) if tfaces.get(t)
                               else (float(t), None) for t in TEMPS])],
                  "Sampling temperature", "Decode speed (tok/s)", yfmt="{:.0f}",
                  hlines=SPEED_HL, label_all=True)
            L.append(img("{}_temp_decode".format(slug),
                         "Decode speed vs sampling temperature — {}".format(cfg["mtp"])))
            L.append("")
        if t1ok:
            gctx = sorted(set(list(t1ok.keys()) +
                              [c for c in conc if _conc_face_ok(conc[c].get(1))]))
            groups = [f"{c // 1024}K" for c in gctx]
            vals = []
            for c in gctx:
                d0 = conc.get(c, {}).get(1)
                vals.append([d0["per_user_mean_tps"] if _conc_face_ok(d0) else None,
                             t1ok[c]["per_user_mean_tps"] if c in t1ok else None])
            bars(os.path.join(ch, "{}_temp_cmp.svg".format(slug)),
                 "Solo decode speed: temperature 0.0 vs 1.0 (per context)",
                 groups, ["t0.0", "t1.0"], vals, "Decode speed (tok/s)",
                 yfmt="{:.0f}", hlines=SPEED_HL)
            L.append("### Temperature 0.0 vs 1.0 (per context)")
            L.append("")
            L.append(img("{}_temp_cmp".format(slug),
                         "Solo decode speed t0.0 vs t1.0 — {}, {}".format(qshort, cfg["mtp"])))
            L.append("")

        # ------------------------------------------------ results detail
        L.append("## Results — detail tables")
        L.append("")
        # context sweep
        if good:
            mixed_src = len({f["src"] for f in good.values()}) > 1
            t1col = any(c in t1ok for c in good)
            hdr = ["Context", "TTFT (s)", "Prefill (tok/s)", "Decode (tok/s)",
                   "Tok/step", "ITL p50 (ms)", "ITL p90 (ms)", "ITL p99 (ms)"]
            if t1col:
                hdr.append("Decode @ t1.0 (tok/s)")
            rows = []
            for c in cfg["ctxs"]:
                f = faces.get(c)
                if not f and c not in conc:
                    continue
                r = [ctxrow(c),
                     "{:.2f}".format(f["ttft_s"]) if f else "-",
                     _fmt_comma(f["prefill_tps"]) if f else "-",
                     "{:.1f}".format(f["decode_tps"]) if f else "-",
                     "{:.2f}".format(f["tok_per_step"]) if f else "-",
                     "{:.1f}".format(f["itl_ms_p50"]) if f else "-",
                     "{:.1f}".format(f["itl_ms_p90"]) if f else "-",
                     "{:.1f}".format(f["itl_ms_p99"]) if f else "-"]
                if t1col:
                    d1 = t1.get(c)
                    r.append("{:.1f}".format(d1["per_user_mean_tps"])
                             if _conc_face_ok(d1) else "-")
                rows.append(r)
            L.append("### Context sweep (solo, temperature 0.0)")
            L.append("")
            L.append(md_table(hdr, rows))
            L.append("")
            if mixed_src:
                L.append("*Rows sourced from the r139/r140 concurrency-1 "
                         "faces where no r138 solo face exists (same "
                         "cold-prefill measurement).*")
                L.append("")
        # concurrency tables
        for c in cctxs:
            cds = conc[c]
            ns = sorted(cds)
            if not ns:
                continue
            base = cds.get(1)
            base_agg = base["agg_tps_wall"] if _conc_face_ok(base) else None
            rows = []
            for n in ns:
                d = cds[n]
                if not d or d.get("n_fail", 0):
                    rows.append([n, "-",
                                 f'FACE_FAIL ({d.get("n_fail", "?")})'] +
                                ["-"] * 6)
                    continue
                ia = itl_agg(d) or {}
                eff = ("%.0f%%" % (100 * d["agg_tps_wall"] / (n * base_agg))
                       if base_agg else "-")
                rows.append([n,
                             ttft_fmt(d["ttft_mean_s"]), ttft_fmt(d["ttft_max_s"]),
                             "{:.1f}".format(d["per_user_mean_tps"]),
                             "{:.1f}".format(d["per_user_min_tps"]),
                             "{:.0f}".format(d["agg_tps_wall"]), eff,
                             "{:.1f}".format(ia.get("p50", float("nan"))),
                             "{:.1f}".format(ia.get("p90", float("nan")))])
            L.append("### Concurrency sweep — {} context (temperature 0.0)".format(ctxrow(c)))
            L.append("")
            L.append(md_table(["Concurrency", "TTFT mean (s)", "TTFT max (s)",
                               "Per-user mean (tok/s)", "Per-user worst (tok/s)",
                               "Aggregate (tok/s)", "Scaling eff.",
                               "ITL p50 med (ms)", "ITL p90 max (ms)"], rows))
            L.append("")
        if cctxs:
            L.append("*Scaling eff. = aggregate / (n x aggregate-at-1): 100% = "
                     "perfect linear scaling. ITL p50 med = median of "
                     "per-request p50s; ITL p90 max = worst user. Aggregate "
                     "is wall-clock (n x 600 tokens / wall) and includes the "
                     "slowest-user tail.*")
            L.append("")

    # ------------------------------------------------ configuration
    L.append("## Configuration (copy & run on your own hardware)")
    L.append("")
    L.append(config_block(cfg, date))
    L.append("")

    # ------------------------------------------------ test parameters
    L.append("## Test parameters")
    L.append("")
    L.append("- Context lengths: {} tokens (prompt + template). Solo faces: "
             "concurrency 1, 600 output tokens, `ignore_eos`.".format(", ".join(f"{c // 1024}K" for c in cfg["ctxs"])))
    if conc:
        L.append("- Concurrency sweeps: {} at contexts {} — barrier-started "
                 "simultaneous requests, 600 output tokens each, temperature "
                 "0.0, plus one temperature-1.0 solo face per context.".format("/".join(str(n) for n in CONCS),
                  ", ".join(f"{c // 1024}K" for c in cctxs)))
    if cfg["temps"]:
        L.append("- Temperature sweep (MTP acceptance): {} at 8K context.".format(", ".join(TEMPS)))
    L.append("- Prompts: seeded random-token contexts, unique per face and "
             "per request slot (`p138_face.py` / `p139_conc.py`, tag-seeded) "
             "— every face is a cold prefill (no prompt-cache hits).")
    L.append("- Client-side timing on the rig loopback; TTFT includes the "
             "full engine prefill window (at c=1 queue wait ~0; at c=N the "
             "TTFT spread IS the queue+prefill stagger).")
    L.append("")

    # ------------------------------------------------ notes
    L.append("## Notes")
    L.append("")
    L.append("- Per-user speed = output tokens / decode window (spec-decode "
             "aware; with MTP one stream chunk carries multiple accepted "
             "tokens). Aggregate = n x 600 tokens / wall clock.")
    L.append("- The r139 1K t0.0 face runs first after the boot warmup at a "
             "cold spec-verify batch shape — its acceptance (2.8 tok/step) "
             "sits below the family norm (3.8 at 8K); the 8K/32K faces are "
             "the steady reference. Temperature comparisons are within-harness "
             "(r138 sweep vs r139 faces, both reported).")
    L.append("- Prefill rate is the client-observed ctx/TTFT; the engine-side "
             "device rate (P6.36 decomposition) is ~2.1–2.9k tok/s at fp8 — "
             "the client view includes chunk drain and the decode-transition "
             "tail.")
    L.append("- Metric definitions, UX bands, use-case thresholds and the "
             "honest-limits statement: [METHODOLOGY.md](METHODOLOGY.md).")
    L.append("- Banks: `results/2026-11-10-p637-reports/` (solo + temperature "
             "faces, r138) and `results/2026-11-11-p637b-conc/` (concurrency "
             "faces, r139)%s; tools `tools/ar/p138/p138_face.py`, "
             "`tools/ar/p139/p139_conc.py`." %
             (" and `results/2026-11-11-r140-gemma128k/` (gemma 128K annex)"
              if annex else ""))
    L = drop_missing_imgs(L, out)
    with open(os.path.join(out, slug + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ------------------------------------------------------------- KV report

def _kv_solo_rows(banks, entries, ctxs, src139):
    """Shared row builder: solo + (r139 conc where the cell is an r139 cell)."""
    b138, b139 = banks[0], banks[1:]
    rows = []
    for dt, cell, note in entries:
        pool, pool_src = pool_of(banks, cell) if src139(cell) else \
            POOL_FACTS.get(cell, (None, ""))
        for cv in ctxs:
            if src139(cell):
                d1 = load_conc(b139, cell, cv, 1)
                d8 = load_conc(b139, cell, cv, 8)
                if d1 is None and d8 is None:
                    rows.append([dt, f"{cv // 1024}K", "-", "-", "-", "-", "-",
                                 _fmt_comma(pool) if pool else "-", note])
                    continue
                ia = itl_agg(d1) if _conc_face_ok(d1) else {}
                rows.append([dt, f"{cv // 1024}K",
                             ttft_fmt(d1["ttft_mean_s"]) if _conc_face_ok(d1) else "-",
                             "{:.1f}".format(d1["per_user_mean_tps"]) if _conc_face_ok(d1) else "-",
                             ("{:.0f}".format(d8["agg_tps_wall"])) if _conc_face_ok(d8) else "-",
                             ("{:.1f}".format(d8["per_user_mean_tps"])) if _conc_face_ok(d8) else "-",
                             "{:.1f}".format(ia.get("p50", float("nan"))) if ia else "-",
                             _fmt_comma(pool) if pool else "-", note])
            else:
                f = load_face(b138, cell, f"{cell}_c{cv}_t0.0")
                if f is None:
                    rows.append([dt, f"{cv // 1024}K", "-", "-", "-", "-", "-",
                                 _fmt_comma(pool) if pool else "-", note])
                    continue
                rows.append([dt, f"{cv // 1024}K",
                             ttft_fmt(f["ttft_s"]), "{:.1f}".format(f["decode_tps"]),
                             "-", "-", "{:.1f}".format(f["itl_ms_p50"]),
                             _fmt_comma(pool) if pool else "-", note])
    return rows


def kv_report(banks, out, date):
    b138, b139 = banks[0], banks[1:]
    ch = os.path.join(out, "charts")
    annex_pool, annex_src = pool_of(banks, "gm4x")
    annex_ok = any(load_conc(b139, "gm4x", cv, 1) is not None
                   for cv in (8192, 32768, 131072))
    L = ["# KV-cache dtype comparison — both models, TurboQuant, capacity cliffs",
         "",
         md_table(["Field", "Value"], [
             ["Models", "Qwen3.8-27B FP8 + MTP k3 (certified posture); "
                        "Gemma-4-31B (certified posture, no MTP)"],
             ["Knob", "`ARC_KVDTYPE` in `tools/serve_arc_b70.sh` (default `fp8_e4m3`) "
                      "→ `--kv-cache-dtype`"],
             ["Menu", "fp8_e4m3 · fp8 · auto (fp16 KV) · turboquant_k8v4 · "
                      "turboquant_4bit_nc · turboquant_k3v4_nc · turboquant_3bit_nc "
                      "(qwen); fp8_e4m3 · auto (gemma, maxlen-capped)"],
             ["Faces", "r139: conc {1,2,3,4,8} at 1K/8K/32K per dtype; "
                       "r138: solo faces for fp8/auto/fp8/e5m2"],
             ["Engine", "vLLM v0.30.0 — vllm-arc v0.30.0-b70-cert23, TP2"],
             ["Date", date]]),
         "",
         "A dtype that refuses to boot is a valid outcome and is banked as "
         "such — two of the boundaries below are refusals.",
         ""]

    # ------------------------------------------------------------ qwen
    L += ["## Qwen3.8-27B FP8 + MTP k3",
          "",
          "`auto` (fp16 KV — 2 bytes/elem) refuses at the certified "
          "`max_model_len 262144`: one max-len request needs 8.73 GiB of KV "
          "against 6.78 GiB available at GMU 0.8 / TP2, so the engine raises "
          "at boot (cell `kvauto` holds the refusal log). Its rows below are "
          "measured at `max_model_len=131072` (cell `kvauto131`) — still "
          "covering the full 1K–128K report sweep; its pool (198,735) is "
          "**exactly half** the fp8 pool — the fp16 byte-doubling law made "
          "measurable. `fp8_e5m2` is the second refusal and a hard engine "
          "boundary, not a capacity one: `NotImplementedError: "
          "FlashAttention does not support fp8_e5m2 kv-cache on this device` "
          "(cell `kve5m2`) — the esimd FA2 backend ships no e5m2 KV dequant "
          "path on Xe2.",
          ""]
    rows = _kv_solo_rows(banks, KVDTS_Q, (8192, 32768),
                         lambda c: c in ("qf3",))
    L.append(md_table(["--kv-cache-dtype", "Context", "TTFT solo (s)",
                       "Decode solo (tok/s)", "Agg @ c8 (tok/s)",
                       "Per-user @ c8 (tok/s)", "ITL p50 (ms)", "KV pool (tok)",
                       "Note"], rows))
    L.append("")

    # ------------------------------------------------------------ turboquant
    L += ["## TurboQuant presets (Qwen3.8-27B FP8 + MTP k3, r139 concurrency sweeps)",
          "",
          "The wheel's four TurboQuant KV dtypes serve via the same knob; the "
          "r139 gate ran the full conc {1,2,3,4,8} sweep per preset (fresh "
          "boot per cell, `VLLM_ARC_TQ_SPEC_FIX=1` default posture):",
          ""]
    tq_rows = []
    for tp in TQ_PRESETS:
        cell = tp["cell"]
        pool, pool_src = pool_of(banks, cell)
        for cv in (8192, 32768):
            d1 = load_conc(b139, cell, cv, 1)
            d8 = load_conc(b139, cell, cv, 8)
            tq_rows.append([
                "`{}`".format(tp["dtype"]), f"{cv // 1024}K",
                ttft_fmt(d1["ttft_mean_s"]) if _conc_face_ok(d1) else "-",
                "{:.1f}".format(d1["per_user_mean_tps"]) if _conc_face_ok(d1) else "-",
                ("{:.0f}".format(d8["agg_tps_wall"])) if _conc_face_ok(d8) else "-",
                ("{:.1f}".format(d8["per_user_mean_tps"])) if _conc_face_ok(d8) else "-",
                ("{} ({})".format(_fmt_comma(pool), tp["poolx"])) if pool else tp["poolx"],
                tp["note"]])
    L.append(md_table(["dtype", "Context", "TTFT solo (s)", "Decode solo (tok/s)",
                       "Agg @ c8 (tok/s)", "Per-user @ c8 (tok/s)",
                       "KV pool (tok, x vs fp8)", "Note"], tq_rows))
    L.append("")
    # per-preset conc tables
    for tp in TQ_PRESETS:
        cell = tp["cell"]
        for cv in (8192, 32768):
            ns = conc_concs(b139, cell, cv)
            if not ns:
                continue
            base = load_conc(b139, cell, cv, 1)
            base_agg = base["agg_tps_wall"] if _conc_face_ok(base) else None
            rows = []
            for n in ns:
                d = load_conc(b139, cell, cv, n)
                if not _conc_face_ok(d):
                    rows.append([n, "-", "-"] + ["-"] * 5)
                    continue
                ia = itl_agg(d) or {}
                rows.append([n, ttft_fmt(d["ttft_mean_s"]), ttft_fmt(d["ttft_max_s"]),
                             "{:.1f}".format(d["per_user_mean_tps"]),
                             "{:.1f}".format(d["per_user_min_tps"]), "{:.0f}".format(d["agg_tps_wall"]),
                             ("%.0f%%" % (100 * d["agg_tps_wall"] / (n * base_agg))
                              if base_agg else "-"),
                             "{:.1f}".format(ia.get("p90", float("nan")))])
            L.append(f'### `{tp["dtype"]}` — {cv // 1024}K context')
            L.append("")
            L.append(md_table(["Conc", "TTFT mean (s)", "TTFT max (s)",
                               "Per-user mean", "Per-user worst", "Aggregate (tok/s)",
                               "Scaling eff.", "ITL p90 max (ms)"], rows))
            L.append("")

    # ------------------------------------------------------------ gemma
    L += ["## Gemma-4-31B (no MTP)",
          "",
          "`auto` refuses at the certified gemma posture (`ARC_MAXLEN=40960`: "
          "the fp16 KV pool cannot cover one 40K session on the 31B geometry "
          "— cell `gkv_auto` banks the refusal); it serves at "
          "`ARC_MAXLEN=16384` (cell `gkv_auto16`). The incumbent "
          "fp8_e4m3 rows are the r139 `gm8` cell.",
          ""]
    g_rows = _kv_solo_rows(banks, KVDTS_G, (8192, 32768),
                           lambda c: c in ("gm8", "gkv_auto16"))
    L.append(md_table(["--kv-cache-dtype", "Context", "TTFT solo (s)",
                       "Decode solo (tok/s)", "Agg @ c8 (tok/s)",
                       "Per-user @ c8 (tok/s)", "ITL p50 (ms)", "KV pool (tok)",
                       "Note"], g_rows))
    L.append("")

    # ------------------------------------------------------------ gemma 128K annex
    if annex_ok or annex_pool:
        L += ["## Gemma 128K annex (r140): 131072 maxlen needs MXFP4 weights",
              "",
              "The certified gemma posture caps `max_model_len` at 40960. "
              "Raising it to 131072 is a KV-headroom question: the annex "
              "boots the 8-bit lane (`gm8x`) and the 4-bit lane (`gm4x`) at "
              "`ARC_MAXLEN=131072` — the MXFP4 weight savings are what buy "
              "the pool for a full 128K session.",
              ""]
        gm8x_oc = outcome(banks, "gm8x")
        gm8x_pool, _ = pool_of(banks, "gm8x")
        gm8x_rows = []
        for cv in (8192, 32768, 131072):
            d1 = load_conc(b139, "gm4x", cv, 1)
            d8 = load_conc(b139, "gm4x", cv, 8)
            gm8x_rows.append([
                "MXFP4 (gm4x)", f"{cv // 1024}K",
                ttft_fmt(d1["ttft_mean_s"]) if _conc_face_ok(d1) else "-",
                "{:.1f}".format(d1["per_user_mean_tps"]) if _conc_face_ok(d1) else "-",
                ("{:.0f}".format(d8["agg_tps_wall"])) if _conc_face_ok(d8) else "-",
                ("{:.1f}".format(d8["per_user_mean_tps"])) if _conc_face_ok(d8) else "-",
                _fmt_comma(annex_pool) if annex_pool else "-"])
        tail = (", pool {} tokens".format(_fmt_comma(gm8x_pool))
                if gm8x_pool else " — the boot refusal banks the boundary")
        L.append("8-bit control (`gm8x`): outcome `{}`{}.".format(
            gm8x_oc, tail))
        L.append("")
        L.append(md_table(["dtype (cell)", "Context", "TTFT solo (s)",
                           "Decode solo (tok/s)", "Agg @ c8 (tok/s)",
                           "Per-user @ c8 (tok/s)", "KV pool (tok)"], gm8x_rows))
        L.append("")
        L.append("Full report: [gemma-4-31b-mxfp4-128k](gemma-4-31b-mxfp4-128k.md).")
        L.append("")

    # ------------------------------------------------------------ capacity
    L += ["## Capacity cliffs (the whole point of the dtype menu)",
          "",
          "Session footprint = ctx + 600 + 64 tokens; pool = per-boot parsed "
          "(`posture.txt`). Max concurrent **full-length** sessions per dtype:",
          ""]
    cap_rows = []
    entries = ([("fp8_e4m3", "qf3")] + [(tp["dtype"], tp["cell"]) for tp in TQ_PRESETS]
               + [("fp8", "kvfp8"), ("auto@131072", "kvauto131"),
                  ("fp8_e4m3 (gemma)", "gm8"), ("auto@16384 (gemma)", "gkv_auto16")])
    if annex_pool:
        entries.append(("mxfp4@131072 (gemma)", "gm4x"))
    for dt, cell in entries:
        pool, pool_src = pool_of(banks, cell)
        if not pool:
            cap_rows.append([dt, "-", "-", "-", "-", POOL_FACTS.get(cell, ("", ""))[1]])
            continue
        cap_rows.append([dt, _fmt_comma(pool),
                         f"{pool // (1024 + OVERHEAD)}",
                         f"{pool // (8192 + OVERHEAD)}",
                         f"{pool // (32768 + OVERHEAD)}",
                         pool_src])
    L.append(md_table(["dtype (cell)", "Pool (tok)", "@1K", "@8K", "@32K",
                       "pool source"], cap_rows))
    L.append("")
    L.append("Refusal boundaries: `auto` at max_model_len 262144 (qwen, fp16 "
             "capacity) and 40960 (gemma); `fp8_e5m2` at boot (kernel, not "
             "memory). Every refusal is banked with its serve log.")
    L.append("")

    # ------------------------------------------------------------ config deltas
    L += ["## Configuration (copyable deltas on the certified posture)",
          "",
          md_table(["dtype", "cell", "delta", "max_model_len"], [
              ["fp8_e4m3", "qf3 / gm8", "(none — incumbent)", "262144 / 40960"],
              ["fp8", "kvfp8", "ARC_KVDTYPE=fp8", "262144"],
              ["auto", "kvauto131", "ARC_KVDTYPE=auto", "131072"],
              ["auto", "kvauto", "ARC_KVDTYPE=auto — refuses", "262144"],
              ["auto (gemma)", "gkv_auto16", "ARC_KVDTYPE=auto ARC_MAXLEN=16384", "16384"],
              ["auto (gemma)", "gkv_auto", "ARC_KVDTYPE=auto — refuses at 40960", "40960"],
              ["turboquant_*", "tq_*", "ARC_KVDTYPE=turboquant_<preset>", "262144"],
              ["mxfp4 (gemma 128K)", "gm4x",
               "ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_MAXLEN=131072", "131072"],
              ["fp8_e5m2", "kve5m2", "ARC_KVDTYPE=fp8_e5m2 — refuses (no kernel)", "-"]]),
          "",
          "`ARC_MAXLEN` is what makes fp16-KV `auto` servable — the refusal "
          "is the capacity cliff itself. The full qwen3.8-27b-fp8-k3 invocation: "
          "[qwen3.8-27b-fp8-k3.md](qwen3.8-27b-fp8-k3.md).",
          ""]

    # ------------------------------------------------------------ charts
    cats = [dt for dt, _c, _n in KVDTS_Q]
    for met, agg, ylab, yfmt, fname, cap in [
            ("ttft_mean_s", False, "TTFT (s)", "{:.1f}", "kvdt_ttft",
             "TTFT vs KV-cache dtype (solo, 8K/32K)"),
            ("per_user_mean_tps", False, "Decode speed (tok/s)", "{:.0f}", "kvdt_decode",
             "Solo decode speed vs KV-cache dtype (8K/32K)"),
            ("agg_tps_wall", True, "Aggregate @ c8 (tok/s)", "{:.0f}", "kvdt_agg8",
             "Aggregate throughput @ c8 vs KV-cache dtype (8K/32K)")]:
        series = []
        for cv in (8192, 32768):
            pts = []
            for i, (dt, cell, _n) in enumerate(KVDTS_Q):
                f = ctx_face(b138, cell, cv) if cell == "qf3" else load_face(
                    b138, cell, f"{cell}_c{cv}_t0.0")
                if cell == "qf3":
                    d = load_conc(b139, cell, cv, 8 if agg else 1)
                    if _conc_face_ok(d):
                        pts.append((i, d[met] if agg else d["ttft_mean_s"]))
                        continue
                if f:
                    pts.append((i, f["ttft_s"] if met == "ttft_mean_s"
                                else f["decode_tps"] if not agg else None))
                else:
                    pts.append((i, None))
            series.append((f"{cv // 1024}K ctx", pts))
        chart(os.path.join(ch, fname + ".svg"),
              cap + " — qwen fp8 k3", series, "--kv-cache-dtype", ylab,
              xcat=cats, yfmt=yfmt, hlines=TTFT_HL if "TTFT" in cap else ())
        L.append(img(fname, cap + " — qwen fp8 k3"))
        L.append("")
    # pool bars
    groups, vals = [], []
    for dt, cell in entries:
        pool, _src = pool_of(banks, cell)
        groups.append(dt.replace("turboquant_", "tq_"))
        vals.append([pool])
    bars(os.path.join(ch, "kvdt_pool.svg"),
         "KV pool by dtype (tokens, per-boot parsed)",
         groups, ["KV pool"], vals, "KV pool (tokens)", yfmt="{:,.0f}")
    L.append(img("kvdt_pool", "KV pool by dtype — capacity is the dtype menu's axis"))
    L.append("")
    # turboquant conc overlay vs fp8
    overlay = [("fp8_e4m3 (qf3)", "qf3")] + [(tp["dtype"], tp["cell"]) for tp in TQ_PRESETS]
    for met, ylab, yfmt, fname, cap, ideal in [
            ("per_user_mean_tps", "Per-user mean speed (tok/s)", "{:.0f}",
             "tq_conc_speed", "Per-user speed vs concurrency — fp8 vs TurboQuant (8K ctx)", False),
            ("agg_tps_wall", "Aggregate throughput (tok/s)", "{:.0f}",
             "tq_conc_agg", "Aggregate throughput vs concurrency — fp8 vs TurboQuant (8K ctx)", True),
            ("ttft_max_s", "Worst-user TTFT (s)", "{:.1f}",
             "tq_conc_ttft", "Worst-user TTFT vs concurrency — fp8 vs TurboQuant (8K ctx)", False)]:
        series = []
        for lab, cell in overlay:
            ns = conc_concs(b139, cell, 8192)
            pts = []
            for n in ns:
                d = load_conc(b139, cell, 8192, n)
                pts.append((n, d[met]) if _conc_face_ok(d) else (n, None))
            series.append((lab, pts))
        chart(os.path.join(ch, fname + ".svg"), cap, series,
              "Concurrent requests", ylab, yfmt=yfmt, hlines=(
                  TTFT_HL if "TTFT" in cap else (SPEED_HL if "Per-user" in ylab else ())),
              ideal=ideal, label_all=True)
        L.append(img(fname, cap))
        L.append("")
    L.append("Banks: r139 `results/2026-11-11-p637b-conc/` (conc cells + "
             "posture pools), r138 `results/2026-11-10-p637-reports/` (solo "
             "dtype cells + refusals); knob: `ARC_KVDTYPE` in "
             "`tools/serve_arc_b70.sh`. Identity records per TQ preset: "
             "CONFIGURATION-arc \"KV-cache dtype\".")
    L = drop_missing_imgs(L, out)
    with open(os.path.join(out, "qwen3.8-27b-fp8-kv-cache-dtype.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# --------------------------------------------------------- methodology

def methodology_report(out, date):
    L = ["# Benchmark methodology — the performance reports",
         "",
         md_table(["Field", "Value"], [
             ["Harnesses", "`tools/ar/p138/p138_face.py` (solo), "
              "`tools/ar/p139/p139_conc.py` (concurrency)"],
             ["Banks", "`results/2026-11-10-p637-reports/` (r138), "
              "`results/2026-11-11-p637b-conc/` (r139), "
              "`results/2026-11-11-r140-gemma128k/` (gemma 128K annex)"],
             ["Engine", "vLLM v0.30.0 — vllm-arc fork, image "
              "`vllm-arc:v0.30.0-b70-cert23`, XPU graphs, TP2"],
             ["Hardware", "2x Intel Arc Pro B70 (BMG-G31, 32 GB each), "
              "host 64 GB RAM, close-out F defaults"],
             ["Date", date]]),
         "",
         "## What each report measures",
         "",
         "User-facing performance: how many concurrent requests a "
         "configuration supports at a given context length before streaming "
         "speed or first-token wait degrade below the user-experience bands. "
         "Every number comes from a banked face JSON; nothing is estimated "
         "except where explicitly marked (`~` capacity estimates, derived "
         "queue decomposition).",
         "",
         "## Metrics",
         "",
         md_table(["Metric", "Definition"], [
             ["TTFT", "Request submission (common barrier) to the first "
              "streamed token, client-side on the rig loopback. At "
              "concurrency 1 this is pure prefill (queue ~0)."],
             ["Per-user speed", "600 output tokens / decode window "
              "(`total - ttft`) per request — spec-aware: with MTP one "
              "stream chunk carries multiple accepted tokens, and the "
              "numerator is the true token count, not the chunk count."],
             ["Aggregate throughput", "n x 600 tokens / wall clock, where "
              "wall is the slowest request's total. Includes the "
              "staggered-start and slowest-user tail: the honest "
              "saturated-burst number."],
             ["Prefill rate", "Context length / TTFT at concurrency 1 "
              "(client-observed; includes chunk drain and the "
              "decode-transition tail — the engine-side device rate is "
              "higher)."],
             ["ITL p50/p90/p99", "Gaps between consecutive streamed chunks, "
              "per request; aggregated as median-of-p50s and max-of-p90/p99 "
              "(worst user)."],
             ["Added wait (n)", "ttft_mean(n) - ttft_mean(1): derived "
              "queue+prefill-share stagger under the barrier burst."],
             ["Scaling efficiency", "aggregate(n) / (n x aggregate(1)); "
              "100% = ideal linear scaling."],
             ["KV pool", "Parsed from the booted engine log "
              "(`KV cache size: N tokens`) and banked per cell in "
              "`posture.txt`."]]),
         "",
         "## User-experience bands",
         "",
         "### Streaming speed",
         "",
         md_table(["Speed", "Experience"], SPEED_BANDS),
         "",
         "### Time to first token",
         "",
         md_table(["TTFT", "Experience"], TTFT_BANDS),
         "",
         "## Use-case thresholds",
         "",
         "The use-case guidance in each report applies the band edges: "
         "interactive work holds TTFT < 10 s (top of the *slow* band) and "
         "speed >= 15 tok/s (the *acceptable* floor); batch document "
         "processing holds the speed bar only. Capacity is the largest "
         "measured concurrency where both bars hold; `~` marks a linear "
         "estimate between two measured points.",
         "",
         "## Prompt law (cold prefill, always)",
         "",
         "Every face and every concurrency slot uses a unique seeded-random "
         "prompt (seed derives from the face tag and slot). Two reasons: "
         "shared prefixes would retrieve from the prefix cache and stop "
         "measuring prefill; and identical prompts across slots would "
         "corrupt the concurrency measurement. The flip side: all TTFT "
         "numbers here are worst-case cold prefill — production traffic "
         "with prefix caching (LMCache, enabled in the certified posture) "
         "pays TTFT only for new tokens.",
         "",
         "## Honest limits",
         "",
         "- Client-side timing only: queue wait is derived "
         "(ttft_mean(n) - ttft_mean(1)), not exported by the engine.",
         "- No GPU telemetry (utilization, power, temperature) is captured "
         "in these banks.",
         "- A concurrency face requires all n requests to succeed "
         "(`n_fail = 0`); failed faces are banked and shown, never averaged "
         "away.",
         "- By-design capacity skips (pool < n x (ctx + 664)) are recorded "
         "as `KV_CAP` rows in the gate log — the measured capacity wall, "
         "not a failure.",
         "- A dtype or posture that refuses to boot is a valid, banked "
         "outcome with its serve log.",
         "",
         "## Reproducing",
         "",
         "Each report's **Configuration** block is the byte-exact validated "
         "invocation (`tools/serve_arc_b70.sh` + the banked posture echo). "
         "The gate scripts are `tools/ar/r139_conc.sh` (concurrency matrix) "
         "and `tools/ar/r140_gemma128k.sh` (gemma 128K annex); reports "
         "regenerate with `tools/ar/p139/p139_report.py`."]
    with open(os.path.join(out, "METHODOLOGY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------- index

def _peak(banks, cfg):
    f = None
    for ctxv in cfg["ctxs"]:
        g = solo_face(banks, cfg["cell"], ctxv)
        if g and (f is None or g["decode_tps"] > f["decode_tps"]):
            f = g
    return f'{f["decode_tps"]:.1f} tok/s @ {f["ctx"] // 1024}K' if f else "-"


def _uc_short(banks, cfg, uc, lastc):
    b139 = banks[1:]
    uctx = uc["ctx"] if uc["ctx"] is not None else lastc
    cd = {n: d for n, d in
          ((n, load_conc(b139, cfg["cell"], uctx, n))
           for n in conc_concs(b139, cfg["cell"], uctx)) }
    if any(_conc_face_ok(d) for d in cd.values()):
        cap, note = uc_verdict(cd, uc)
        if note:
            return note.split(" (")[0]
        return f"{cap} req"
    return "-"


def index_report(banks, out, date):
    ch = os.path.join(out, "charts")
    qcfgs = [c for c in CONFIGS if c["plan"].startswith("qwen")]
    gcfgs = [c for c in CONFIGS if c["plan"] == "gemma"]
    for fam, cfgs, tag in [(qcfgs, qcfgs, "qwen3.8-27b"), (gcfgs, gcfgs, "gemma-4-31b")]:
        for met, ylab, yfmt, fname, cap, zeroy in [
                ("decode_tps", "Decode speed (tok/s)", "{:.0f}", tag + "_all_decode",
                 "{}: solo decode speed vs context (all configs, t0.0)".format(tag.capitalize()), True),
                ("ttft_s", "TTFT (s)", "{:.1f}", tag + "_all_ttft",
                 "{}: TTFT vs context (all configs, t0.0)".format(tag.capitalize()), False)]:
            series = []
            for c in cfgs:
                pts = []
                for ctxv in c["ctxs"]:
                    f = solo_face(banks, c["cell"], ctxv)
                    pts.append((ctxv, f[met] if f else None))
                series.append(("{} {}".format(c["quant"].split(" (")[0], c["mtp"]), pts))
            chart(os.path.join(ch, fname + ".svg"), cap, series,
                  "Context length (tokens)", ylab, yfmt=yfmt, y_from_zero=zeroy,
                  hlines=SPEED_HL if "speed" in cap.lower() else TTFT_HL)
        # conc-8 comparison bars (per-user mean + aggregate) at 8K
        groups, pv, av = [], [], []
        for c in cfgs:
            d8 = load_conc(banks[1:], c["cell"], 8192, 8)
            groups.append("{} {}".format(c["quant"].split(" (")[0].replace(" W4A16", ""),
                                     c["mtp"].replace("no MTP", "k0")))
            pv.append([d8["per_user_mean_tps"] if _conc_face_ok(d8) else None])
            av.append([d8["agg_tps_wall"] if _conc_face_ok(d8) else None])
        if any(v[0] is not None for v in pv):
            bars(os.path.join(ch, "{}_all_c8user.svg".format(tag)),
                 "{}: per-user speed @ concurrency 8 (8K ctx, t0.0)".format(tag.capitalize()),
                 groups, ["per-user mean"], pv, "Per-user mean speed (tok/s)",
                 yfmt="{:.0f}", hlines=SPEED_HL)
        if any(v[0] is not None for v in av):
            bars(os.path.join(ch, "{}_all_c8agg.svg".format(tag)),
                 "{}: aggregate throughput @ concurrency 8 (8K ctx, t0.0)".format(tag.capitalize()),
                 groups, ["aggregate"], av, "Aggregate throughput (tok/s)", yfmt="{:.0f}")

    L = ["# Performance reports — 2x Intel Arc Pro B70 (vllm-arc cert23)",
         "",
         "World-class close-out reports per model x quantization x MTP "
         "posture: executive summary and key findings, use-case guidance "
         "with capacity verdicts, system throughput, UX-banded speed and "
         "TTFT, capacity analysis, technical analysis (queue decomposition, "
         "prefill, ITL, scaling efficiency), concurrency sweeps {{1,2,3,4,8}}, "
         "temperature analysis, the full copyable configuration, and charts. "
         "Generated {} from the r138 bank "
         "(`results/2026-11-10-p637-reports/`) + r139 bank "
         "(`results/2026-11-11-p637b-conc/`) + gemma 128K annex "
         "(`results/2026-11-11-r140-gemma128k/`).".format(date),
         "",
         "Metric definitions, UX bands and thresholds: "
         "[METHODOLOGY.md](METHODOLOGY.md).",
         ""]

    def cfg_rows(cfgs):
        rows = []
        for c in cfgs:
            d8 = load_conc(banks[1:], c["cell"], 8192, 8)
            lastc = max(c["ctxs"])
            rows.append([
                "[{}]({}.md)".format(c["slug"], c["slug"]), c["quant"], c["mtp"],
                _peak(banks, c),
                ("{:.1f}".format(d8["per_user_mean_tps"])) if _conc_face_ok(d8) else "-",
                ("{:.0f}".format(d8["agg_tps_wall"])) if _conc_face_ok(d8) else "-",
                _hr_col(d8["agg_tps_wall"]) if _conc_face_ok(d8) else "-",
                ttft_fmt(d8["ttft_max_s"]) if _conc_face_ok(d8) else "-",
                _uc_short(banks, c, USE_CASES[0], lastc),
                outcome(banks, c["cell"])])
        return rows

    hdr = ["Report", "Precision", "MTP", "Peak decode (solo)",
           "Per-user @ c8, 8K", "Agg @ c8, 8K", "Tokens/hr @ c8",
           "TTFT max @ c8 (s)", "Chat capacity (8K)", "Outcome"]

    def _hr_col(tps):
        v = tps * 3600
        return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{int(v / 1e3)}K"
    L += ["## Qwen3.8-27B (dense hybrid-GDN, MTP k3 and no-MTP)", ""]
    L.append(md_table(hdr, cfg_rows(qcfgs)))
    L += ["", "## Gemma-4-31B (dense hybrid, no MTP head; 128K annex on MXFP4)", ""]
    L.append(md_table(hdr, cfg_rows(gcfgs)))
    L += ["", "## Cross-cutting", "",
          "- [KV-cache dtype comparison — both models + TurboQuant + capacity "
          "cliffs](qwen3.8-27b-fp8-kv-cache-dtype.md)",
          "- [Benchmark methodology](METHODOLOGY.md)",
          ""]
    for nm, cap in [("qwen3.8-27b_all_decode", "Qwen: solo decode speed vs context (all configs)"),
                    ("qwen3.8-27b_all_ttft", "Qwen: TTFT vs context (all configs)"),
                    ("gemma-4-31b_all_decode", "Gemma: solo decode speed vs context (all configs)"),
                    ("gemma-4-31b_all_ttft", "Gemma: TTFT vs context (all configs)"),
                    ("qwen3.8-27b_all_c8user", "Qwen: per-user speed @ c8 (8K ctx)"),
                    ("qwen3.8-27b_all_c8agg", "Qwen: aggregate throughput @ c8 (8K ctx)"),
                    ("gemma-4-31b_all_c8user", "Gemma: per-user speed @ c8 (8K ctx)"),
                    ("gemma-4-31b_all_c8agg", "Gemma: aggregate throughput @ c8 (8K ctx)")]:
        if not os.path.isfile(os.path.join(ch, nm + ".svg")):
            continue
        L.append(img(nm, cap))
        L.append("")
    L += ["## Reading the numbers", "",
          "- **Per-user speed (streaming UX)**: < 15 tok/s slow · 15–25 "
          "acceptable · 25–50 fast · ≥ 50 very fast.",
          "- **TTFT**: < 0.5 s instant · 0.5–2 s responsive · 2–5 s "
          "noticeable · 5–10 s slow · > 10 s frustrating.",
          "- **Concurrency**: r139 barrier-starts N simultaneous requests, "
          "each a full cold prefill + 600 streamed tokens — the worst-case "
          "burst; sustained interactive user counts run well below request "
          "concurrency.",
          "- Solo faces (r138): concurrency 1, 600 output tokens, "
          "temperature 0.0 unless stated; unique per-face prompts (cold "
          "prefill). MTP lanes add the 6-point temperature sweep; r139 adds "
          "a t1.0 solo face per context on every config.",
          "- Hardware: 2x Arc Pro B70 (32 GB each, one card per socket, TP2), "
          "host 64 GB RAM; close-out F defaults (--shm-size 16g, 16 GB "
          "LMCache CPU tier)."]
    L = drop_missing_imgs(L, out)
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    b138 = Bank(sys.argv[1])
    b139 = Bank(sys.argv[2])
    out, rest = sys.argv[3], sys.argv[4:]
    b140 = None
    if rest and os.path.isdir(rest[0]):
        b140 = Bank(rest[0])
        date = rest[1] if len(rest) > 1 else "2026-11-11"
    else:
        date = rest[0] if rest else "2026-11-11"
    banks139 = [b for b in (b139, b140) if b and b.ok]
    banks = [b138] + banks139
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    for cfg in CONFIGS:
        per_config_report(banks, out, cfg, date)
        print("report: {}.md".format(cfg["slug"]))
    kv_report(banks, out, date)
    print("report: qwen3.8-27b-fp8-kv-cache-dtype.md")
    methodology_report(out, date)
    print("report: METHODOLOGY.md")
    index_report(banks, out, date)
    print("report: README.md (index)")


if __name__ == "__main__":
    main()
