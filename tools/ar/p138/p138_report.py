#!/usr/bin/env python3
"""p138_report.py -- P6.37 close-out report generator (local, post-pull).

Reads the r138 bank (faces + postures + outcomes) and emits the owner
close-out deliverable: per-config performance reports in docs/reports/
with copyable configuration blocks and SVG line charts of ALL data
(millstoneai-style: TTFT / decode speed / prefill speed / ITL vs context
length, decode speed vs sampling temperature), a kv-cache-dtype
comparison report, and an index with cross-config overlay charts.

Derived metrics (client-side, solo streaming, concurrency 1):
  decode tok/s = max_tokens / (total_s - ttft_s)   -- spec-decode aware
                 (numerator law: rate-accounting-spec-sse-chunks memory)
  prefill tok/s = ctx / ttft_s                     -- queue ~0 at c=1
  tok/step     = max_tokens / n_stream_chunks      -- MTP acceptance view

Usage: p138_report.py <bank_dir> <docs_out_dir> [date]
"""
import glob
import json
import os
import sys

QCTX = [1024, 8192, 32768, 65536, 131072]
GCTX = [1024, 8192, 32768]
TEMPS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
QCFG = '{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'

QF8 = "ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1"
QMX = ("ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b "
       "ARC_QUANTCFG=" + QCFG + " ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0")
GM = ("ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 "
      "ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0")

# cell -> report identity. env = the exact words the gate passed (parity
# with tools/ar/r138_reports.sh; the banked posture.txt echo is the
# runtime evidence for each).
CONFIGS = [
    dict(cell="qf3", slug="qwen3.8-27b-fp8-k3", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="FP8 (fp8 checkpoint, compressed-tensors)", plan="qwen3", mtp="MTP k3",
         env=QF8 + " ARC_SPEC=3", ctxs=QCTX, temps=True,
         blurb="the certified incumbent posture"),
    dict(cell="qf0", slug="qwen3.8-27b-fp8-k0", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="FP8 (fp8 checkpoint, compressed-tensors)", plan="qwen0", mtp="no MTP",
         env=QF8 + " ARC_SPEC=0", ctxs=QCTX, temps=False,
         blurb="the MTP-off ablation of the incumbent"),
    dict(cell="q83", slug="qwen3.8-27b-mxfp8-k3", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="MXFP8 (online, bf16 base checkpoint)", plan="qwen3", mtp="MTP k3",
         env=QMX + " ARC_QUANT=mxfp8 ARC_SPEC=3", ctxs=QCTX, temps=True,
         blurb="the online-MXFP8 lane"),
    dict(cell="q80", slug="qwen3.8-27b-mxfp8-k0", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="MXFP8 (online, bf16 base checkpoint)", plan="qwen0", mtp="no MTP",
         env=QMX + " ARC_QUANT=mxfp8 ARC_SPEC=0", ctxs=QCTX, temps=False,
         blurb="the MTP-off ablation of the MXFP8 lane"),
    dict(cell="q43", slug="qwen3.8-27b-mxfp4-k3", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="MXFP4 W4A16 (online, bf16 base checkpoint)", plan="qwen3", mtp="MTP k3",
         env=QMX + " ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=3", ctxs=QCTX, temps=True,
         blurb="the decode champion posture (P6.35-V winner)"),
    dict(cell="q40", slug="qwen3.8-27b-mxfp4-k0", model="Qwen3.8-27B", org="Qwen (Alibaba)",
         quant="MXFP4 W4A16 (online, bf16 base checkpoint)", plan="qwen0", mtp="no MTP",
         env=QMX + " ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=0", ctxs=QCTX, temps=False,
         blurb="the MTP-off ablation of the MXFP4 lane"),
    dict(cell="gf8", slug="gemma-4-31b-fp8", model="Gemma-4-31B", org="Google",
         quant="FP8 (online, bf16 base checkpoint)", plan="gemma", mtp="no MTP",
         env=GM + " ARC_QUANT=fp8", ctxs=GCTX, temps=False,
         blurb="the gemma online-FP8 capability lane"),
    dict(cell="gm8", slug="gemma-4-31b-mxfp8", model="Gemma-4-31B", org="Google",
         quant="MXFP8 (online, bf16 base checkpoint)", plan="gemma", mtp="no MTP",
         env=GM + " ARC_QUANT=mxfp8", ctxs=GCTX, temps=False,
         blurb="the gemma MXFP8 lane"),
    dict(cell="gm4", slug="gemma-4-31b-mxfp4", model="Gemma-4-31B", org="Google",
         quant="MXFP4 W4A16 (online, bf16 base checkpoint)", plan="gemma", mtp="no MTP",
         env=GM + " ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16", ctxs=GCTX, temps=False,
         blurb="the gemma MXFP4 lane"),
]

KVDTS = [("fp8_e4m3", "qf3", "the certified incumbent"),
         ("auto", "kvauto131",
          "fp16 KV; refuses at max_model_len 262144, measured at 131072"),
         ("fp8", "kvfp8", ""),
         ("fp8_e5m2", "kve5m2",
          "refuses to boot — FA2 has no e5m2 KV path on Xe2")]

# ---------------------------------------------------------------- faces

def load_face(bank, cell, tag):
    p = os.path.join(bank, cell, "p138_face_%s.json" % tag)
    if not os.path.isfile(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    if d.get("ttft_s", -1) <= 0 or d.get("total_s", -1) <= 0:
        return None
    dw = d["total_s"] - d["ttft_s"]
    d["decode_tps"] = d["max_tokens"] / dw if dw > 0 else 0.0
    d["prefill_tps"] = d["ctx"] / d["ttft_s"]
    d["tok_per_step"] = (d["max_tokens"] / d["n_stream_chunks"]
                         if d.get("n_stream_chunks") else 0.0)
    d["decode_window_s"] = dw
    return d


def ctx_face(bank, cell, c):
    return load_face(bank, cell, "%s_c%d_t0.0" % (cell, c))


def temp_face(bank, cell, t):
    return load_face(bank, cell, "%s_s8192_t%s" % (cell, t))


def outcome(bank, cell):
    p = os.path.join(bank, cell, "outcome.txt")
    if not os.path.isfile(p):
        return "MISSING"
    return open(p, encoding="utf-8").read().strip()

# ---------------------------------------------------------------- SVG

PALETTE = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c",
           "#0891b2", "#65a30d", "#be185d"]
MARKERS = ["circle", "square", "triangle", "diamond"]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_tick(v):
    if v >= 1048576 and v % 1048576 == 0:
        return "%dM" % (v // 1048576)
    if v >= 1024 and v % 1024 == 0:
        return "%dK" % (v // 1024)
    return str(int(v))


def _niceTicks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** int(__import__("math").floor(__import__("math").log10(raw)))
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
          y_from_zero=True):
    """Hand-rolled SVG line chart. series = [(label, [(x, y), ...])].
    xcat = list of category labels -> x values are indices into it."""
    import math
    W, H, ML, MR, MT, MB = 780, 430, 78, 18, 46, 52
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
    yhi = max(ys) * 1.12 if max(ys) > 0 else 1.0

    def Y(v):
        return MT + PH - (v - ylo) / (yhi - ylo) * PH

    out = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
           'viewBox="0 0 %d %d" font-family="Segoe UI, Helvetica, Arial, '
           'sans-serif">' % (W, H, W, H)]
    out.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    out.append('<text x="%d" y="24" font-size="16" font-weight="600" '
               'fill="#101828">%s</text>' % (ML, esc(title)))
    # y grid + ticks
    for t in _niceTicks(ylo, yhi, 5):
        yy = Y(t)
        if yy < MT - 2 or yy > MT + PH + 2:
            continue
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#f0f0f0" '
                   'stroke-width="1"/>' % (ML, yy, ML + PW, yy))
        out.append('<text x="%d" y="%.1f" font-size="11" fill="#667085" '
                   'text-anchor="end">%s</text>' % (ML - 8, yy + 4, yfmt.format(t)))
    # x ticks
    if xcat is not None:
        for i, lab in enumerate(xcat):
            xx = X(i)
            out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#e5e5e5" '
                       'stroke-width="1"/>' % (xx, MT, xx, MT + PH))
            out.append('<text x="%.1f" y="%d" font-size="11" fill="#667085" '
                       'text-anchor="middle">%s</text>' % (xx, MT + PH + 18, esc(lab)))
    else:
        for xv in xs:
            xx = X(xv)
            out.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#e5e5e5" '
                       'stroke-width="1"/>' % (xx, MT, xx, MT + PH))
            out.append('<text x="%.1f" y="%d" font-size="11" fill="#667085" '
                       'text-anchor="middle">%s</text>' % (xx, MT + PH + 18,
                                                               _fmt_tick(xv)))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#d0d5dd" '
               'stroke-width="1"/>' % (ML, MT + PH, ML + PW, MT + PH))
    out.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#d0d5dd" '
               'stroke-width="1"/>' % (ML, MT, ML, MT + PH))
    # series
    for si, (lab, ps) in enumerate(series):
        col = PALETTE[si % len(PALETTE)]
        mk = MARKERS[si % len(MARKERS)]
        good = [(xv, yv) for xv, yv in ps if yv is not None]
        if len(good) >= 2:
            pl = " ".join("%.1f,%.1f" % (X(xv), Y(yv)) for xv, yv in good)
            out.append('<polyline points="%s" fill="none" stroke="%s" '
                       'stroke-width="2.2"/>' % (pl, col))
        for xv, yv in good:
            xx, yy = X(xv), Y(yv)
            if mk == "circle":
                out.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s"/>' % (xx, yy, col))
            elif mk == "square":
                out.append('<rect x="%.1f" y="%.1f" width="8" height="8" fill="%s"/>' % (xx - 4, yy - 4, col))
            elif mk == "triangle":
                out.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="%s"/>'
                           % (xx, yy - 5, xx - 5, yy + 4, xx + 5, yy + 4, col))
            else:
                out.append('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" fill="%s"/>'
                           % (xx, yy - 5, xx + 5, yy, xx, yy + 5, xx - 5, yy, col))
        if good:  # last-point value label
            lx, ly = good[-1]
            out.append('<text x="%.1f" y="%.1f" font-size="10.5" fill="%s" '
                       'text-anchor="end" font-weight="600">%s</text>'
                       % (min(X(lx) + 44, ML + PW), Y(ly) - 8, col, yfmt.format(ly)))
    # legend (top, horizontal)
    lx = ML
    for si, (lab, _ps) in enumerate(series):
        col = PALETTE[si % len(PALETTE)]
        w = 16 + 8 * min(len(lab), 26) + 18
        if lx + w > ML + PW:
            break
        out.append('<line x1="%d" y1="34" x2="%d" y2="34" stroke="%s" '
                   'stroke-width="3"/>' % (lx, lx + 16, col))
        out.append('<text x="%d" y="38" font-size="11.5" fill="#1d2939">%s</text>'
                   % (lx + 20, esc(lab[:26])))
        lx += w
    out.append('<text x="%d" y="%d" font-size="12.5" fill="#475467" '
               'text-anchor="middle">%s</text>' % (ML + PW // 2, H - 10, esc(xlabel)))
    out.append('<text x="14" y="%d" font-size="12.5" fill="#475467" '
               'text-anchor="middle" transform="rotate(-90 14 %d)">%s</text>'
               % (MT + PH // 2, MT + PH // 2, esc(ylabel)))
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")

# ------------------------------------------------------------ md helpers

def ctxrow(c):
    return "%d (%sK)" % (c, c // 1024)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def img(slug, name, caption):
    return ('![%s](charts/%s.svg)\n\n<i>%s</i>' % (caption, name, caption))

# ------------------------------------------------------- config rendering

def config_block(cfg, date):
    env = cfg["env"]
    words = dict(w.split("=", 1) for w in env.split())
    spec = words.get("ARC_SPEC", "0")
    gmu = words.get("ARC_GMU", "0.8")
    maxlen = words.get("ARC_MAXLEN", "262144")
    quant = words.get("ARC_QUANT", "")
    is_gemma = "gemma" in cfg["cell"]
    kvdt = "fp8_e4m3"
    lmc = words.get("ARC_LMC", "1") == "1"
    spec_flag = ("--speculative-config "
                 "'{\"method\":\"mtp\",\"num_speculative_tokens\":%s,"
                 "\"draft_sample_method\":\"probabilistic\"}'" % spec
                 if spec != "0" else "")
    quant_flag = ("--quantization %s" % quant) if quant else ""
    qcfg_flag = ("--quantization-config '%s'" % QCFG) if "ARC_QUANTCFG" in words else ""
    mode_flag = ("ARC_MXFP4_MODE=w4a16 " if "ARC_MXFP4_MODE" in words else "")
    lines = []
    lines.append("#### Validated invocation (what the r138 gate ran, byte-exact)\n")
    lines.append("```bash")
    lines.append("# image vllm-arc:v0.30.0-b70-cert23 ; %s ; TP2 on 2x Arc Pro B70" % date)
    lines.append("ARC_IMAGE=vllm-arc:v0.30.0-b70-cert23 ARC_NAME=arc-b70 \\")
    lines.append(("  %s%s \\" % (mode_flag, env)).replace("  \\ ", " \\"))
    lines.append("  bash serve_arc_b70.sh     # tools/serve_arc_b70.sh in this repo")
    lines.append("```")
    lines.append("\nThe wrapper is the audited source of truth (banked posture echo: "
                 "`%s/posture.txt`). What it expands to:\n" % cfg["cell"])
    lines.append("```bash")
    lines.append("# 1) container boot layer  [serve_arc_b70.sh:199-210]")
    lines.append("docker run -d --name arc-b70 \\")
    lines.append("  --privileged --network host --ipc host \\")
    lines.append("  --shm-size 16g \\              # close-out F: 64 GB-RAM hosts")
    lines.append("  -v %s:/models/target:ro \\" % words.get("ARC_MODEL", "/models/<model>"))
    if is_gemma:
        lines.append("  -v <chat-template.json>:/root/chat_template.json:ro \\")
    lines.append("  -v ./neo_cache:/root/.neo_cache -v ./triton_cache:/root/.triton_cache \\")
    lines.append("  -v ./inductor_cache:/root/.inductor_cache \\")
    lines.append("  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \\")
    lines.append("  --entrypoint /bin/bash vllm-arc:v0.30.0-b70-cert23 -c 'sleep infinity'")
    lines.append("")
    lines.append("# 2) engine layer (docker exec)  [serve_arc_b70.sh:327-375]")
    lines.append("export PYTHONHASHSEED=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=%s ZE_AFFINITY_MASK=0,1"
                 % words.get("ARC_RPC", "300"))
    lines.append("export CCL_ENABLE_SYCL_KERNELS=1 CCL_ZE_IPC_EXCHANGE=pidfd CCL_TOPO_P2P_ACCESS=1")
    lines.append("export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0 CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072")
    lines.append("export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576 CCL_SYCL_ALLGATHERV_TMP_BUF=0 CCL_SYCL_ALLREDUCE_TMP_BUF=0")
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
    lines.append("  --served-model-name %s --tensor-parallel-size 2 \\" % words.get("ARC_SERVED", "<model>"))
    lines.append("  --gpu-memory-utilization %s --max-model-len %s \\" % (gmu, maxlen))
    lines.append("  --max-num-batched-tokens 8192 --max-num-seqs 64 --block-size 64 \\")
    lines.append("  --dtype float16 %s %s \\" % (quant_flag, qcfg_flag))
    lines.append("  --kv-cache-dtype %s \\" % kvdt)
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
        lines.append("  %s \\" % spec_flag)
    lines.append("  --port 8000")
    lines.append("```")
    return "\n".join(lines)

# ------------------------------------------------------------- reports

def per_config_report(bank, out, cfg, date):
    cell = cfg["cell"]
    ch = os.path.join(out, "charts")
    slug = cfg["slug"]
    oc = outcome(bank, cell)
    faces = {c: ctx_face(bank, cell, c) for c in cfg["ctxs"]}
    tfaces = {t: temp_face(bank, cell, t) for t in TEMPS} if cfg["temps"] else {}
    L = []
    L.append("# %s on 2x Intel Arc Pro B70 — %s, %s" %
             (cfg["model"], cfg["quant"].split(" (")[0], cfg["mtp"]))
    L.append("")
    L.append(md_table(["Field", "Value"], [
        ["Organization", cfg["org"]],
        ["Parameters", "27B dense hybrid-GDN (+MTP head)" if "Qwen" in cfg["model"]
         else "31B dense hybrid (two-head-size GDN, no MTP head)"],
        ["Precision", cfg["quant"]],
        ["Accelerators", "2x Intel Arc Pro B70 (BMG-G31, 32 GB each, 64 GB total), TP2"],
        ["Host RAM", "64 GB (close-out F posture: --shm-size 16g, 16 GB LMCache CPU tier)"],
        ["Engine", "vLLM v0.30.0 — vllm-arc fork, image v0.30.0-b70-cert23, XPU graphs"],
        ["KV cache dtype", "fp8_e4m3"],
        ["Speculative decoding", cfg["mtp"] + ", probabilistic drafting" if "k3" in cfg["mtp"] else "disabled"],
        ["Date", date],
        ["Cell outcome", oc],
    ]))
    L.append("")
    if oc != "FACES_OK":
        L.append("> **Cell did not complete** (`%s`). See `%s/serve.log` in the bank — "
                 "a refusal/boot failure is a valid matrix outcome." % (oc, cell))
    good = {c: f for c, f in faces.items() if f}
    if good:
        c1k = good.get(1024)
        peak = max(good.values(), key=lambda f: f["decode_tps"])
        L.append("## Executive summary")
        L.append("")
        L.append("%s — %s — measured on the shipping cert23 image, solo streaming, "
                 "seeded-random unique prompts (no cache hits), `ignore_eos`." %
                 (cfg["model"], cfg["blurb"]))
        L.append("")
        rows = []
        if c1k:
            rows.append(["TTFT @ 1K ctx", "%.2f s" % c1k["ttft_s"]])
        lastc = max(good)
        if good[lastc]:
            rows.append(["TTFT @ %dK ctx" % (lastc // 1024), "%.1f s" % good[lastc]["ttft_s"]])
        rows.append(["Peak decode speed", "%.1f tok/s @ %sK ctx" %
                     (peak["decode_tps"], peak["ctx"] // 1024)])
        rows.append(["Decode @ largest ctx", "%.1f tok/s" % good[lastc]["decode_tps"]])
        pf = [f["prefill_tps"] for f in good.values()]
        rows.append(["Prefill rate (client, ctx/TTFT)",
                     "{:,.0f} (min) … {:,.0f} tok/s (max)".format(min(pf), max(pf))])
        if cfg["temps"] and tfaces:
            t0 = tfaces.get("0.0")
            t1 = tfaces.get("1.0")
            if t0 and t1:
                rows.append(["Temperature effect on decode (8K ctx)",
                             "%.1f tok/s @ t0.0 → %.1f tok/s @ t1.0 (%+.0f%%)" %
                             (t0["decode_tps"], t1["decode_tps"],
                              100 * (t1["decode_tps"] / t0["decode_tps"] - 1))])
        nok = sum(1 for f in faces.values() if f) + sum(
            1 for f in tfaces.values() if f)
        ntot = len(faces) + len(tfaces)
        rows.append(["Success rate", "%d/%d faces (ctx sweep + temp sweep)"
                     % (nok, ntot)])
        L.append(md_table(["Metric", "Result"], rows))
        L.append("")
        L.append("## Test parameters")
        L.append("")
        L.append("- Context lengths: %s tokens (prompt + template). Concurrency 1, "
                 "600 output tokens per request, `ignore_eos`." %
                 ", ".join("%dK" % (c // 1024) for c in cfg["ctxs"]))
        if cfg["temps"]:
            L.append("- Temperature sweep (MTP acceptance): %s at 8K context." %
                     ", ".join(TEMPS))
        L.append("- Prompts: seeded random-token contexts, unique per face "
                 "(`p138_face.py`, tag-seeded) — every face is a cold prefill "
                 "(no prompt-cache hits).")
        L.append("- Client-side timing on the rig loopback; TTFT includes the full "
                 "engine prefill window (queue ~0 at concurrency 1).")
        L.append("")
        L.append("## Configuration (copy & run on your own hardware)")
        L.append("")
        L.append(config_block(cfg, date))
        L.append("")
        L.append("## Results — context sweep (temperature 0.0)")
        L.append("")
        L.append(md_table(
            ["Context", "TTFT (s)", "Prefill (tok/s)", "Decode (tok/s)",
             "Tok/step", "ITL p50 (ms)", "ITL p90 (ms)", "ITL p99 (ms)"],
            [[ctxrow(c),
              "%.2f" % f["ttft_s"] if f else "—",
              "{:,.0f}".format(f["prefill_tps"]) if f else "—",
              "%.1f" % f["decode_tps"] if f else "—",
              "%.2f" % f["tok_per_step"] if f else "—",
              "%.1f" % f["itl_ms_p50"] if f else "—",
              "%.1f" % f["itl_ms_p90"] if f else "—",
              "%.1f" % f["itl_ms_p99"] if f else "—"]
             for c, f in ((c, faces.get(c)) for c in cfg["ctxs"])]))
        L.append("")
        if cfg["temps"] and tfaces:
            L.append("## Results — temperature sweep (8K context, %s)" % cfg["mtp"])
            L.append("")
            L.append(md_table(
                ["Temp", "TTFT (s)", "Decode (tok/s)", "Tok/step",
                 "ITL p50 (ms)", "ITL p90 (ms)"],
                [[t,
                  "%.2f" % f["ttft_s"] if f else "—",
                  "%.1f" % f["decode_tps"] if f else "—",
                  "%.2f" % f["tok_per_step"] if f else "—",
                  "%.1f" % f["itl_ms_p50"] if f else "—",
                  "%.1f" % f["itl_ms_p90"] if f else "—"]
                 for t, f in ((t, tfaces.get(t)) for t in TEMPS)]))
            L.append("")
            L.append("Higher temperature lowers draft acceptance (tok/step falls), "
                     "so decode speed drops as sampling sharpness rises — the "
                     "certified probabilistic drafting keeps the target "
                     "distribution correct at every temperature.")
            L.append("")
        # charts
        L.append("## Charts")
        L.append("")
        xs = cfg["ctxs"]
        chart(os.path.join(ch, "%s_ttft.svg" % slug),
              "Time to first token vs context length (temperature 0.0)",
              [(cfg["mtp"], [(c, faces[c]["ttft_s"]) if faces.get(c) else (c, None)
                             for c in xs])],
              "Context length (tokens)", "TTFT (s)", yfmt="{:.1f}")
        chart(os.path.join(ch, "%s_decode.svg" % slug),
              "Decode speed vs context length (temperature 0.0)",
              [(cfg["mtp"], [(c, faces[c]["decode_tps"]) if faces.get(c) else (c, None)
                             for c in xs])],
              "Context length (tokens)", "Decode speed (tok/s)", yfmt="{:.0f}")
        chart(os.path.join(ch, "%s_prefill.svg" % slug),
              "Client-observed prefill rate vs context length (temperature 0.0)",
              [(cfg["mtp"], [(c, faces[c]["prefill_tps"]) if faces.get(c) else (c, None)
                             for c in xs])],
              "Context length (tokens)", "Prefill rate (tok/s)", yfmt="{:,.0f}")
        chart(os.path.join(ch, "%s_itl.svg" % slug),
              "Inter-token latency vs context length (temperature 0.0)",
              [("ITL p50", [(c, faces[c]["itl_ms_p50"]) if faces.get(c) else (c, None)
                            for c in xs]),
               ("ITL p90", [(c, faces[c]["itl_ms_p90"]) if faces.get(c) else (c, None)
                            for c in xs])],
              "Context length (tokens)", "ITL (ms)", yfmt="{:.0f}")
        for nm, cap in [("ttft", "TTFT vs context length"),
                        ("decode", "Decode speed vs context length"),
                        ("prefill", "Prefill rate vs context length"),
                        ("itl", "Inter-token latency vs context length")]:
            L.append(img(slug, "%s_%s" % (slug, nm), "%s — %s, %s" %
                         (cap, cfg["quant"].split(" (")[0], cfg["mtp"])))
            L.append("")
        if cfg["temps"] and tfaces:
            chart(os.path.join(ch, "%s_temp_decode.svg" % slug),
                  "Decode speed vs sampling temperature (8K context)",
                  [("decode", [(float(t), tfaces[t]["decode_tps"]) if tfaces.get(t)
                               else (float(t), None) for t in TEMPS])],
                  "Sampling temperature", "Decode speed (tok/s)", yfmt="{:.0f}")
            chart(os.path.join(ch, "%s_temp_ttft.svg" % slug),
                  "TTFT vs sampling temperature (8K context)",
                  [("TTFT", [(float(t), tfaces[t]["ttft_s"]) if tfaces.get(t)
                             else (float(t), None) for t in TEMPS])],
                  "Sampling temperature", "TTFT (s)", yfmt="{:.1f}")
            L.append(img(slug, "%s_temp_decode" % slug,
                         "Decode speed vs sampling temperature — %s" % cfg["mtp"]))
            L.append("")
            L.append(img(slug, "%s_temp_ttft" % slug,
                         "TTFT vs sampling temperature — %s" % cfg["mtp"]))
            L.append("")
        L.append("## Notes")
        L.append("")
        L.append("- Decode speed = output tokens / decode window (spec-decode aware; "
                 "with MTP one stream chunk carries multiple accepted tokens — "
                 "Tok/step is the acceptance view).")
        L.append("- Prefill rate is the client-observed ctx/TTFT; the engine-side "
                 "device rate (P6.36 decomposition) is ~2.1–2.9k tok/s at fp8 — "
                 "the client view includes chunk drain and the decode-transition tail.")
        L.append("- Bank: `results/%s/` (faces, postures, serve logs); face tool: "
                 "`tools/ar/p138/p138_face.py`." % os.path.basename(bank))
    with open(os.path.join(out, slug + ".md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return faces


def kv_report(bank, out, date):
    ch = os.path.join(out, "charts")
    rows = []
    ser_d = []
    ser_t = []
    for dt, cell, note in KVDTS:
        f8 = ctx_face(bank, cell, 8192) if cell == "qf3" else load_face(
            bank, cell, "%s_c8192_t0.0" % cell)
        f32 = ctx_face(bank, cell, 32768) if cell == "qf3" else load_face(
            bank, cell, "%s_c32768_t0.0" % cell)
        for ctxv, f in [(8192, f8), (32768, f32)]:
            rows.append([dt, "%dK" % (ctxv // 1024),
                         "%.2f" % f["ttft_s"] if f else "BOOT_FAIL/—",
                         "%.1f" % f["decode_tps"] if f else "—",
                         "%.1f" % f["itl_ms_p50"] if f else "—",
                         note])
    L = ["# Qwen3.8-27B FP8 + MTP k3 — KV-cache dtype comparison (close-out G)",
         "",
         md_table(["Field", "Value"], [
             ["Model", "Qwen3.8-27B FP8 checkpoint, MTP k3 (certified posture)"],
             ["Sweep", "--kv-cache-dtype in {fp8_e4m3, auto, fp8, fp8_e5m2}"],
             ["Contexts", "8K, 32K (600 out tokens, temperature 0.0)"],
             ["Engine", "vLLM v0.30.0 — vllm-arc v0.30.0-b70-cert23, TP2"],
             ["Date", date]]),
         "",
         "`fp8_e4m3` is the certified incumbent (cells `qf3`); the other dtypes "
         "are cells `kvfp8` / `kve5m2` via the `ARC_KVDTYPE` knob. "
         "A dtype that refuses to boot is a valid outcome and is banked as such. "
         "`auto` (fp16 KV — 2 bytes/elem) is exactly that case at the certified "
         "posture: serving one max-len request needs 8.73 GiB of KV cache "
         "against 6.78 GiB available at GMU 0.8 / TP2, so `--kv-cache-dtype "
         "auto` raises at boot (vLLM estimates max_model_len 202048 would fit; "
         "cell `kvauto` holds the refusal log). Its row below is measured at "
         "`max_model_len=131072` (cell `kvauto131`), which still covers the "
         "full 1K-128K report sweep. `fp8_e5m2` is the second refusal and a "
         "hard engine boundary, not a capacity one: "
         "`NotImplementedError: FlashAttention does not support fp8_e5m2 "
         "kv-cache on this device` (cell `kve5m2`) — the esimd FA2 attention "
         "backend ships no e5m2 KV dequant path on Xe2.",
         "",
         md_table(["--kv-cache-dtype", "Context", "TTFT (s)", "Decode (tok/s)",
                   "ITL p50 (ms)", "Note"], rows),
         ""]
    # copyable config deltas per dtype (config-parity law: full invocation)
    L += ["",
          "## Configuration (copyable deltas on the certified posture)",
          "",
          "Base = the Qwen3.8-27B FP8 + MTP k3 block from "
          "[qwen3.8-27b-fp8-k3.md](qwen3.8-27b-fp8-k3.md); each cell changes only:",
          "",
          md_table(["dtype", "cell", "delta", "max_model_len"], [
              ["fp8_e4m3", "qf3", "(none — incumbent)", "262144"],
              ["auto", "kvauto131", "ARC_KVDTYPE=auto", "131072"],
              ["fp8", "kvfp8", "ARC_KVDTYPE=fp8", "262144"],
              ["fp8_e5m2", "kve5m2", "ARC_KVDTYPE=fp8_e5m2 — refuses", "—"]]),
          "",
          "`ARC_MAXLEN=131072` is what makes `auto` servable — the 262144 "
          "refusal (cell `kvauto`) is the fp16-KV capacity cliff itself. "
          "`fp8_e5m2` has no working configuration on this engine "
          "(attention-kernel boundary, not memory)."]
    # charts: x categorical dtype, two series (8K, 32K)
    cats = [dt for dt, _, _ in KVDTS]
    for met, ylab, yfmt, fname, cap in [
            ("decode_tps", "Decode speed (tok/s)", "{:.0f}", "kvdt_decode",
             "Decode speed vs KV-cache dtype"),
            ("ttft_s", "TTFT (s)", "{:.1f}", "kvdt_ttft", "TTFT vs KV-cache dtype"),
            ("itl_ms_p50", "ITL p50 (ms)", "{:.0f}", "kvdt_itl",
             "Inter-token latency vs KV-cache dtype")]:
        series = []
        for ctxv in (8192, 32768):
            pts = []
            for i, (dt, cell, _n) in enumerate(KVDTS):
                f = ctx_face(bank, cell, ctxv) if cell == "qf3" else load_face(
                    bank, cell, "%s_c%d_t0.0" % (cell, ctxv))
                pts.append((i, f[met] if f else None))
            series.append(("%dK ctx" % (ctxv // 1024), pts))
        chart(os.path.join(ch, fname + ".svg"), cap + " (qwen fp8, MTP k3)",
              series, "--kv-cache-dtype", ylab, xcat=cats, yfmt=yfmt)
        L.append(img("kvdt", fname, cap + " — qwen fp8 k3"))
        L.append("")
    L.append("Bank: `results/%s/` (cells qf3, kvauto=refusal, kvauto131, "
             "kvfp8, kve5m2); knob: `ARC_KVDTYPE` in `tools/serve_arc_b70.sh`."
             % os.path.basename(bank))
    # banked TurboQuant matrix (cert8/cert9 gate campaigns; unchanged on
    # cert23 — cited, not re-run; law-6 economy)
    L += ["",
          "## The `turboquant_*` presets (banked cert8/cert9 matrix)",
          "",
          "The wheel's four TurboQuant KV dtypes remain servable via the same "
          "flag. Their full performance matrix was banked at the cert8/cert9 "
          "gates (P6.14/P6.15 campaigns — unchanged on cert23, so cited here, "
          "not re-run):",
          "",
          md_table(["dtype", "KV pool ×", "t0 solo decode (tok/s)",
                    "t1.0 agg8 (tok/s)", "U32k/U64k TTFT (s)"],
                   [["turboquant_k8v4", "1.98×", "88.5 / 88.6", "172.95", "15.29 / 33.03"],
                    ["turboquant_4bit_nc", "2.86×", "86.2 / 86.0", "134.42", "15.91 / 34.02"],
                    ["turboquant_k3v4_nc", "3.22×", "80.8 / 80.5", "137.11", "15.48 / 33.78"],
                    ["turboquant_3bit_nc", "3.68×", "85.8 / 85.6", "171.32", "16.02 / 35.74"]]),
          "",
          "Presets trade aggregate speed for KV capacity (up to 3.68× pool); "
          "the identity records and per-preset guidance: "
          "CONFIGURATION-arc \"KV-cache dtype\"; evidence: "
          "`results/2026-11-04-p619-tqverify/`."]
    with open(os.path.join(out, "qwen3.8-27b-fp8-kv-cache-dtype.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def index_report(bank, out, date):
    ch = os.path.join(out, "charts")
    byslug = {c["slug"]: c for c in CONFIGS}
    qcfgs = [c for c in CONFIGS if c["plan"].startswith("qwen")]
    gcfgs = [c for c in CONFIGS if c["plan"] == "gemma"]
    for fam, cfgs, tag in [(qcfgs, qcfgs, "qwen"), (gcfgs, gcfgs, "gemma")]:
        for met, ylab, yfmt, fname, cap, zeroy in [
                ("decode_tps", "Decode speed (tok/s)", "{:.0f}", tag + "_all_decode",
                 "%s: decode speed vs context (all configs, t0.0)" % tag.capitalize(), True),
                ("ttft_s", "TTFT (s)", "{:.1f}", tag + "_all_ttft",
                 "%s: TTFT vs context (all configs, t0.0)" % tag.capitalize(), False)]:
            series = []
            for c in cfgs:
                pts = []
                for ctxv in c["ctxs"]:
                    f = ctx_face(bank, c["cell"], ctxv)
                    pts.append((ctxv, f[met] if f else None))
                series.append(("%s %s" % (c["quant"].split(" (")[0], c["mtp"]), pts))
            chart(os.path.join(ch, fname + ".svg"), cap, series,
                  "Context length (tokens)", ylab, yfmt=yfmt, y_from_zero=zeroy)
    L = ["# Performance reports — 2x Intel Arc Pro B70 (vllm-arc cert23)",
         "",
         "Individual close-out reports per model x quantization x MTP posture, "
         "each with the full copyable configuration, all decode/TTFT/prefill/ITL "
         "data, and line charts. Generated %s from the r138 bank "
         "(`results/%s/`)." % (date, os.path.basename(bank)),
         "",
         "## Qwen3.8-27B (dense hybrid-GDN, MTP k3 and no-MTP)",
         ""]
    L.append(md_table(["Report", "Precision", "MTP", "Peak decode", "Outcome"], [
        ["[%s](%s.md)" % (c["slug"], c["slug"]), c["quant"], c["mtp"],
         _peak(bank, c), outcome(bank, c["cell"])] for c in qcfgs]))
    L += ["", "## Gemma-4-31B (dense hybrid, no MTP head)", ""]
    L.append(md_table(["Report", "Precision", "MTP", "Peak decode", "Outcome"], [
        ["[%s](%s.md)" % (c["slug"], c["slug"]), c["quant"], c["mtp"],
         _peak(bank, c), outcome(bank, c["cell"])] for c in gcfgs]))
    L += ["", "## Cross-cutting", "",
          "- [Qwen FP8 + MTP k3 — KV-cache dtype comparison](qwen3.8-27b-fp8-kv-cache-dtype.md)",
          ""]
    for nm in ["qwen3.8-27b_all_decode", "qwen3.8-27b_all_ttft", "gemma-4-31b_all_decode", "gemma-4-31b_all_ttft"]:
        L.append(img("idx", nm, nm.replace("_", " ")))
        L.append("")
    L += ["## Reading the numbers", "",
          "- **Decode speed**: < 15 tok/s slow · 15–25 acceptable · 25–50 fast · "
          "> 50 very fast (streaming UX).",
          "- **TTFT**: < 0.5 s instant · 0.5–2 s responsive · 2–5 s noticeable · "
          "5–10 s slow · > 10 s frustrating.",
          "- All cells: concurrency 1, 600 output tokens, temperature 0.0 unless "
          "the chart/table says otherwise; unique per-face prompts (cold prefill).",
          "- Hardware: 2x Arc Pro B70 (32 GB each), host 64 GB RAM; "
          "close-out F defaults (--shm-size 16g, 16 GB LMCache CPU tier)."]
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def _peak(bank, cfg):
    f = None
    for ctxv in cfg["ctxs"]:
        g = ctx_face(bank, cfg["cell"], ctxv)
        if g and (f is None or g["decode_tps"] > f["decode_tps"]):
            f = g
    return "%.1f tok/s @ %dK" % (f["decode_tps"], f["ctx"] // 1024) if f else "—"


def main():
    bank = sys.argv[1]
    out = sys.argv[2]
    date = sys.argv[3] if len(sys.argv) > 3 else "2026-11-10"
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    for cfg in CONFIGS:
        per_config_report(bank, out, cfg, date)
        print("report: %s.md" % cfg["slug"])
    kv_report(bank, out, date)
    print("report: qwen3.8-27b-fp8-kv-cache-dtype.md")
    index_report(bank, out, date)
    print("report: README.md (index)")


if __name__ == "__main__":
    main()
