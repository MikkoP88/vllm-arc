"""r27_analyze.py — dual-mode attribution from rank chrome traces.

Phase A (graphed product posture): replayed kernels are INVISIBLE to the XPU
torch profiler (r21b law) — report the visible eager-side per-step kernel set
and collective counts only (anchored on single-kernel varlen-FA clusters =
draft step 0).

Phase B (eager posture, cudagraph NONE): every kernel visible; wall is
launch-bound, CONTENT = per-kernel duration sums. lm_head GEMMs are isolated
by duration (vocab 124160/rank, ~5-10x any layer GEMM, BW-bound so duration
~constant across 1-5 rows) — expect exactly 5 per step: [verify, draft x4].
Step segment = (V_k, V_{k+1}]; collectives inside are split draft/forward by
count separation and by position around the draft-lmhead run.

usage: python3 r27_analyze.py [profdir]   (default /root/prof)
"""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

STEADY_MIN_US = 30_000
STEADY_MAX_US = 2_000_000


def cat(name):
    n = name.lower()
    if ("allreduce" in n or "all_gather" in n or "allgather" in n
            or "reducecopy" in n or "barrier" in n or "ccl" in n
            or "reduce_scatter" in n or "reducescatter" in n):
        return "collective"
    if "gemm" in n:
        return "gemm-fp8" if ("fp8" in n or n.startswith("gemm_kernel")) else "gemm"
    if "matmul" in n:
        return "gemm"
    if "varlen_fwd" in n or "fa2" in n or "flash" in n or "fmha" in n:
        return "fa-attn"
    if "gdn" in n or "chunk" in n or "gated_delta" in n:
        return "gdn"
    if "quant" in n:
        return "quant"
    if "rms_norm" in n or "layer_norm" in n:
        return "norm"
    if "silu" in n or "act_and_mul" in n:
        return "act"
    if "copy" in n or "memcpy" in n or "fill" in n or "elementwise" in n:
        return "copy/elementwise"
    if "rotary" in n or "rope" in n:
        return "rope"
    return "other"


def load_kerns(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        tr = json.load(f)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    ks = [e for e in evs
          if e.get("ph") == "X" and e.get("cat", "").lower() in
          ("kernel", "xpu_kernel", "gpu_op")]
    ks.sort(key=lambda e: e["ts"])
    return ks


def agg_print(label, evs, n):
    a = defaultdict(lambda: [0, 0.0])
    for e in evs:
        c = cat(e["name"])
        a[c][0] += 1
        a[c][1] += e["dur"]
    tot = sum(v[1] for v in a.values()) or 1.0
    print(f"\n== {label} (n/step={len(evs)/max(1,n):.0f} kernels) ==")
    print(f"  content total {tot/max(1,n)/1000:7.2f} ms/step (dur-units)")
    for c, (cnt, us) in sorted(a.items(), key=lambda kv: -kv[1][1]):
        print(f"  {c:16s} {us/n/1000:7.2f} ms/step {100*us/tot:5.1f}% "
              f"n/step={cnt/n:6.1f} avg={us/cnt/1000:7.1f}u")
    return a


def analyze_graphed(path, ks):
    """Phase A: visible eager-side kernels only; anchors = 1-kernel varlen
    clusters (draft step 0 FA)."""
    print(f"\n########## {os.path.basename(path)} [GRAPHED] ##########")
    vl = [e for e in ks if "varlen_fwd" in e["name"].lower()]
    clusters = []
    for e in vl:
        if clusters and e["ts"] - clusters[-1][1] < 10_000:
            clusters[-1][1] = e["ts"] + e["dur"]
            clusters[-1][2] += 1
        else:
            clusters.append([e["ts"], e["ts"] + e["dur"], 1])
    hist = defaultdict(int)
    for _, _, c in clusters:
        hist[min(c, 5)] += 1
    print(f"kernels={len(ks)} varlen={len(vl)} clusters={len(clusters)} "
          f"cluster-size hist (5=>=5): {dict(sorted(hist.items()))}")
    anchors = [c for c in clusters if c[2] == 1]
    if len(anchors) < 10:
        print(f"TOO FEW single anchors ({len(anchors)})")
        return
    steady = []
    for i in range(len(anchors) - 1):
        d = anchors[i + 1][0] - anchors[i][0]
        if STEADY_MIN_US <= d <= STEADY_MAX_US:
            steady.append((anchors[i][0], anchors[i + 1][0]))
    n = len(steady)
    deltas = sorted(x[1] - x[0] for x in steady)
    print(f"steady pairs={n} step wall p50={deltas[len(deltas)//2]/1000:.1f}ms")
    seg = []
    j = 0
    for d0, d1 in steady:
        while j < len(ks) and ks[j]["ts"] < d0:
            j += 1
        while j < len(ks) and ks[j]["ts"] < d1:
            seg.append(ks[j])
            j += 1
    agg_print("VISIBLE per-step set (outside-graph eager kernels)", seg, n)
    byname = defaultdict(lambda: [0, 0.0])
    for e in seg:
        byname[e["name"][:72]][0] += 1
        byname[e["name"][:72]][1] += e["dur"]
    print("\n== top visible kernels ==")
    for nm, (cnt, us) in sorted(byname.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {us/n/1000:6.3f} ms/step n/step={cnt/n:4.1f} "
              f"avg={us/cnt/1000:7.1f}u  {nm}")


def analyze_eager(path, ks):
    print(f"\n########## {os.path.basename(path)} [EAGER] ##########")
    gemms = [e for e in ks if cat(e["name"]).startswith("gemm")]
    if len(gemms) < 50:
        print(f"too few gemms ({len(gemms)}) for lm_head segmentation")
        return
    durs = sorted(e["dur"] for e in gemms)
    dmax = durs[-1]
    lm = [e for e in gemms if e["dur"] >= 0.5 * dmax]
    print(f"kernels={len(ks)} gemms={len(gemms)} "
          f"lm_head-class (>=0.5*max, max={dmax/1000:.1f}u): {len(lm)}")
    if len(lm) % 5:
        lm = lm[: len(lm) - (len(lm) % 5)]
        print(f"  trimmed to {len(lm)} (multiple of 5; cyclic [V,d1..d4])")
    lm.sort(key=lambda e: e["ts"])
    n = len(lm) // 5
    if n < 5:
        print("TOO FEW steps for attribution")
        return
    V = lm[0::5]
    vgap = sorted(V[i + 1]["ts"] - V[i]["ts"] for i in range(len(V) - 1))
    print(f"steps={n} verify-lm_head gap p50={vgap[len(vgap)//2]/1000:.1f}ms "
          f"p90={vgap[int(len(vgap)*0.9)]/1000:.1f}ms "
          f"(eager step wall; launch-bound)")
    print(f"lm_head dur: verify avg={sum(e['dur'] for e in V)/len(V)/1000:.1f}u "
          f"draft avg={sum(e['dur'] for e in lm[1::5])/max(1,n)/1000:.1f}u "
          f"(BW-bound -> ~equal)")

    # step segment = (V_k, V_{k+1}]: forward tail + sample + verify + draft
    seg = []
    j = 0
    for i in range(n):
        t0 = V[i]["ts"]
        t1 = V[i + 1]["ts"] if i + 1 < len(V) else ks[-1]["ts"]
        while j < len(ks) and ks[j]["ts"] <= t0:
            j += 1
        while j < len(ks) and ks[j]["ts"] < t1:
            seg.append(ks[j])
            j += 1
    a = agg_print("STEP segment (V_k, V_{k+1}] — forward+sample+verify+draft",
                  seg, n)

    # draft segment: around the 4 draft lm_heads (d1-3ms .. d4+3ms wall in
    # eager launch-gap time; captures the draft cluster)
    dseg = []
    j = 0
    for i in range(n):
        d1 = lm[5 * i + 1]["ts"] - 3000
        d4 = lm[5 * i + 4]["ts"] + 3000
        while j < len(ks) and ks[j]["ts"] < d1:
            j += 1
        while j < len(ks) and ks[j]["ts"] < d4:
            dseg.append(ks[j])
            j += 1
    da = agg_print("DRAFT cluster (d1-3ms .. d4+3ms)", dseg, n)

    fcoll = a.get("collective", (0, 0.0))[1] - da.get("collective", (0, 0.0))[1]
    fcnt = a.get("collective", (0, 0.0))[0] - da.get("collective", (0, 0.0))[0]
    print(f"\n== ASKS (content units) ==\n"
          f"  lm_head GEMMs: {sum(e['dur'] for e in lm)/n/1000:.2f} "
          f"ms/step total (draft {sum(e['dur'] for e in lm[1:])/n/1000:.2f} "
          f"+ verify {sum(e['dur'] for e in V)/n/1000:.2f})\n"
          f"  draft collectives: {da.get('collective',(0,0.0))[1]/n/1000:.2f} "
          f"ms/step ({da.get('collective',(0,0.0))[0]/n:.0f} calls/step)\n"
          f"  forward collectives (step-draft): {fcoll/n/1000:.2f} ms/step "
          f"({fcnt/n:.0f} calls/step)")
    byname = defaultdict(lambda: [0, 0.0])
    for e in dseg:
        byname[e["name"][:72]][0] += 1
        byname[e["name"][:72]][1] += e["dur"]
    print("\n== top DRAFT-cluster kernels ==")
    for nm, (cnt, us) in sorted(byname.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {us/n/1000:6.2f} ms/step n/step={cnt/n:4.1f} "
              f"avg={us/cnt/1000:7.1f}u  {nm}")


def analyze(path, mode=None):
    ks = load_kerns(path)
    if not ks:
        print(f"{os.path.basename(path)}: NO KERNEL EVENTS")
        return
    # mode detect: eager has hundreds of gemms/step; graphed ~1/step visible
    gemms = sum(1 for e in ks if cat(e["name"]).startswith("gemm"))
    if mode is None:
        mode = "eager" if gemms > 120 else "graphed"
    span_s = (ks[-1]["ts"] - ks[0]["ts"]) / 1e6
    print(f"\n>>> {os.path.basename(path)}: kernels={len(ks)} gemms={gemms} "
          f"span={span_s:.1f}s -> {mode}")
    if mode == "eager":
        analyze_eager(path, ks)
    else:
        analyze_graphed(path, ks)


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "/root/prof"
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    files = sorted(glob.glob(os.path.join(d, "*.gz"))
                   + glob.glob(os.path.join(d, "*.json")))
    files = [f for f in files if "rank" in os.path.basename(f)]
    if not files:
        print(f"no traces in {d}")
        sys.exit(1)
    print(f"traces: {[os.path.basename(f) for f in files]}")
    for f in files:
        try:
            analyze(f, mode)
        except Exception as exc:  # noqa: BLE001
            print(f"{f}: FAILED {exc!r}")
