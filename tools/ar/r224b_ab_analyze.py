#!/usr/bin/env python3
"""r224b_ab_analyze.py — in-container analyzer for the r224a A/B payloads.

Loads every mxkv_ab.*.pt written by the ARC_MXKV_AB instrument in
vllm_gaudi/xpu/patches/mx_kv/engine.py (run inside the arc container,
whose venv has torch — the workstation python does not) and answers,
per capture and in aggregate:

  1. POOL LAYOUT — the real kv_cache view metadata (dim / shape / stride
     / storage_offset / dtype), whether the strides are dense-contiguous
     for that shape, and if not, whether they are a pure axis permutation
     of dense (the classic fused-pool K/V split-view hazard) or a genuinely
     strided view. The r220c offline value gate built a dense synthetic
     pool and was structurally blind to all of this.
  2. DIVERGENCE — native vs certified-Triton decode on identical inputs:
     dmax / dmean / exact-match fraction / tolerance exceed counts / NaN
     counts.
  3. LOCALIZATION — per-row max divergence paired with that row's seq_len
     and block-table head (growth with sl points at a walk/extent defect;
     row-independent structure points at a layout defect), and per-head
     (D=128 for qwen3.8-27b) max divergence.
  4. CONTEXT BUCKETS — captures bucketed by banked seq_lens (SHORT <100,
     MID, LONG >=1000): the cert26 refusal was long-return-class only, so
     the decisive fact is whether divergence appears/magnifies in LONG.

Verdict: R224B_VERDICT MATCH (native reproduced certified Triton values
on the engine's own pool view — root cause moves off the kernel) or
DIVERGENT (with bucket + layout attribution), printed as the last line.
Diagnostic tool, not a gate: exit 0 on MATCH or DIVERGENT, 3 when no
payloads were found.

Usage: r224b_ab_analyze.py [payload_dir]   (default /tmp)
"""

import glob
import itertools
import os
import sys

import torch

HEAD_DIM = 128  # legacy 2D fallback; qwen3.8-27b attn D=256 (scale 1/sqrt(256))
TOL_MATCH = 2e-2  # verdict threshold; raw numbers always printed regardless


def dense_stride(shape) -> tuple:
    st = []
    acc = 1
    for d in reversed(shape):
        st.append(acc)
        acc *= d
    return tuple(reversed(st))


def classify_layout(shape, stride) -> str:
    """dense / permuted-dense / strided attribution for a view."""
    if tuple(stride) == dense_stride(shape):
        return "dense-contiguous"
    if sorted(stride) == sorted(dense_stride(shape)):
        for perm in itertools.permutations(range(len(shape))):
            pshape = tuple(shape[i] for i in perm)
            if tuple(stride) == dense_stride(pshape):
                return f"permuted-dense(memory axes follow view axes {perm})"
    return "strided-non-dense (gaps/dilation)"


def bucket_of(mean_sl: int) -> str:
    if mean_sl < 100:
        return "SHORT"
    if mean_sl < 1000:
        return "MID"
    return "LONG"


def head_profile(out_native, out_triton):
    """max |native - triton| per head, or None if not decomposable."""
    d = (out_native.float() - out_triton.float()).abs()
    if d.dim() == 3:  # (rows, H, D) — the r224a engine dispatch shape
        return d.amax(dim=(0, 2)).tolist()
    w = d.shape[-1]
    if d.dim() == 2 and w % HEAD_DIM == 0:
        per = d.view(d.shape[0], w // HEAD_DIM, HEAD_DIM).amax(dim=(0, 2))
        return per.tolist()
    return None


def analyze(path: str) -> dict:
    p = torch.load(path, map_location="cpu", weights_only=False)
    on, ot = p["out_native"], p["out_triton"]
    d8 = (on.float() - ot.float()).abs()
    rows = min(8, on.shape[0])
    per_row = []
    for i in range(rows):
        per_row.append((int(p["sl"][i]), float(d8[i].max())))
    exact = float((d8 == 0).float().mean())
    t1 = float((d8 > 1e-3).float().mean())
    t2 = float((d8 > 1e-2).float().mean())
    sl_mean = int(p["sl"][:rows].float().mean())
    return {
        "name": os.path.basename(path),
        "fmt": p["fmt"], "scale": p["scale"], "rows": p["rows"],
        "kv_dim": p["kv_dim"], "kv_shape": tuple(p["kv_shape"]),
        "kv_stride": tuple(p["kv_stride"]), "kv_off": p["kv_off"],
        "kv_dtype": p["kv_dtype"],
        "layout": classify_layout(p["kv_shape"], p["kv_stride"]),
        "dmax": float(p["dmax"]), "dmean": float(p["dmean"]),
        "dmax8": float(d8.max()), "exact": exact, "t1": t1, "t2": t2,
        "nan_native": int(p["nan_native"]), "nan_triton": int(p["nan_triton"]),
        "sl_mean": sl_mean, "bucket": bucket_of(sl_mean),
        "per_row": per_row, "heads": head_profile(on[:rows], ot[:rows]),
        "bt0": p["bt"][:rows, :4].tolist(),
    }


def main() -> int:
    pd = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    paths = sorted(glob.glob(os.path.join(pd, "mxkv_ab.*.pt")))
    if not paths:
        print(f"R224B_VERDICT: NO_PAYLOADS (no mxkv_ab.*.pt under {pd})")
        return 3
    caps = []
    for path in paths:
        try:
            caps.append(analyze(path))
        except Exception as exc:  # noqa: BLE001 — report, keep analyzing
            print(f"!! {os.path.basename(path)}: load/analyze failed: {exc!r}")
    if not caps:
        print("R224B_VERDICT: NO_PAYLOADS (all loads failed)")
        return 3

    layouts = {}
    for c in caps:
        layouts.setdefault((c["kv_shape"], c["kv_stride"], c["kv_off"]),
                           c["layout"])
    print(f"== pool views seen ({len(layouts)} distinct):")
    for (shape, stride, off), lay in sorted(layouts.items()):
        print(f"   shape={shape} stride={stride} off={off} -> {lay}")

    buckets = {}
    for c in caps:
        buckets.setdefault(c["bucket"], []).append(c)

    for c in caps:
        pr = " ".join(f"r{i}(sl={sl},dmax={dm:.2e})"
                      for i, (sl, dm) in enumerate(c["per_row"]))
        heads = c["heads"]
        hs = (" ".join(f"h{i}={v:.2e}" for i, v in enumerate(heads))
              if heads else "n/a (not head-divisible)")
        print(f"== {c['name']} fmt={c['fmt']} rows={c['rows']} "
              f"bucket={c['bucket']} sl_mean={c['sl_mean']}")
        print(f"   kv dim={c['kv_dim']} dtype={c['kv_dtype']} "
              f"layout={c['layout']} off={c['kv_off']}")
        print(f"   dmax={c['dmax']:.3e} dmax8={c['dmax8']:.3e} "
              f"dmean={c['dmean']:.3e} exact={c['exact']:.4f} "
              f">1e-3={c['t1']:.4f} >1e-2={c['t2']:.4f} "
              f"nan={c['nan_native']}/{c['nan_triton']}")
        print(f"   rows: {pr}")
        print(f"   heads: {hs}")
        print(f"   bt0: {c['bt0']}")

    print("== bucket rollup:")
    for b in ("SHORT", "MID", "LONG"):
        if b not in buckets:
            print(f"   {b}: 0 captures")
            continue
        g = buckets[b]
        worst = max(g, key=lambda c: c["dmax"])
        ndiv = sum(1 for c in g
                   if c["dmax"] > TOL_MATCH or c["nan_native"] > 0)
        print(f"   {b}: {len(g)} captures, divergent={ndiv}, "
              f"max_dmax={worst['dmax']:.3e} ({worst['name']}), "
              f"sl_mean range "
              f"{min(c['sl_mean'] for c in g)}-{max(c['sl_mean'] for c in g)}")

    ndiv = sum(1 for c in caps if c["dmax"] > TOL_MATCH or c["nan_native"] > 0)
    any_nan = sum(c["nan_native"] for c in caps)
    pools_dense = all(v == "dense-contiguous" for v in layouts.values())
    bs = " ".join(f"{b}={sum(1 for c in buckets.get(b, []) if c['dmax'] > TOL_MATCH)}"
                  for b in ("SHORT", "MID", "LONG"))
    if ndiv == 0:
        print(f"R224B_VERDICT: MATCH captures={len(caps)} "
              f"max_dmax={max(c['dmax'] for c in caps):.3e} "
              f"nan={any_nan} pools_dense={pools_dense} — native reproduces "
              f"certified Triton on the engine's own pool view")
        return 0
    print(f"R224B_VERDICT: DIVERGENT captures={len(caps)} divergent={ndiv} "
          f"max_dmax={max(c['dmax'] for c in caps):.3e} nan={any_nan} "
          f"pools_dense={pools_dense} divergent_by_bucket: {bs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
