#!/usr/bin/env python3
"""r225l_splitprobe.py — split-combine native::exp2 probe (r225 static
finding): mx_reduce_split.hpp:359 rescales every KV-split partial with
sycl::native::exp2 — the ~1e-3-relative instruction r198 evicted from
the mainloop — and chunk_prefill_epilogue.hpp:185/304 carry the same
class inside reduce_A (live whenever ReduceK > 1, i.e. even at ONE KV
split).

Regime law: in a PEAKED softmax one split/subgroup dominates
(local_max - global_max = 0, and native::exp2(0) == 1 exactly) and the
rest rescale to ~0, so every synthetic sweep was bitwise clean — the
REAL payloads sit in a FLAT regime (Ks e8m0 codes 119-122 make logits
small), where every rescale factor is nontrivial and the 2^-10-class
approximation error lands on the output: the residual's err/|O| p50 =
2^-9.4 signature.

Discriminator: replay at num_splits_kv in {auto, 1, 2, 8} on the real
payloads.  n1 clean + auto wrong pins the combine (mx_reduce_split:359);
n1 wrong too pins reduce_A (epilogue 185/304) as a second live site.
Flat synthetics (Ks pinned to 119) must turn erroneous at auto —
closing the r225i coverage gap (its [120,134] codes were peaked-only) —
with a pinned-134 peaked control.

Usage: r225l_splitprobe.py <payload.pt> [payload2.pt ...]
"""

import sys

import torch

sys.path.insert(0, "/tmp")
from r225j_iso import dmax, grid_stats, model_row, synth_window  # noqa: E402

from vllm_gaudi.xpu.patches.mx_kv.mx_layout import offsets as _mx_off  # noqa: E402
from vllm_gaudi.xpu.patches.mx_kv.mx_quant import decode_kv  # noqa: E402

BLOCK = 64
D = 256
BLOCK_STRIDE = 51200
ENGINE_BT_WIDTH = 4114
SPLITS = (0, 1, 2, 8)


class ReplayNS:
    """Engine-geometry GPU replay on an arbitrary window, with the KV
    split count forced (0 = the engine's auto heuristic)."""

    def __init__(self, template, dev):
        self.shape = tuple(template["kv_shape"])
        self.stride = tuple(template["kv_stride"])
        self.off_i = int(template["kv_off"])
        self.bt_dtype = template["bt"].dtype
        self.dev = dev

    def __call__(self, q_rows, win, sl_vals, scale, nsplit):
        from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode
        nb = win.shape[0]
        r = len(sl_vals)
        storage = torch.zeros(self.off_i + self.shape[0] * BLOCK_STRIDE,
                              dtype=torch.uint8, device=self.dev)
        for b in range(nb):
            storage[self.off_i + b * BLOCK_STRIDE:
                    self.off_i + (b + 1) * BLOCK_STRIDE] = \
                win[b].reshape(-1).to(self.dev)
        pool = torch.as_strided(storage, self.shape, self.stride,
                                self.off_i)
        bt = torch.zeros(r, ENGINE_BT_WIDTH, dtype=self.bt_dtype,
                         device=self.dev)
        bt[:, :nb] = torch.arange(nb, dtype=self.bt_dtype, device=self.dev)
        sl32 = torch.tensor(sl_vals, dtype=torch.int32, device=self.dev)
        cu_q = torch.arange(r + 1, dtype=torch.int32, device=self.dev)
        out = mx_paged_decode(
            q_rows.to(self.dev), pool, "mxk8v4", cu_q, sl32, bt,
            max_seqlen_k=ENGINE_BT_WIDTH * 64,
            softmax_scale=scale, num_splits_kv=nsplit,
        )[:r].cpu()
        del pool, storage
        torch.xpu.empty_cache()
        return out


def synth_pinned(nb, seed, off, code):
    """synth_window with the Ks surface pinned to ONE e8m0 code."""
    win = synth_window(nb, seed, off)
    win[..., off["ks"]:off["ks"] + 8] = code
    return win


def flatness(q_row, K, sl_val, scale):
    """p50/p90 of the per-head max-p (softmax concentration)."""
    nb_i = (sl_val + BLOCK - 1) // BLOCK
    last = sl_val - (nb_i - 1) * BLOCK
    k_seq = torch.cat([K[:nb_i - 1].reshape(-1, 2, D), K[nb_i - 1][:last]],
                      dim=0)
    maxes = []
    for h in range(q_row.shape[0]):
        s = (q_row[h] @ k_seq[:, h // 6].T) * scale
        p = torch.softmax(s.double(), dim=-1)
        maxes.append(float(p.max()))
    t = torch.tensor(maxes)
    return float(t.median()), float(t.quantile(0.9))


def score(n_out, model, label):
    mism, mul = grid_stats(n_out, model)
    print(f"   {label:<14}: dmax={dmax(n_out, model):.3e}"
          f"  mism={mism}/{12 * D}  max_ulp={mul:.2f}")
    return mism, mul


def main() -> int:
    dev = torch.device("xpu")
    off = _mx_off(D, "mxk8v4")
    first = True

    for path in sys.argv[1:]:
        p = torch.load(path, map_location="cpu", weights_only=False)
        r = p["q"].shape[0]
        sl = p["sl"][:r].to(torch.long).tolist()
        win = p["kv_win"].to(torch.uint8)
        scale = float(p["scale"])
        q = p["q"].to(torch.bfloat16)
        rep = ReplayNS(p, dev)
        K, V = decode_kv(win[..., : off["v"]], win[..., off["v"]:],
                         "mxk8v4", D)
        K, V = K.double(), V.double()
        f50, f90 = flatness(q[0].double(), K, sl[0], scale)
        name = path.split("/")[-1]
        print(f"== {name}  rows={r} sl={sl}  flatness(max-p "
              f"p50={f50:.4f} p90={f90:.4f})")

        models = [model_row(q[i].double(), K, V, sl[i], scale)
                  for i in range(r)]
        for ns in SPLITS:
            lab = "auto" if ns == 0 else f"nsplit{ns}"
            n_out = rep(q, win, sl, scale, ns)
            worst = (0, 0.0)
            for i in range(r):
                mism, mul = grid_stats(n_out[i], models[i])
                worst = (max(worst[0], mism), max(worst[1], mul))
            dm = max(dmax(n_out[i], models[i]) for i in range(r))
            print(f"   {lab:<14}: dmax={dm:.3e}"
                  f"  mism={worst[0]}/{12 * D}  max_ulp={worst[1]:.2f}")

        if first:
            # auto-S identification: the kernel is deterministic per S,
            # so a forced-S replay that BITWISE-matches the auto output
            # pins the heuristic's choice (coverage math prints whether
            # that S leaves EMPTY splits -- S*nbps exceeding the row's
            # kv-block count -- the r225 fourth-source suspect).
            auto_out = rep(q, win, sl, scale, 0)
            wkb = [(s + BLOCK - 1) // BLOCK for s in sl]
            print(f"   -- auto-S sweep (wkb blocks={wkb})")
            for s_try in (2, 4, 8, 12, 16, 24, 32, 43, 48, 64, 69, 86,
                          96, 128):
                try:
                    cand = rep(q, win, sl, scale, s_try)
                except Exception as exc:  # noqa: BLE001
                    print(f"   S={s_try:3d}: replay failed "
                          f"{type(exc).__name__}: {exc}")
                    continue
                eq = all(torch.equal(auto_out[i], cand[i])
                         for i in range(r))
                nbps = -(-wkb[0] // s_try)
                cover = -(-wkb[0] // nbps)
                dm = max(dmax(auto_out[i], cand[i]) for i in range(r))
                print(f"   S={s_try:3d}: bitwise={eq} dmax={dm:.1e} "
                      f"nbps={nbps} empty={s_try - cover}")

            # flat synthetic (Ks pinned 119) vs peaked control (134)
            nb = win.shape[0]
            for code in (119, 134):
                w = synth_pinned(nb, 11, off, code)
                hK, hV = decode_kv(w[..., : off["v"]], w[..., off["v"]:],
                                   "mxk8v4", D)
                model = model_row(q[0].double(), hK.double(), hV.double(),
                                  sl[0], scale)
                sf50, sf90 = flatness(q[0].double(), hK.double(), sl[0],
                                      scale)
                print(f"   -- synth Ks={code} (max-p p50={sf50:.4f}"
                      f" p90={sf90:.4f})")
                for ns in (0, 1):
                    lab = "auto" if ns == 0 else "nsplit1"
                    n_out = rep(q[0:1], w, [sl[0]], scale, ns)[0]
                    score(n_out, model, f"synth{code}-{lab}")
            first = False

    print("R225L_SPLITPROBE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
