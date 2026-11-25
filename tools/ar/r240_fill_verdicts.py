#!/usr/bin/env python3
"""r240b/r240c VERDICT fill extraction (local, post-pull audit 2026-11-25).

Prints every value the two VERDICT drafts need, recomputed from the
banked face JSONs (gate-script law: recompute from banked artifacts,
never from gate lines). Run from the repository root:

  python tools/ar/r240_fill_verdicts.py
"""
import json
import os

B240B = "results/2026-11-24-p647-r240b-conc"
B240C = "results/2026-11-24-p647-r240c-gemma128k"


def load(bank, cell, ctx, n, temp="0.0"):
    suffix = "_t{}".format(temp) if temp != "0.0" else ""
    p = os.path.join(bank, cell,
                     "p139_conc_{}_c{}_n{}{}.json".format(cell, ctx, n, suffix))
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf8") as f:
        return json.load(f)


def ok(d):
    return d is not None and d.get("n_fail", 1) == 0 and d.get("n_ok", 0) > 0


def row(d):
    if not ok(d):
        return "DEAD"
    return ("n_ok={} ttft={:.1f}s puser={:.1f} agg={:.0f} "
            "min={:.1f} itl_p50={:.0f}ms".format(
                d["n_ok"], d["ttft_mean_s"], d["per_user_mean_tps"],
                d["agg_tps_wall"], d["per_user_min_tps"],
                d.get("itl_ms_p50_max") or -1))


def pool(bank, cell):
    p = os.path.join(bank, cell, "posture.txt")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf8") as f:
        for line in f:
            if line.startswith("kv_pool_tokens="):
                return int(line.split("=", 1)[1])
    return None


print("== r240b pools (kv_pool_tokens per posture.txt)")
for c in ["qf3", "qf0", "q83", "q80", "q43", "q40",
          "tq_k8v4", "tq_4bit", "tq_k3v4", "tq_3bit",
          "kvm4", "kvm8", "kvmk", "gf8", "gm8", "gm4",
          "gkv_m4", "gkv_m8", "gkv_mk"]:
    print("  {:8s} {:>8s}".format(c, "{:,}".format(pool(B240B, c) or 0)))

print("== r240c pools")
for c in ["gm4x", "gm8x", "gkv_m4x", "gkv_mkx",
          "gf8x64", "gm8x64", "gkv_m8x64", "gkv_m8x131"]:
    print("  {:10s} {:>8s}".format(c, "{:,}".format(pool(B240C, c) or 0)))

print("== r240b: 3x2 quant x spec matrix, 8K n1")
for c in ["qf3", "qf0", "q83", "q80", "q43", "q40"]:
    d = load(B240B, c, 8192, 1)
    print("  {:4s} {}".format(c, row(d)))

print("== r240b: 3x2 matrix, 8K n8 (agg)")
for c in ["qf3", "qf0", "q83", "q80", "q43", "q40"]:
    d = load(B240B, c, 8192, 8)
    print("  {:4s} {}".format(c, row(d)))

print("== r240b: MX-native kvface grids, per lane per ctx per n")
for c in ["kvm4", "kvm8", "kvmk"]:
    for cv in [1024, 8192, 32768, 65536, 131072]:
        for n in [1, 2, 4, 8]:
            d = load(B240B, c, cv, n)
            if d is not None:
                print("  {:4s} c{:6d} n{} {}".format(c, cv, n, row(d)))

print("== r240b: gemma lanes n1")
for c in ["gf8", "gm8", "gm4", "gkv_m4", "gkv_m8", "gkv_mk"]:
    for cv in [1024, 8192, 32768]:
        d = load(B240B, c, cv, 1)
        if d is not None:
            print("  {:7s} c{:6d} {}".format(c, cv, row(d)))

print("== r240b: gf8 full ladder (agg @ n)")
for cv in [1024, 8192, 32768]:
    line = []
    for n in [1, 2, 3, 4, 5, 6, 7, 8]:
        d = load(B240B, "gf8", cv, n)
        line.append("n{}={}".format(n, "{:.0f}".format(d["agg_tps_wall"])
                                    if ok(d) else "-"))
    print("  gf8 c{:6d} {}".format(cv, " ".join(line)))

print("== r240b: TQ presets 8K/32K n1 + n8")
for c in ["tq_k8v4", "tq_4bit", "tq_k3v4", "tq_3bit"]:
    for cv in [8192, 32768]:
        for n in [1, 8]:
            d = load(B240B, c, cv, n)
            print("  {:8s} c{:6d} n{} {}".format(c, cv, n, row(d)))

print("== r240c: gm4x ladder (W4A16 @ maxlen 131072)")
for cv in [1024, 8192, 32768, 65536, 131072]:
    for n in [1, 2, 3, 4, 8]:
        d = load(B240C, "gm4x", cv, n)
        if d is not None:
            print("  gm4x c{:6d} n{} {}".format(cv, n, row(d)))
for cv in [1024, 8192, 32768, 65536, 131072]:
    d = load(B240C, "gm4x", cv, 1, "1.0")
    if d is not None:
        print("  gm4x c{:6d} t1.0 {}".format(cv, row(d)))

print("== r240c: 64K closure cells")
for c in ["gf8x64", "gm8x64", "gkv_m8x64"]:
    for n in [1, 2, 4, 8]:
        d = load(B240C, c, 65536, n)
        if d is not None:
            print("  {:9s} n{} {}".format(c, n, row(d)))
    d1 = load(B240C, c, 65536, 1, "1.0")
    if d1 is not None:
        print("  {:9s} t1.0 {}".format(c, row(d1)))

print("== r240c: gm4x KV_CAP evidence (cell conc.log POOL/NEED lines)")
p = os.path.join(B240C, "gm4x", "conc.log")
if os.path.isfile(p):
    with open(p, encoding="utf8", errors="replace") as f:
        for line in f:
            if "KV_CAP" in line or "POOL" in line or "NEED" in line:
                print("  " + line.rstrip()[:160])
