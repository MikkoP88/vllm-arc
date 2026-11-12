#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p147_telemetry_report.py - P6.47 A10 attribution over r162 telemetry.

Reads the banked p139_conc JSONs (r162_telemetry_ladder.sh) and emits the
end-to-end attribution table the owner directive asks for: per rung and
concurrency, ITL p50/p90 (decode step time at growing ctx), TTFT (prefill
axis) and per-user tps for q8 (native fp8_e4m3 KV) vs m8 (mxfp8) vs m4
(mxfp4). At conc=1 the non-attention layers are identical between cells,
so itl(m8) - itl(q8) isolates the attention-path per-step delta; divided
by N_ATTN (~16-18 full-attention layers; derived from the ctrl KV pool:
6.8 GiB / 398,363 tok / 1024 B-per-layer-per-rank ~= 18 minus the mamba
pool share) it becomes a per-layer decode cost directly comparable to the
A9 kernel timings. TTFT deltas carry the prefill/store axis.

Usage: p147_telemetry_report.py <bank_dir> [n_attn]
Stdlib only. Writes markdown to stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CELLS = ("q8", "m8", "m4")
RUNGS = (1024, 8192, 32768)
CONCS = (1, 8)
N_ATTN_DEFAULT = 16


def load_cell(cell_dir: Path, ctx: int, nc: int):
    fn = cell_dir / f"p139_conc_{cell_dir.name}_c{ctx}_n{nc}.json"
    if not fn.exists():
        return None
    try:
        with open(fn) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def itl_stats(j):
    rs = [r for r in j.get("requests", []) if r.get("error") is None]
    if not rs:
        return None
    def mean(k):
        vals = [r[k] for r in rs if r.get(k, 0) > 0]
        return sum(vals) / len(vals) if vals else -1.0
    return mean("itl_ms_p50"), mean("itl_ms_p90")


def main():
    bank = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    n_attn = int(sys.argv[2]) if len(sys.argv) > 2 else N_ATTN_DEFAULT
    print(f"# A10 telemetry attribution - {bank.name} (n_attn={n_attn})")
    print()
    print("| rung | conc | q8 itl p50/p90 | m8 itl p50/p90 | m4 itl p50/p90"
          " | q8 ttft | m8 ttft | m4 ttft | q8 tps | m8 tps | m4 tps |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = {}
    for ctx in RUNGS:
        for nc in CONCS:
            cells = {}
            for c in CELLS:
                j = load_cell(bank / c, ctx, nc)
                cells[c] = j
                if j is None:
                    continue
                rows[(ctx, nc, c)] = j
            def fmt(c, cells=cells):
                j = cells.get(c)
                if j is None:
                    return "-"
                it = itl_stats(j)
                it_s = f"{it[0]:.1f}/{it[1]:.1f}" if it else "-"
                return it_s
            def ttft(c, cells=cells):
                j = cells.get(c)
                return f"{j['ttft_mean_s']:.2f}" if j else "-"
            def tps(c, cells=cells):
                j = cells.get(c)
                return f"{j.get('per_user_mean_tps', -1):.1f}" if j else "-"
            print(f"| {ctx} | {nc} | {fmt('q8')} | {fmt('m8')} | {fmt('m4')}"
                  f" | {ttft('q8')} | {ttft('m8')} | {ttft('m4')}"
                  f" | {tps('q8')} | {tps('m8')} | {tps('m4')} |")
    print()
    print("## Attribution (conc=1: attn-path delta per step, per layer)")
    print()
    print("| rung | m8 dITL ms | m8 per-layer ms | m4 dITL ms | m4 per-layer ms"
          " | m8 dTTFT s | m4 dTTFT s |")
    print("|---|---|---|---|---|---|---|")
    for ctx in RUNGS:
        q = rows.get((ctx, 1, "q8"))
        m8 = rows.get((ctx, 1, "m8"))
        m4 = rows.get((ctx, 1, "m4"))
        if not (q and m8):
            continue
        qit = itl_stats(q)
        m8it = itl_stats(m8)
        m4it = itl_stats(m4) if m4 else None
        d8 = m8it[0] - qit[0] if (qit and m8it) else None
        d4 = m4it[0] - qit[0] if (qit and m4it) else None
        dt8 = m8["ttft_mean_s"] - q["ttft_mean_s"]
        cells = [f"| {ctx} "]
        cells.append(f"| {d8:.1f} | {d8 / n_attn:.2f} "
                     if d8 is not None else "| - | - ")
        cells.append(f"| {d4:.1f} | {d4 / n_attn:.2f} "
                     if d4 is not None else "| - | - ")
        cells.append(f"| {dt8:.2f} |")
        if m4:
            cells.append(f" {m4['ttft_mean_s'] - q['ttft_mean_s']:.2f} |")
        else:
            cells.append(" - |")
        print("".join(cells))
    print()
    print("Reading: per-layer ms compares directly against the A9 kernel")
    print("timings at the same shape (D=256, HK=4, N_REP=6). A per-layer")
    print("delta far above the kernel ms means the cost is NOT in the")
    print("decode kernel - look to launches, engine casts, store path, or")
    print("the spec/draft loop before touching the kernel again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
