#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""p181_acceptknee.py — P6.47 S1: the e2m1-V acceptance ctx-knee map report.

S1 lever (plan Appendix A15 / r161 evidence): on the e2m1-V lanes
(mxfp4, mxk8v4) MTP verify acceptance died to exactly 0.0% in the 32K
longprobe windows of cert25y while short-ctx held 50-80% — the m4/mkv4
e2e gap driver (dead acceptance = 4-way verify + draft cost per single
token). r168 (LMC auto-off) shipped AFTER that evidence, so the CURRENT
curve on cert25zb must be re-established before any adaptive ARC_SPEC
off-switch is designed: this report reads the r185 bank
(<bank>/<cell>/c{ctx}_n{nc}/ with metrics_before.txt + metrics_after.txt
bracketing each p139_conc group) and emits the acceptance-vs-ctx curve
per dtype plus the knee verdict.

Definitions (arc_spec_gate.py, the certified signature):
  acceptance%   = accepted/draft_tokens
  mean len      = accepted/drafts + 1  (the +1 bonus token)
  eff ms/token  = itl_p50 / mean len

ACCEPT_KNEE_PASS = every expected cell carries a valid window
(drafts > 0 in the delta). The knee itself is REPORTED, never gated:
KNEE = smallest ctx where acceptance < 15% (either conc); "none" when
the curve stays alive through 32K — the adaptive off-switch question
then moves to rungs beyond this ladder.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROM_BASE = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted": "vllm:spec_decode_num_accepted_tokens",
}

LINE_RE = re.compile(
    r'^(vllm:spec_decode_[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$'
)

CELLS = ("m4q", "mkq", "m8q")
CTXS = (4096, 8192, 16384, 32768)
CONCS = (1, 8)
KNEE_PCT = 15.0


def parse_metrics(path: Path) -> dict[str, float]:
    """Sum /metrics counter families by name (label sets collapse)."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.startswith("vllm:spec_decode_"):
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(3))
        if name.endswith("_created"):
            continue  # prometheus _created gauge (epoch ts), not a counter
        if name.endswith("_total"):
            name = name[: -len("_total")]  # exposition suffix on counters
        out[name] = out.get(name, 0.0) + val
    return {k: out.get(v, 0.0) for k, v in PROM_BASE.items()}


def window(cell_dir: Path) -> dict | None:
    """Counter deltas for one bracketed cell, or None when absent."""
    before = parse_metrics(cell_dir / "metrics_before.txt")
    after = parse_metrics(cell_dir / "metrics_after.txt")
    if not before or not after:
        return None
    d = {k: after[k] - before[k] for k in PROM_BASE}
    if d["drafts"] <= 0 or d["draft_tokens"] <= 0:
        return None
    d["accept_pct"] = 100.0 * d["accepted"] / d["draft_tokens"]
    d["mean_len"] = d["accepted"] / d["drafts"] + 1.0
    return d


def itl_p50(cell_dir: Path, ctx: int, nc: int) -> float | None:
    fn = cell_dir / f"p139_conc_c{ctx}_n{nc}.json"
    if not fn.exists():
        return None
    try:
        reqs = json.loads(fn.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(reqs, dict):
        reqs = reqs.get("requests", [])
    vals = sorted(r.get("itl_ms_p50") for r in reqs
                  if isinstance(r, dict) and r.get("itl_ms_p50"))
    if not vals:
        return None
    return vals[len(vals) // 2]


def steady_state(cell_dir: Path, ctx: int, nc: int) -> float | None:
    """Median per-slot steady-state inter-chunk gap in ms.

    Ramp-interleaved ITL law (r185 forensics, 2026-11-14): aggregate p50s
    on ramp-interleaved cells mix prefill-stall chunk gaps into decode.
    The steady decode rate is (total_s - ttft_s)/(n_stream_chunks - 1)
    per slot; the median across slots is the honest column next to the
    aggregate p50 (equals it exactly on n1, where there is no ramp).
    """
    fn = cell_dir / f"p139_conc_c{ctx}_n{nc}.json"
    if not fn.exists():
        return None
    try:
        reqs = json.loads(fn.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(reqs, dict):
        reqs = reqs.get("requests", [])
    vals = []
    for r in reqs:
        if not isinstance(r, dict) or r.get("error"):
            continue
        ch = r.get("n_stream_chunks", 0)
        if not ch or ch < 2 or not r.get("total_s") or not r.get("ttft_s"):
            continue
        vals.append((r["total_s"] - r["ttft_s"]) / (ch - 1) * 1e3)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def main() -> int:
    bank = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/out")
    cells = tuple((sys.argv[2] if len(sys.argv) > 2 else ",".join(CELLS)).split(","))
    if not bank.is_dir():
        print(f"P181_NO_BANK {bank}")
        return 2
    ok = True
    knees: dict[str, str] = {}
    print(f"p181_acceptknee — bank={bank} (acceptance% / mean-len / "
          f"itl p50 / eff ms-per-token)")
    for cell in cells:
        cdir = bank / cell
        if not cdir.is_dir():
            print(f"[{cell}] MISSING")
            ok = False
            continue
        knee = "none"
        print(f"[{cell}]")
        print("| ctx | n1 acc% / len / itl / eff / ss |"
              " n8 acc% / len / itl / eff / ss |")
        print("|---|---|---|")
        for ctx in CTXS:
            row = [f"| {ctx} "]
            for nc in CONCS:
                g = cdir / f"c{ctx}_n{nc}"
                w = window(g)
                if w is None:
                    row.append("| — (no window) ")
                    ok = False
                    continue
                itl = itl_p50(g, ctx, nc)
                ss = steady_state(g, ctx, nc)
                eff = f"{itl / w['mean_len']:.1f}" if itl else "—"
                accs = f"{w['accept_pct']:.1f}%"
                itls = f"{itl:.0f}" if itl else "—"
                sss = f"{ss:.0f}" if ss is not None else "—"
                row.append(
                    f"| {accs} / {w['mean_len']:.2f} / {itls} / {eff} / {sss} "
                )
                if w["accept_pct"] < KNEE_PCT and knee == "none":
                    knee = f"{ctx}@n{nc}"
            print("".join(row) + "|")
        knees[cell] = knee
        print(f"[{cell}] KNEE (first acc < {KNEE_PCT:.0f}%): {knee}")
    print("--- knees ---")
    for cell in cells:
        print(f"[{cell}] {knees.get(cell, '?')}")
    print(f"ACCEPT_KNEE_{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
