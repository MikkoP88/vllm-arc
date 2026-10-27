#!/usr/bin/env python3
"""arc_spec_gate.py — p31 spec-acceptance gate (plan §10 P4 row; superiority
contract acceptance >= 0.85; hard gate pos1 >= 0.5).

Telemetry is upstream-0.30 native (vllm/v1/spec_decode/metrics.py, verified
in image vllm-arc:v0.30.0-b70-p12):

  /metrics counters
    vllm:spec_decode_num_drafts_total              (draft steps)
    vllm:spec_decode_num_draft_tokens_total
    vllm:spec_decode_num_accepted_tokens_total
    vllm:spec_decode_num_accepted_tokens_per_pos_total{...,position="i"}
  engine log window (cross-check, D-law lineage — rig p2r3_leg_v1.sh:387)
    "SpecDecoding metrics: ... Per-position acceptance rate: p1, p2, ...,
     Avg Draft acceptance rate: X%"

Definitions (D certified signature: pos1 0.914-0.954, avg draft 86-87.5%):
  pos1        acceptance of the FIRST drafted token = per_pos[0]/drafts
              (the first scalar of the log's per-position vector). Hard
              gate >= 0.5: below is the wrong-logits/mojibake family (D r1
              signature 0.000x4; coherent output was never observed under
              0.5 — probe B vs probe D, no middle state).
  acceptance  accepted/draft tokens overall — the contract target >= 0.85
              (champion 0.871 gated).

Modes:
  default          fire a fixed light load (Run-7n prompt parity), gate on
                   counter deltas
  --expect-off     assert spec counters ABSENT — non-spec leg posture purity
  --strict-target  exit 4 when overall acceptance < target (certification)

Runs on the rig host (localhost metrics + docker exec for the log probe).

Exit codes: 0 pass · 2 pos1 hard-gate miss · 3 telemetry absent on a spec
leg (D fail-law: absence = telemetry break, not silence-by-design) ·
4 target miss (--strict-target).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request

PROM_BASE = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted": "vllm:spec_decode_num_accepted_tokens",
    "per_pos": "vllm:spec_decode_num_accepted_tokens_per_pos",
}


def fetch_metrics(url: str) -> dict:
    """Sum /metrics counters by metric name (and position for per_pos).

    Sums across label sets (TP engines expose one APIServer endpoint; label
    variants still collapse to the same totals).
    """
    out: dict[str, float] = {}
    txt = urllib.request.urlopen(url + "/metrics", timeout=30).read().decode()
    for line in txt.splitlines():
        if not line.startswith("vllm:spec_decode_"):
            continue
        m = re.match(r'^(vllm:spec_decode_[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$',
                     line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        # prometheus_client appends _total to Counter exposition names —
        # strip it so the base names above match.
        if name.endswith("_total"):
            name = name[: -len("_total")]
        if name == PROM_BASE["per_pos"]:
            pm = re.search(r'position="(\d+)"', labels)
            if not pm:
                continue
            key = f"per_pos_{int(pm.group(1))}"
        else:
            key = next((k for k, v in PROM_BASE.items() if v == name), None)
            if key is None:
                continue
        out[key] = out.get(key, 0.0) + val
    return out


def fire_load(url: str, model: str, nstreams: int, mt: int, out: list) -> None:
    """Run-7n parity prompt/params: temp 0, thinking off, non-streaming."""
    def call() -> None:
        body = {"model": model, "max_tokens": mt, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user",
                              "content": "Count from 1 to 1000."}]}
        req = urllib.request.Request(
            url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=600).read()
            out.append(True)
        except Exception as e:  # noqa: BLE001
            print(f"SPEC_GATE call_FAIL {type(e).__name__}: {e}", flush=True)
            out.append(False)

    ths = [threading.Thread(target=call) for _ in range(nstreams)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()


def log_window(container: str, log: str) -> str:
    """Last SpecDecoding window from the engine log (D-law cross-check)."""
    try:
        r = subprocess.run(
            ["docker", "exec", container, "sh", "-c",
             f"grep -a 'Per-position acceptance rate' {log} | tail -1"],
            capture_output=True, text=True, timeout=30, check=True)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001 — cross-check is best-effort
        return f"(log probe unavailable: {e})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8-27b-fp8")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--hard", type=float, default=0.5)
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--expect-off", action="store_true",
                    help="non-spec leg: counters must be ABSENT")
    ap.add_argument("--strict-target", action="store_true",
                    help="exit 4 when overall acceptance < target")
    ap.add_argument("--container", default="arc-test")
    ap.add_argument("--log", default="/root/serve_full.log")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    m0 = fetch_metrics(args.url)
    has_spec = any(k.startswith("per_pos_") or k in
                   ("drafts", "draft_tokens", "accepted") for k in m0)

    if args.expect_off:
        if has_spec:
            print(f"SPEC_GATE FAIL: spec counters present on a non-spec leg "
                  f"(posture leak): {sorted(m0)}")
            return 2
        print("SPEC_GATE expect-off PASS: no spec counters (non-spec posture "
              "pure)")
        return 0

    if not has_spec:
        print("SPEC_GATE FAIL: no spec counters on a spec leg — telemetry "
              "absent (D fail-law: absence = telemetry break, not "
              "silence-by-design)")
        return 3

    ok_flags: list[bool] = []
    t0 = time.perf_counter()
    fire_load(args.url, args.model, args.streams, args.max_tokens, ok_flags)
    wall = time.perf_counter() - t0
    m1 = fetch_metrics(args.url)

    drafts = m1.get("drafts", 0.0) - m0.get("drafts", 0.0)
    dtk = m1.get("draft_tokens", 0.0) - m0.get("draft_tokens", 0.0)
    acc = m1.get("accepted", 0.0) - m0.get("accepted", 0.0)
    pos = {p: m1.get(f"per_pos_{p}", 0.0) - m0.get(f"per_pos_{p}", 0.0)
           for p in range(4)}
    pos1 = pos[0] / drafts if drafts > 0 else 0.0
    acceptance = acc / dtk if dtk > 0 else 0.0
    mean_len = 1 + acc / drafts if drafts > 0 else 0.0

    row = {"wall_s": round(wall, 2), "streams_ok": f"{sum(ok_flags)}/{args.streams}",
           "drafts": int(drafts), "draft_tokens": int(dtk),
           "accepted": int(acc),
           "pos": {str(p): round(pos[p] / drafts, 4) if drafts else None
                   for p in pos},
           "pos1": round(pos1, 4), "acceptance": round(acceptance, 4),
           "mean_acceptance_length": round(mean_len, 2),
           "hard": args.hard, "target": args.target}
    print("SPEC_GATE " + json.dumps(row))
    print("SPEC_GATE log_window: " + log_window(args.container, args.log))

    if sum(ok_flags) < args.streams:
        print(f"SPEC_GATE FAIL: load incomplete "
              f"({sum(ok_flags)}/{args.streams})")
        return 3
    if drafts <= 0:
        print("SPEC_GATE FAIL: zero draft steps measured (telemetry present "
              "but idle — engine served without drafting?)")
        return 3
    if pos1 < args.hard:
        print(f"SPEC_GATE FAIL: pos1={pos1:.4f} < {args.hard} hard gate — "
              f"wrong-logits family (D r1 mojibake signature)")
        return 2
    verdict = (f"SPEC_GATE PASS: pos1={pos1:.4f} >= {args.hard} hard, "
               f"acceptance={acceptance:.4f} vs {args.target} target")
    if acceptance < args.target:
        verdict += " (TARGET MISS"
        if args.strict_target:
            print(verdict + ")")
            return 4
        verdict += " — recorded, --strict-target for certification)"
    print(verdict)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
