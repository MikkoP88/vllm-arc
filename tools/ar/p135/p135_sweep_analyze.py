#!/usr/bin/env python3
"""p135_sweep_analyze.py -- P6.35-P posture sweep analyzer (runs on the banked
p135_face_*.json artifacts; local, no rig needed).

Per cell: effective decode tok/s = (n_stream_chunks-1)/(total_s-ttft_s)
(client-view, the robust metric under spec-decode burst streaming: within a
step, chunks arrive back-to-back so per-chunk ITLs are burst-shaped — the
total-time ratio is the honest rate), ITL percentiles for jitter shape, and
baseline deltas. Baseline = fp8 spec3 solo U8k600 (banked leg-F face:
~40.5 ms step cadence).

Usage: p135_sweep_analyze.py <dir-with-p135_face_*.json> [baseline_tag]
"""
import glob
import json
import os
import sys

BASELINE_TAG = sys.argv[2] if len(sys.argv) > 2 else None


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def row(tag, d):
    n = d.get("n_stream_chunks", 0)
    mt = d.get("max_tokens", 0)
    ttft = d.get("ttft_s", -1)
    total = d.get("total_s", -1)
    if n < 10 or ttft < 0 or total <= ttft or mt < 10:
        return "  %-10s INVALID n=%d ttft=%.3f total=%.3f" % (tag, n, ttft, total)
    # Spec-decode streaming batches multiple tokens per SSE chunk (fp8k4:
    # 600 tokens in 127 chunks), so chunks UNDERCOUNT tokens. The honest
    # numerator is max_tokens (faces run ignore_eos, finish=length; metrics
    # cross-check: generation_tokens_total & request_success length match).
    rate = (mt - 1) / (total - ttft)
    tok_per_chunk = (mt - 1) / max(n - 1, 1)
    return ("  %-10s tok/s=%6.1f  ttft=%6.2fs  tokens=%3d(%4.2f/chunk)  ITL ms: "
            "mean=%5.2f p50=%5.2f p90=%6.2f p99=%7.2f max=%7.2f" % (
                tag, rate, ttft, mt, tok_per_chunk, d["itl_ms_mean"],
                d["itl_ms_p50"], d["itl_ms_p90"], d["itl_ms_p99"],
                d["itl_ms_max"]))


def main():
    d = sys.argv[1]
    files = sorted(glob.glob(os.path.join(d, "p135_face_*.json")))
    if not files:
        print("no face jsons in", d)
        return 1
    base = None
    rows = []
    for p in files:
        tag = os.path.basename(p)[len("p135_face_"):-len(".json")]
        try:
            j = load(p)
        except Exception as e:  # noqa: BLE001
            rows.append("  %-10s UNREADABLE %r" % (tag, e))
            continue
        rows.append(row(tag, j))
        if BASELINE_TAG and tag == BASELINE_TAG:
            base = (j["max_tokens"] - 1) / (j["total_s"] - j["ttft_s"])
    print("=== P6.35-P posture sweep (solo U8k ctx=8192 max=600) ===")
    print("\n".join(rows))
    if base:
        print("\n  baseline %s = %.1f tok/s; deltas:" % (BASELINE_TAG, base))
        for p in files:
            tag = os.path.basename(p)[len("p135_face_"):-len(".json")]
            try:
                j = load(p)
            except Exception:  # noqa: BLE001
                continue
            if tag == BASELINE_TAG or j.get("n_stream_chunks", 0) < 10:
                continue
            r = (j["max_tokens"] - 1) / (j["total_s"] - j["ttft_s"])
            print("    %-10s %+6.1f%%  (%.1f -> %.1f tok/s)" % (
                tag, (r / base - 1) * 100.0, base, r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
