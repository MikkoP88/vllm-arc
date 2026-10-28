"""r8b bare-torch probe matrix: capture-stream poisoning + replay-stream check.

Phase A: capture on the SIDE stream (champion discipline, eagerly inited) —
         does the DEFAULT stay clean? Then replay on the default — does the
         launcher get flagged? (torch 2.13.0+xpu / oneAPI 2026.0.0)
Phase B: capture directly on the DEFAULT (upstream 0.30 behavior) — expect
         default flagged post-capture (p13 law reproduction = probe sanity).
Usage: probe.py a|b
"""
import sys

import torch

print("torch", torch.__version__, flush=True)


def cap(tag):
    try:
        print(f"{tag:38s} capturing="
              f"{torch.xpu.is_current_stream_capturing()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{tag:38s} capturing=PROBE_ERR {e!r}", flush=True)


phase = sys.argv[1] if len(sys.argv) > 1 else "a"
default = torch.xpu.current_stream()
# Eager init — mirrors torch's own lazy init inside graph.__init__ (r8 bug:
# reading the class attr before the first context entry yields None).
side = torch.xpu.graph.default_capture_stream
if side is None:
    side = torch.xpu.Stream()
    torch.xpu.graph.default_capture_stream = side
print("phase", phase, "| default:", default, "| side:", side, flush=True)
cap("fresh default")

buf = torch.ones(8, device="xpu")
g = torch.xpu.graphs.XPUGraph()
cap_stream = side if phase == "a" else default
with torch.xpu.graph(g, stream=cap_stream):
    buf.mul_(2.0)
cap(f"post-capture[{phase}] (default)")
with torch.xpu.stream(side):
    cap(f"post-capture[{phase}] (side)")

if phase == "b":
    try:
        g.replay()  # r8's exact shape: replay on the CAPTURE stream (default)
        torch.xpu.synchronize()
        print("phase-b replay (on default): OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print("phase-b replay (on default): RAISED", repr(e), flush=True)
    cap("post-b-replay (default)")
    try:
        y = buf.mul_(3.0)
        torch.xpu.synchronize()
        print("phase-b plain kernel: OK vals=", y.tolist(), flush=True)
    except Exception as e:  # noqa: BLE001
        print("phase-b plain kernel: RAISED", repr(e), flush=True)

if phase == "a":
    try:
        g.replay()
        torch.xpu.synchronize()
        print("replay #1 (on default): OK", flush=True)
    except Exception as e:  # noqa: BLE001
        print("replay #1 (on default): RAISED", repr(e), flush=True)
    cap("post-REPLAY-on-default (default)")
    try:
        y = buf.mul_(3.0)
        torch.xpu.synchronize()
        print("plain kernel on default: OK vals=", y.tolist(), flush=True)
    except Exception as e:  # noqa: BLE001
        print("plain kernel on default: RAISED", repr(e), flush=True)
    try:
        g.replay()
        torch.xpu.synchronize()
        print("replay #2: OK buf=", buf.tolist(), flush=True)
    except Exception as e:  # noqa: BLE001
        print("replay #2: RAISED", repr(e), flush=True)
    cap("post-replay-2 (default)")
print("done", flush=True)
