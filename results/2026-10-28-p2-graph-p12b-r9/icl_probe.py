"""r9b probe: eager/replay INTERLEAVING shapes + ICL knob sensitivity.

New theory after r9 (all captures on side stream, flip STILL at first real
request): the poison is runtime-side — first request = eager prefill kernels
followed by graph REPLAY on the same (default) queue; boot warmup never
interleaves. Second candidate: oneAPI 2026 immediate command lists (ICL)
implement graph launch as command-graph append → queue reads "recording".
Run: python probe.py c|d|e   (optionally with
UR_L0_USE_IMMEDIATE_COMMANDLISTS=0 to compare)
"""
import os
import sys

import torch

print("torch", torch.__version__, "| ICL:",
      os.environ.get("UR_L0_USE_IMMEDIATE_COMMANDLISTS", "<unset>"),
      flush=True)


def cap(tag):
    try:
        print(f"{tag:40s} capturing="
              f"{torch.xpu.is_current_stream_capturing()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{tag:40s} capturing=PROBE_ERR {e!r}", flush=True)


shape = sys.argv[1] if len(sys.argv) > 1 else "c"
default = torch.xpu.current_stream()
side = torch.xpu.graph.default_capture_stream
if side is None:
    side = torch.xpu.Stream()
    torch.xpu.graph.default_capture_stream = side
print("shape", shape, "| default:", default, "| side:", side, flush=True)

buf = torch.ones(8, device="xpu")
g1 = torch.xpu.graphs.XPUGraph()
g2 = torch.xpu.graphs.XPUGraph()
pool = torch.xpu.graph_pool_handle()
with torch.xpu.graph(g1, pool=pool, stream=side):
    buf.mul_(2.0)
with torch.xpu.graph(g2, pool=pool, stream=side):
    buf.mul_(1.5)
cap("post-captures (default)")

if shape == "c":  # eager kernels first, then replay — the first-request shape
    buf.add_(1.0)
    torch.xpu.synchronize()
    cap("post-eager (default)")
    g1.replay()
    torch.xpu.synchronize()
    cap("post-eager-then-replay (default)")
    buf.add_(1.0)
    torch.xpu.synchronize()
    cap("post-replay-then-eager (default)")
elif shape == "d":  # repeated eager/replay alternation
    for i in range(3):
        buf.add_(1.0)
        g1.replay()
        torch.xpu.synchronize()
        cap(f"cycle {i} eager+replay (default)")
elif shape == "e":  # two pool-shared graphs replayed back to back after eager
    buf.add_(1.0)
    g1.replay()
    g2.replay()
    torch.xpu.synchronize()
    cap("post-eager+2 replays (default)")
    g1.replay()
    torch.xpu.synchronize()
    cap("post-replay-3 (default)")

try:
    y = buf.mul_(1.0)
    torch.xpu.synchronize()
    print("final plain kernel: OK", flush=True)
except Exception as e:  # noqa: BLE001
    print("final plain kernel: RAISED", repr(e), flush=True)
cap("final (default)")
print("buf:", buf.tolist(), flush=True)
