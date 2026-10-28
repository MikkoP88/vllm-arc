# VERDICT — p12b rounds 1+2: dezombie validated in production; barrier law + flip-window revision

Rounds 1 and 2 of the p12b graph-stream-hygiene patch (branch
`feat/arc-p12b-graph-stream-hygiene`) ran as evidence legs on the p2-graph
posture. Both FAIL by design: the serve-time alarm raises (integrity law — the
in-flight request's forward was enqueued on the dead queue) and vllm v1
fail-fast kills the engine. The findings below are the payload.

## 1. Dezombie works in production (both rounds, both TP workers)

The serve-time output alarm fired exactly as designed: probe (capturing=True,
no capture open) → `torch.xpu.set_stream(torch.xpu.Stream())` → post-swap
probe False → canary on the fresh stream **executed=True, vals=[123.456×4]** →
our `RuntimeError` raised (5 occurrences in round 2: 2×TP0 raise, 2×TP1 raise,
1× EngineCore aggregate) → engine death, no garbage tokens ever returned.

Stream identities (round 2, 00:28:03):
- TP0: poisoned serving stream `sycl_queue=0x1d443410`, side capture stream
  `0x1d2ec410`, post-swap fresh `0x1d2f9f80`
- TP1: poisoned `0x1c4aa570`, side `0x1cb11570`, post-swap fresh `0x1c9ce6d0`
- `default=ERR` in the ids line: **`torch.xpu.default_stream()` raises on this
  build** (torch 2.13.0+xpu). The round-2 onset watchdog called it at its first
  probe and its `except: return` silently killed the thread — ZERO onset lines
  were collected. Round 3 (90aee3b3) probes the watchdog thread's OWN current
  stream (a fresh thread's current IS the device default) instead.

## 2. NEW LAW — the device barrier sees the abandoned queue

After the dezombie swap, vllm's error-drain path executed FURTHER
`AsyncOutput` constructions on the fresh (clean, probe=False) stream. Each took
the healthy path and died at `torch.xpu.synchronize()`:

```
graph_safe_output.py:409 __init__ → torch.xpu.synchronize()
→ RuntimeError: wait cannot be called for a queue which is recording to a command graph.
```

`torch.xpu.synchronize()` is a DEVICE-WIDE barrier: it refuses while ANY queue
in the process still records. The swap hides the zombie from the per-stream
probe but the abandoned queue keeps its L0 recording state and blocks the
device barrier forever. Meanwhile clean-STREAM-scoped ops keep working (the
dezombie canary's D2H copy on a fresh stream succeeds while the poisoned queue
exists) — **barrier law: per-stream syncs tolerate the zombie; device syncs do
not.** Consequence for any survival design: the healthy-path barrier must be
stream-scoped (e.g. wait the copy stream), and the ultimate fix must CLOSE the
recorder, not just abandon the queue.

## 3. Flip-window revision — not deterministic at boot

Round-2 timeline (TP0):
- 00:22:12 patches installed (worker), captures complete during boot
- 00:26:52, 00:26:54, 00:26:57 — THREE budgeted "graph-safe output applied"
  constructions: probe False, device sync SUCCEEDED. A request (A) was fully
  served: prefill + decode steps clean, `POST /v1/completions 200 OK`.
- 00:28:03 — request B's construction: probe True → alarm → dezombie → raise.

So the poison entered between A's last sample (00:26:57) and B's construction
(00:28:03) — a window containing only `/metrics` polls and then B's forward.
The answer leg (2026-10-27, p13 image) flipped BEFORE its first request; round
2 flipped AFTER a fully-served request. **The flip is not pinned to boot
capture; it correlates with serving activity and its exact op is still
unknown.** Notably A's decode (graph replays included, FULL_DECODE_ONLY) ran
clean — replays alone do not guarantee the flip.

## 4. Round 3 instrumentation (built from 90aee3b3)

- Watchdog probes its own thread's current stream (= device default) + the
  side capture stream via explicit switch; logs `ARC_GRAPH_ZOMBIE_ONSET` with
  timestamps (≤100 ms resolution) bracketed against vllm's request logs.
- Post-flip tripwires wrap `torch.xpu.synchronize`, `Event.record`,
  `Stream.wait_stream/wait_event/synchronize` (+ the `torch.cuda.synchronize`
  alias when the donor alias law bound it to the same original): the FIRST
  call made with a stuck flag and no capture open logs its caller stack —
  naming the poisoning neighborhood. CPU-only processes never arm.

Expected round-3 readout: onset timestamp vs request-B arrival + the first
tripwire stack. Raws for both rounds on the rig leg dirs; key lines in
`excerpts.txt` (copied to this dir as `r2_excerpts.txt`).
