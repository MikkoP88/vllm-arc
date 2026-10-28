# VERDICT — p2r3d-stream-scope (2026-10-27)

**LEG: diagnostic (hot-patch overlay, image ce0ee65f3766 @ 17ae84a7).
DECISIVE: the L0 recording flag is PER-STREAM.**

## Posture

graph=1, spec=0, lmc=0, TP2. Container hot-patch overlay (defer branch
probes a brand-new torch.xpu.Stream at the defer moment). First boot in a
freshly recreated container (image unchanged from p2r3c) — AOT compile at
15:35–15:38 (fresh /root/.cache).

## The probe result (verbatim from the worker WARNINGS)

    (Worker_TP1 pid=3800) WARNING ... deferred to upstream (capture window
    active). current_stream=torch.xpu.Stream(device=xpu:1 sycl_queue=0x28779b80)
    capturing(cur)=True capturing(fresh stream)=False
    (Worker_TP0 pid=3799) WARNING ... current_stream=torch.xpu.Stream(device=xpu:0
    sycl_queue=0x10544750) capturing(cur)=True capturing(fresh stream)=False

## Consequences

1. The MAIN compute stream of each worker is left in recording mode at the
   first real request; a brand-new stream is clean. Not device-global.
2. Fix decision splits on one open question: do kernels enqueued on the
   flagged main stream EXECUTE (stale flag / waits-only broken →
   fresh-stream copy path is a complete fix) or get appended and never run
   (true zombie → serving garbage → the recording state itself must be
   cleared/prevented)? Fresh-stream copies in the garbage case would be
   silent corruption — integrity trap, unacceptable.
3. Instrument added (commit 20536a5d): execution canary in the defer branch
   — distinctive fill on the flagged current stream, 1 s settle, blocking
   readback on a clean stream. Also the first actual fresh-stream COPY
   test (prior probes only queried the flag).

## Harness note

Same as p2r3c: serve log overwritten by the p2r3e serves before mirroring;
the probe lines above were captured in-session. Mirror law restored from
p2r3e.
