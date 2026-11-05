#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ar_ceiling.py — P6.22 r95: TP2 allreduce primitives ceiling bench.

Answers ONE question per message size: how much headroom does stock
oneCCL allreduce (the serve path: c10d::allreduce_ -> topo sycl ring,
r67b/r71 evidence) leave on the table versus the algorithmic floor of
a one-shot peer-write + fused local add at the proven P2P rate
(10.23 GB/s both directions, r71b)?

Two roles (ceiling-first law — the P6.21c lesson: measure the bound
BEFORE building integration):
  dist     torchrun --nproc-per-node 2 (backend xccl — the serve
           engine's own backend name, the serve env's CCL block
           exported by the wrapper): stock dist.all_reduce, events on
           each rank, median-of-N; numerics guard (SUM == a+b) at the
           first size.
  oneshot  single process owning BOTH GPUs — the IPC-free UPPER BOUND
           of a two-process one-shot AR: both staging copies issued
           (per-device streams when available, else sequential), sync,
           local fused adds out_i = a_i + st_i. Reports the end-to-end
           wall plus the pieces (one-direction copy, add) for the
           composed model. Real integration pays IPC handles + a
           completion protocol on top — this role prices the ceiling,
           not the product.

Sizes: fp16 elements; --sizes-bytes overrides with r94's measured
message sizes (comma-separated bytes).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

import torch

DEFAULT_BYTES = [4096, 65536, 1 << 20, 8 << 20, 16 << 20, 32 << 20,
                 48 << 20, 64 << 20, 80 << 20, 96 << 20, 128 << 20]


def med_events(fn, iters=9, warmup=3):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    evs = []
    for _ in range(iters):
        s = torch.xpu.Event(enable_timing=True)
        e = torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        evs.append((s, e))
    torch.xpu.synchronize()
    return statistics.median([s.elapsed_time(e) for s, e in evs])


def role_dist(sizes_bytes, out, guard="const", label="stock",
              tol_mean=0.02, tol_cos=0.9999):
    import os

    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    torch.xpu.set_device(rank)
    # Backend name: "xccl" — natively registered by the Intel xpu torch
    # build (2.13.0+xpu: backend_list = fake/gloo/mpi/nccl/ucc/xccl) and
    # exactly what the serve path uses (vllm/platforms/xpu.py:111
    # `dist_backend = "xccl"`; engine log "backend=xccl"). NOT "ccl" —
    # that name needs a oneCCL-bindings import this image does not carry
    # (r95 attempt-1/2 defects, banked in r95_solo*.log).
    dist.init_process_group(backend="xccl")
    report = []
    for nbytes in sizes_bytes:
        numel = nbytes // 2
        if guard == "randn":
            # P6.24 fp8 lane: realistic values + locally-constructed fp32
            # reference (CPU-seeded per rank, deterministic across runs and
            # ranks — no extra collective needed). Bars below.
            a = (torch.randn(numel, generator=torch.Generator().manual_seed(
                     1000 + rank)) * 0.5).to(device=f"xpu:{rank}",
                                             dtype=torch.float16)
            ref = None
            for r in (0, 1):
                part = (torch.randn(numel, generator=torch.Generator(
                            ).manual_seed(1000 + r)) * 0.5)
                ref = part if ref is None else ref + part
            dist.all_reduce(a)
            # cosine + RMS-normalized mean-abs bars: per-element relative
            # error is unbounded near zero even for stock fp16; these two
            # catch wrong-half/scale/garbage corruption while passing the
            # e4m3 noise floor (~1-2% of RMS).
            af, rf = a.float().cpu(), ref.float()
            cos = ((af * rf).sum() /
                   (af.norm() * rf.norm()).clamp_min(1e-9)).item()
            rel = ((af - rf).abs().mean() / rf.std()).item()
            ok = cos >= tol_cos and rel <= tol_mean
            if nbytes == sizes_bytes[0]:
                print(f"[{'PASS' if ok else 'FAIL'}] numerics randn "
                      f"(cos {cos:.6f} rel-rms {rel:.5f}) rank{rank}")
        else:
            a = torch.full((numel,), float(rank + 1), device=f"xpu:{rank}",
                           dtype=torch.float16)
            # numerics guard FIRST, on a fresh single call (the timed reps
            # below accumulate into the same buffer — SUM == a0 + a1 == 3)
            a.fill_(float(rank + 1))
            dist.all_reduce(a)
            ok = torch.allclose(a.float(), torch.full_like(a.float(), 3.0))
            if nbytes == sizes_bytes[0]:
                print(f"[{'PASS' if ok else 'FAIL'}] numerics (allreduce "
                      f"sum == 3) rank{rank}")
        t = med_events(lambda: dist.all_reduce(a))
        # n=2: each rank ships S bytes over the link per AR, so S/t is
        # the per-rank link utilization — directly comparable to the
        # r71b 10.23 GB/s P2P rate.
        link_gbs = nbytes / (t * 1e-3) / 1e9
        row = {"bytes": nbytes, "label": label, "ms": round(t, 3),
               "stock_ms": round(t, 3), "link_gbs": round(link_gbs, 2)}
        report.append(row)
        if rank == 0:
            print(f"{label.upper():>10} S={nbytes >> 20:>3d}MiB({nbytes:>9d}B): "
                  f"{t:8.3f} ms  {link_gbs:6.2f} GB/rank/s")
    if rank == 0:
        json.dump(report, open(out, "w"), indent=1)
        print(f"wrote {out}")
    dist.destroy_process_group()


def role_oneshot(sizes_bytes, out):
    assert torch.xpu.device_count() == 2, "needs both GPUs"
    report = []
    streams = []
    try:
        streams = [torch.xpu.Stream(device=i) for i in range(2)]
    except Exception as e:  # noqa: BLE001 — stream API absence -> seq mode
        print(f"per-device streams unavailable ({e!r}) — sequential "
              f"copies ( UNDER-states the ceiling; pieces still valid)")
    for nbytes in sizes_bytes:
        numel = nbytes // 2
        a = [torch.full((numel,), float(i + 1), device=f"xpu:{i}",
                        dtype=torch.float16) for i in range(2)]
        st = [torch.empty_like(a[i]) for i in range(2)]
        out_t = [torch.empty_like(a[i]) for i in range(2)]

        def copies():
            # dev0 -> st1 and dev1 -> st0 (peer writes, both directions)
            if streams:
                with torch.xpu.stream(streams[0]):
                    st[1].copy_(a[0])
                with torch.xpu.stream(streams[1]):
                    st[0].copy_(a[1])
                for s in streams:
                    s.synchronize()
            else:
                st[1].copy_(a[0])
                st[0].copy_(a[1])
                torch.xpu.synchronize()

        def adds():
            torch.add(a[0], st[0], out=out_t[0])
            torch.add(a[1], st[1], out=out_t[1])
            torch.xpu.synchronize()

        def full():
            copies()
            adds()

        t_copy = med_events(copies)
        t_add = med_events(adds)
        t_full = med_events(full)
        # a0=1, a1=2; st[i] holds the peer's a -> out = 1+2 = 3 both sides
        ok = all(torch.allclose(o.float(),
                                torch.full_like(o.float(), 3.0))
                 for o in out_t)
        gbs = nbytes / (t_copy * 1e-3) / 1e9
        row = {"bytes": nbytes, "oneshot_ms": round(t_full, 3),
               "copies_ms": round(t_copy, 3), "add_ms": round(t_add, 3),
               "copy_gbs_each_dir": round(gbs, 2), "numerics": ok}
        report.append(row)
        print(f"ONESHOT S={nbytes >> 20:>3d}MiB({nbytes:>9d}B): "
              f"full={t_full:8.3f} ms (copies={t_copy:7.3f} "
              f"[{gbs:5.2f} GB/s/dir] + add={t_add:6.3f}) "
              f"numerics={'OK' if ok else 'FAIL'}")
    json.dump(report, open(out, "w"), indent=1)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["dist", "oneshot"])
    ap.add_argument("--sizes-bytes", default=None,
                    help="comma-separated byte sizes (r94 census values)")
    ap.add_argument("--out", default="/work/ar_ceiling.json")
    ap.add_argument("--guard", default="const", choices=["const", "randn"],
                    help="dist numerics guard: const (sum==3) or randn "
                         "(fp32 reference, fp8-lane bars)")
    ap.add_argument("--label", default="stock",
                    help="row label (stock / rebuilt / arc-fp8-off / "
                         "arc-fp8-on)")
    ap.add_argument("--tol-mean", type=float, default=0.02)
    ap.add_argument("--tol-cos", type=float, default=0.9999)
    a = ap.parse_args()
    sizes = ([int(x) for x in a.sizes_bytes.split(",")]
             if a.sizes_bytes else DEFAULT_BYTES)
    if a.role == "dist":
        role_dist(sizes, a.out, guard=a.guard, label=a.label,
                  tol_mean=a.tol_mean, tol_cos=a.tol_cos)
    else:
        role_oneshot(sizes, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
