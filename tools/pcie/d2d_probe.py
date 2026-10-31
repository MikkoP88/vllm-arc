#!/usr/bin/env python3
"""p71 D2D peer-copy probe — measures the REAL GPU0<->GPU1 path.

Why this exists: sysfs/lspci link attributes misreport on this stack
(xe driver leaves current_link_speed at the Gen1-x1 default on every
device incl. parent bridges — flatly contradicted by the measured
prefill faces), and the oneCCL transport choice is only visible at
CCL_LOG_LEVEL>=2. This probe answers the bandwidth question directly:
copy a tensor across devices and read the effective throughput.

Verdict table (rig slots are Gen3 x16, ~12.5 GB/s per direction):

  ~10-12 GB/s   direct P2P (level-zero peer copy; ACS off, IPC live)
  ~5-6  GB/s    staged through host memory (bounce) or half-duplex
  <=1   GB/s    degraded link (the Gen1-x1 misreport is REAL) —
                BIOS/config investigation, not a software knob

Run only at an idle window (never mid-leg — caution law in
tools/pcie/README.md). Host-side wrapper: b70_probe21_d2d.sh.
"""
import statistics
import time

import torch

MIB = 1024 * 1024
SIZE_MB = 256          # per-direction payload
N_WARM, N_REP = 3, 20


def gb_s(seconds: float) -> float:
    return (SIZE_MB * MIB) / 1e9 / seconds


def main() -> None:
    n = torch.xpu.device_count()
    names = [torch.xpu.get_device_name(i) for i in range(n)]
    print(f"devices={n} {names}")
    peer_api = getattr(torch.xpu, "can_device_access_peer", None)
    if peer_api is not None:
        print(f"can_device_access_peer(0,1)={peer_api(0, 1)} "
              f"(1,0)={peer_api(1, 0)}")
    else:
        print("torch.xpu.can_device_access_peer: absent in this torch build")
    for src, dst in ((0, 1), (1, 0)):
        a = torch.randn(SIZE_MB * MIB // 2, dtype=torch.float16,
                        device=f"xpu:{src}")
        b = torch.empty_like(a, device=f"xpu:{dst}")
        for _ in range(N_WARM):
            b.copy_(a, non_blocking=True)
        torch.xpu.synchronize(src)
        torch.xpu.synchronize(dst)
        ts = []
        for _ in range(N_REP):
            t0 = time.perf_counter()
            b.copy_(a, non_blocking=True)
            torch.xpu.synchronize(dst)
            ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        print(f"xpu:{src} -> xpu:{dst}  {SIZE_MB} MiB  "
              f"median {gb_s(med):6.2f} GB/s  "
              f"min {gb_s(min(ts)):6.2f}  max {gb_s(max(ts)):6.2f}")
        del a, b
    print("D2D_PROBE_DONE")


if __name__ == "__main__":
    main()
