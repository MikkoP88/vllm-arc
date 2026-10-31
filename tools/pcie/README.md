# PCIe ACS / P2P tooling (rig REDACTED-HOST, dual Arc Pro B70)

Host-level service that keeps **ACS (Access Control Services) disabled** on
every PCIe device so peer-to-peer DMA between the two B70s is never blocked
by redirect enforcement. This is **rig/OS tooling, not image content** — it
changes no serve flags and no model numerics.

## Files

| File | What it is |
|---|---|
| `disable-acs.sh` | Dell PowerEdge 16th-Gen setpci practice: zeros the ACS Control register (`ECAP_ACS+0x6.w=0000`) on every ACS-capable device; logs old values; self-verifies at the end. Idempotent. Exact bytes of the deployed rig copy. |
| `disable-acs.service` | systemd oneshot unit, `Before=docker.service`, `WantedBy=multi-user.target` — re-applies the script at every boot before any container starts. |
| `d2d_probe.py` | D2D peer-copy microbench (p71): 256 MiB fp16 tensor, both directions, median/min/max GB/s + `torch.xpu.can_device_access_peer` when present. Verdict table in the file header. |
| `b70_probe21_d2d.sh` | Host wrapper for the probe — extends the rig's `b70_probe*` family; captures xpu-smi discovery + sysfs link state + the timed copies in one transcript. Idle window only. |
| `r71a_ccl_diag.sh` | CCL_LOG_LEVEL=debug diagnostic boot (p71 lane-1 confirm; enum-name level — numerics throw at env parse on this build): certified serve line + exactly one inserted env; captures the oneCCL transport/IPC selection lines; W14 restored by trap. Idle window only. |

## Provenance (no-unevidenced-customization law)

- Dell PowerEdge 16th-generation guidance: manual `setpci` ACS disable when
  the BIOS toggle is unavailable; must be re-run after each reboot → the
  systemd unit is Dell's own recommended persistence mechanism.
- Why it matters here: peer TLPs (GPU→GPU direct copies for TP2 allreduce)
  are redirected to the root complex when ACS redirect bits are set —
  the "host-bounce" regime. External evidence: Level1Techs multi-GPU Arc
  Pro B60 testing (P2P enabled = prefill/TTFT 65–70% faster in vLLM).

## Rig deployment record (2026-10-31, REDACTED-HOST = Ubuntu 24.04.4, kernel 6.17)

- B70s at `b1:00.0` + `da:00.0`; board is dual-socket (two root complexes,
  bus 00 + bus 17; BMC Matrox VGA at `03:00.0`). Slots are Gen3 x16
  (~12.5 GB/s per direction); cards are PCIe5-capable.
- Deployed: `/usr/local/sbin/disable-acs.sh` +
  `/etc/systemd/system/disable-acs.service`, `systemctl enable --now` →
  **active + enabled**. First run: 18 ACS devices zeroed, 300 without ACS
  skipped, self-verify "all ACSCtl = 0000".
- **Finding: ACS was ALREADY fully disabled on this rig** (every device read
  `was 0000`). The service is insurance against a BIOS update/reset
  re-enabling ACS — it changes nothing about today's behavior.

## Re-deploy from this repo (fresh host, or after rig reinstall)

```bash
# from this directory, over the rig ssh path (askpass law applies).
# RIG = the rig address — lives ONLY in the untracked layer
# (CLAUDE.local.md "Rig" section); tracked files never carry it.
RIG=10.20.3.REDACTED
scp disable-acs.sh root@$RIG:/usr/local/sbin/disable-acs.sh
scp disable-acs.service root@$RIG:/etc/systemd/system/disable-acs.service
ssh root@$RIG 'chmod 755 /usr/local/sbin/disable-acs.sh && \
  systemctl daemon-reload && systemctl enable --now disable-acs.service && \
  sleep 2 && journalctl -u disable-acs.service --no-pager -n 20'
```

## Verify (any time)

```bash
systemctl is-active disable-acs.service && systemctl is-enabled disable-acs.service
lspci -vvv | grep ACSCtl          # every line must show all bits '-' (= 0000)
journalctl -u disable-acs.service --no-pager -n 20   # per-device log + final verify
```

## Open P2P items (status after the 2026-10-31 p71 analysis + executed
## idle-window legs — ALL LANES CLOSED, verdicts below)

Owner question: is the TP2 allreduce on the direct P2P path or
host-bounced — and is each cause solvable? Evidence:
`results/2026-10-31-p71-r71-p2p-analysis/` (VERDICT + transcripts +
full CCL-diag logs).

1. **IOMMU blocking P2P — RULED OUT (dead)**. Read-only host checks:
   `/sys/kernel/iommu_groups/` and `/sys/class/iommu/` both empty, zero
   DMAR/IOMMU lines in dmesg, no `intel_iommu=` anywhere on the kernel
   cmdline → IOMMU is OFF on this rig; nothing at the IOMMU layer can
   refuse a peer DMA. (ACS was likewise already fully disabled — the
   insurance service above just keeps it that way.)
2. **oneCCL config missing the P2P path — CLOSED: SOLVED-ALREADY**
   (r71a executed, `CCL_LOG_LEVEL=debug`): every collective in the
   window selected the device-kernel P2P path — 32,082 allreduce +
   1,028 allgather calls, ALL `algo topo sycl`, zero host/offload
   selections; pidfd IPC exchange "verified successfully" AND live
   (280 `pidfd_open`, 292/280 IPC-handle get/open, 2,334 peer handle
   fills); `has_shm 0` on all 2,876 provider-selection lines. The
   champion chain (`CCL_TOPO_P2P_ACCESS=1` + `CCL_ZE_IPC_EXCHANGE=pidfd`
   + `CCL_ATL_TRANSPORT=ofi`) was never on the host-bounce path.
   **LEVEL LAW** (attempt-1 evidence): this build takes enum names
   (`error|warn|info|debug|trace`) — numeric `CCL_LOG_LEVEL=2` throws
   at env parse (`env_parser.hpp:100`) and kills both workers at the
   first all_reduce.
3. **Driver/link reality — CLOSED: DIRECT P2P, no defect** (r71b
   executed): D2D 256 MiB fp16 copies **10.23 GB/s both directions**
   (median of 20; min/max within 0.02), `can_device_access_peer` True
   both ways — ~82 % of the Gen3×16 per-direction ceiling with the
   cross-socket topology baked in. The sysfs/lspci 2.5 GT/s ×1
   attributes are conclusively a MISREPORT. Nothing to fix at
   driver/link/BIOS.
4. **Topology**: parent bridges are `b0:01.0` / `d9:01.0` (one per
   socket) — the cross-socket UPI hop is real and baked into the
   10.23 GB/s; no separate topology leg needed.
5. **AR attribution leg** (queued in the plan from p67): still queued,
   but its question is narrowed — with P2P confirmed (~2 GB/s headroom
   at most vs measured need), the leg would attribute the ~30.4 % AR
   share between **algorithm/latency** (ring, 64 serialized hops,
   small decode payloads) and residual link saturation at prefill Ms.
   Any AR gain is a kernel/algorithm project (fused-AR, algo
   thresholds — posture changes, own evidence-gated A/B legs), not
   config or host state.

Bottom line (owner's diagram): traffic moves like the **P2P (direct)**
row — one hop, ~10 GB/s class — not the host-bounce row. The
PCIe/Gen4/Gen5 lane is closed and prefill remains compute-bound (fp8
GEMM rung closed negative on r40; FA2 varlen 20.3 % is the live rung;
decode AR is latency-bound, 2–6 ms/step swing class).

## Cautions

- Never run the D2D probe or flip host state while a bench leg is in flight
  (measurement contamination). The 2026-10-31 deployment ran once at
  10:40 UTC (`enable --now`) inside the r70 campaign window (legs a
  pre-date it, leg b overlapped, leg c followed) — contamination ruled
  out because it was a **verified no-op**: every ACS register already
  read `0000`, so the writes changed zero bits of live PCIe state. The
  steady-state design is boot-time-only (`Before=docker.service`);
  future host-state changes wait for idle windows, never mid-leg.
- After any rig reboot: re-verify interleave + BW matrix (standing law) —
  the service itself needs no attention, it runs before docker.
- Link-attribute misreport — RESOLVED 2026-10-31 (r71b): sysfs/lspci
  print 2.5 GT/s ×1 on GPUs AND parent bridges, but the D2D probe
  measured 10.23 GB/s both directions — the kernel's PCIe link
  attributes on this xe stack simply do not reflect the operational
  link. Do not "fix" the link config; measure with the probe instead.
