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
| `r71a_ccl_diag.sh` | CCL_LOG_LEVEL=2 diagnostic boot (p71 lane-1 confirm): certified serve line + exactly one inserted env; captures the oneCCL transport/IPC selection lines; W14 restored by trap. Idle window only. |

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

## Open P2P items (status after the 2026-10-31 p71 analysis — instruments
## run at the next idle window, NEVER mid-leg)

Owner question: is the TP2 allreduce on the direct P2P path or
host-bounced — and is each cause solvable? Analysis evidence:
`results/2026-10-31-p71-r71-p2p-analysis/`.

1. **IOMMU blocking P2P — RULED OUT (dead)**. Read-only host checks:
   `/sys/kernel/iommu_groups/` and `/sys/class/iommu/` both empty, zero
   DMAR/IOMMU lines in dmesg, no `intel_iommu=` anywhere on the kernel
   cmdline → IOMMU is OFF on this rig; nothing at the IOMMU layer can
   refuse a peer DMA. (ACS was likewise already fully disabled — the
   insurance service above just keeps it that way.)
2. **oneCCL config missing the P2P path — EMPTY so far (champion chain
   already maximal)**. The image's oneCCL is **2022.0.0**; its env
   surface was extracted from `libccl.so` (296 `CCL_*` names, evidence
   file in the results dir). The frozen champion block already sets the
   three knobs that matter: `CCL_TOPO_P2P_ACCESS=1`,
   `CCL_ZE_IPC_EXCHANGE=pidfd`, `CCL_ATL_TRANSPORT=ofi` (log-confirmed),
   and the bundled libfabric embeds the `ze` + `shm` providers. No
   obvious unset knob remains — the runtime selection still needs
   confirmation: **`r71a_ccl_diag.sh`** (CCL_LOG_LEVEL=2 boot, one-env
   change, zero cost). If the log shows the ze/IPC path active, lane 1
   closes as "already solved" and any further gain is algorithmic
   (e.g. `CCL_ZE_BIDIR_ALGO` / `CCL_ATL_RMA` class knobs — those would
   be posture changes needing their own evidence-gated A/B legs).
3. **Driver/link reality — OPEN, the decisive instrument is the D2D
   probe**. sysfs/lspci link attributes misreport on this stack (xe
   leaves `current_link_speed` at 2.5 GT/s ×1 on every device incl.
   parent bridges — flatly contradicted by the measured faces: U8k
   prefill at 3.86 s is impossible over a 250 MB/s link).
   **`b70_probe21_d2d.sh` + `d2d_probe.py`** measure the effective
   GPU0↔GPU1 bandwidth: ~10–12 GB/s = direct P2P; ~5–6 = host-staged;
   ≤1 = real degraded link (BIOS/config investigation).
4. **Topology**: parent bridges are `b0:01.0` / `d9:01.0` (one per
   socket) — cross-socket P2P adds a UPI hop; the D2D number bakes
   that reality in, so no separate topology leg is needed unless the
   probe shows bounce.
5. **AR attribution leg** (queued in the plan from p67): per-AR timing
   vs payload/link-rate — decides independently of probes whether the
   ~30.4 % AR share is link-bound or algorithm/latency-bound.

Expected magnitudes (AR model: prefill ≈ 0.66 MB/token, decode ≈ 5 MB/step
at c8, 64 serialized hops): Gen3 P2P ≈ 9% of prefill wall, bounced ≈
18–25%; decode AR is latency-bound (2–6 ms/step swing). If the probe shows
direct P2P already active, the whole PCIe/Gen4/Gen5 lane closes and prefill
remains compute-bound (fp8 GEMM rung closed negative on r40; FA2 varlen
20.3% is the live rung).

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
- Known oddity to resolve once: `lspci -s b1:00.0 -vv` grep of LnkCap
  printed "Port #0, Speed 2.5GT/s, Width x1" — likely the wrong capability
  line; confirm link speed on the PARENT bridge port, not the endpoint.
