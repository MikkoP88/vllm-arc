# P6.11 / r71 — TP2 allreduce P2P-path solvability analysis

**Date**: 2026-10-31 · **Owner directives**: (1) commit + host-validate
the ACS tooling; (2) "oneCCL never selected the P2P path (most common,
pure config) — check: boot log with CCL_LOG_LEVEL=2; fix: enable the
level-zero IPC/P2P knobs verified against the image's build"; (3)
"driver/IOMMU refusing P2P DMA — check IOMMU groups + a D2D peer-copy
probe (~12 GB/s = P2P, ~5–6 = bounce); fix per driver docs". Each lane:
**deep-analyze and check whether it is solvable.**

**Headline verdicts** (all evidence read-only on the live rig during
the r70 soak; dynamic legs queued for the idle window):

| Lane | Verdict |
|---|---|
| ACS (directive 1) | **DONE + validated on host** — active+enabled, 18 devices zeroed, self-verify "all ACSCtl = 0000", 0 control bits set (lspci recount). Banked as `tools/pcie/` (commit 21c73802 / PR #26); validation transcript in `acs_validation.txt`. |
| IOMMU refuses P2P (directive 3a) | **RULED OUT — dead**: IOMMU is fully OFF on this rig (empty `/sys/kernel/iommu_groups` + `/sys/class/iommu`, zero DMAR lines, no `intel_iommu=` on cmdline). Nothing at the IOMMU layer can block a peer DMA. |
| Driver/link reality (directive 3b) | **OPEN — decisive instrument authored**: sysfs/lspci link attributes MISREPORT on this stack (xe leaves 2.5 GT/s ×1 on every device incl. both parent bridges — contradicted by the faces: U8k prefill 3.86 s is impossible over 250 MB/s). `b70_probe21_d2d.sh` + `d2d_probe.py` measure the effective GPU↔GPU bandwidth at the next idle window. |
| oneCCL never selected P2P (directive 2) | **No missing config found — champion chain already maximal**: image oneCCL = **2022.0.0**; env surface extracted from the image's `libccl.so` (296 `CCL_*` names; `ccl_env_surface.txt`). The frozen champion block already sets `CCL_TOPO_P2P_ACCESS=1` + `CCL_ZE_IPC_EXCHANGE=pidfd` + ATL=ofi (boot-log confirmed: "CCL_ATL_TRANSPORT changed to be ofi"), and the bundled libfabric embeds the `ze` + `shm` providers. The runtime *selection* is still unconfirmed → `r71a_ccl_diag.sh` (CCL_LOG_LEVEL=2, one-env change, zero cost) at the idle window. |

## What was checked, exactly (all read-only during the soak)

1. **Champion env chain** (`tools/serve_arc_b70.sh`, frozen): the CCL
   block is 8 vars, provenance "rig container env inspect 2026-10-27,
   W14 == lsv-test byte-compared". `CCL_ZE_IPC_EXCHANGE=pidfd` is
   load-bearing (keeps oneCCL's ze_fd_manager off the opendir device
   walk that kills non-privileged TP2 init — the `--privileged` pair).
2. **Banked boot logs** (r70 legs a/b): both ranks print
   `CCL_ATL_TRANSPORT changed to be ofi (default:mpi)` +
   `CCL_TOPO_P2P_ACCESS changed to be 1 (default:-1)` → the env IS
   parsed and applied by this build (the "knob doesn't exist in this
   version" failure mode is excluded for the two key knobs).
3. **Image oneCCL inventory** (read-only container, no GPU):
   `oneccl-2022.0.0`; real binary `libccl.so` (210 MB, 296 env names)
   vs the small `libccl.so.2.0` shim (12 names — first-extraction
   trap, noted for honesty); libfabric 472 KB with `ze`, `shm`,
   `tcp`, `sockets`, `verbs`, `psm` provider tokens.
4. **Host state**: ACS 0 bits set (recount); IOMMU off (above);
   kernel cmdline bare (`ro` only); GPUs at `b1:00.0`/`da:00.0` behind
   per-socket bridges `b0:01.0`/`d9:01.0` (cross-socket UPI hop if
   direct — the D2D number bakes that in); xpu-smi v2.0 present
   (discovery works; no link-stats subcommand on this build).

## Queued idle-window legs (in order; one change each, E-law)

- **r71a** — `r71a_ccl_diag.sh`: certified line + one env
  (`CCL_LOG_LEVEL=2`); boot, health, one request, capture transport
  lines, teardown, W14 restore (trap). Decides lane "oneCCL selection".
- **r71b** — `b70_probe21_d2d.sh`: throwaway container, both GPUs,
  256 MiB both directions, median GB/s. Decides lane "driver/link".
- Interpretation matrix: P2P confirmed both → AR lever moves to
  algorithmic knobs (own A/B legs, posture-change law); D2D bounce →
  driver/level-zero peer-access fix per the image's compute-runtime
  docs (evidence-gated); D2D ≤1 GB/s → BIOS link config investigation.

## Files

`acs_validation.txt` (host validation transcript), `host_checks.txt`
(IOMMU/cmdline/sysfs/link reads), `ccl_env_surface.txt` (extracted env
names — captured subset + the exact re-extraction one-liner),
`kit/` (byte-exact copies of the three authored instruments, also
tracked at `tools/pcie/`).

**No engine-code change; cert4 remains the certified image (no
cert5).**
