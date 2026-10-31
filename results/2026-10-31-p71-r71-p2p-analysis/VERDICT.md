# P6.11 / r71 — TP2 allreduce P2P-path solvability analysis + executed legs

**Date**: 2026-10-31 · **Owner directives**: (1) commit + host-validate
the ACS tooling; (2) "oneCCL never selected the P2P path (most common,
pure config) — check: boot log with CCL_LOG_LEVEL=2; fix: enable the
level-zero IPC/P2P knobs verified against the image's build"; (3)
"driver/IOMMU refusing P2P DMA — check IOMMU groups + a D2D peer-copy
probe (~12 GB/s = P2P, ~5–6 = bounce); fix per driver docs". Each lane:
**deep-analyze and check whether it is solvable.**

**FINAL headline verdicts** (analysis phase read-only during the r70
soak; the two dynamic legs executed at the idle window after it,
14:55–15:22 UTC):

| Lane | Verdict |
|---|---|
| ACS (directive 1) | **DONE + validated on host** — active+enabled, 18 devices zeroed, self-verify "all ACSCtl = 0000", 0 control bits set (lspci recount). Banked as `tools/pcie/` (commit 21c73802 / PR #26); validation transcript in `acs_validation.txt`. |
| IOMMU refuses P2P (directive 3a) | **RULED OUT — dead**: IOMMU is fully OFF on this rig (empty `/sys/kernel/iommu_groups` + `/sys/class/iommu`, zero DMAR lines, no `intel_iommu=` on cmdline). Nothing at the IOMMU layer can block a peer DMA. |
| Driver/link reality (directive 3b) | **CLOSED — DIRECT P2P, no defect** (r71b, executed): D2D 256 MiB fp16 copies **10.23 GB/s BOTH directions** (median of 20, min/max within 0.02), `torch.xpu.can_device_access_peer` **True** both ways. ~82 % of the Gen3×16 per-direction ceiling with the cross-socket topology baked in. The sysfs/lspci 2.5 GT/s ×1 attributes are conclusively a MISREPORT. Nothing to fix at driver/link/BIOS. Evidence: `r71b_d2d.txt`. |
| oneCCL never selected P2P (directive 2) | **CLOSED — SOLVED-ALREADY: the champion chain selects the P2P path** (r71a, executed at `CCL_LOG_LEVEL=debug`): **every collective in the window chose the device-kernel P2P path** — 32,082 allreduce + 1,028 allgather calls, ALL `selected algo: … algo topo sycl`, zero host/offload selections; **pidfd IPC exchange "verified successfully" AND exercised** (280 `pidfd_open`, 292 `zeMemGetIpcHandle` / 280 `zeMemOpenIpcHandle`, 2,334 `fill_remote_handle` peer exports); `zeDeviceCanAccessPeer`/`zeDeviceGetP2PProperties` consulted for the p2p matrix; **`has_shm 0` on all 2,876 provider-selection lines** (no host-staged path anywhere). Evidence: `r71a_extracts.txt` + full gz logs. |

**Combined answer to the owner's question**: the TP2 allreduce is **NOT
host-bounced** — it already rides direct GPU↔GPU P2P at ~10 GB/s, driven
by oneCCL's `topo sycl` kernels over pidfd-exchanged ze IPC handles. The
PCIe/Gen4/Gen5 "fix the link" lane closes; the ~30.4 % AR slice
(r67b op-class trace) is **algorithmic/latency-bound, not link-bound** —
any further gain is a kernel/algorithm project (e.g. fused-AR,
algo-threshold A/Bs — posture changes needing their own evidence-gated
legs), not a config or host fix.

## What was checked, exactly (analysis phase, read-only during the soak)

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
   `tcp`, `sockets`, `verbs`, `psm` provider tokens. Runtime log:
   libfabric 2.4.0-impi loads; `ze fabric ports: 0` (the B70s are
   PCIe peers only, no fabric).
4. **Host state**: ACS 0 bits set (recount); IOMMU off (above);
   kernel cmdline bare (`ro` only); GPUs at `b1:00.0`/`da:00.0` behind
   per-socket bridges `b0:01.0`/`d9:01.0` (cross-socket UPI hop —
   baked into the 10.23 GB/s number); xpu-smi v2.0 present
   (discovery works; no link-stats subcommand on this build).

## Executed idle-window legs (2026-10-31, 14:55–15:22 UTC)

- **r71b — D2D peer-copy probe** (`b70_probe21_d2d.sh`):
  throwaway cert4 container, `--privileged`, `ZE_AFFINITY_MASK=0,1`,
  256 MiB fp16 both directions, 3 warm + 20 timed, median.
  **10.23 / 10.23 GB/s; peer-access True both ways.** Transcript:
  `r71b_d2d.txt`.
- **r71a — CCL diagnostic boot** (`r71a_ccl_diag.sh`, level `debug`):
  certified line + exactly one inserted env; health 200 in ~5 min; one
  request served; 415,094 CCL/transport lines captured; W14 restored
  by trap. Extracts: `r71a_extracts.txt`; full redacted logs:
  `r71a_ccl_diag_lines.txt.gz` (rig-raw md5 `032546041daf…`) +
  `r71a_serve_diag.log.gz` (rig-raw md5 `ff7ed22a7e82…`).
  **Instrument-defect note (attempt 1, banked)**: `CCL_LOG_LEVEL=2` is
  INVALID on this build — oneCCL 2022.0 takes enum names
  (`error|warn|info|debug|trace`, the env_parser.hpp:100 exception
  text is itself the evidence) and numeric `2` killed both workers at
  the first all_reduce. Attempt-1 log:
  `r71a_attempt1_invalid_numeric_level/serve_diag.log`. The LEVEL LAW
  is now baked into the script header (no-unevidenced law honored the
  hard way).
- **W14 end-state**: restored by the r71a EXIT trap at 15:22:48 UTC
  (`W14_RESTORED`, radiance-test up, health verified before banking).

## Interpretation matrix (resolved)

P2P confirmed on BOTH legs (the actual outcome) → **AR lever moves to
algorithmic/kernel territory** (own A/B legs, posture-change law; the
p67b ~30.4 % AR slice at 64 serialized hops × small decode payloads is
latency-dominated — the link has ~2 GB/s headroom at most and is not
the constraint). The D2D-bounce and degraded-link branches are moot.

## Files

`acs_validation.txt` (host validation transcript), `host_checks.txt`
(IOMMU/cmdline/sysfs/link reads), `ccl_env_surface.txt` (extracted env
names — captured subset + the exact re-extraction one-liner),
`r71b_d2d.txt` (D2D transcript + verdict), `r71a_extracts.txt`
(decisive CCL-diag lines + counts), `r71a_ccl_diag_lines.txt.gz` +
`r71a_serve_diag.log.gz` (full redacted logs; rig-raw md5s recorded
above), `r71a_first_resp.json` (the diagnostic-boot request),
`r71a_attempt1_invalid_numeric_level/` (numeric-level defect evidence),
`kit/` (byte-exact copies of the three instruments, also tracked at
`tools/pcie/`).

**No engine-code change; cert4 remains the certified image (no
cert5).**
