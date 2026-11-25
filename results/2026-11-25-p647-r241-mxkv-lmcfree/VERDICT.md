# VERDICT — r241: the LMC-free re-measurement of the qwen MX-KV lanes (posture-confound correction leg)

**Date**: 2026-11-25 (leg 15:35–17:12 UTC; bank pulled the same day)
**Image**: `vllm-arc:v0.30.0-b70-cert28` (tag `rel/v0.30.0-b70-cert28`)
**fork_sha**: `43f84cf977f9444900ff21387715ca5865db4c1a`
**.so pins**: libattn `61d2f5a365f05084b3dd72c4ae1da3e0` / libmqa `190e261edc79b3b497d8d8e801e65757`
**Model**: qwen3.8-27b-fp8 (dense hybrid-GDN, MTP spec=3, maxlen 262144, seqs 64, graph capture ON)
**Gate script**: `tools/ar/r241_mxkv_lmcfree_cert28.sh` (refused on image / fork_sha / .so pins / p139_conc.py sha / serve script; W14 quiesced, EXIT-trap restored — `W14_RESTORED` + health 200 banked in `launch.log`)

## Purpose

r240b Finding 2 ("qwen packed-KV sustained-decode stability boundary")
was re-classified as posture-confounded on 2026-11-25: the r240b gate
script carried `ARC_LMC=1` into the MX cells via its `$QF8` base env,
against the certified LMC-free MX posture and each boot's own WARN,
and LMCache stored through both death windows (274/306 store lines).
This leg re-runs the exact r240b face lists under explicit
`ARC_LMC=0`, with two new gates so the confound can never recur
silently: a semantic per-face gate (n_ok/n_fail parsed;
CONC_ENGINE_DEAD discrimination) and a posture refuse (any LMCache
line in a serve log aborts the leg, exit 3).

## VERDICT: CLEAN — Finding 2 is retracted in full

- **Census (launch.log:99)**: `R241_CENSUS real_faces=68
  dead_or_partial_faces=0` — every face in all three grids served.
- **Every r240b death face survived LMC-free**, full n_ok:
  - mxfp4-KV (kvm4) 32K × n4 — n_ok=4, TTFT mean 35.6 s (the exact
    r240b silent-hang face)
  - mxfp4-KV (kvm4) 32K × n8 — n_ok=8, TTFT mean 63.2 s
  - mxfp4-KV (kvm4) 64K × n8 — n_ok=8, TTFT mean 138.1 s, per-user
    min 2.5 tok/s (slow but alive; the r240b watchdog death)
  - mxk8v4 (kvmk) 32K × n4 — n_ok=4, TTFT mean 36.2 s (the exact
    r240b `UR_RESULT_ERROR_DEVICE_LOST` face)
  - mxk8v4 (kvmk) 32K × n8 — n_ok=8, TTFT mean 63.2 s
- **Posture proof (per-cell `posture.txt`)**: `lmc_lines=0` at boot
  and `lmc_lines_end=0` at leg end for all three cells; `tracebacks=0`
  everywhere; `R241_POSTURE_DEFECT` never fired.
- **Capacity is the only boundary** (the retraction's positive half):
  pools kvm4 721,994 / kvm8 386,484 / kvmk 504,832 tokens; mxfp4-KV
  serves 128K context at n1–n4 (34.6 / 16.3 / 7.0 tok/s per-user) —
  a packed-KV capacity record; kvm8/kvmk serve 128K at n1–n2. The
  grids end where the pool ends, not where the engine dies.

## What stands and what changes

- **Retracted**: the ≥16K-context/low-concurrency restriction on
  4-bit-V packed KV. The lanes are capacity-limited only.
- **Unchanged certified law**: LMCache OFF on every MX-KV lane —
  the posture the r240b cells violated is the one the r241 cells
  prove correct. P-D (next phases) hardens this as a boot-time
  refusal so the combination cannot ship silently again.
- **Unchanged, separate class**: the gemma hd512 MX-KV first-decode
  refusal (r240b `gkv_m*` cells) is a kernel-instantiation gap, not
  a posture effect — P-C owns it. Not fixed by this leg or cert28.
- The r240b bank and its artifacts stand as banked; its VERDICT
  carries the correction addendum pointing here.

## Evidence (this bank; SHA256SUMS covers the data files)

- `launch.log` — refuses (line 1), per-boot serve lines, 68 ×
  CONC_OK, census + `R241_COMPLETE 2026-11-25T17:12:14Z`,
  `W14_RESTORED` + health 200.
- `kvm4/ kvm8/ kvmk/` — per-cell `posture.txt` (pool, lmc_lines,
  lmc_lines_end, tracebacks), `boot.txt`, `conc.log`,
  `p139_conc_<cell>_c<ctx>_n<n>.json` faces (68 conc + t1.0 solos),
  serve logs.
- Rejected-vehicle note: the conc-face JSONs carry TTFT / per-user /
  aggregate / n_ok fields; the conc harness banks no per-request ITL
  percentiles (solo-face field), matching the r240b/r240c conc faces.

## Post-leg actions

Reports regenerated from the r241 bank (MX KV lanes owned by it via
the generator's cell-ownership law); PERFORMANCE/STATUS/CHANGELOG
passages moved from "provisional pending r241" to "retracted, measured
clean"; generator + overlap-audit tooling landed with this bank.
