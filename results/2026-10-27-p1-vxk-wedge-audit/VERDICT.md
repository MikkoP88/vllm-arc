# VERDICT — p1-vxk-wedge-audit (2026-10-27)

**AUDIT: NOT CLEAN — §6 contingency invoked (v88-lineage wheel rebuild +
device-gate). Triggered by owner question mid-p2r3e; the §10 P1 gate row's
wedge-audit item had never been executed/recorded (P1 VERDICT gate table
omitted it).**

## Question asked

"Check the p2r3 first-request death is not caused by the vxk wedge-site
issue (§6)."

**Answer: it is NOT.** Five independent discriminators:

1. **Error signature.** p2r3 defect-b raises L0 API-state RuntimeErrors
   ("wait method cannot be used for an event associated with a command
   graph"; "wait cannot be called for a queue which is recording to a
   command graph") in the event/copy path. The wedge defect produces
   silent state corruption or UR_RESULT_ERROR_DEVICE_LOST inside the conv
   kernel — never a clean Python RuntimeError about stream/event state.
2. **Trigger is a state query, not a kernel.** The p2r3 defer fires when
   `torch.xpu.is_current_stream_capturing()` returns True — before any
   copy or GDN kernel in that path. An int32 pointer overflow inside a
   kernel cannot set a stream capture flag.
3. **Overflow condition unreachable at the failure point.** The wedge
   fires at state-pool row id ≥ 4097 (stride 524,288 × id in int32; 4096
   wraps to INT32_MIN). The p2r3 deaths occur on the FIRST request after
   boot — the pool is nearly empty, row ids are tiny.
4. **Boot captures run the same conv kernels 27× cleanly** (FULL campaign
   27/27 + trailing device syncs succeed). Wedge corruption would manifest
   during the campaign or under load, not exactly at the first AsyncOutput.
5. **Same wheel serves correctly** on the p2r3e boots (coherent text,
   health 200, 5-tok and 832-tok requests) — defect-b tracks a per-boot
   stream-capture state, not a pool-row-dependent corruption.

## What the audit found (the real result of checking)

- Image pins `vllm-xpu-kernels==0.1.14.1` (PyPI wheel; Dockerfile stack
  swap, P0 venv ground truth).
- PyPI publishes only a bdist_wheel for 0.1.14.1 (no sdist), uploaded
  2026-08-28.
- GitHub `v0.1.14` tag: **6 bare int32 sites** (2/file —
  causal_conv1d.hpp:232,452; chunk_causal_conv1d_xe2.hpp:203,627;
  chunk_causal_conv1d_tiled_xe2.hpp:220,585), zero `static_cast<int64_t>`
  (audit_site_grep.txt).
- GitHub commits API: zero commits touching those headers 2026-08-01 →
  2026-09-15; the 0.1.14.1 release (2026-08-28) sits inside the window →
  its source is the v0.1.14 tag's → **the shipped wheel is bare**.
- v0.1.15 and (by tag list) 0.1.16.x: also bare — the fix never landed
  upstream through the latest release.
- Exposure: stride(0)=524,288 (1-MiB padded pages) overflows int32 at
  id ≥ 4097; mamba_pool_ratio 2 pushes dense qwen3.8-27b rows past 4096
  (plan §6); every real prefill ≥ 8 tok takes the tiled path. This is a
  silent-corruption / DEVICE_LOST landmine for P4 integrity and the P5
  soak — must be fixed BEFORE any P2 banking.

## Consequence (plan §6 contingency — expected "audit clean" falsified)

1. `external_patches/vllm-xpu-kernels/` — ported patcher (6-site int64
   widen, `[VXKI64]` mark) + torch-matched build script (torch 2.13.0+xpu,
   version 0.1.14.1+arc1). Donor kit read-only import, provenance in its
   README.
2. Device-gate (all before the image pin swaps): donor repro cap727.pt
   LO+HI+BND noseed AND seeded → exit 0 CLEAN; then on the rebuilt image
   serialized-24 + burst_harsh ×3, zero new dmesg "Engine reset"; k1k
   sanity unchanged.
3. Wheel leg dir with the gate evidence; Dockerfile/pins.md5 swap; image
   rebuild; P1 record amended.

## Discrimination note for the p12 branch

The p2r3 zombie investigation (defect-b, command-graph event family)
continues unaffected — different defect, different fix
(graph_safe_output.py). Both must land before the P2 gate banks.

## Artifacts

env.txt (method + sources), audit_site_grep.txt (raw site greps v0.1.14 /
v0.1.15), summary.txt. Wheel-build + device-gate legs follow in
results/2026-10-27-p1b-vxk-wheel/.
