# P6.20 — prefill-path optimization evidence (2026-11-04)

Item: eliminate host-side prefill bottlenecks and the U32k/U64k p95 gap
(plan §P6.20; owner directive: code optimization, telemetry-driven,
certified image at the end). Code delta: `vllm_gaudi/xpu/patches/lmcint/
vllm_v1_adapter.py` (L1 evsync + L3 off-thread store + L6 dispatch-late),
new `vllm_gaudi/xpu/patches/step_trace.py` (ARC_STEP_TRACE per-step host
timeline), `vllm_gaudi/xpu/platform.py` (step_trace registration),
`vllm_gaudi/envs.py` + `docs/envs.md` (registry rows).

## Legs (all on rig 10.20.3.REDACTED, cert8 image + bind-mount probe posture)

| leg | what | verdict |
|---|---|---|
| r80 | L1 evsync A/B (`VLLM_ARC_LMC_EVSYNC` on/off) + first step_trace attempt | **NEUTRAL** — on the in-order XPU stream a flush-time event sits behind the just-submitted chunk forward; instrument wrapped nothing (targeted hpu_* classes that never load on Arc) |
| r81 | L3 async-store A/B/A/B (l3on1/off1/l3on2/off2), p65 v5.1 faces per phase | **ADOPTED** — U32k 17.42/16.88 vs 18.13/17.65 (−0.74 s mean), U64k 38.95/38.94 vs 39.64/41.54 (−1.64 s), U8k −0.08 s both pairs; C32k mixed (cached face); zero ERROR all phases; store worker ACTIVE iff async |
| r82 | full gate battery on L3 (first run aborted at INSTRUMENT_V2 — lazy lmcint wrap imported the patch module, not the source-override core module; fixed, relaunched) | **GREEN** — instrument V2 all 9 segments; integrity 5/5 + det 3/3; run7n t0/t1 in-family; spec PASS; LONG-RETURN t0 faces 100 % clean; ONE flag: essay t1-seed pair |
| r82b | t1-seed re-characterization (cert7 annex method) + p65 + audit | **KNOWN_FACE** — 8/8 pairs identical, 16/16 single hash, t0 spot = boot-stable 648d66ae32c9, seed-999 control differs; ERROR=0/TRACEBACK=0; WARN set = stock + registerable env notice |
| r83 | full gate battery on L3+L6 (dispatch-late) | **FULL GREEN, no annex** — LONGRETURN_GATE PASS (4/4 probe verdicts CLEAN); spec at exactly cert8 banked values (pos1 0.9258, acc 0.837); p65 U32k 17.17 / C32k 1.11 / U8k 3.76 / U64k 38.92; ERROR=0 |

## Key artifacts

- `rig/r83/trace_mechanism.txt` — per-worker segment dists; L3 proof:
  `wfs_flush` (engine handoff) p99 = 0.1 ms, `wfs_body` (worker) p50 ~87 ms;
  chunk-step `wfs_total` p90 ~2.3 s.
- `rig/r83/trace_u64k_window.txt` — the lane-closing measurement: U64k
  window `wfs_total` waits 4.14→5.12 s per chunk (prefix-scaling growth)
  while worker bodies in-window are ~108 ms max and `execute_model` walls
  ~400 ms (submission) — **the wait is device-queue drain behind the
  just-submitted chunk forward; the face is device-compute-bound** (~89 %
  of TTFT inside wfs drains, ~11 % submits/gaps). Host path is clean; the
  residual U32k/U64k gap is chunk compute (r67b kernel mix: AR 30.4 % /
  FA 20.3 % of rank0 device time).
- `rig/r82/trace_mechanism.txt` — same table pre-L6 (stall attribution
  that motivated L6; r83 shows the stall persists but is device-bound).
- `rig/r82/t1seed_rechar.{out,json}` — the annex evidence.
- `rig/r8*/longprobe_*.json` — LONG-RETURN classes (cross-boot code/multi
  hash drift is the documented persistent-cache-path face; essay/simple
  boot-stable since the p619 fix).

## Disposition

L1 neutral (kept as cheap default inside the body); L3 adopted (default
ON, `VLLM_ARC_LMC_ASYNC_STORE=0` kill switch); L6 kept (restores pre-L3
store landing order, idle-worker mamba section, TTFT-neutral, gates
green); L6a (unpin deferral) + L6b (mamba-store queueing) NOT built —
zero TTFT value on a device-bound face. Remaining lever on the gap is
kernel-level compute (separate scope-law review). Step_trace ships
silent-off in the diagnostics class (`docs/envs.md`).
