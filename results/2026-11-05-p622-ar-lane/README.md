# P6.22 — TP2 allreduce lane (owner directive 2026-11-05, task #61)

Item: close the last un-dispositioned prefill device slice — the TP2
allreduce family (~30.4 % of rank0 device time at U32k, r67b) — with
telemetry-first evidence (message-size census + device trace +
primitives ceiling), then either a product-form fix or an honest
negative close. Directive: code optimization only, no knob testing, no
hardware legs; certified image + full metrics + docs flip at the end
iff something serve-affecting ships.

Code delta (instrument): `vllm_gaudi/xpu/patches/ar_trace.py`
(ARC_AR_TRACE, silent-off diagnostics class — same registry-exempt law
as ARC_STEP_TRACE), registered in `vllm_gaudi/xpu/platform.py`
`_XPU_PATCH_MODULES`, documented in `docs/envs.md` diagnostics.
Rig kit: `tools/ar/{r94_ar_capture.sh, r95_solo.sh, ar_ceiling.py,
ar_summarize.py, trace_ar_fold.js}`.

## The starting ledger (deep-analysis phase, 2026-11-05)

U32k rank0 device time (r67b): GEMM 39.7 % (closed — fastest measured
route deployed, r40/r41), **AR 30.4 % (OPEN — this lane)**, FA 20.3 %
(closed negative P6.21c), GDN 4.7 %, norms/act/misc ~5.4 %, host path
closed at P6.20 (r83: device-compute-bound). AR transport proven clean
by p71 (direct P2P 10.23 GB/s both directions; every collective on the
device-kernel `topo sycl` path; zero host/offload) — config/transport
are DEAD by evidence; what remains is the ALGORITHM (message sizes
never captured — p67's outstanding note).

## r94a — the message-size census (attempt 3, survived a rig power cut)

A rig-wide power outage killed the window mid-[5] (after warm U8k +
profiled U32k face, before U64k). The line-buffered census JSONL
survived in the dead container's /tmp (docker cp from the corpse) —
the instrument's durability proof. Files:
`rig/r94a_census_p{1040,1152}.jsonl` (4.2 MB each, the two TP workers),
`rig/r94a_census_summary_attempt3.json`, `rig/r94_window_attempt3_powercut.log`.

Model dims (from the served checkpoint config, `rig/model_config.json`):
hidden_size **5120**, 64 layers (48 GDN + 16 FA, interval 4),
24 q / 4 kv heads, head_dim 256, MTP 1 layer, vocab 248320.

Census decode (all_reduce, window = warmup + U8k + U32k faces):

| bytes | shape (rows x 5120 x fp16) | count | class |
|---|---|---|---|
| 150,994,944 (144 MiB) | not rows*5120 (TBD from trace) | 54 | large misc |
| 83,886,080 (80 MiB) | 8192 rows | 131 | full-bucket forward |
| **76,677,120 (73.1 MiB)** | **7488 rows** | **528** | **the dominant prefill chunk AR** |
| 68,157,440 (65 MiB) | 6656 rows | 264 | partial-chunk AR |
| 15,779,840 / 15,380,480 | 1541 / 1502 rows | 132+132 | tail chunks |
| 2,621,440 … 0.08 KiB | decode/draft classes | ~15k | latency class |

Totals: 16,968 all_reduce + 536 all_gather_into_tensor calls,
**101.4 GB of all_reduce input bytes** in the window. Caller for every
model AR: `vllm.distributed.device_communicators.xpu_communicator:all_reduce`
(stock vllm's own communicator seam — the integration point for any
custom XPU AR).

Call chain (stock 0.30, verified from image sources):
RowParallelLinear -> communication_op.tensor_model_parallel_all_reduce
-> GroupCoordinator.all_reduce -> torch.ops.vllm.all_reduce (out-of-place
contract: "PyTorch custom ops do not support mutation ... so we always
make the all-reduce operation out-of-place", parallel_state.py:721)
-> XpuCommunicator.all_reduce = **`output = input_.clone(); dist.all_reduce(output)`**.
The clone is the out-of-place contract's price: an EXTRA 2S local DRAM
round trip per AR (~1.3 ms at 73.1 MiB) hidden in r67b's "elementwise
misc", not the AR family. CudaCommunicator pays no naive clone — it has
an optimized ladder (symm-mem AR with copy into registered buffers,
quick-AR, fi-AR, pynccl); XpuCommunicator is the naive fallback
(clone + c10d) with NO ladder. An out-of-place c10d path cannot avoid
the copy (all_reduce is in-place-only) — killing it requires the
product-form custom AR (registered symmetric buffers + one-shot peer
write + fused local add, writing the output directly).

Floor math at the dominant 76,677,120 B size: per-rank link traffic =
S bytes; at the 10.23 GB/s P2P rate ≈ 7.5 ms unidirectional, ≈ 3.8 ms
if the 2-rank ring's RS+AG phases use both link directions
concurrently; + local reduce ~1.3 ms (overlappable). r67b-derived stock
≈ 5.5-6 ms/call → the naive estimate says stock may already sit within
~1.3x of the link floor at prefill sizes — r95 decides with measured
numbers (the build gate is >= 1.3x on the AR slice).

## r94 — full capture (attempt 4, post-outage re-run)

Window (cert9 image, ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0, profiler=torch,
bind-mounted kit; log `rig/r94_window_attempt4.log`): instrument
installed in **4/4** engine processes (`ar_trace_install_count.txt` =
4, unchanged after the window); faces under profiler: warm U8k TTFT
**3.747 s** (cert9 3.76 — parity), U32k **17.552 s** (r67b 17.53,
+1.2 % — the kineto tax), U64k **39.504 s** (cert9 unprofiled 38.93,
+1.5 %). Census (attempt 4, both TP workers + draft): **18,717
all_reduce calls, 193.20 GB input bytes**; big calls (>= 8 MiB):
2,826 calls / 173.69 GB. Traces: rank0 kineto
`rig/rank0.1793879884978252561.pt.trace.json.gz` (16 MB gz), folded by
`tools/ar/trace_ar_fold.js` -> `rig/r94b_trace_fold.json`.

Rank0 device-kernel classes across the profiled window
(kernel total 51.818 s):

| class | device s | share | events |
|---|---|---|---|
| GEMM | 18.100 | 34.9 % | 5,636 |
| **AR family (ring+rc+barrier+misc)** | **14.129** | **27.3 %** | — |
| — `allreduce_large_su_ring_write_multi_kernel` | 6.757 | 13.0 % | 2,112 |
| — `ReduceCopyKernel` (oneCCL) | 6.787 | 13.1 % | 2,112 |
| — `oneccl_invoke_barrier` | 0.492 | 0.9 % | 2,128 |
| — misc ccl | 0.094 | 0.2 % | 4,634 |
| FA | 15.214 | 29.4 % | 272 |
| GDN | 2.063 | 4.0 % | 6,048 |
| norm / act / elementwise / other | 2.30 | 4.4 % | — |

Population fact (trace-vs-census): the trace carries only **2,414
eager `c10d::allreduce_` cpu ops** (linked 1:1 to ring kernels by
external id) while the census logged 18,717 calls — the ~15.9k decode/
draft ARs execute **inside XPU-graph replays** (no cpu ops, no Python).
Per-size attribution therefore uses **duration clustering with census
counts as fingerprints**, not order-merge (the fold's per-size table
is banked but known-unreliable for this reason; see
`rig/r94b_trace_fold.json` `per_size`).

Dominant-class decode (fingerprint n=1,584 per worker = the 76,677,120 B
class; ring clusters: 3.92 ms x1,584 / 2.0 ms x131 / 1.0 ms x132 /
0.5 ms x264): **ring 3.917 ms + ReduceCopy 3.933 ms = 7.85 ms per AR**
in situ. The dominant class alone is 6.20 + 6.23 = **12.43 s = 88 % of
the AR family** — the AR slice IS the 73.1 MiB prefill chunk AR.

## r95 — AR primitives ceiling bench

Posture: standalone throwaway cert9 container, same CCL env block as
serve, ZE_AFFINITY_MASK=0,1 (`tools/ar/r95_solo.sh`; sizes = the
dominant census classes). Two roles in `tools/ar/ar_ceiling.py`:

- **oneshot** (upper bound of a custom one-shot AR; one process owning
  both GPUs, concurrent per-device-stream peer staging copies + local
  adds) — measured (`rig/r95_oneshot.json`):

| bytes | class | oneshot ms | copies ms | GB/s/dir | add ms |
|---|---|---|---|---|---|
| 150,994,944 | 144 MiB misc | 30.966 | 30.063 | 5.02 | 1.391 |
| 83,886,080 | 8192 rows | 17.502 | 17.039 | 4.92 | 1.087 |
| **76,677,120** | **7488 rows (dominant)** | **16.053** | **15.590** | **4.92** | **0.990** |
| 68,157,440 | 6656 rows | 14.335 | 13.965 | 4.88 | 1.186 |
| 15,779,840 | 1541 rows | 3.783 | 3.624 | 4.35 | 0.811 |
| 2,621,440 | decode | 1.231 | 1.061 | 2.47 | 0.940 |

The decisive discovery: **concurrent** peer copies sustain only
**4.92 GB/s per direction** — the two torch copy engines SHARE ~10
GB/s aggregate over the link, HALF the 10.23 GB/s solo rate p71b
measured. The one-shot algorithm's floor is copies-bound: 2S bytes at
~10 GB/s aggregate = 16.05 ms at the dominant size, before any IPC
handles or completion protocol a real integration must add.

- **stock** (2-proc torchrun, the serve path in isolation): backend
  name discovery cost two attempts — the engine uses **`xccl`**
  (`vllm/platforms/xpu.py:111 dist_backend = "xccl"`; engine log
  `parallel_state.py:1827 backend=xccl`; natively registered by the
  Intel xpu torch 2.13.0 build — `backend_list` =
  fake/gloo/mpi/nccl/ucc/**xccl**), NOT "ccl" (attempt-1:
  `Unknown backend type ccl`; attempt-2: this image carries oneccl
  2022.0.0 runtime but NO `oneccl_bindings_for_pytorch` python module).
  Attempt-3 (`rig/r95_dist.json`, `rig/r95_solo3.log`; numerics PASS
  both ranks):

| bytes | stock dist ms | GB/rank/s | oneshot ms | stock faster by |
|---|---|---|---|---|
| 150,994,944 | 15.305 | 9.87 | 30.962 | 2.02x |
| 83,886,080 | 8.513 | 9.85 | 17.471 | 2.05x |
| **76,677,120** | **7.783** | **9.85** | **16.028** | **2.06x** |
| 68,157,440 | 6.922 | 9.85 | 14.344 | 2.07x |
| 15,779,840 | 1.614 | 9.78 | 3.770 | 2.34x |
| 2,621,440 | 0.332 | 7.90 | 1.206 | 3.63x |

  Stock sustains **9.85 GB/rank/s CONCURRENT** — each rank at ~96 % of
  the 10.23 GB/s solo rate simultaneously (~19.7 GB/s aggregate over
  the link) versus the torch copy engines' 4.92 GB/s/dir (9.85
  aggregate): oneCCL's ring drives both link directions at 2x the
  torch primitives' concurrent rate, uniformly across all sizes.
  In situ serve measurement (r94 trace, above): **7.85 ms** at the
  dominant size — **0.9 % agreement with the isolated microbench**:
  the measurement is closed three ways (microbench, serve trace,
  floor math).

## Verdict — lane CLOSED NEGATIVE (decision gate r95)

Pre-registered gate: build a custom AR iff the measured one-shot
ceiling leaves >= 1.3x headroom on the AR slice (ceiling <= stock/1.3
= 6.0 ms at the dominant size). Measured: **ceiling 16.03-16.05 ms vs
stock 7.78 (microbench) / 7.85 (in-situ trace) ms — stock oneCCL is
2.06x FASTER than the achievable torch-primitive ceiling**; the gate
fails by 2.66x. Family-level: measured
AR family 14.13 s vs naive unidirectional bytes floor **18.89 s**
(193.20 GB census bytes / 10.23 GB/s) — oneCCL already runs at 1.34x
the solo-rate single-direction bound via both-directions ring +
kernel overlap. A custom ESIMD kernel would need to out-pump
tuned closed-source oneCCL at pure link transfer — the exact race
P6.21c lost against the closed FA kernel (0.33x), with even less
compute to win on. The ReduceCopy inefficiency (3.93 ms vs the 0.99 ms
measured local-add floor) lives INSIDE oneCCL's closed .so — not
patchable through the c10d seam, and subsumed by the ring-rate
advantage anyway.

Remaining routes (all out of scope for this lane, banked for the
record): (a) quantized AR — halves the bytes but changes numerics;
owner has not named it; (b) AR/compute overlap — r83 proved the
device compute-saturated stream at prefill (nothing to overlap with
on-stream); (c) the out-of-place clone before every AR (the
XpuCommunicator seam) — measured inside r67b elementwise misc at
~0.4 s / 0.9 % of device — not a lane.

What ships from P6.22: the **census instrument**
(`vllm_gaudi/xpu/patches/ar_trace.py`, ARC_AR_TRACE — silent-off
diagnostics class, zero cost off, validated across a full serve window
AND a rig power cut), the rig kit (`tools/ar/*`), and this evidence
pack. Nothing serve-affecting changed -> **no cert10** (plan gate: no
image unless something serve-affecting ships); cert9 remains the
certified image, docs currency unchanged.

## Defect log (banked for reruns)

- attempt-1 r95 size picker took the six SMALLEST >= 1 MiB classes
  (`sorted(sizes)[:6]`) -> r95_solo.sh pins the dominant classes.
- attempt-1 r95 ran from /kit — `platform.py` shadowed stdlib `platform`
  via sys.path[0] (circular import death) -> copy to /bench.
- ar_ceiling.py numerics guards expected 5.0 but both roles produce
  a0+a1 = 3.0 (1+2); timings unaffected (banked attempt-1 runs carry
  numerics:false — known-bad check, correct arithmetic; attempt-2
  re-run: numerics OK at all sizes, timings reproduce within noise).
- ar_ceiling.py dist role backend name: "ccl" is NOT registered in
  this stack — the serve path uses **"xccl"** (natively in the xpu
  torch build). Attempt-1 `Unknown backend type ccl`; attempt-2's
  `oneccl_bindings_for_pytorch` import does not exist in the image
  (oneccl 2022.0.0 ships runtime libs only). Fixed to backend="xccl".
- ar_summarize.py `human_bytes` double-divided every unit >= KiB
  (76,677,120 B printed as 0.07 MiB) — fixed post-hoc; summary.json
  rows carry raw bytes (unaffected).

## Reproduce

1. Kit: scp `tools/ar/{ar_trace.py, platform.py}` (from the repo's
   `vllm_gaudi/xpu/`) + `tools/ar/{ar_ceiling.py, ar_summarize.py}` to
   `$R/p622/kit/` on the rig; `r94_ar_capture.sh` md5-logs the kit.
2. r94 capture: `bash r94_ar_capture.sh` (W14 stop -> cert9 serve with
   bind-mounted instrument + ARC_AR_TRACE=1 + profiler=torch ->
   warm U8k -> profiled U32k + U64k -> census + traces out; W14
   restore by trap). Fold census with `ar_summarize.py`, trace with
   `node trace_ar_fold.js` (16 MB gz traces need
   `--max-old-space-size=8192`).
3. r95 bench: `bash r95_solo.sh` (standalone short window; oneshot +
   dist roles at the dominant census sizes; W14 restore by trap).
   Verify restore before the next window: FRANCE + W13H_BOOT_DONE in
   the restore log + health 200.
4. Posture law: every window discards the probe container; cert9
   image untouched; never leave the rig on an arc image.

## Post-outage recovery record (2026-11-05)

Rig power loss at ~10:37 during attempt 3. Recovery: containers triaged
(all corpses, no arc image serving), D2D/interleave probe re-verified
post-reboot (10.22/10.23 GB/s both directions, peer access True — r71b
baseline intact), W14 restored via w14_boot.sh fullres (FRANCE +
W13H_BOOT_DONE + health 200 verified) BEFORE the attempt-4 window.
Census + model config extracted from the attempt-3 corpse and banked
above; the kineto trace did NOT survive (writes only at stop_profile —
known cost of an abrupt cut).
