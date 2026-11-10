# P6.35 — End-to-end telemetry: decode/prefill/TTFT root-cause (leg A banked)

Owner directive 2026-11-10: comprehensive telemetry end-to-end to locate
root issues in **Decode, Prefill, TTFT**; plan every improvement; full
validation matrix (gemma-4-31B + qwen3.8-27b × MTP on/off × FP8/MXFP8/
MXFP4) at certification.

## Method (leg A, 2026-11-10)

Probe image = cert22 + ARC_P135 wrappers (p66b pattern: host-side python
patcher `tools/ar/p135/patch_p135a.py` → docker cp → commit
`vllm-arc:v0.30.0-b70-cert22-p135probe`). Certified posture: TP2,
`ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1`, qwen3.8-27b-fp8. Wrappers: file-append
O_APPEND `/root/p135_steps.log` (engine + both TP ranks; line-atomic),
walls on STEP/SCHED/PINP/PATT/META/PMAM/RFG/RPW/EXEC/WEX/SAMP/SMPL/WSM +
torch.profiler XPU window armed by `/root/PROF_ON` (N execute_model calls;
rank-lost-race noted below). Faces: solo streaming completions, seeded
random-token prompts (arc_pybench law), ignore_eos.

Leg dir (rig): `/root/build/p635/legA-20261110-053607`; kit:
`tools/ar/p135/` (patch_p135a.py, p135_face.py, p635_gate1.sh,
trace_summary.py, p135b_replay_bench.py, p135d_instantiate_bench.py,
p135e_isolated_bench.py + p135e_run.sh [W14-stop/restore wrapper]).

## Artifacts (sha256 in leg dir SHA256SUMS)

| file | sha256 (head) | note |
|---|---|---|
| p135_steps.log | 83bc4171… | 8,686 wall lines, all 3 faces |
| p135_prof_rank1.json | db34e3ab… | 28.0 MB chrome trace, U32k-decode window (2nd window overwrote 1st) |
| p135_face_u8k/u32k/u8kb.json | — | client ITL series |
| metrics_warm/u8k/u32k.txt | — | /metrics scrapes |

## Findings

### Decode: HOST-bound at 2.25× the device floor (root cause pinned)

Steady solo decode (MTP k3, acceptance ≈ 3.85 tok/step):

| metric | value |
|---|---|
| step wall (client p50 = STEP p50) | **40.5 ms** |
| tokens/step | 3.85 → ~95 tok/s solo (10.5 ms/tok; matches cert22 battery — probe did not regress serving) |
| device busy (32-step XPU window) | **27.7%** (574 ms kernels / 2,074 ms span) |
| device per step | ~18 ms ≈ weight-streaming floor (TP2: ~13.5 GB/rank @ 589 GB/s ≈ 23 ms envelope incl. L2 hits) |
| device-idle gaps per step | **29.34 ms** + 5.78 ms + 2.94 ms |

Kernel mix (rank1, U32k window): gemm 282.7 ms (49.2%, ~8.8 ms/step),
AR (ring+ReduceCopy+barriers) ~165 ms (~5.0 ms/step), GDN attention
36.2 ms (1.1 ms/step), norm/act 15.9 ms, sampler kernels **0.38 ms total**.

Gap placement (kernel-bracketed, every step, perfectly periodic):
`…oneccl_allreduce_pcie (logits) → _prepare_rope_positions_kernel →
[29.34 ms IDLE] → VectorizedGatherKernel<long> → lm_head gemm 2.14 ms →
oneccl_allgatherv (vocab-parallel logits) → rejection/resample/postprocess
kernels (µs each) → [2.94 ms] → _prepare_decode_inputs/slot_mappings →
[5.78 ms] → next-step input-prep burst → verify layers`.

With async scheduling this means the host path per step is ~40 ms of
nearly-pure Python: sample_tokens glue (sampler + draft propose +
postprocess, ≈29 ms) + scheduler/input-prep (≈6 ms) + replay enqueues
(~1 ms), while the device needs ~18 ms and idles the rest.

### Falsified suspects (evidence-banked, do not re-litigate)

1. **torch XPU-graph replay overhead** (the PR #188874 watch item) —
   EXONERATED TWICE, second time with a correction of our own bench:
   leg-A micro-bench `p135b_replay_bench.py` measured 609 µs @ 135 nodes
   (serving-scale replays ≈ 0.6 ms). A later scale extension of the same
   bench (tight replay loop, W14 serving on the same GPUs) appeared to show
   linear ~4.8 µs/node host cost (28.8 ms @ 6,000 nodes ≈ the serving RFG
   wall) — but `p135d_instantiate_bench.py` + `p135e_isolated_bench.py`
   (idle device: W14 stopped for the bench window, isolated single calls
   on an empty queue) falsified that: **`replay()` host cost is 7.1–7.6 µs
   FLAT at both 1,000 and 6,000 nodes**, for `keep_graph=False` (serving
   path via the `torch.cuda.CUDAGraph = torch.xpu.XPUGraph` shim) and
   `keep_graph=True` + explicit `instantiate()` alike; first-replay
   auto-instantiate = 4.6/31.8 ms (one-time, linear in nodes — the only
   linear term, paid at capture not per step). The earlier linear numbers
   were contention/backpressure artifacts. Also: `raw_xpu_graph_exec()`
   returns an opaque int ("XPU doesn't provide APIs to manipulate this
   object") — no direct-launch shim exists in torch 2.13, and none is
   needed. The serving RFG wall of ~28 ms therefore lives in
   run_fullgraph's other layers (offloader sync, our hygiene wrapper, or
   launch-blocking on busy queue state) — leg F splits exactly those
   (XRP torch-level wrap vs RFG vs SYN).
2. **Our graph hygiene wrapper** (`graph_safe_output.py` `_arc_replay` →
   `_hygiene` ×2 per replay): `is_current_stream_capturing` = **0.17 µs**;
   dezombie path returns immediately when healthy. ≈ 1 µs/replay. Not ours.
3. **LMCache offloader sync** (`sync_prev_onload`): no weight offload in
   posture → wait_stream no-op. Not it.
4. **Device-side sampler**: 0.38 ms total kernels. The sampler cost is
   host-side Python + eager small collectives
   (`oneccl_allreduce_pcie` appears eagerly in the sampler path).

### Prefill / TTFT: the engine is NOT the TTFT

| face | TTFT | engine exec (chunks) | non-engine |
|---|---|---|---|
| U8k (8,192 tok) | 4.549 s | ~1.0 s (7,488-chunk EXEC 867 ms + 1,094-chunk + first decode) | **~3.5 s** |
| U32k (32,768 tok) | 15.094 s | ~4.3 s (4 chunks; 8.6 tok/ms device-side) | **~10.8 s** |

Device prefill rate 8.6 tok/ms per chunk; the inter-chunk + tokenize +
queue + client path dominates TTFT. Leg C (ADDR/SCHED epoch-ts brackets +
standalone tokenizer bench `p135_tok.py`) decomposes the non-engine slice.

### Probe defects (noted for future legs)

- `/root/PROF_ON` trigger raced between TP ranks (rank0 ARM_ERR, rank1
  won; trace = rank1 only — sufficient, ranks symmetric). Future: per-rank
  trigger files.
- Second profiler window overwrote the first (same filename) — surviving
  trace = U32k window; U8k walls equivalent (40.5 ms p50 both).
- EXEC/PINP lines log `r?` — V2 GPUModelRunner has no `rank` attr
  (Worker lines do).

## Improvement plan (owner: "comprehensive plan for every improvement")

D-decode host-stall lanes (one fix per leg, gated e2e, upstream-shaped):

1. **P6.35-D1 — eager-collective + sampler/draft host path**: leg C
   brackets split the 29 ms (propose vs sampler vs postprocess vs CCL
   calls); fix = capture the eager per-step collectives + sampler/draft
   region into the FULL graphs (precedent: verify graph already holds AR
   ring kernels; our speculator_full_capture patch already forces FULL
   for the draft manager) + steady-state metadata caching at constant
   decode shapes. Target: host step 40.5 → ~20 ms → decode ~95 → ~150+
   tok/s solo (ceiling: device floor 18 ms ≈ 222 tok/s).
2. **P6.35-D2 — input-prep/scheduler python** (5.78+2.94 ms gaps):
   batch/fuse the per-step prep op storm (fused prep kernels exist;
   extend coverage), cache constant-shape metadata.
3. **P6.35-T1 — TTFT non-engine slice** (~3.5 s @ U8k, ~10.8 s @ U32k):
   leg C tokenizer bench + request-path ts → then: tokenizer fastpath /
   chunk-scheduling overlap / detokenizer streaming fix, whichever the
   split names. Device prefill (8.6 tok/ms) is near its own envelope
   (chunk EXEC 867 ms/7,488 tok at 27B TP2 is the weight+KV floor for
   chunked prefill; secondary target: chunk-gap overlap).
4. **P6.35-V — full validation matrix** (owner-specified): gemma-4-31B
   (no MTP) × {fp8, mxfp8, mxfp4} + qwen3.8-27b × {MTP k3, MTP off} ×
   {fp8, mxfp8, mxfp4} — arc_pybench battery both models; cert23 if any
   engine code changes; all docs + both remotes + gh release
   (cert-release-sequence law).

Answer to owner's Rust question (2026-11-10): solvable in Python —
upstream runs this architecture on CUDA at 5–15 ms/step host; our 40 ms
is stack-specific (eager CCL Python calls, per-op glue, metadata
rebuilds), not a language wall; a Rust engine would violate the fork
model. Evidence above.

## Leg F (gate2b, 2026-11-10): full attribution + root cause PINNED

Probe `p135fprobe` = cert22 + 13-file telemetry (leg C set + XPUGraph
torch-level replay wall XRP w/ graph id + AutoRegressiveSpeculator
propose/_prefill/_run_model/_multi_step_decode/_fused_multi_step_decode
ARP/ARPF/ARRM/MSD/FMSD + prepare_prefill_inputs/prepare_decode_inputs
PPF/PDC + offloader SYN/GETO + batch_shard BSA/BSG/BSL + race-free
PROF_ON). Boots: SPEC=3 (certified) and SPEC=0 (MTP off). Leg dir (rig):
`/root/build/p635/legF-20261110-072856`; local `legF/`.

### Decode decomposition (steady solo decode, per-rank max view)

| | SPEC3 (MTP k3) | SPEC0 (no MTP) |
|---|---|---|
| step wall p50 | 40.47 ms | **29.90 ms** |
| WEX (execute_model) max | 28.46 | 27.70 |
| — XRP max (torch replay) | **~24 ms** | **~23.6 ms** |
| RFG−XRP (hygiene+offloader+glue) | ~0.15 ms | ~0.15 ms |
| SYN (sync_prev_onload) | 0.01 | 0.00 |
| WSM (sample_tokens) max | 12.28 | 2.63 |
| — ARP propose | 8.16 | — |
| — — FMSD (fused draft steps) | 5.11 | — |
| — SAMP | 1.63 | 1.08 |
| SCHED | 0.15 | 0.14 |
| engine residual (STEP−WEX−WSM−SCHED) | ≈ 0 | ≈ 0 |

Engine thread 100% accounted: `STEP = WEX + WSM + SCHED`. Scheduler and
prep python are SMALL (leg-A's ~6 ms estimate was wrong; PINP+PATT+META
+PMAM ≈ 4.5 ms combined). MTP draft tax ≈ 10.6 ms/step (WSM 12.3 vs 2.6)
— real but secondary: **the dominant term in BOTH postures is one
XPUGraph.replay() call blocking ~23–24 ms.**

### Root cause (micro-bench chain, all banked in this dir)

| bench | result | falsified/supported |
|---|---|---|
| p135d (W14 contended) | inconsistent "linear" numbers | contention artifact — corrected by p135e |
| p135e (idle device, isolated calls) | replay host cost 7.1–7.6 µs FLAT @ 1k/6k nodes, both keep_graph modes; first-replay auto-instantiate 4.6/31.8 ms (one-time, linear) | per-node replay cost: FALSIFIED; raw_xpu_graph_exec = opaque int (no direct-launch API) |
| p135g (2-rank xccl) | AR-in-graph isolated 14.4 µs; AR replay w/ peer held 200 ms = 15.3 µs; tiny-after-big backlog 7.6 µs | ccl-in-graph sync, peer rendezvous, small-launch queue throttle: ALL FALSIFIED |
| p135h (single rank) | big(20k-node)-after-big launch = 7–9 µs; empty-queue big launch 15.5 µs | queue-depth throttling for big launches: FALSIFIED |
| **p135i (single rank)** | **same-exec re-launch while in flight = 13,957 µs (blocks until its own prior launch retires); two execs sharing ONE pool alternating = 8–12 µs** | **ROOT CAUSE: torch 2.13 XPUGraph.replay() on the same exec blocks until that exec's prior launch retires** |

Serving fit: async scheduling is AUTO-ON in core 0.30 (rig-verified
2026-10-28, plan log) → the engine runs one step ahead → every step's
verify replay lands on the SAME exec still executing the previous step →
the replay call blocks ≈ the graph's own device time (~23–24 ms; matches
XRP p50 23.6 SPEC0, p90 24.4 SPEC3; matches the old r24 probe's
"~3.5 ms fixed per-replay overhead" on tiny draft graphs; matches leg-A
device busy 27.7% — host and device fully serialized through the call;
the 29.34 ms "idle gap" = the host-only remainder). Tokenizer bench
(p135_tok): 32k warm encode 124 ms — tokenization is NOT the TTFT slice.

### D1 fix (leg J, gate2c, 2026-11-10): double-buffer WORKS — the wait MOVES

p135i B8's fix shape, implemented as probe patch `_arc_db_capture`/
`_arc_db_rfg` on base CudaGraphManager (both Model and Speculator
managers funnel through base capture/run_fullgraph): capture every graph
TWICE into the same shared pool, ALTERNATE replays per desc. Boot cost
+29 s (6 arc-db capture passes, serve_full.log). Leg dir (rig):
`/root/build/p635/legJ-20261110-081404`; local `legJ/`.

**Launch-level validation (p135_steps_spec3.log, 119 steady steps):**

| metric | leg F (blocked) | leg J (double-buffer) |
|---|---|---|
| XRP max/step (verify replay) | ~24 ms | **0.37 ms** |
| XRP id histogram | 1 id per desc | **12 ids × 59 = perfect G1/G2 alternation** |
| WEX max/step | 28.46 | 6.84 |
| WSM max/step (propose+sampler) | 12.28 | 5.69 |
| — ARP propose (inside WSM) | 8.16 | 1.76 |
| engine serial work (WEX+WSM+SCHED) | 40.8 | **~12.7 ms** |
| step wall p50 | 40.47 | **40.43 (unchanged)** |

Replay no longer blocks; ALL tagged engine work now finishes ~9–10 ms
into the step. **But the step wall did not move**: engine residual
(STEP−WEX−WSM−SCHED) p50 = 31.81 ms, all of it in the untagged tail
after `WSM` (log ts-resolution 1 ms; offsets: every tag ≤ +9 ms).

**Device-side truth (p135_prof_spec3_rank0/1.json, 38 MB each):**
the 7.35 s profiler window caught prefill tail (~100 ms kernels/100 ms
bin = device-saturated) then steady decode from ~6.2 s: **29.3 ms
device-idle gaps repeating at 40.5 ms period; device busy only ~11
ms/step (27%)** — same 29.3 ms gap signature as leg A. Host AND device
idle together ~29 ms per step.

**Root cause #2 pinned at the L0 driver level (rank1 trace):**
`zeEventHostSynchronize` — 20 calls of **39.6–40.2 ms each** (one per
step, 744 ms of a 1.15 s window) on **tid=1270, a second thread** in
the worker process (all aten work is tid=1030). This is
`AsyncOutput.get_output()`'s `copy_event.synchronize()`
(vllm/v1/worker/gpu/async_utils.py:168; `torch.cuda.Event(blocking=True)`
sleep-event; D2H enqueued on a side `copy_stream` that first
`wait_stream(main_stream)` then records `copy_event`) — the r22a
pattern at driver resolution. Fast paths in the SAME trace:
`zeEventQueryStatus` ~3.5 µs/call, `zeCommandListHostSynchronize`
(stream sync) ~1.05 ms/call. Also banked: 1.87 s eager `xccl:all_reduce`
host block at a prefill chunk boundary (rank skew — TTFT lane input),
95 ms `aten::copy_` at prefill→decode transition.

Falsified by leg J: "step is device-bound at ~40 ms device path"
(device busy is 11 ms/step); "double-buffer alone lifts decode"
(it removes the block, does not move the wall).

### Leg K (gate2e, 2026-11-10): waiter-mode A/B — the wait is REAL work

Single variable — HOW the waiter waits (baked per image, ARC_EVT_SYNC):
`busy` = poll `copy_event.query()`; `stream` = `copy_stream.synchronize()`;
control = leg J (sleep-event). Images p135kbusy/p135kstream from the
p135j base; walls CEVT/GETO/EVTS (+tid). Leg dir (rig):
`/root/build/p635/legK-20261110-084845`; local `legK/`.

| | leg J (sleep) | K busy (spin) | K stream |
|---|---|---|---|
| step wall p50 | 40.43 | 42.24 | **40.41** |
| ITL p50 (client) | 40.50 | 42.26 | 40.48 |
| face total_s | 14.50 | 11.69 | 11.09 |
| CEVT p50 (the wait) | ~40 (driver) | 23.5 | 34.7 |
| XRP max (db active) | 0.37 | 1.39 | 1.82 |

**Verdict: the event is NOT late.** Spinning on `query()` still waits
23–27 ms (the event signals when the device finishes); stream-sync
returns the step to exactly the same wall. busy is slightly WORSE
(spin steals host CPU from the engine thread's python path). The
40 ms `zeEventHostSynchronize` was the symptom holder: it waits out
the device chain.

### The trace-blindness correction (changes the leg-A reading)

Leg-J decode-region kernels: **62/step, 1 gemm/step, 2.43 ms busy —
all eager glue** (elementwise/index/scatter/slot-mappings). The
graph-replay kernels are INVISIBLE to kineto in db mode (whole trace
7121 kernels < 10 visible steps' worth; host launch enqueues 63/step =
eager only). The "29.3 ms device-idle gaps" of leg A/J are where the
graphs actually run. Leg-A's "27.7% device busy" was measured in the
BLOCKED posture where kineto still saw graph kernels; the db posture
runs them off-trace.

**Final db-mode step decomposition (40.4 ms):**
- device chain ≈ 30–32 ms: verify graph ~23 ms (= TP2 fp8 weight-stream
  floor: 13.5 GB/rank ÷ 589 GB/s) + draft graphs ~7 + eager tail 2.4
- exposed host tail ≈ 8–10 ms (scheduler must see step N's output
  before scheduling N+1 — one-step-lookahead limit; host serial 12.7 ms
  total, ~4 ms overlapped)
- ⇒ **leg J/K sit AT the fp8 pipeline floor (~40 ms ≈ 105 tok/s solo
  spec3 ceiling).** D1 lifted device utilization 27.7% → ~80% and is
  REQUIRED for any further gain to show; decode throughput beyond
  ~105 tok/s now needs POSTURE (fewer weight bytes / more tokens per
  read), not host fixes.

### Next lanes (priority order)

1. **P6.35-P posture sweep** (stock cert22, owner-matrix core):
   spec{3,5} × quant{fp8, mxfp4/W4A16} solo faces — W4A16 quarters
   weight bytes (floor 23 → ~11.5 ms) and k5 raises tokens/read;
   both certified lanes (cert20/cert11). Winner posture → P6.35-V
   full matrix + cert23.
2. **D1 ship**: double-buffer as `vllm_gaudi/xpu/` patch (probe
   validated; boot +29 s; alternation proven; necessary enabler).
3. **D2 host-tail polish** (~8 ms exposed → step ~36–38 ms; ≤ +10%).
4. **T1 prefill rank skew**: 1.87 s eager xccl all_reduce host block
   at a chunk boundary (leg-J trace) — TTFT lane.
