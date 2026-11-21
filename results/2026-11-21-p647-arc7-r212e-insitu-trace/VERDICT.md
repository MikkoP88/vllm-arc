# r212e — in-situ torch-profiler bracket: the kineto graph-blindness law

**Date**: 2026-11-21 · **Leg**: r212e (arc7 Phase-8) · **Stack**:
build21 wheel + mx_kv overlay (`19934f1c…` — the bytes that pass
`int(max_seq_len)` to the kernel, per the r212f audit) +
`vllm-arc:v0.30.0-b70-cert25zc` · **Run**: `r212e_launch1.log`
07:18-07:26 UTC · **Tool**: `tools/ar/p147_mxkv/r212e_prof.sh` (r67b
recipe: `--profiler-config.profiler=torch` flags + one-line
`gpu_worker.py` activities CUDA→XPU bind mount; `/start_profile`
bracket around one c16384 n1 face)

## Verdict

**PASS as a method leg; the in-situ kernel question is UNANSWERABLE
by kineto on this stack — XPU-graph replay kernels do not surface in
the trace.** The decode-phase step wall is confirmed to live inside
the replay; the profiler sees only the eager residue around it.

### What the trace shows (rank0; rank1 identical)

- Prefill phase (t≈2-6 s): thousands of events/s, ~1 s device-busy
  per second — the eager chunked-prefill machinery (gemm 3.02 s,
  allreduce family 1.34 s, gdn 0.31 s, FA-prefill
  `compat::…KernelFunctor` n=51 max 13.2 ms).
- Decode phase (t≈7-31 s, ~2 steps/s at ITL p50 **498.4 ms** — the
  r212b native class reproduced under the profiler, +0.1% vs 497.8):
  **exactly 128 kernel events and 4.8 ms device time per second,
  every second, flat**. Per 498 ms step: 64 events = the eager
  spec-decode machinery (`_rejection/_resample/_insert_resampled/
  _post_update/_compute_slot_mappings/…` all 1-6 µs) + one eager
  logits `gemm_kernel` (2.14 ms) + a small allgather (122 µs).
- Step timeline: verify tail (eager, ends ≈ t+131.2 ms in the
  sampled step) → 30 ms silence → `_prepare_decode_inputs` (t+161)
  → **~465 ms silence** → next verify tail. The silent wall after
  prepare_decode is the full-model XPU-graph replay: ~93% of the
  step, zero kineto events.

### KINETO GRAPH-BLINDNESS LAW (torch 2.13+xpu, this stack)

Kernels launched inside an XPU-graph replay do not appear as
`cat=kernel` events in kineto traces. Consequences for this
program: (a) the mx native decode kernel cannot be timed in situ
with the torch profiler while `FULL_DECODE_ONLY` graphs are on;
(b) r67b's device-side op-class table was a PREFILL table (its
counts match chunked-prefill passes — the decode contribution was
invisible there too); (c) any future in-situ decode kernel timing
needs either eager-mode decode (changes the measured thing — the
capture-freeze laws make eager a different posture) or a
driver-level trace (the r12b kernel-launch tracer class).

### Where this leaves the 40×

Combined with the same-day r212f no-op-falsification audit (the
r212c "fix" never changed the kernel argument), the replay wall is
exactly where the split-starved walk would live: the step's
~465 ms invisible graph wall ≈ 16 MX layers × ~29 ms/layer
serialized-walk class. The decisive causal test is offline:
r212f (`P204_MAXSK` int literal, cells ctx/m9/cap at c32768
eshape) — m9 blowing up confirms the kernel-arg chain end-to-end.

## Method notes

- The script's on-rig parse heredoc ran via `docker exec` without
  `-i` — stdin never reached python, `summary.txt` came back empty
  and the leg exited cleanly. Traces survived (docker-cp'd before
  container removal); re-parsed on the rig host (`summary.txt`,
  banked). Lesson banked: heredoc-into-container needs
  `docker exec -i`.
- Profiler overhead negligible: face ITL 498.4 vs 497.8 ms banked.
- Traces: `traces/rank{0,1}.*.pt.trace.json.gz` (4.8 MB each) +
  the API-server stub trace; `telemetry.txt` shows the trap
  visually — the log line says `max_seqlen_k=262400` (computed)
  while the call passed 9.

## Artifacts

`c16384_n1{w,p}.json`, `telemetry.txt`, `serve.log`, `boot.txt`,
`prof_ls.txt`, `summary.txt`, `gpu_worker_xpu.py` (rig-side patch
source), `traces/*.gz`. Rig: `/root/build/p647/r212e/`, log
`/root/build/p199/logs/r212e_launch1.log`.
