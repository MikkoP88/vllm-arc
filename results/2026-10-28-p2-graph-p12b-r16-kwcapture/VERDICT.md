# VERDICT — r16 kwargs-capture + preamble bisection: DEFECT-b MECHANISM IDENTIFIED. The stream wedges INSIDE the first eager LARGE TP allreduce (embedding AR of the first real request) after decode-graph capture warmup; the GDN kernel, projection gemms, and graph capture/replay are ALL exonerated by real-kwargs standalone replay. Model v3.

## Hook lineage (v1→v7, each failure a lesson banked in the next)

- v1: metadata gate at forward_xpu entry never opened — attn_metadata is
  None at entry during dummy profiling AND the op early-returns when it is
  None (the SYCL kernel never runs during dummy profiling — why warmup
  never flips).
- v2: poll-thread wrap missed workers — vllm v1 workers are FORKED from
  EngineCore before model import; threads die at fork. Fix: fork-safe
  sys.meta_path finder (_WrapOnImport).
- v3/v4: budget eaten by per-layer calls + capture-warmup sweep →
  layer-0-only + tok>512 filter.
- v5: real-kwargs save DIED — torch.save's blocking D2H wait raised
  `RuntimeError('wait method cannot be used for an event associated with a
  command graph')` → the default stream's queue was ALREADY in command-graph
  capture state at the SYCL runtime level when layer-0's save ran. Onset
  stack showed the forward thread inside in_proj_ba (plain F.linear) BEFORE
  this step's GDN op.
- v6: side-stream CPU staging (stream sync, never an event on the poisoned
  queue), two-phase save (entry e / post p), CALL print at entry. Landed
  full captures: real hidden (832x5120), module weights (fp8 in_proj),
  conv/A_log/dt_bias, pool geometry (611x3x5120 conv, 611x24x128x128 ssm),
  real step metadata (dict[64]: qsl [0,832], state_indices [0],
  non_spec_token_indx=None — schema-Optional), recomputed qkvz (832x8192) /
  ba (832x48).
- v7: preamble probes (VocabParallelEmbedding.forward +
  tensor_model_parallel_all_reduce with namespace sweep) — the bisection
  that named the op.

## Facts (all from this leg, arc-test container, r13 posture serve:
## mode=NONE + FULL_DECODE_ONLY + sizes 1..320, TP2, graphs ON)

1. **Timeline** (v6 CALL prints): dummy profile tok=8192 pre_rec=False
   (clean, op early-returns, pools unbound) → capture warmups (kernel runs
   there, clean) → health → first real request layer-0 entry:
   **pre_rec=True ALREADY** — poison precedes the first GDN forward of the
   step (v5's in_proj_ba stack was downstream of the poison, not the
   poisoner).
2. **Smoking gun** (v7, smoking_gun_bracket.txt, both ranks):
   `AR entry (832,5120) 8519680B pre_rec=False` → `AR exit pre_rec=True`
   → `EMB exit pre_rec=True` → `CALL layer-0 pre_rec=True` →
   ARC_GRAPH_ZOMBIE_ONSET. **The recording wedge happens INSIDE the first
   eager xccl allreduce ≥ ~8 MB** (embedding AR of the first real request),
   executed after decode-graph capture warmup. Stream clean at entry,
   recording at exit.
3. **Small-first vaccine** (controlled pair, hook-active fresh boots):
   - LARGE-as-first (832 tok, AR 8.5 MB): flip + HTTP 500 + pre_rec=True
     — reproduced twice (v6 leg, A2 leg); SYCL_CACHE_PERSISTENT=0 does
     NOT cure it (leg B) — arcitech's first-request segfault is a cousin,
     not our mechanism.
   - SMALL-as-first (954 chars ≈ 250 tok, AR 2.6 MB): HTTP 200, zero
     onset — and the SAME 832-token request afterwards returns 200
     (pre_rec=False) plus further requests keep working (n=1 boot; the
     product fix below re-validates at gate time).
4. **Standalone exonerations** (r16_replay{,2}.py, logs banked):
   - REAL-kwargs GDN prefill op: CLEAN with graphs and without, both
     attempts, twice each (r15's synthetic-kwargs DEVICE_LOST was an
     artifact — kernel lane for defect-b stays CLOSED).
   - Real projection gemms (shape-fidelity for fp8 weights), decode-shaped
     GDN graph capture+replays, trivial graphs: clean.
   - 2-proc xccl chain (torchrun, ccl env mirrored): eager AR → completed
     graph + replays → eager 8.5 MB AR AFTER graphs → gemms → real GDN op
     → final AR: ALL CLEAN — the wedge needs vllm's runner context (graph
     pool / capture history / stream dance), not just graphs+comm.
   - Collective capture-illegality proven: capturing an allreduce into an
     XPUGraph raises `RuntimeError: Graph nodes cannot depend on events
     from outside the graph` — serve's decode-graph captures must be
     splitting/breaking around ARs (breakable-cudagraph), so ARs replay
     eagerly between graph segments at decode time; the FIRST BIG EAGER AR
     at prefill is the wedge point.
5. pkill lesson (ops): `vllm serve` rewrites proctitles (`VLLM::Worker_TP0`
   …); `[a]pi_server`/`entrypoints` patterns miss them — kill with
   `pkill -9 -f "[V]LLM::"` + `pkill -9 -f "[v]llm serve"` and verify port
   000 before relaunch. Two invalid leg attempts came from surviving
   engines (health-from-old-engine looks identical to a fast boot — always
   verify wrap prints + fresh CALL lines + fresh pids).

## Defect-b model v3 (supersedes v2 and plan Appendix C)

With XPU graphs live on TP2, vllm 0.30 performs decode-graph capture
warmup at boot (ARs excluded via graph-break) and reports healthy WITHOUT
any real-size eager collective afterwards (no warmup generate in 0.30).
The first real request's embedding allreduce — the first EAGER large AR —
leaves the device-default stream's SYCL queue in command-graph recording
state with no capture open (runtime-level: event waits on it fail with
"queue which is recording"); every subsequent op enqueues into the
phantom capture; our integrity guard kills the step (500s). A first
request small enough (≤ ~512 tok / < 8 MB AR) initializes the path
cleanly and the engine stays stable for large requests afterwards.

## Fix lane (recommendation to owner)

**Boot-time warmup in our platform layer (vllm_gaudi/xpu/patches)**: after
graph capture warmup completes and before serving traffic, run an internal
2-stage warmup — small (≤ 512 tok) then max-batch-sized (8192) — fail-loud
at boot if the wedge hits (integrity law fires at deploy time, never in
user traffic). Evidence: this leg's controlled pair. NOT arcitech's env
(refuted by leg B), NOT the kernel wheel (closed), NOT lane-A prefill
routing (kernel exonerated — lane A demoted to perf option only).
Upstream issue candidate (after owner sign-off): vllm 0.30 XPU TP>1:
first eager large AR after decode-graph capture wedges the default stream
(queue reads recording; no capture open) — vllm#49924/#62781 class.

## Artifacts (this dir; rig raws in arc-test /root/*.log, caps /root/r16cap/out)

- r16_sitecustomize.py (v7 final: meta-path fork-safe wrap, side-stream
  staging, e/p two-phase save, AR/EMB preamble probes)
- r16_replay.py (real-kwargs standalone replay ± graphs)
- r16_replay2.py + r16b_run.sh (2-proc xccl mini-serve)
- r16_serve.sh / r16b_serve.sh (leg postures; b = SYCL_CACHE_PERSISTENT=0)
- smoking_gun_bracket.txt (the entry/exit AR bracket + onset)
- replay_{graph,nograph}.log, replay2_xccl.log (all-clean exonerations)
