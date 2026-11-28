# VERDICT — r268 eager trace leg (the non-reproduction + the dead-letter)

Banked 2026-11-28 from rig bank `/root/build/p647/r268_trace` after
`R268_DONE rc=0 09:13:16Z` + `W14_RESTORE rc=0` (transfer oracle 9/9
byte-match; no address-bearing files). Boot: cert29, gemma lawful g3
skin, `kv_cache_dtype=mxfp8` + native windowed lane (P-C2 telemetry
live), `ARC_GRAPH=0` (eager), traced overlay
`ARC_MXKV_OVERLAY=/root/build/r268_overlay` (engine.py `0ce0954c`,
md5-verified in the BEGIN line) + `ARC_MXKV_TRACE=1`.

## Result 1 — THE NON-REPRODUCTION (the headline)

The boot was **CLEAN on all three workload requests**:

- gen1 raw "The capital of France is" (ptok=6, max_tokens=8, temp=0)
  → `" a city of romance, art, and"` — 8/8 sensible tokens.
- gen2 raw "def fibonacci(n):" → `" \n if n <= 1:"` — clean.
- gen3 the same prompt as gen1 **with logprobs** → HTTP 200, all 8
  token_logprobs finite (−1.82 … −0.89) — **no nan**.

The serve args are **byte-identical to e2** (18/18 keys diffed: model,
template, dtype, max_model_len 40960, block_size 64, gmu 0.8,
kv_cache_dtype mxfp8, prefix caching on, mnbt 8192, seqs 64, TP2);
the MXKV impl telemetry matches (fmt=mxfp8 head=256 slot=528 both
pools + P-C2 windowed W=1024). e2 — the same posture — was RED from
its very FIRST request (all 324 probe requests HTTP-400
`Out of range float values are not JSON compliant: nan`; the
no-logprobs gate battery `'Paris our our our…'` at ptok=40).
**The corruption is request-CONDITIONAL, not boot-inherent.**

Candidate axes that differ between the clean r268 workload and the
RED e2 workload (bisected next, r268b): prompt length (6 vs 37-41+
tokens), chat templating (raw vs g3-wrapped), sampling params
(temp=0 bare vs temp=1.0/top_p=0.95/top_k=64/seed/stop_token_ids),
skip_special_tokens, max_tokens (8 vs 64-96), and CONCURRENCY (r268
strictly sequential; the probe fires `--workers 8`).

## Result 2 — the trace never fired (the ARC_MXKV_TRACE dead letter)

`trace lines: 0 total, 0 store/decx` — despite `ARC_MXKV_TRACE=1` in
the runner env. Root cause: `serve_arc_b70.sh` passes env into the
engine container through a curated export list in its docker-exec
block; `ARC_MXKV_TRACE` is **not in the list** (MXKV_NATIVE,
MXKV_NATIVE_MINLEN, MXKV_VPROBE, DIRECTBF16, NSPLIT, P64NATIVE, K2,
K3, AB*, V5_* are). The overlay mounted fine (the `-v` mount uses
ARC_MXKV_OVERLAY directly), so the engine ran the traced engine.py —
but `_TRACE` read an env var the container never received, so the
decx/store prints were inert. The overlay itself is proven
behaviorally inert by diff (30 lines: the store-telemetry fields +
the `_TRACE`-gated decx block) — the clean generations are NOT an
overlay artifact. Fix for any future trace leg: add the export line
to the serve script or set the var via `docker exec env`.

## Consequence

The G3 hypothesis is refined: the engine hands the kernels wrong
metadata VALUES **only for some request shapes** (or under
concurrent batching) — the kernels/store math are exact (r267
canary) and the plain small-sequential-raw path is exact in the
engine too (this leg). **r268b** (launched 09:27Z, staged with this
bank) bisects the axes on ONE plain-image c3-posture boot — ladder
A (length) / B (exact e2 shapes) / C (2-way and 8-way concurrency) /
D (poisoning recheck) / E (determinism) — then runs the FULL p266
matrix probe on the SAME boot (the same-boot arbiter:
ladder-clean + probe-RED = the trigger lives in the probe's request
set; both clean = boot-nondeterminism).
