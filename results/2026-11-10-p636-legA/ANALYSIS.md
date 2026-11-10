# P6.36 leg A — prefill-window decomposition (probe image on cert23)

Probe: cert23 + `tools/ar/p136/patch_p136a.py` wrappers (ARR/SCHE/ESTEP/WEXE/REXE,
epoch timestamps, O_APPEND shared log). Image
`vllm-arc:v0.30.0-b70-cert23-p136probe` — probe only, never ships.

Legs: `l1fp8` = qwen3.8-27b-fp8, spec3 (k3), LMC=1, graph=1 — the certified
posture. U8k face (8192 tok prompt + 390 template = 8582 ARR) and U32k face
(32768 + 1488 = 34256 ARR). Analysis tool: `tools/ar/p136/p136_decompose.py`
(pid = rank proxy; workers carry no RANK env; SchedulerOutput exposes no
total_scheduled_tokens at these hooks so chunks are classified by wall).

## Verdict — the prefill window is device-bound, not host-bound

U8k (client TTFT 4.275 s; window ARR→first decode 4156 ms):

| phase | ms |
|---|---|
| ARR → first dispatch | 8.7 |
| chunk-1 exec (8192 tok; enqueue wall) | 908.9 |
| **engine-core wait (future.result) → chunk-2 dispatch** | **2519.4** |
| chunk-2 exec (390 tok remainder + transition) | 381.3 |
| last exec → first decode done | 337.6 |

U32k (server-side prefill (10,15] s bucket — matches the 13 556 ms window):

| chunk | enqueue wall (ms) | gap to next dispatch (ms) |
|---|---|---|
| 1 (8192) | 407.9 | 2778.9 |
| 2 (8192) | 386.3 | 3118.1 |
| 3 (8192) | 410.6 | 3851.4 |
| 4 (8192) | 403.7 | 1540.2 |
| 5 (~1488) | 542.3 | 503.6 (→ first decode) |

Window split (U32k): dispatch 22 ms | exec walls 2578 ms | EC inter-chunk
waits **10 880 ms (80 %)** | tail 504 ms. Schedule walls: 14 ms total.
Queue time: sub-ms (metrics). Cross-rank skew: **zero** — both ranks'
WEXE/REXE walls agree to 0.1 ms at every chunk (lockstep).

Findings:

1. **Rank skew is dead as a lever.** The leg-F "1.87 s eager-xccl rank-skew
   block" was the device drain seen through a host-blocking profiler; TP2
   ranks run in lockstep at every chunk boundary.
2. **Scheduler/queue are dead as levers** (sub-ms queue, 14 ms scheduling,
   schedule walls ≤ 6.6 ms).
3. **The window is device drain**: `step_with_batch_queue` blocks in
   `future.result()` on the chunk's sample future; the future resolves when
   the device finishes the chunk forward. Decode ITL is 40.4 ms flat with
   the same round trip ⇒ round-trip overhead ≤ ~2 ms; the 2.8–3.9 s gaps
   are device completion time. Prefill device throughput ≈ 2.1–2.9 k tok/s
   (fp8-k3, ~3.1–4.3 s per 8192-token chunk incl. enqueue overlap).
4. **Per-chunk device time varies 2.5×** for identical 8192-token chunks
   (gaps 1540–3851 ms) — candidate causes: sustained-burst power behavior,
   full-attention layers over growing KV, or per-chunk boundary effects.
5. Chunked at MNBT=8192 ⇒ U8k (8582 tok) always pays two chunks: 909 ms
   first + 381 ms remainder + the full first-chunk drain before the 390-token
   remainder even starts.

## l2mx4 leg (mx4-W4A16 k5 winner posture, U8k only)

Client: TTFT 5.263 s, decode 46.7 ms/chunk × 5.93 tok/chunk ≈ 127 tok/s
(= P6.35-V winner within probe tax). Telemetry (window 7285 ms):

| execute | wall (ms) | gap to next dispatch (ms) |
|---|---|---|
| 1 | 144.3 | 2730.3 |
| 2 | 400.5 | 3682.2 |
| 3 | 309.1 | 30.0 (→ decode cadence) |

Window split: waits **6413 ms (88 %)** vs exec walls 1353 ms; schedule
15.5 ms total; ranks lockstep. Same device-drain verdict as fp8 — the mx4
prefill tax is device-side too. Open detail: the mx4 chunk sequence shows
three chunk-class executes with much smaller enqueue walls (144 ms first vs
908 ms at fp8) — a different host-enqueue path (W4A16 lane) whose device
still drains 2.7–3.7 s behind; per-chunk semantics unresolved, does not
change the verdict.

## Incident + defect notes (gate script)

`face_u32k.json` in `l1fp8/` is a stale copy of the U8k JSON — the U32k
section copied `$CD/p135_face_$LEGN.json` (the U8k pull already sitting in
the rig results dir) instead of docker-cp'ing the fresh container file.
Engine-side `prefill_u32k.log` and `metrics_u32k.txt` were pulled correctly
(direct docker cp); U32k client TTFT recovered from the metrics histogram:
prefill time in (10,15] s, consistent with the 13.56 s telemetry window.
Fixed in r137b (distinct face tags + docker cp before local cp).

Second defect: r137 `run_leg` invoked `p135_face.py 8192 600 "$LEGN"`
without the 4th model argument — the script defaults to
`qwen3.8-27b-fp8`, so the l2mx4 face 404'd (model not found) before
reaching the engine and produced no JSON. Caught live: re-ran the face
manually with `qwen3.8-27b` while the gate's 500 s wait loop was still
polling; the gate picked up the JSON at 408 s and banked everything
itself. r137b passes `$SERVED` through to every face call.

## P6.36-B discriminators — ALL NEGATIVE (r137b, probe image)

Client TTFTs (probe-laden; U32k face = 32k-token prompt, bf16 = 30k):

| leg | posture | U8k TTFT | U32k TTFT | verdict |
|---|---|---|---|---|
| l1fp8 | MNBT 8192 (certified) | 4.275 s | 13.6 s (window) | baseline |
| b1m16 | MNBT 16384 | **3.653 s** | **21.674 s** | U8k −0.6 s, U32k +8 s → regression |
| b2bf16 | bf16 no-quant, MAXLEN 32k | 5.320 s | 13.625 s (30k ctx) | refuted |
| b3m4k | MNBT 4096 | 4.218 s | 14.776 s | neutral |

- **B1 (chunk size)**: 16k chunks make U8k single-chunk (−0.6 s) but U32k
  collapses (+8 s — per-16k-chunk device ~7.5 s, sublinear efficiency, plus
  a 7.6 s post-prefill tail). 4k chunks: neutral U8k, +1.2 s U32k. The
  certified 8192 is the optimum; device prefill rate ≈ 2.1–2.9 k tok/s
  across all chunk sizes.
- **B2 (wide math)**: bf16 loses everywhere: U8k 5.32 vs 4.275 s; normalized
  30k rate ~2.2 vs ~2.5 k tok/s. Xe2's fp8 counter-operand-conversion GEMM
  path is prefill-**optimal** — the wide-math lane has no case (also 2×
  weights → MAXLEN capped at 32k). Side observations: bf16-k3 decode ≈
  100 tok/s (vs fp8-k3 94) — not worth −25 % TTFT and ¼ context; ~1.5 s
  unexplained between bf16 final-chunk completion and first token (parked;
  fp8 books balance).
- **B3 (burst length)**: neutral — the 2.5× identical-chunk gap variance is
  not burst-length-addressable.

## D2 decode host-tail refinement (from the same telemetry)

Steady-decode cadence per step (median): enqueue wall 4.4 ms + e→b wait
36.0 ms = 40.5 ms b→b at fp8-k3 (client ITL 40.4–41.3 flat); mx4-k5:
4.3 + 39.9 = 44.2. Device floor ≈ 34–35 ms/step ⇒ addressable host ≈
6 ms/step ⇒ **decode ceiling ≈ +8–10 %** — the banked ≤+10 % D2 bound
holds, now with the split (4.4 enqueue + ~1.5 round trip vs device floor).

## Improvement-room conclusion (P6.36)

Prefill/TTFT: **at the device floor** — every lawful dtype (fp8/bf16/W4A16)
and chunk size (4k/8k/16k) measured; no host, scheduler, skew, or queue
lever exists (all sub-ms or zero). Decode: device weight-stream floor +
bounded ≤10 % host tail (D2), previously and now confirmed not worth the
risk to the certified champion. **No remaining lawful lever ≥ 10 %.**
P6.36 closes as evidence-only (no engine bytes changed; cert23 stands).
→ Proceed to owner close-out: phases E (per-config reports), F (64 GB-RAM
host compatibility), G (kv-cache-dtype reports), full docs overhaul.
