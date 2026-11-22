# VERDICT — r221b cold-graph A/A: hot-L2 hypothesis REFUTED — cold WIDENS `_64`'s offline win (2026-11-22)

Leg: arc7 Phase-8 r221b — the hot-L2 test defined by r221a. Harness
gained `P204_COLDMB` (r221b edit, md5 `8a61098024920a5aba608a9f4ceba7be`,
additive env-gated guards only, default 0 = banked behavior
untouched): a 160 MB device buffer swept in-place between EVERY
captured call, evicting L2 so each replayed call reads its KV pages
cold from HBM — reproducing the engine's layer-interleaved reality.
Sweep-only measured 624.9 µs/slot = 160 MB × 2 / 625 µs ≈ 512 GB/s
(HBM-realistic); the report subtracts it for per-call cold numbers.

## Result — 0/6 sign flips; cold margin WIDER at 32K

```
    fmt    ctx |      c64 |      c16 |  64vs16 | hot 64vs16
  mxfp8   1024 |    381.3 |    803.9 |  -52.6% |     -53.1%
  mxfp8  32768 |    677.4 |    820.4 |  -17.4% |      -9.6%
  mxfp4   1024 |    282.2 |    625.6 |  -54.9% |     -55.5%
  mxfp4  32768 |    505.9 |    650.8 |  -22.3% |     -17.2%
  mxk8v4   1024 |    338.9 |    764.1 |  -55.7% |     -56.2%
  mxk8v4  32768 |    598.6 |    766.1 |  -21.9% |     -13.0%
```

Canaries clean. THE INVERSION OF THE HYPOTHESIS: `_16` is the
cold-sensitive route (+8-13% slower cold-vs-hot: 754→820, 608→651,
694→766) while `_64` barely moves (682→677, 504→506, 604→599). If
anything, engine-like cold conditions favor `_64` offline HARDER —
the exact opposite of the e2e ordering.

## Verdict

Hot-L2 blindness is refuted as the inversion mechanism. Three
hypothesis families are now eliminated with same-wheel evidence:
shape knobs (r221a: bt 4100/stale/pool/cap extent), row count
(r221a: NCAP=4), and cache state (r221b: cold graph). The offline
instrument — engine-shaped, engine-row-counted, cold-cached — still
says `_64` wins by 17-22% @32K while the engine says +13.4% worse.

The residual differences the isolated single-GPU probe structurally
cannot contain, for the record:
- TP2 concurrency: the engine's native call runs on BOTH B70s
  simultaneously (per-layer, oneCCL-allreduce-bracketed) — shared
  power/thermal envelope; `_64`'s wider instantaneous access
  pattern could clock-throttle only under dual-GPU load.
- The engine's step-graph surrounds the native kernel with weights
  GEMMs, GDN state ops, and allreduces — in-graph command/event
  structure and memory-pool state differ from a native-only capture
  (cf. the torch-2.14 XPU-graph replay overhead watch item).

The e2e arbiter stands: posture `_16` + auto-128; the tile lever
remains CLOSED. Reopening requires a dual-GPU lockstep probe (two
harness processes, one per device, r221c candidate) or an in-engine
A/B instrument — both are real builds; the payoff ceiling is the
offline-claimed −17..−22% @32K per call (≈ −2.8 ms/step ≈ −4.9% itl
IF it transferred, which three instrument generations have failed to
demonstrate).
