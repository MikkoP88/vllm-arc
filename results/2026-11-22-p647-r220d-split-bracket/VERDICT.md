# VERDICT — r220d split bracket: no split depth rescues the `_64` tile — the lever closes e2e-negative (2026-11-22)

Leg: arc7 Phase-8 r220d — p64 (`ARC_MXKV_P64NATIVE=1`) with the split
env pinned at 64 and 256, face c32768 n1, mxfp4, wheel
`62f234c8d0aa89c4aec3e8d2c6fa22ad` — the decisive bracket after
r220b's e2e inversion of r220a's offline win.

## Result

```
     cell |    itl | vs ctl | vs p64@128 |    tps |  chunks | qual | errs
   n64s64 |   63.6 | +12.9% |      -0.5% |   57.3 | ~   53 | OK | 0
  n64s256 |   67.7 | +20.0% |      +5.8% |   54.8 | ~   53 | OK | 0
BEST n64s64 itl=63.6 -> NOT_RESCUED (tile lever e2e-negative at every split depth)
```

Anchors (same wheel, r220b): ctl (`_16`, splits auto→128) 56.4;
p64 (splits auto→128) 64.0; banked posture 56.8.

## Reading

- The `_64` split surface under engine load is FLAT from 64 to 128
  (-0.5%) and regresses at 256 (+5.8%) — the r219 `_16` optimum
  (128) carries over, and no depth recovers the ~7-11 ms/tok penalty
  vs `_16`. Quality healthy (chunks ~53, 0 errors) — the loss is
  scheduling/residency, not value.
- Combined with r220b: the kv_tile 16→64 walk is OFFLINE-faster
  (-11..-50% per call, r220a) but E2E-SLOWER at every split depth —
  the r213 probe/e2e sign-inversion class with the occupancy
  mechanism (per-WG SLM V image 4× taller → 4× fewer resident WGs
  under the engine's 16 concurrent calls; the single-call probe never
  sees it).

## Line verdict

The kv_tile 16→64 lever is CLOSED as an engine posture change. What
SURVIVES from r220: the value-correct multi-SG route (r220c, env-gated
off), the corrected coordinate-frame laws (WG-spanning svB copies,
wi16/sg_row_off scope), and the r221 road definition — IF the walk
line continues, the candidate is V sub-staging (16-row ping-pong V
slices restoring `_16`-level per-WG SLM while keeping the 4-SG K
walk), priced against its partial capture (~half the offline K+V win
≈ −0.7 ms/step ≈ −1.2% itl @32K — small vs the 14.3 ms/step MX total;
weigh against other levers before spending a build).

Posture unchanged: `_16` sub-tile frame, splits auto (→128).

## Redaction

`perl -pi -e 's/10\.20\.3\.65/RIG_ADDR/g; s/10\.100\.8\.6/CLIENT_ADDR/g'`
at file-write time; 18 literal replacements pre-pass across
`{n64s64,n64s256}/{serve.log,c32768_n1/serve_snapshot.log}`; zero
matches post-pass; `tools/leak_sweep.sh` exit 0.
