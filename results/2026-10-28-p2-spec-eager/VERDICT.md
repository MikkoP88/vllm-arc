# VERDICT — eager+MTP k4: acceptance EXCELLENT (pos1 0.87, len 3.88/4) but perf COLLAPSES — the contract needs the graph lane back

## The leg (arc-p2-graph-20261028-034620, image 02961955b15a, GRAPH=0 SPEC=1)

First spec leg on the pivoted (eager) posture. MTP k4 via
--speculative-config (CLI law, p30). Zero zombie/storm/recording lines —
eager stays immune. Full matrix ok=N/N, `ARC_LEG_DONE`.

## p31 acceptance gate (SPEC_GATE)

- pos1 = **0.8702** — hard gate ≥0.5 PASS, and pos1 alone exceeds the 0.85
  target
- overall acceptance 0.7188 vs 0.85 target (positions 2/3 drag: 0.644/0.572)
- mean acceptance length **3.88 / 4** — the MTP head drafts near-perfectly
  on this model
- log-window cross-check: 4/4 streams ok; metrics line per-position 0.769/
  0.692/0.615 agrees with the gate probe
- Verdict line: `SPEC_GATE PASS: pos1=0.8702 >= 0.5 hard, acceptance=0.7188
  vs 0.85 target (TARGET MISS — recorded, --strict-target for certification)`

## Performance — the eager tax × spec multiplier

| cell | eager non-spec TPOT | eager+MTP TPOT | eager agg | MTP agg |
|---|---|---|---|---|
| k16k-c1 | 91.8 | 115.2 | 4.8 | 1.6 |
| k16k-c4 | 94.5 | 158.3 | 37.2 | 14.9 |
| k16k-c8 | 99.8 | 164.6 | 65.0 | 18.4 |
| k16k-c16 | 123.4 | 313.7 | 97.1 | 23.8 |
| k8k-c16 | 99.4 | 338.1 | 129.8 | 23.0 |

Spec decode on eager is a REGRESSION, not a lever: a spec step runs the
draft chain + verify (≈5 eager forwards) and eager pays full price for each
— graphs (FULL_DECODE_ONLY) were the mechanism that amortized that cost.
At mean length 3.88, break-even needs per-forward cost ≈¼ of non-spec;
eager gives no such discount (c16 spec batches are LARGER: 16×5 tokens).

## Consequence (decision tree resolved)

- eager non-spec: TPOT c8 94 ms vs ≤60 target — miss
- eager+MTP: TPOT c8 165 ms — worse
- → **the graph lane is load-bearing for the contract** (TPOT ≤60,
  agg ≥274.41). Root-cause isolation is now the critical path: the
  prefix-cache-write trigger lead is being tested by the graphs-on +
  prefix-off isolation leg (ARC_PREFIX=0 knob added to serve_arc_b70.sh —
  host-side, no image rebuild; default stays 1 = D champion).
- Acceptance target (0.85 overall) may also improve on graphs (D champion
  measured 0.85+ WITH graphs); recorded as target-miss for now.

Raws: this directory (spec_gate.json = machine-readable gate probe,
serve_full.log, k1k_sanity.txt; rig leg dir arc-p2-graph-20261028-034620).
