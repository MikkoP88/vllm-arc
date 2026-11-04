# REDACTION note — P6.19 campaign + cert8 banking (2026-11-04)

Same convention and per-token deltas as
`REDACTION-2026-11-02-p615.md` (`10.20.3.REDACTED` +6/occ,
`10.100.8.REDACTED` +7/occ, `REDACTED-HOST` +5/occ; perl `-pi` —
MSYS sed injects CRLF). Byte arithmetic must reproduce each delta
exactly; raw-ref grep must be 0 after. First-bank redaction (the p619
tree was never previously committed — E-LAW n/a; the pass ran before
the bytes ever entered git).

`2026-11-04-p619-tqverify/` total **+2341** = 137×6 + 217×7 + 0×5.

Verified: raw-ref grep 0 across the tree after the pass (per-token
`grep -ro` counts, 137 rig + 217 node before → 0 after; the +2341 byte
delta reproduces the arithmetic exactly).

## Append (cert8 gate + annex evidence, 2026-11-04) — second pass

The cert8 release-gate + post-gate annex legs banked under
`2026-11-04-p619-tqverify/rig/` (gate/, repchar/, mqannex2/, w14ctrl/,
incidents/, scripts, consoles) — same first-bank rule (redacted before
entering git). This pass adds the `ainode01` hostname class to the
standard two masks (manifest `build_host=` + build log). Total
**+816** = 56×6 + 65×7 + 5×5.

| File | rig | node | host | delta |
|---|---|---|---|---|
| `rig/build.out` | 0 | 0 | 3 | +15 |
| `rig/gate/manifest.txt` | 0 | 0 | 1 | +5 |
| `rig/gate/fp8/manifest.txt` | 0 | 0 | 1 | +5 |
| `rig/gate/fp8/serve_full_default.log` | 8 | 12 | 0 | +132 |
| `rig/gate/k3v4/serve_full.log` | 8 | 12 | 0 | +132 |
| `rig/gate/k8v4/serve_full.log` | 8 | 11 | 0 | +125 |
| `rig/gate/q3nc/serve_full.log` | 8 | 7 | 0 | +97 |
| `rig/gate/q4nc/serve_full.log` | 8 | 11 | 0 | +125 |
| `rig/mqannex2/serve_full.log` | 8 | 0 | 0 | +48 |
| `rig/repchar/serve_full.log` | 8 | 12 | 0 | +132 |

Verified: raw-ref grep 0 across the tree after the pass (per-token
`grep -ro` counts, 56 rig + 65 node + 5 host before → 0 after; the
+816 delta reproduces the arithmetic exactly).
