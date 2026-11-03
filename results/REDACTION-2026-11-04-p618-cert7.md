# REDACTION note — P6.18d cert7 gate banking (2026-11-04)

Same convention and per-token deltas as
`REDACTION-2026-11-02-p615.md` (`10.20.3.REDACTED` +6/occ,
`10.100.8.REDACTED` +7/occ, `REDACTED-HOST` +5/occ; perl `-pi` —
MSYS sed injects CRLF). Byte arithmetic must reproduce each delta
exactly; raw-ref grep must be 0 after. First-bank redaction (the cert7
tree was never previously committed — E-LAW n/a; the pass ran before
the bytes ever entered git).

`2026-11-03-p618-cert7/` total **+132** = 8×6 + 7×7 + 7×5.

| File | rig | node | host | delta |
|---|---|---|---|---|
| `rig/gate/serve_full_default.log` | 8 | 7 | 0 | +97 |
| `rig/cert7_build1_brokentag.out` | 0 | 0 | 3 | +15 |
| `rig/cert7b_build2_fixed.out` | 0 | 0 | 3 | +15 |
| `rig/gate/manifest.txt` | 0 | 0 | 1 | +5 |

Verify: per-file ripgrep raw-ref total 0 across the tree after the
pass (directory-mode rg silently skips NUL-bearing logs — the sweep ran
per file).
