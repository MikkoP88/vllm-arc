# REDACTION 2026-11-01 — P6.13 gate serve log

Same transformation and law as `REDACTION-2026-10-31.md` (plan §11 rule 9):
tracked == pushable; raw rig/client addresses never enter git.

## File

`2026-11-01-p65-rung3-slice/rig/serve_full_p73.log` (the p73 gate's engine
log, mirrored from the container; banked with `git add -f` per the Pass-5
law).

## Transformation (deterministic, auditable)

Two string substitutions via `perl -pi` (MSYS sed injects CRLF — law),
nothing else:

- `10.20.3.<host>` → `10.20.3.REDACTED` — 29 occurrences (uvicorn access
  lines / `mq_connect_ip=`)
- `10.100.8.<client>` → `10.100.8.REDACTED` — 44 occurrences (harness
  client address in uvicorn access lines)

Byte-exactness verified after the pass: delta **+482 bytes exactly** =
29 × (REDACTED 8 − "65" 2 = +6) + 44 × (REDACTED 8 − "6" 1 = +7) =
174 + 308. (The first mental arithmetic — +540 — double-counted the
replaced digit pair; the correct per-token derivation reproduces the
observed delta exactly.) Raw-address grep after: **0**. Token counts after:
44 + 29, matching the pre-pass grep. No measurement, timestamp, counter, or
verdict byte altered; all other mirrored artifacts in the same dir grepped
clean pre-check (0 hits each).
