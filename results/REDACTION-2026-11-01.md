# REDACTION 2026-11-01 — P6.13 gate + cert5 release logs

Same transformation and law as `REDACTION-2026-10-31.md` (plan §11 rule 9):
tracked == pushable; raw rig/client addresses never enter git.

## Files (pass 1 — P6.13 probe gate)

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

## Files (pass 2 — cert5 release gates)

`2026-11-01-p65-cert5/rig/` (cert5 tag-built-image gates; `.log` files
banked with `git add -f` per the Pass-5 law). Two substitutions, same
tokens as pass 1, plus the established hostname token:

| File | `10.20.3.<rig>` → `10.20.3.REDACTED` | `10.100.8.<client>` → `10.100.8.REDACTED` | `<hostname>` → `REDACTED-HOST` | delta (expect) |
|---|---|---|---|---|
| `serve_full_default.log` | 8 (+48) | 12 (+84) | 0 | **+132 (+132)** |
| `serve_full_p73face.log` | 1 (+6) | 4 (+28) | 0 | **+34 (+34)** |
| `build.out` | 0 | 0 | 3 (+15) | **+15 (+15)** |
| `manifest.txt` | 0 | 0 | 1 (+5) | **+5 (+5)** |

Hostname occurrences: `build.out` carries `build_host=` three times (the
docker-build RUN script text, its echo, and its output line — the
`--build-arg BUILD_HOST` of `tools/build_image.sh`); `manifest.txt` carries
the same field once. All four deltas reproduce the per-token arithmetic
exactly (REDACTED 8 − rig last octet 2 chars = +6; 8 − client last octet
1 char = +7; REDACTED-HOST 13 − the 8-char hostname = +5). Raw-ref grep
across the whole dir after: **0**; token
counts after = pre-pass counts (8/12, 1/4, 3, 1). The other 13 mirrored
artifacts grepped clean pre-check (0 hits each).
