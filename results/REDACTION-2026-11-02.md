# REDACTION 2026-11-02 — TQ4 A/B leg

Same transformation and law as `REDACTION-2026-10-31.md` / `-11-01.md`
(plan §11 rule 9): tracked == pushable; raw rig/client addresses and
the rig hostname never enter git.

## File

`2026-11-02-p65-tq4-ab/rig/` (the turboquant_4bit_nc A/B leg; `.log`
evidence banked with `git add -f` per the Pass-5 law).

| File | `10.20.3.<rig>` → `10.20.3.REDACTED` | `10.100.8.<client>` → `10.100.8.REDACTED` | `<hostname>` → `REDACTED-HOST` | delta (expect) |
|---|---|---|---|---|
| `serve_full_tq4.log` | 8 (+48) | 13 (+91) | 0 | **+139 (+139)** |
| `preexisting_arc-test.json` | 0 | 0 | 1 (+5) | **+5 (+5)** |

`preexisting_arc-test.json` is the docker-inspect snapshot of the
non-leg arc-test container found running at leg start (started
2026-11-02T11:02Z, not by this leg); its hostname field is the only raw
ref. Both deltas reproduce the per-token arithmetic exactly (REDACTED 8
− rig last octet 2 = +6; 8 − client last octet 1 = +7; REDACTED-HOST 13
− the 8-char hostname = +5). Raw-ref grep across the whole dir after:
**0**; token counts after = pre-pass counts (8/13, 1). The other 13
mirrored artifacts grepped clean pre-check (0 hits each). No
measurement, timestamp, counter, or verdict byte altered.
