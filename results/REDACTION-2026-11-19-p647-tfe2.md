# REDACTION 2026-11-19 — P6.47 Phase 7 tf2 e2e (r195) bank scrubbed

Scope: `results/2026-11-19-p647-tf2-e2e/tfe2A26/` (the r195
ARC_MXKV_V5_TFLOOR=2 e2e bank: m4qtf2/m8qtf2/mkqtf2 ladder cells +
tfe2.txt report), as pulled from the rig on 2026-11-19.

## Transformation (deterministic, auditable)

- Two string substitutions, nothing else:
  `10.20.3.<host>` → `10.20.3.REDACTED` and `10.100.8.<client>` →
  `10.100.8.REDACTED` — the rig's own address and the harness client's
  intranet address, in uvicorn access/client lines and `mq_connect_ip=`
  engine lines inside banked `serve.log` / `serve_snapshot.log` files.
- 27 files touched (24 `serve_snapshot.log` + 3 cell `serve.log`).
  Integrity against the rig-computed `SHA256SUMS` was verified BEFORE
  the pass (`sha256sum -c` clean over the whole bank; first-fire pass
  was capped by a `head` filter at 12 files — caught and completed over
  all 27 in a second pass before any commit).
- Byte-exactness verified after the pass: digit-anchored verification
  (`10\.20\.3\.[0-9]` / `10\.100\.8\.[0-9]`) reports 0 raw-IP sites
  across all logs; the naive prefix grep matches only the
  `10.20.3.REDACTED` literals it is supposed to. No measurement,
  timestamp, counter, or verdict byte was altered (`perl -pi`, not MSYS
  sed, per the 2026-10-31 CRLF-injection lesson).
- The bank-level `SHA256SUMS` is the rig-side bank-time sum
  (pre-redaction bytes); post-redaction bytes differ from it by exactly
  the substitutions recorded here.

## E-law note

Banked results bytes are immutable by the evidence law; host-address
hygiene (plan §11 rule 9) is the one sanctioned exception, recorded
here so the audit trail stays complete. `tools/leak_sweep.sh` exits 0
on the tracked tree after this pass.
