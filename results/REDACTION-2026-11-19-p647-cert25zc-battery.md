# REDACTION 2026-11-19 — P6.47 cert25zc full battery (r197) bank scrubbed

Scope: `results/2026-11-19-p647-cert25zc-battery/` (the r197 cert25zc
full-par10 battery bank: ctrl/g0/m4g/m4q/m8g/m8q/mkg/mkq legs +
gemma_compare.txt), as pulled from the rig on 2026-11-19.

## Transformation (deterministic, auditable)

- Two string substitutions, nothing else:
  `10.20.3.<host>` → `10.20.3.REDACTED` and `10.100.8.<client>` →
  `10.100.8.REDACTED` — the rig's own address and the harness client's
  intranet address, in uvicorn access/client lines and `mq_connect_ip=`
  engine lines inside the eight banked `serve.log` files.
- 351 substitution sites across the bank (single un-capped pass over
  every grep match; full per-file list captured in the session record —
  all 8 legs' serve.logs, no other file touched).
- Integrity against the rig-computed `SHA256SUMS` was verified BEFORE
  the pass (`sha256sum -c` clean: 192/192 OK, 0 non-OK lines).
- Byte-exactness verified after the pass: digit-anchored verification
  (`10\.20\.3\.[0-9]` / `10\.100\.8\.[0-9]`) reports 0 raw sites; 351
  `*.REDACTED` literals are present (balanced against the before
  count). No measurement, timestamp, counter, or verdict byte was
  altered (`perl -pi`, not MSYS sed, per the 2026-10-31 CRLF-injection
  lesson).
- The bank-level `SHA256SUMS` is the rig-side bank-time sum
  (pre-redaction bytes); post-redaction bytes differ from it by exactly
  the substitutions recorded here.

## E-law note

Banked results bytes are immutable by the evidence law; host-address
hygiene (plan §11 rule 9) is the one sanctioned exception, recorded
here so the audit trail stays complete. `tools/leak_sweep.sh` exits 0
on the tracked tree after this pass.
