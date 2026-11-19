# REDACTION 2026-11-19 — P6.47 cert25zc fastcycle (r196b) bank scrubbed

Scope: `results/2026-11-19-p647-zc-fastcycle/zcA27/` (the r196b
cert25zc validation fastcycle bank: m4qzc/m8qzc/mkqzc ladder cells +
zc.txt report), as pulled from the rig on 2026-11-19.

## Transformation (deterministic, auditable)

- Two string substitutions, nothing else:
  `10.20.3.<host>` → `10.20.3.REDACTED` and `10.100.8.<client>` →
  `10.100.8.REDACTED` — the rig's own address and the harness client's
  intranet address, in uvicorn access/client lines and `mq_connect_ip=`
  engine lines inside banked `serve.log` / `serve_snapshot.log` files.
- 202 substitution sites across the bank (single un-capped pass over
  every grep match — the r195 head-cap lesson held).
- Integrity against the rig-computed `SHA256SUMS` was verified BEFORE
  the pass (`sha256sum -c` clean over the whole bank, 0 non-OK lines).
- Byte-exactness verified after the pass: digit-anchored verification
  (`10\.20\.3\.[0-9]` / `10\.100\.8\.[0-9]`) reports 0 files with raw
  sites; 202 `*.REDACTED` literals are present. No measurement,
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
