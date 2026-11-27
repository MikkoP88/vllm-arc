# REDACTION — 2026-11-27 — the r264 cert29 gate bank: audit clean

`results/2026-11-27-p647-cert29-gate-r264/` (pulled 2026-11-27 by
`tools/ar/r264_pull.sh`): **zero files required the redaction pass.**

- The pull's law-8 block (`perl -pi` byte-exact, rig + client subnets) ran
  over every text file holding a host literal — it found none: the bank's
  serve logs carry container-internal API lines only (probes ran on the
  rig itself; no client-LAN access lines this run).
- Post-pull address audit over the bank
  (`grep -rIlE '10\.20\.3\.[0-9]{1,3}|10\.100\.8\.[0-9]{1,3}'`): clean.
- `SHA256SUMS.rig` (238 entries, pre-redaction rig-side sums) matches the
  pulled bytes on every entry with zero address-bearing exceptions —
  consistent with no redaction having been needed. `SHA256SUMS` is the
  post-pull sum set over the same bytes.
