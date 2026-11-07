# Redaction note — 2026-11-07, phaseC boot logs (r108r/r108s)

Files (4): `phaseC/cert14_boot/serve.log`, `phaseC/cert14_boot/serve_end.log`,
`phaseC/cert14dbg_boot/serve.log`, `phaseC/cert14dbg_boot/serve_end.log`

Pass: byte-exact perl in-place (`perl -pi -e`), MSYS sed avoided (CRLF
injection):
- rig host `10.20.3.<rig>` → `10.20.3.REDACTED`
- client host `10.100.8.<client>` → `10.100.8.REDACTED`

Occurrences were LMCache/kv-transfer and engine-init log lines that embed
host addresses. No other content altered; post-pass grep for both raw
addresses returns zero matches across the banked tree.
