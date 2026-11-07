# Redaction note — 2026-11-07, phaseC cert15 boot logs (r108t)

Files (2): `phaseC/cert15_boot/serve.log`,
`phaseC/cert15_boot/serve_end.log`

Pass: byte-exact perl in-place (`perl -pi -e`), MSYS sed avoided (CRLF
injection):
- rig host `10.20.3.<rig>` → `10.20.3.REDACTED`
- client host `10.100.8.<client>` → `10.100.8.REDACTED`

Occurrences were engine-init/LMCache log lines embedding host
addresses. No other content altered; post-pass grep for both raw
addresses returns zero matches across the banked tree. Bundle md5
ad92910248fd7f80cdad16f2fb02f69c matched rig-side before redaction.
