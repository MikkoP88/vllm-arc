# Redaction note — serve.log (law 8, plan section 11 rule 9)

Two lines (72, 522: `multiproc_executor.py:153 DP group leader ...`)
carried the rig's real address in the `dist_init_addr`-class field.
Byte-exact perl replacement on the rig (`s/10\.20\.3\.65/RIG_ADDR/g`,
`s/10\.100\.8\.6/CLIENT_ADDR/g`; MSYS sed injects CRLF — perl only).
Post-redaction grep count 0; sha256 after redaction
f1eab251a4d0737329f481cd00697fb1df63c9de48a8a71df0609148d7c1550c.
No other banked file contained a host literal (grep count 0).
