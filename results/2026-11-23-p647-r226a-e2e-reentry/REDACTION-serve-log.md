# Redaction note — serve.log, first/serve.log (law 8, plan section 11 rule 9)

Both banked serve logs carried the rig's real address in
`dist_init_addr`-class lines (serve.log: 9 occurrences; first/serve.log:
2). Byte-exact perl replacement on the rig
(`s/10\.20\.3\.65/RIG_ADDR/g`, `s/10\.100\.8\.6/CLIENT_ADDR/g`;
MSYS sed injects CRLF — perl only). Post-redaction grep count 0 on
both; sha256 after redaction:

- `serve.log` 3b5034942d9097db8eb5402f58851bbc76247021cdd4034c9b1be6204f1f3602
- `first/serve.log` 8de8c311a4095fddf737fcc2b319c036ef9f13e4ac6df0cc23906a82fbcc4c7c

The transfer to the repo bank was verified byte-exact against these
hashes. No other banked file contained a host literal (grep count 0).
