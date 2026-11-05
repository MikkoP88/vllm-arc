# Redaction note — 2026-11-05 P6.21 ar_quant evidence (plan §11 rule 9)

Banked serve logs for p621 leg r86 contain the rig and client host
addresses in distributed-init / request-source log lines. Redacted with
`perl -pi` (MSYS sed injects CRLF) using the standing masks — same
convention and per-token deltas as `REDACTION-2026-11-02-p615.md`:
`10.20.3.REDACTED` (+6 bytes/occ), `10.100.8.REDACTED` (+7 bytes/occ).
All other log bytes untouched; post-pass sweep over the whole p621
evidence tree = 0 remaining raw addresses.

| file | rig ×65 | client ×8.6 | bytes before → after | delta (expected) |
|---|---|---|---|---|
| rig/r86/arq1/serve_full.log | 8 | 12 | 263675 → 263807 | 132 (8·6+12·7 = 132) |
| rig/r86/arq2/serve_full.log | 8 | 11 | 262558 → 262683 | 125 (125) |
| rig/r86/cf1/serve_full.log | 8 | 12 | 256881 → 257013 | 132 (132) |
| rig/r86/off1/serve_full.log | 8 | 12 | 260249 → 260381 | 132 (132) |
| rig/r86/off2/serve_full.log | 8 | 11 | 260187 → 260312 | 125 (125) |

Total: 40 rig + 58 client = 98 occurrences, +646 bytes; post-pass
residual over the whole p621 tree = 0. E-LAW scope: host-address
redaction is the sanctioned exception to banked-bytes immutability.
