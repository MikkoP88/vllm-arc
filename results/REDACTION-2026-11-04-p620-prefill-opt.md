# Redaction note — 2026-11-04 P6.20 evidence (plan §11 rule 9)

Banked serve logs for p620 legs r80–r83 contain the rig (10.20.3.65) and
client (10.100.8.6) host addresses in distributed-init / request-source
log lines. Redacted with `perl -pi` (MSYS sed injects CRLF) using the
standing masks: `10.20.3.65` → `10.20.3.REDACTED` (+6 bytes/occurrence),
`10.100.8.6` → `10.100.8.REDACTED` (+7 bytes/occurrence). All other log
bytes untouched; post-pass sweep over the whole evidence tree = 0
remaining raw addresses. The evidence README's prose reference got the
rig mask directly.

| file | rig ×65 | client ×8.6 | bytes before → after | delta (expected) |
|---|---|---|---|---|
| rig/r80/a/serve_full.log | 8 | 7 | 173338 → 173435 | 97 (8·6+7·7 = 97) |
| rig/r80/b/serve_full.log | 1 | 4 | 168567 → 168601 | 34 (34) |
| rig/r81/l3on1/serve_full.log | 1 | 4 | 169134 → 169168 | 34 (34) |
| rig/r81/l3on2/serve_full.log | 8 | 5 | 171645 → 171728 | 83 (83) |
| rig/r81/off1/serve_full.log | 1 | 7 | 168763 → 168818 | 55 (55) |
| rig/r81/off2/serve_full.log | 1 | 1 | 169876 → 169889 | 13 (13) |
| rig/r82/serve_full.log | 15 | 24 | 399372 → 399630 | 258 (258) |
| rig/r83/serve_full.log | 15 | 12 | 327021 → 327195 | 174 (174) |

Total: 50 rig + 64 client = 114 occurrences, +748 bytes. E-LAW scope:
host-address redaction is the sanctioned exception to banked-bytes
immutability.
