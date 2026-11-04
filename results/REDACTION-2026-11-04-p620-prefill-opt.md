# Redaction note — 2026-11-04 P6.20 evidence (plan §11 rule 9)

Banked serve logs for p620 legs r80–r83 contain the rig and client
host addresses in distributed-init / request-source log lines.
Redacted with `perl -pi` (MSYS sed injects CRLF) using the standing
masks — same convention and per-token deltas as
`REDACTION-2026-11-02-p615.md`: `10.20.3.REDACTED` (+6 bytes/occ),
`10.100.8.REDACTED` (+7 bytes/occ). All other log
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

## r84/r84b — cert9 image-native gate + annexes (appended 2026-11-04)

Same masks, same method, over the banked `rig/r84/cert9/gate/` tree:

| file | rig ×65 | client ×8.6 | bytes before → after | delta (expected) |
|---|---|---|---|---|
| r84/cert9/gate/fp8_serve_leg1.log | 1 | 1 | 164865 → 164878 | 13 (13) |
| r84/cert9/gate/k3v4/serve_full.log | 8 | 7 | 186855 → 186952 | 97 (97) |
| r84/cert9/gate/k3v4_rechar/serve_full.log | 8 | 12 | 209669 → 209801 | 132 (132) |
| r84/cert9/gate/k8v4/serve_full.log | 8 | 12 | 279213 → 279345 | 132 (132) |
| r84/cert9/gate/mqannex2/serve_full.log | 1 | 4 | 128583 → 128617 | 34 (34) |
| r84/cert9/gate/q3nc/serve_full.log | 8 | 7 | 175935 → 176032 | 97 (97) |
| r84/cert9/gate/q4nc/serve_full.log | 8 | 7 | 178284 → 178381 | 97 (97) |
| r84/cert9/gate/q4nc_rechar/serve_full.log | 8 | 12 | 206109 → 206241 | 132 (132) |

Total: 50 rig + 55 client = 105 occurrences, +734 bytes; post-pass
residual over the whole p620 tree = 0.
