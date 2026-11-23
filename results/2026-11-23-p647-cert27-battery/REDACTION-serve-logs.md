cert27 battery serve.log redaction (rig address -> REDACTED-RIG-ADDR)

Late addition (pre-close-out sweep): the uvicorn access lines also
carried the proxy client source address (a 10.100.x.x host) — a second
private-infra host-address class the rig-side pass did not cover
(it swept only the rig address). Scrubbed bytes-exact locally,
168 occurrences across 7 leg serve.logs (g0 28, m4g 35, m4q 7,
m8g 42, m8q 7, mkg 42, mkq 7) -> REDACTED-CLIENT-ADDR; SHA256SUMS
regenerated post-scrub. Lesson banked: the bank-time sweep must run
the full leak_sweep pattern set, not a single-address grep.
