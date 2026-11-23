r230 serve.log redaction

Applied rig-side with perl (byte-exact; MSYS sed injects CRLF) BEFORE
the bank pull, covering BOTH private address classes from the start
(the cert27-battery lesson applied at bank time this round): the rig
address and the proxy client source (a 10.100.x.x host) in uvicorn
access lines. Occurrences redacted: rig 11 (ctrl 1, m4q 1, m8q 8,
mkq 1), client 14 (ctrl 7, mkq 7) — replaced with REDACTED-RIG-ADDR /
REDACTED-CLIENT-ADDR. Post-redaction full rfc1918 scan on the rig:
zero matches; local post-pull scan: zero. The report tail saved as
r230_report_tail.txt was address-filtered at capture (filtered-stream
md5 81c8297cde31e07967debab6d4dbaa37).
