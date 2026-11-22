# Redaction note — cert26 battery bank (2026-11-22)

Bank: `results/2026-11-22-p647-cert26-battery/` (pulled from the rig's
`/root/build/p647/cert26`). The nine serve logs (ctrl, m8q, m4q, mkq,
mkq_native_refused, m8g, m4g, mkg, g0) carried the rig and client
addresses in their container-registration and request-log lines —
**366 hits** total — redacted at file-write time with the byte-exact
perl pass (MSYS sed injects CRLF; perl only):

```
perl -pi -e 's/10\.20\.3\.65/RIG_ADDR/g; s/10\.100\.8\.6/CLIENT_ADDR/g' <serve logs>
```

Post-pass sweep: 0 remaining hits. The bank's SHA256SUMS was
regenerated AFTER the redaction pass (the rig-side pre-redaction sums
do not apply to the banked bytes; the recompute-from-banked-artifacts
law). All other artifacts (probe outs, longprobe JSONs, markers,
battery.txt, metrics, boot.txt) were clean as pulled.
