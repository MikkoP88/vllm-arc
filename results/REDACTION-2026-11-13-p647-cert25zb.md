# REDACTION 2026-11-13 — P6.47 cert25zb certification bank

## r184 battery bank (results/2026-11-13-p647-cert25zb-certification/)

- Same two perl substitutions over the pulled bank tree (byte-exact `perl -pi`,
  no CRLF injection): `10.20.3.<host>` → `10.20.3.REDACTED` and
  `10.100.8.<host>` → `10.100.8.REDACTED`. Post-pass scan of the whole bank:
  zero numeric-octet matches for either subnet.
- Counts: `raw/cert25zb/m4g/serve.log` 46 (the external LAN client's access
  lines — the same client whose 404s are the erraudit re-adjudication subject),
  `raw/cert25zb/m4g/erraudit_readjudication.txt` 4 (context lines around the
  ERROR lines), the remaining occurrences spread across the other seven legs'
  serve logs (Uvicorn access lines + engine `mq_connect_ip=`/bind lines).
  Longprobe faces, battery/metrics/markers/probe files verified address-free.
- Sums: rig-side `raw/cert25zb/rig_serve_sums.txt` + the rig bank's
  `SHA256SUMS` refer to the PRE-redaction originals (which remain on the rig
  under `/root/build/p647/cert25zb/`); the local `SHA256SUMS` at the bank root
  hashes the REDACTED bytes (authoritative for the tracked tree).
- Only `raw/cert25zb/m4g/serve.log` ships tracked (force-add — it is the
  re-adjudication evidence); other serve logs stay untracked locally per the
  cert24 bank pattern, rig holds the originals.
