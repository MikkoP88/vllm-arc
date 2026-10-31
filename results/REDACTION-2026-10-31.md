# REDACTION 2026-10-31 — rig host address scrubbed from banked results

**Directive**: owner, 2026-10-31 — "run full sweep and add plan: never
upload any secrets, ip address and etc." Tracked tree == pushable tree
(plan §11 rule 9, added the same day); the rig's real address
(`10.20.3.<host>`) appeared in 32 banked `results/` files and was scrubbed.

## Transformation (deterministic, auditable)

- Two string substitutions, nothing else:
  `10.20.3.<host>` → `10.20.3.REDACTED` (88 occurrences, 32 files — the
  rig's own address: uvicorn client lines, `mq_connect_ip=`, prose), and
  `10.100.8.<client>` → `10.100.8.REDACTED` (119 occurrences, 18 files —
  the harness/bench client's intranet address in uvicorn access lines;
  caught by `tools/leak_sweep.sh`'s RFC1918 pattern during this same
  pass, not by the first targeted grep — the gate earned its keep on
  day one).
- Byte-exactness verified after the pass: `git diff` contains ONLY
  IP-bearing lines (0 non-IP changed lines; line-ending/CR bytes
  unchanged everywhere — 14 files carry original CRs from terminal
  capture, counts identical before/after). First attempt with `sed -i`
  was REVERTED and redone with `perl -pi` because MSYS sed injected
  CRLF into untouched lines (evidence-law violation; caught by the
  non-IP-diff-line check before commit).
- No measurement, timestamp, counter, or verdict byte was altered.
  Where the address appeared: uvicorn access-log client addresses
  (`10.20.3.REDACTED:55852 - "GET /v1/models"`), the
  `multiproc_executor` `mq_connect_ip=` line, and prose
  (`Rig: 10.20.3.REDACTED`) in env.txt / VERDICT.md / audit notes.

## E-law note

Banked results bytes are immutable by the evidence law; this edit is the
one sanctioned exception (owner directive, hygiene > byte-preservation
for HOST ADDRESSES only), and it is recorded here + in the squash commit
so the audit trail stays complete. The address itself remains valid
locally: it lives in the untracked layer (`CLAUDE.local.md`, plan §9).

## Already-pushed history (caveat, deliberate non-action)

Commits merged before 2026-10-31 still contain the address in these same
log files (origin mirror + github branches/tags). History rewrite is
REJECTED: certified `rel/*` tags and IMAGE_MANIFEST `fork_sha` lines pin
commit SHAs — rewriting invalidates the certification chain. The law
(plan §11 rule 9) protects the tree forward; if the repo's visibility
ever changes, the owner decides on a rebuild-from-scratch re-publish.

## Files

Rig address: 32 files listed below. Client address: 18 files, all serve
logs in the 2026-10-28 p2 legs_pull, 2026-10-31 p65 (cert4 + r48a
sub-legs), and 2026-10-31 p66-r66a dirs — largely the same logs.

2026-10-27-p1-vxk-wedge-audit/{audit_site_grep.txt,env.txt},
2026-10-27-p13-vxk-wheel/{VERDICT.md,env.txt},
2026-10-28-p2-graph-p12b-r1r2/{env.txt}, 2026-10-28-p2-graph-p12b-r3/{env.txt},
2026-10-28-p2-graph-p12b-r4r5/{env.txt}, 2026-10-27-p2-graph-defectb-answer/{env.txt},
2026-10-28-p2-p14-prefill-r29/legs_pull/{serve_r29_final.log},
2026-10-28-p2-p33-draft-structure-r26/legs_pull/{serve_r26_final.log},
2026-10-28-p2-r27-draft-ar-attribution/legs_pull/{serve_eager_final.log},
2026-10-28-p2-r28-local-argmax-ab/legs_pull/{serve_r28_final.log},
2026-10-31-p65-cert4/{serve_full_default.log, serve_full_p65face.log,
cert4tag/serve_full_p65face.log},
2026-10-31-p65-r48a-stall-capture/{serve_full_r48b..r48l variants + r48f/r48g/
r48h/r48i/r48iv2/r48j/r48k sub-dirs},
2026-10-31-p66-r66a-prefill-capture/{serve_thr2048.log, serve_thr4096.log,
serve_thr8192.log, serve_thrnone.log}.

(Exact list recoverable with:
`git show <this-commit> --stat -- results` — 32 files, 88 line pairs.)

## Non-findings (swept, clean)

No credentials/passwords/tokens/keys (the gate's credential patterns —
ssh password helpers, credential assignments, private-key blocks, bearer
strings — all zero), no MAC addresses, no Windows
usernames, no other RFC1918 addresses in fork-own files (upstream
placeholders `${{ secrets.* }}` in `.github/` and `api_key="EMPTY"`
examples are upstream content and contain no real values). The owner's
git identity email in `2026-10-30-p5-r45-env-log-audit/evidence.txt` is
commit-metadata-public by design (marker-free commit law) and was left
as banked bytes.

Enforcement going forward: `tools/leak_sweep.sh` (exit 0 required
before every push) + allowlist `tools/leak_sweep.allow`.
