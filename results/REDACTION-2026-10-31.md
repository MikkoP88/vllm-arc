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

## Pass 2 (same day, later) — r70 evidence mirror (p70, P6.10)

The r70 comprehensive cert4 campaign banked
`results/2026-10-31-p70-r70-cert4-comprehensive/` (legs a, b,
a_invalid_compile_posture; leg c mirrored later, see Pass 3 if
applicable). Same two substitutions, `perl -pi` (MSYS sed law), applied
before first commit — the banked bytes never contained the raw
addresses:

- `10.20.3.<host>` → `10.20.3.REDACTED`: **40 occurrences** —
  5 serve-log files × 8 each: `a/serve_full.log`,
  `a/serve_pre_warm.log`, `a/serve_post_warm.log`,
  `a_invalid_compile_posture/serve_pre_warm.log`, `b/serve_full.log`.
- `10.100.8.<client>` → `10.100.8.REDACTED`: **42 occurrences** — same
  5 files (12 / 8 / 8 / 7 / 7).
- Byte-exactness verified by occurrence arithmetic: rig substitution
  +6 bytes each (16−10), client +7 each (17−10) → 40×6+42×7 = **534**
  total bytes, matching the measured size delta exactly (an earlier
  "+6/+6 → expect 492" scratch figure was the author's arithmetic slip,
  not a file discrepancy). Post-pass grep: 0 raw rig, 0 raw client;
  `tools/leak_sweep.sh` clean.

## Pass 3 (same day, evening) — r71 P2P legs evidence (p71, P6.11)

The r71 idle-window legs banked into
`results/2026-10-31-p71-r71-p2p-analysis/`. Same `perl -pi` law, applied
before first commit; a THIRD substitution joins the set here — the rig's
**hostname** (printed by oneCCL's `build_host_info` debug line and the
engine's topology prints):

- `10.20.3.<host>` → `10.20.3.REDACTED` and
  `<hostname>` → `REDACTED-HOST` (hostname = 8 bytes → 13, +5 each).
- `r71a_attempt1_invalid_numeric_level/serve_diag.log`: 1 rig address
  (+6). 
- `r71a_ccl_diag_lines.txt` (pre-gzip): 4 rig (+24) + 6 hostname
  (+30) = +54 — verified 56,478,532 → 56,478,586.
- `r71a_serve_diag.log` (pre-gzip): 5 rig (+30) + 12 hostname (+60) =
  +90 — verified 56,586,510 → 56,586,600.
- The two full logs are banked **gzipped** (byte-exact payload of the
  redacted files; 687 KB / 708 KB); rig-raw md5s recorded in
  `r71a_extracts.txt` for provenance. `r71b_d2d.txt`,
  `r71a_extracts.txt`, `r71a_first_resp.json` contain no raw
  addresses/hostnames (checked).
- Post-pass grep: 0 raw rig, 0 raw hostname in the dir;
  `tools/leak_sweep.sh` clean.

## Pass 4 (2026-11-01) — r70 leg c soak mirror (p70, P6.10 close-out)

The 4 h soak leg's mirror
`results/2026-10-31-p70-r70-cert4-comprehensive/c/` (124 files, 1.7 MB:
56 cycle batteries, 56 integrity probes, soak_log/soak_summary, final
spec gate, metrics, serve_full.log 9,444 lines, warm-v2 captures).
Same `perl -pi` law, applied before first commit — the banked bytes
never contained the raw addresses:

- `serve_full.log`: 169 rig (+1,014) + 287 client (+2,009) = **+3,023**
  bytes expected delta (prior session's byte-arithmetic verification;
  token counts re-verified at close-out: 169/287).
- `serve_pre_warm2.log`: 1 rig (+6) + 4 client (+28) = **+34**.
- `serve_post_warm2.log`: 1 rig (+6) + 11 client (+77) = **+83**.
- Total: 171 rig + 302 client occurrences = **+3,140 bytes**. Only
  these 3 of 124 files carry address bytes; the other 121 (cycles,
  integ probes, soak_log, soak_summary, spec_gate_final, metrics,
  boot, survival, warm_rehit2) swept clean — 0 raw addresses, 0
  hostnames (rig hostname checked: 0) in the whole `c/` tree.
  Post-pass `tools/leak_sweep.sh` clean.
  (Correction, Pass 5: the close-out's plain `git add` silently
  skipped these three `.log` files — upstream `.gitignore` `*.log` —
  so the close-out commit (and its pushes) held 121 of 124 files:
  Pass 4's "banked" was true of the disk, not of git. No unredacted
  bytes were ever pushed — the three logs were absent, not raw;
  banked redacted by Pass 5 the same day.)

## Pass 5 (2026-11-01) — referenced-but-unbanked log evidence (full-phase audit finding)

The 2026-11-01 all-phase validation audit found that **plain
`git add` silently drops `.log` files** (upstream `.gitignore:65
*.log`): 65 evidence logs **referenced by name from banked
write-ups** (each leg's own VERDICT/README "log: X.log (same dir)")
had never entered git on any branch — invisible to every earlier
sweep precisely because they were untracked (leak_sweep greps the
tracked tree). Two more were missing from the banked p72 branch
(the probe ALLPASS logs cited by PROBE.md). Fix: banked all 67
(`git add -f`, byte-identical to the on-disk evidence, under the
names as cited); the 37 carrying raw addresses/hostnames were
redacted FIRST with the same `perl -pi` law, then byte-arithmetic
verified:

- **124 rig (+744) + 112 client (+784) + 8 hostname (+40) = +1,568
  bytes**, per-file delta == token arithmetic ×37/37 (pre-sizes
  snapshotted before mutation; 0 raw refs after).
- The other 30 of 67 — incl. the Pass-4 c/ logs (already redacted)
  and the two p72 probe logs — contained no address/hostname bytes
  (checked before add).
- Largest classes: p2-leg `serve_full.log`s (8 rig + 10–14 client
  each, the uvicorn-access pattern), p3 r31/r32 capture logs,
  p67 serve/prof logs, build logs carrying the rig hostname
  (`build.log`, p47/cert2 builds, r67 gate2).
- Exact per-file counts recoverable with
  `git show <pass-5-commit> --stat -- results`. The audit also
  classified the remaining ~40 ignored-unreferenced files as
  quote-inline residue (the established non-banking practice for
  raw captures whose key lines are already quoted in banked
  write-ups) — left unbanked deliberately.

**Law going forward (the root cause)**: `.log` evidence must be
added with `git add -f` — a plain `git add <dir>` reports success
while silently skipping every log; every evidence commit should
diff `git status --ignored results/` against the intended list.
