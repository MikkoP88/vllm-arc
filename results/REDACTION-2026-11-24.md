# REDACTION 2026-11-24 — r231de banked serve logs

**Scope**: `results/2026-11-24-p647-r231de-e2e-faces/bank/{m4q25,
m4q28a,m4q28b}/serve.log` — the three e2e face-leg serve logs
committed in the r231de increment (PR #123).

## Cause

The serve logs carry the rig's own address and the proxy client's
address in uvicorn access lines and the `multiproc_executor`
`mq_connect_ip=` line. They were force-added (`*.log` is
gitignored) AFTER the pre-commit sweep pass — and the sweep's
untracked pass respects gitignore, so the still-untracked logs were
invisible to both passes at commit time. The tracked-tree pass
caught them on the next sweep (during cert28 tool prep), 40 min
after the merge. **Process fix**: force-added files get an explicit
`sweep <file>` (the gate's single-file mode) at `git add -f` time —
never rely on the directory pass for ignored files.

## Transformation (deterministic, auditable)

- Two substitutions, nothing else: `10.20.3.<host>` →
  `10.20.3.REDACTED` and `10.100.8.<client>` → `10.100.8.REDACTED`
  (24 occurrences across the three files: 3 `mq_connect_ip=` lines,
  21 uvicorn client-address tokens). Applied with `perl -pi`
  (MSYS sed injects CRLF into untouched lines).
- Byte-exactness verified: `git diff --numstat` = 8/8 lines per
  file (exactly the address-bearing lines; the only other diff
  lines are the three file headers). No measurement, timestamp,
  counter, or verdict byte altered.
- `SHA256SUMS` regenerated for the bank after the pass.
- `tools/leak_sweep.sh` exits 0 on the full pattern set after the
  pass.

The addresses remain recorded where the law allows them:
`CLAUDE.local.md` and the untracked plan.
