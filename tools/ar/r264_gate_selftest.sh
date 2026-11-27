#!/bin/bash
# r264 selftest -- exercises the r264b gate helpers (garble_check,
# leg_verdict, run_one, state_of) OUTSIDE the gate script, on banked
# evidence: the crashed gemma leg of the first cert29 image
# (results/2026-11-26-p647-cert29-gate/m8g) and healthy legs (m8q, g0).
# Functions are extracted by name from the gate script so the test runs
# the shipped bytes. Exit 0 = every expectation held.
# Run: bash r264_selftest.sh <repo>
set -uo pipefail
REPO=${1:?repo path}
GATE=$REPO/tools/ar/r264_cert29_gate.sh
BANK=$REPO/results/2026-11-26-p647-cert29-gate
T=$(mktemp -d)
FAILS=0
ok() { echo "[PASS] $1"; }
bad() { echo "[FAIL] $1"; FAILS=$((FAILS + 1)); }
check() {  # $1 = expectation $2 = got $3 = label
  if [ "$1" = "$2" ]; then ok "$3 ($2)"; else bad "$3 (want $1, got $2)"; fi
}
count_up() { GATE_FAIL=$((GATE_FAIL + 1)); }

extract() {  # $1 = function name -> body on stdout
  awk -v fn="$1" '$0 ~ "^"fn"\\(\\) \\{" { on = 1 } on { print } on && $0 == "}" { exit }' "$GATE"
}

# ---- the functions under test + their globals, exactly as the gate defines them
{
  echo "GATE_FAIL=0"
  echo "P=$T/bank"
  extract garble_check
  extract leg_verdict
  extract run_one
  extract state_of
} > "$T/fns.sh"
mkdir -p "$T/bank/_battery"
: >> "$T/bank/_battery/leg_verdicts.txt"
CNAME=arc-selftest
# shellcheck disable=SC1091
source "$T/fns.sh"
for fn in garble_check leg_verdict run_one state_of; do
  type "$fn" >/dev/null 2>&1 || { bad "function $fn not extracted"; }
done

echo "== garble_check: crashed gemma leg (r255 m8g, the first cert29 image), bank-only"
GATE_FAIL=0
garble_check "$BANK/m8g/longprobe_full.out" 0
check 1 "$GATE_FAIL" "crashed full output counts exactly one failed check (probe errors)"
GATE_FAIL=0
garble_check "$BANK/m8g/longprobe_rehit1.out" 0
check 1 "$GATE_FAIL" "crashed rehit output counts one failed check"

echo "== garble_check: healthy qwen leg (r255 m8q), absolute mode"
GATE_FAIL=0
for f in full rehit1 rehit2 rehit3; do garble_check "$BANK/m8q/longprobe_$f.out" 1; done
check 0 "$GATE_FAIL" "healthy qwen leg: zero failed checks"

echo "== garble_check: healthy native gemma leg (r255 g0), bank-only mode"
GATE_FAIL=0
for f in full rehit1 rehit2 rehit3; do garble_check "$BANK/g0/longprobe_$f.out" 0; done
check 0 "$GATE_FAIL" "healthy g0 leg: zero failed checks"

echo "== garble_check: truncated output (3 class lines) must fail"
head -3 "$BANK/m8q/longprobe_full.out" > "$T/trunc.out"
GATE_FAIL=0
garble_check "$T/trunc.out" 0
check 1 "$GATE_FAIL" "truncated output: one failed check (ride completion)"

echo "== garble_check: an ERROR line among four class lines must fail"
sed '2s/.*/[code] ERROR HTTPError(500)/' "$BANK/m8q/longprobe_full.out" > "$T/err.out"
GATE_FAIL=0
garble_check "$T/err.out" 0
check 1 "$GATE_FAIL" "error line: one failed check (probe errors)"

echo "== leg_verdict / state_of"
GATE_FAIL=0; leg_verdict m8g 0
count_up; leg_verdict m4g 0
count_up; count_up; leg_verdict mkg 1
count_up; leg_verdict lmc 0
leg_verdict compare 3
check GREEN "$(state_of m8g)" "state_of m8g (0 new failures)"
check RED "$(state_of m4g)" "state_of m4g (1 new failure)"
check RED "$(state_of mkg)" "state_of mkg (2 new failures after an earlier one)"
check RED "$(state_of lmc)" "state_of lmc"
check NOT-RUN "$(state_of g0)" "state_of g0 (never run)"
echo "-- verdict lines:"; cat "$T/bank/_battery/leg_verdicts.txt"
check 5 "$(wc -l < "$T/bank/_battery/leg_verdicts.txt")" "one verdict line per leg_verdict call"

echo "== precondition logic (as in the battery tail)"
PRE_OK=1; REQ=""; REST=""
for l in m8g m4g mkg lmc; do REQ="$REQ $l:$(state_of $l)"; done
for l in gref g0 compare ctrl m8q m4q mkq; do REST="$REST $l:$(state_of $l)"; done
for l in m8g m4g mkg lmc gref g0 compare ctrl m8q m4q mkq; do
  [ "$(state_of $l)" = GREEN ] || PRE_OK=0
done
echo "R264_CERT_PRECONDITION gemma_mx+lmcache=[$REQ ] rest=[$REST ] -> $([ "$PRE_OK" = 1 ] && echo CERT_CUT_ALLOWED || echo CERT_CUT_BLOCKED)"
check 0 "$PRE_OK" "red lanes block the certificate"

echo "== all lanes green -> allowed"
: > "$T/bank/_battery/leg_verdicts.txt"
for l in m8g m4g mkg lmc gref g0 compare ctrl m8q m4q mkq; do
  echo "2026-11-27T00:00:00Z $l green=1 failed_checks=0" >> "$T/bank/_battery/leg_verdicts.txt"
done
PRE_OK=1
for l in m8g m4g mkg lmc gref g0 compare ctrl m8q m4q mkq; do
  [ "$(state_of $l)" = GREEN ] || PRE_OK=0
done
check 1 "$PRE_OK" "eleven green legs allow the certificate"

echo "== run_one: an unknown leg counts a failure and gets a RED verdict"
GATE_FAIL=0; run_one bogus
check RED "$(state_of bogus)" "unknown leg is RED"

echo "== R264_SELFTEST fails=$FAILS"
rm -rf "$T"
[ "$FAILS" = 0 ]
