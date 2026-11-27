#!/bin/bash
# r266_selftest.sh -- offline selftest of the gate logic of
# tools/ar/r266_gemma_skin.sh (local: no docker, no rig, no GPU).
#
# The r266 functions are lifted verbatim from the script (every top-level
# `name() { ... }` body, by awk) and run AGAINST FIXTURE EVIDENCE in this
# shell (never in a command substitution: the failed-check counter
# GATE_FAIL must be visible to the assertions) -- a gate is proven on
# known evidence before it is trusted on the rig (the vacuous-gate laws:
# boot-snapshot markers, existence-only greens, zero-match clobbers):
#   skin_check        a gemma-skin serve log passes a gemma leg; the
#                     banked qwen-skin gemma serve log (r255-era gate,
#                     m8g) fails a gemma leg on BOTH checks (template,
#                     parser) and passes a qwen control; a gemma log does
#                     not pass a qwen control; a missing log fails
#   qa_check          P6.28-shaped answers pass all seven; r264-shaped
#                     degenerate answers fail six of seven; a missing
#                     answer fails the count as well
#   garble_check      ride completion + zero probe errors gate at gate=0
#                     and gate=1 alike; ZERO-GARBLE only at gate=1
#   longreturn_summary  full + one rehit cycle: PASS when every hash
#                     agrees; FAIL on cross-cycle drift, on a garble
#                     flag, on a truncation; T1_SEED MISS is its own gate
#   leg_verdict       the failed-check count between entry and exit
#   leg_summaries     NOT-RUN vs the five-line summary
#   run_one           an unknown leg name is a named failed check
#   script hygiene    bash -n, LF-only, the functions the flow calls exist
# Usage: bash tools/ar/r266_selftest.sh   Exit 0 = every case holds.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$HERE/r266_gemma_skin.sh
REPO=$(cd "$HERE/../.." && pwd)

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# the Windows python3 is a store stub: the lifted longreturn_summary needs
# a real interpreter under that name
mkdir -p "$TMP/bin"
printf '#!/bin/sh\nexec python "$@"\n' > "$TMP/bin/python3"
chmod +x "$TMP/bin/python3"
PATH="$TMP/bin:$PATH"

FUNCS=$(awk '/^[a-zA-Z_0-9]+\(\) \{/{p=1} p{print} p&&/^\}/{p=0}' "$SCRIPT")
eval "$FUNCS"

P=$TMP/bank
mkdir -p "$P"
GATE_FAIL=0
N_CASES=0
N_FAILS=0
O=""

check() {  # $1 = 0|1 (1 = holds) $2 = label $3 = detail
  N_CASES=$((N_CASES + 1))
  if [ "$1" = 1 ]; then
    echo "  ok   $2"
  else
    N_FAILS=$((N_FAILS + 1))
    echo "  FAIL $2 -- $3"
  fi
}
run_cap() {  # run in THIS shell (GATE_FAIL stays visible); output -> $O
  "$@" > "$TMP/last.out" 2>&1
  O=$(cat "$TMP/last.out")
}
risen() {  # $1 = expected failed checks since GATE_FAIL=0 $2 = label
  check "$([ "$GATE_FAIL" = "$1" ] && echo 1 || echo 0)" "$2" \
    "GATE_FAIL=$GATE_FAIL (want $1) out: $(echo "$O" | tr '\n' '|' | cut -c1-320)"
}
holds() {  # $1 = exit status of the assertion command (0 = holds) $2 = label $3 = detail
  check "$([ "$1" = 0 ] && echo 1 || echo 0)" "$2" "$3"
}
has() {  # $1 = fixed text $2 = label -- the last output carries it
  grep -qF -- "$1" <<< "$O"
  holds $? "$2" "missing [$1] in: $(echo "$O" | tr '\n' '|' | cut -c1-320)"
}
has_re() {  # $1 = regex $2 = label
  grep -qE -- "$1" <<< "$O"
  holds $? "$2" "missing /$1/ in: $(echo "$O" | tr '\n' '|' | cut -c1-320)"
}

# ---- fixtures ----
# the real qwen-skin non-default args line of a gemma leg (banked r255-era
# gate, m8g); synthesized from the same shape when the bank is absent
BANKED=$REPO/results/2026-11-26-p647-cert29-gate/m8g/serve.log
if [ -f "$BANKED" ]; then
  QWEN_LINE=$(grep -m1 "non-default args" "$BANKED")
  SRC_LINE="the banked r255-era m8g serve log"
else
  QWEN_LINE="non-default args: {'model': '/models/target', 'max_model_len': 40960, 'chat_template': '/root/chat_template_qwen38_high.jinja', 'reasoning_parser': 'qwen3', 'tool_call_parser': 'qwen3_coder', 'enable_auto_tool_choice': True}"
  SRC_LINE="a synthesized qwen-skin line (bank absent)"
fi
# what the same engine must print under the faithful skin: vLLM's
# non-default args drop every field left at its default (parser None,
# tool choice off), so the three parser keys disappear
GEMMA_LINE=$(printf '%s\n' "$QWEN_LINE" \
  | sed -e "s#'chat_template': '/root/chat_template_qwen38_high.jinja'#'chat_template': '/root/chat_template_gemma4.jinja'#" \
        -e "s/, 'reasoning_parser': 'qwen3'//" \
        -e "s/'reasoning_parser': 'qwen3', //" \
        -e "s/, 'tool_call_parser': 'qwen3_coder'//" \
        -e "s/'tool_call_parser': 'qwen3_coder', //" \
        -e "s/, 'enable_auto_tool_choice': True//" \
        -e "s/'enable_auto_tool_choice': True, //")
GEMMA_KEY="'chat_template': '/root/chat_template_gemma4.jinja'"
mk_skin() {  # $1 = dir $2 = line (empty = no log)
  mkdir -p "$1"
  [ -n "$2" ] && printf '%s\n' "$2" > "$1/serve_boot.log"
  return 0
}

QA_GOOD='PROMPT_OK P1 stop 5 '"'"'Paris'"'"'
PROMPT_OK P2 stop 210 '"'"'Here is a Python function that returns the n-th Fibonacci number using iteration:\n\n```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n```'"'"'
PROMPT_OK P3 stop 46 '"'"'Hydrogen, nitrogen, oxygen, fluorine and chlorine are gases at room temperature.'"'"'
PROMPT_OK P4 stop 12 '"'"'4:40pm'"'"'
PROMPT_OK P5 stop 5 '"'"'seven'"'"'
PROMPT_OK P6 stop 10 '"'"'s[::-1]'"'"'
PROMPT_OK P7 stop 15 '"'"'HEALTH CHECK OK'"'"'
QA_DONE'
QA_BAD='PROMPT_OK P1 stop 7 '"'"'\n\nParis'"'"'
PROMPT_OK P2 length 209 '"'"'\n\n<think>\nThe user wants a Python function that computes the n-th Fibonacci number using an iterative approach (not recursion).\n\nLet me think about the Fibonacc'"'"'
PROMPT_OK P3 length 236 '"'"'\n\n1. **Identify the criterion** -- an element must be in the gaseous state at typical room temperature (about 20 C / 68 F). \n2. **Recall the periodic table** --'"'"'
PROMPT_OK P4 length 64 '"'"'</ our our our our our our our our our o'"'"'
PROMPT_OK P5 length 24 '"'"'\n\n\n\n\n\n\n\n\n\n\n\n'"'"'
PROMPT_OK P6 length 64 '"'"'our our our our our our our our our our our our our'"'"'
PROMPT_OK P7 length 24 '"'"'</ our our our our our our our our our o'"'"'
QA_DONE'
mk_out() { mkdir -p "$1"; printf '%s\n' "$2" > "$1/probe.out"; }

mk_lp() {  # $1 = dir $2 = file -- four clean class lines
  mkdir -p "$1"
  {
    for c in essay code multi simple; do
      echo "[$c] t0_identical=True hashes=['aaaa1111', 'aaaa1111', 'aaaa1111'] garble={'cold': [], 'rehit': [], 'rehit2': []} t1_seed_identical=True prompt_tok=2273 chars=[912, 912, 912]"
    done
    echo "LONGPROBE x full: CLEAN"
  } > "$1/$2"
}
mk_json() {  # $1 = dir $2..=kwargs: drift=<class> garble=<class> trunc=<class> t1miss=<class> norehit=1
  mkdir -p "$1"
  python - "$1" "${@:2}" << 'PYEOF'
import json, sys
d, kw = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
full = {c: {"t0_identical": True, "hashes": ["aaaa1111"] * 3, "content_hashes": ["aaaa1111"] * 3,
            "garble": {"cold": [], "rehit": [], "rehit2": []}, "finish": ["stop"] * 3,
            "prompt_tok": 2273, "chars": [912] * 3, "t1_seed_identical": True,
            "t1_hashes": ["bbbb2222", "bbbb2222"], "t1_garble": [], "dt_cold": 1.0}
        for c in ("essay", "code", "multi", "simple")}
rehit = {c: {"post_churn_rehit_identical": True, "hashes": ["aaaa1111", "aaaa1111"],
             "garble": {"r1": [], "r2": []}, "prompt_tok": 2273}
         for c in ("essay", "code", "multi", "simple")}
if kw.get("drift"):            # one class answers differently after the churn
    rehit[kw["drift"]]["hashes"] = ["cccc3333", "cccc3333"]
if kw.get("garble"):           # a flag on one class's cold answer
    full[kw["garble"]]["garble"]["cold"] = ["REP-LOOP"]
if kw.get("trunc"):            # one class ran out of budget
    full[kw["trunc"]]["finish"] = ["length", "length", "length"]
if kw.get("t1miss"):           # one class ignores its seed
    full[kw["t1miss"]]["t1_seed_identical"] = False
    full[kw["t1miss"]]["t1_hashes"] = ["bbbb2222", "dddd4444"]
json.dump(full, open(d + "/longprobe_t_full.json", "w"))
if kw.get("norehit") != "1":
    json.dump(rehit, open(d + "/longprobe_tr1_rehit.json", "w"))
PYEOF
}

echo "[0] script hygiene"
bash -n "$SCRIPT"; holds $? "bash -n r266_gemma_skin.sh" "syntax error"
# the CR count is taken with tr: inside a script on this (MSYS) host a bare
# CR pattern does not survive the trip to grep, which then matches every line
CR=$(tr -cd '\015' < "$SCRIPT" | wc -c)
check "$([ "${CR:-0}" = 0 ] && echo 1 || echo 0)" "LF-only" "CR bytes=${CR:-0}"
for fn in kmark khealth w14_restore verify_removed gcount boot_wait garble_check \
          longreturn_summary churn skin_check qa_battery qa_check erraudit \
          run_skin_leg leg_verdict leg_summaries leg_g0f leg_mkgf leg_mkgq \
          leg_m8gf leg_m4gf run_one; do
  type "$fn" >/dev/null 2>&1; holds $? "function $fn defined" "missing"
done
grep -qF "$GEMMA_KEY" <<< "$GEMMA_LINE" && ! grep -qE "'reasoning_parser'|'tool_call_parser'|'enable_auto_tool_choice'" <<< "$GEMMA_LINE"
holds $? "fixture: the gemma-skin line carries the template and no parser key ($SRC_LINE)" "$GEMMA_LINE"
grep -qF "'reasoning_parser': 'qwen3'" <<< "$QWEN_LINE" && grep -qF "'tool_call_parser': 'qwen3_coder'" <<< "$QWEN_LINE"
holds $? "fixture: the qwen-skin line carries both parsers ($SRC_LINE)" "$QWEN_LINE"

echo "[1] skin_check"
GATE_FAIL=0; mk_skin "$P/a" "$GEMMA_LINE"
run_cap skin_check "$P/a" mkgf gemma; risen 0 "gemma log, gemma leg: no failed check"
has "SKIN[mkgf]: want=gemma $GEMMA_KEY" "gemma log: the SKIN line names the served template"
GATE_FAIL=0; mk_skin "$P/b" "$QWEN_LINE"
run_cap skin_check "$P/b" mkgf gemma; risen 2 "qwen log, gemma leg: TWO failed checks (template + parser)"
has "the gemma template is not the served template" "qwen log: the template failure is named"
has "a qwen parser rode the gemma leg" "qwen log: the parser failure is named"
GATE_FAIL=0
run_cap skin_check "$P/b" mkgq qwen; risen 0 "qwen log, qwen control: no failed check"
GATE_FAIL=0; mk_skin "$P/c" "$GEMMA_LINE"
run_cap skin_check "$P/c" mkgq qwen; risen 1 "gemma log, qwen control: ONE failed check"
has "the qwen control skin is not the served skin" "gemma log on a qwen control: named"
GATE_FAIL=0; mk_skin "$P/d" ""
run_cap skin_check "$P/d" m8gf gemma; risen 1 "missing log, gemma leg: fails (existence is not a pass)"
GATE_FAIL=0; mk_skin "$P/e" "$GEMMA_LINE"
run_cap skin_check "$P/e" g0f gemma
check "$([ -s "$P/e/skin.txt" ] && echo 1 || echo 0)" "the checked line is banked in skin.txt" ""

echo "[2] qa_check"
GATE_FAIL=0; mk_out "$P/q1" "$QA_GOOD"
run_cap qa_check "$P/q1"; risen 0 "P6.28-shaped answers: all seven pass"
has "QA_SUMMARY answers=7/7 gases=" "QA_SUMMARY line printed"
check "$([ -s "$P/q1/qa_check.txt" ] && echo 1 || echo 0)" "checks banked in qa_check.txt" ""
GATE_FAIL=0; mk_out "$P/q2" "$QA_BAD"
run_cap qa_check "$P/q2"; risen 6 "r264-shaped answers: six of seven fail (Paris still present)"
for k in P2_FIB P3_GASES P4_TRAIN P5_SEVEN P6_REVERSE P7_SANITY; do
  has "QA_$k: FAIL" "degenerate: QA_$k named"
done
! grep -q "QA_P1_PARIS: FAIL" <<< "$O"; holds $? "degenerate: Paris NOT flagged" "$O"
GATE_FAIL=0; mk_out "$P/q3" "$(printf '%s\n' "$QA_GOOD" | grep -v "P4 ")"
run_cap qa_check "$P/q3"; risen 2 "a missing answer: the count AND its check fail"
GATE_FAIL=0; mk_out "$P/q4" "PROMPT_ERR P1 HTTPError('500')"
run_cap qa_check "$P/q4"; risen 8 "all requests refused: count + seven checks = 8"

echo "[3] garble_check"
mk_lp "$P/g" lp_full.out
GATE_FAIL=0; run_cap garble_check "$P/g/lp_full.out" 1; risen 0 "four clean class lines, gate=1: pass"
GATE_FAIL=0; run_cap garble_check "$P/g/lp_full.out" 0; risen 0 "four clean class lines, gate=0: pass"
sed "s/^\[multi\].*/[multi] ERROR <HTTPError 500: Internal Server Error>/" "$P/g/lp_full.out" > "$P/g/lp_err.out"
GATE_FAIL=0; run_cap garble_check "$P/g/lp_err.out" 1; risen 2 "a crashed class, gate=1: probe-error + zero-garble"
has "PROBE-ERROR: FAIL" "crashed class: PROBE-ERROR named"
has "ZERO-GARBLE: FAIL" "crashed class: ZERO-GARBLE named"
GATE_FAIL=0; run_cap garble_check "$P/g/lp_err.out" 0; risen 1 "a crashed class, gate=0: probe-error still gates"
sed "s/^\[simple\].*/[simple] t0_identical=False hashes=['1', '2', '1'] garble={'cold': ['REP-LOOP'], 'rehit': [], 'rehit2': []} t1_seed_identical=True/" "$P/g/lp_full.out" > "$P/g/lp_flag.out"
GATE_FAIL=0; run_cap garble_check "$P/g/lp_flag.out" 1; risen 1 "a garble flag, gate=1: ZERO-GARBLE only"
GATE_FAIL=0; run_cap garble_check "$P/g/lp_flag.out" 0; risen 0 "a garble flag, gate=0: bank-only"
: > "$P/g/lp_empty.out"
GATE_FAIL=0; run_cap garble_check "$P/g/lp_empty.out" 1; risen 2 "an EMPTY out file: ride + zero-garble (never a vacuous pass)"
has "RIDE-COMPLETION: FAIL" "empty out file: RIDE-COMPLETION named"

echo "[4] longreturn_summary"
mk_json "$P/l1"
GATE_FAIL=0; run_cap longreturn_summary t "$P/l1" 1; risen 0 "clean full + clean rehit: LONGRETURN PASS, T1_SEED PASS"
grep -q "LONGRETURN: PASS" "$P/l1/longreturn_summary.txt" && grep -q "T1_SEED: PASS" "$P/l1/longreturn_summary.txt"
holds $? "summary file carries both PASS lines" "$O"
GATE_FAIL=0; run_cap longreturn_summary t "$P/l1" 0; risen 0 "same evidence ungated (gate=0): no failed check"
mk_json "$P/l2" drift=multi
GATE_FAIL=0; run_cap longreturn_summary t "$P/l2" 1; risen 1 "post-churn drift: LONGRETURN FAIL"
has "multi: cross-cycle hash drift ['aaaa1111', 'cccc3333']" "drift: the class and its hashes are named"
mk_json "$P/l3" garble=essay
GATE_FAIL=0; run_cap longreturn_summary t "$P/l3" 1; risen 1 "a garble flag: LONGRETURN FAIL"
mk_json "$P/l4" trunc=code
GATE_FAIL=0; run_cap longreturn_summary t "$P/l4" 1; risen 1 "a truncation: LONGRETURN FAIL"
has "code: truncation finish=['length', 'length', 'length']" "truncation: named with the finish reasons"
mk_json "$P/l5" t1miss=simple
GATE_FAIL=0; run_cap longreturn_summary t "$P/l5" 1; risen 1 "a seed miss: T1_SEED MISS (its own gate)"
has "LONGRETURN: PASS" "seed miss: LONGRETURN stays PASS"
has "T1_SEED: MISS simple" "seed miss: T1_SEED names the class"
mk_json "$P/l6" norehit=1
GATE_FAIL=0; run_cap longreturn_summary t "$P/l6" 1; risen 0 "full only (no rehit file): still a verdict"
has "LONGRETURN: PASS" "full only: PASS on clean evidence"

echo "[5] leg_verdict"
GATE_FAIL=3; run_cap leg_verdict mkgf 3
has "LEG_VERDICT[mkgf]: GREEN" "no failed check in the leg: GREEN"
GATE_FAIL=5; run_cap leg_verdict g0f 3
has "LEG_VERDICT[g0f]: RED failed_checks=2" "two failed checks in the leg: RED failed_checks=2"
GATE_FAIL=1; run_cap leg_verdict m8gf 0
has "LEG_VERDICT[m8gf]: RED failed_checks=1" "a failed check is charged to its own leg"

echo "[6] leg_summaries"
GATE_FAIL=0; run_cap leg_summaries
check "$([ "$(grep -c "NOT-RUN" <<< "$O" || true)" = 5 ] && echo 1 || echo 0)" "empty bank: five NOT-RUN lines" "$O"
mkdir -p "$P/mkgf"
cp "$P/e/skin.txt" "$P/mkgf/skin.txt"; cp "$P/q2/qa_check.txt" "$P/mkgf/qa_check.txt"
printf 'P266_DETERMINISM mkgf: cold_stable=3/3 warm_stable=3/3 churn_stable=3/3 cold_eq_warm=0/3 all_200=True salt_forces_cold=True\n' > "$P/mkgf/determinism.out"
cat > "$P/mkgf/garbage.out" << 'EOF'
[state_stability] 200:65b41dc505b622a1:'</ our our our' == 200:fbfa5ad7d9aa0afe:'\n\n\n' -> False
GARBAGE_SUITE_FAIL
EOF
cp "$P/l2/longreturn_summary.txt" "$P/mkgf/longreturn_summary.txt"
run_cap leg_summaries
grep -q "^R266_LEG_SUMMARY\[mkgf\]:" <<< "$O" && [ "$(grep -c "NOT-RUN" <<< "$O" || true)" = 4 ]
holds $? "one leg banked: its summary + four NOT-RUN" "$O"
has "skin:  'chat_template': '/root/chat_template_gemma4.jinja' " "summary: the skin"
has "qa:    QA_SUMMARY answers=7/7 gases=0 QA_P2_FIB: FAIL;QA_P3_GASES: FAIL" "summary: the QA verdict and its failed checks"
has "det:   P266_DETERMINISM mkgf: cold_stable=3/3" "summary: the determinism line"
has_re "suite: \[state_stability\] 200:65b41dc505b622a1.*-> False \| GARBAGE_SUITE_FAIL" "summary: state_stability + the suite verdict"
has "long:  LONGRETURN: FAIL multi: cross-cycle hash drift" "summary: LONGRETURN + T1_SEED"

echo "[7] run_one dispatch"
GATE_FAIL=1; run_one nosuch > /dev/null 2>&1
check "$([ "$GATE_FAIL" = 2 ] && echo 1 || echo 0)" "an unknown leg name: one failed check" "GATE_FAIL=$GATE_FAIL"

if [ "$N_FAILS" -gt 0 ]; then
  echo "R266_SELFTEST FAIL n_cases=$N_CASES fails=$N_FAILS"
  exit 1
fi
echo "R266_SELFTEST PASS n_cases=$N_CASES"
exit 0
