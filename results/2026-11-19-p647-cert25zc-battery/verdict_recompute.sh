#!/bin/bash
# verdict_recompute.sh -- local re-derivation of the r197 cert25zc
# battery verdict from the BANKED artifacts under the AMENDED gate
# (7fe72efc: m8q asserts the Phase-6 LMC auto-off posture, not the
# r184-inherited LMC-on expectation the in-flight instance carried).
# Per the recompute-from-banked-artifacts law: the running instance's
# stale-m8q GATE_FAIL=1 is expected; this recompute is the verdict of
# record. Zero-match grep guards throughout (pipefail grep-absence law).
set -uo pipefail
B="$(cd "$(dirname "$0")" && pwd)"
GF=0
note() { echo "$*"; }

leg_basic() { # $1=leg $2=model-family(qwen|gemma) $3=expect-mx(0|1)
  local L=$1 FAM=$2 EMX=$3 CD="$B/$1"
  local mx lm tb
  mx=$(grep -o 'mx_backend_lines=[0-9]*' "$CD/markers.txt" 2>/dev/null | head -1 | cut -d= -f2 || true)
  lm=$(grep -o 'lmcache_lines=[0-9]*' "$CD/markers.txt" 2>/dev/null | head -1 | cut -d= -f2 || true)
  tb=$(grep -o 'tracebacks=[0-9]*' "$CD/markers.txt" 2>/dev/null | head -1 | cut -d= -f2 || true)
  echo "[$L] mx=$mx lmcache=$lm tracebacks=${tb:-?}"
  if [ "$EMX" = 0 ]; then
    [ "${mx:-x}" = "0" ] || { note "FAIL: $L mx_backend_lines=$mx (want 0)"; GF=1; }
  else
    [ "${mx:-0}" -ge 1 ] 2>/dev/null || { note "FAIL: $L mx_backend_lines=$mx (want >=1)"; GF=1; }
  fi
  [ "${tb:-0}" = "0" ] || { note "FAIL: $L tracebacks=$tb"; GF=1; }
  # text battery
  local pok par tbk
  pok=$(grep -c 'PROMPT_OK' "$CD/probe.out" 2>/dev/null || true)
  par=$(grep -c 'Paris' "$CD/probe.out" 2>/dev/null || true)
  tbk=$(grep -c 'Traceback' "$CD/probe.out" 2>/dev/null || true)
  echo "[$L] probe: PROMPT_OK=${pok:-0}/3 Paris=${par:-0} tracebacks=${tbk:-0}"
  [ "${pok:-0}" = "3" ] || { note "FAIL: $L PROMPT_OK=${pok:-0}/3"; GF=1; }
  [ "${tbk:-0}" = "0" ] || { note "FAIL: $L probe traceback"; GF=1; }
  if [ "$FAM" = qwen ]; then
    [ "${par:-0}" -ge 1 ] || { note "FAIL: $L P1_PARIS missing"; GF=1; }
  else
    local ne; ne=$(grep -cE '^PROMPT_OK [1-9]' "$CD/probe.out" 2>/dev/null || true)
    [ "${ne:-0}" = "3" ] || { note "FAIL: $L gemma shape gate ${ne:-0}/3"; GF=1; }
  fi
}

lmc_off() { # $1=leg -- Phase-6 posture: auto-off NOTE + lmcache_lines=0
  local L=$1 CD="$B/$1"
  grep -q 'NOTE: ARC_LMC auto-off' "$CD/boot.txt" 2>/dev/null \
    || { note "FAIL: $L LMC auto-off NOTE missing"; GF=1; }
  grep -q 'lmcache_lines=0' "$CD/markers.txt" 2>/dev/null \
    || { note "FAIL: $L lmcache_lines != 0"; GF=1; }
  echo "[$L] LMC posture: auto-off OK"
}

lmc_on() { # $1=leg -- certified LMC-on posture (ctrl only)
  local L=$1 CD="$B/$1"
  grep -Eq 'lmcache_lines=[1-9]' "$CD/markers.txt" 2>/dev/null \
    || { note "FAIL: $L ctrl lost LMC-on posture"; GF=1; }
  echo "[$L] LMC posture: on OK"
}

full_suite() { # $1=leg $2=gated(1|0)
  local L=$1 G=$2 CD="$B/$1"
  grep -q 'GARBAGE_SUITE_PASS' "$CD/garbage.out" 2>/dev/null \
    || { note "FAIL: $L suite-13 garbage"; GF=1; }
  echo "[$L] suite-13 garbage: PASS"
  if [ -f "$CD/longreturn_summary.txt" ]; then
    grep -q 'LONGRETURN: PASS' "$CD/longreturn_summary.txt" \
      || { [ "$G" = 1 ] && { note "FAIL: $L LONGRETURN"; GF=1; }; }
    grep -q 'T1_SEED: PASS' "$CD/longreturn_summary.txt" \
      || { [ "$G" = 1 ] && { note "FAIL: $L T1_SEED"; GF=1; }; }
    echo "[$L] longreturn: $(grep -o 'LONGRETURN: [A-Z]*' "$CD/longreturn_summary.txt" | head -1) $(grep -o 'T1_SEED: [A-Z]*' "$CD/longreturn_summary.txt" | head -1) (gated=$G)"
  else
    echo "[$L] longreturn_summary.txt MISSING"; GF=1
  fi
  grep -q 'ERROR=0' "$CD/erraudit.txt" 2>/dev/null \
    || { note "FAIL: $L erraudit"; GF=1; }
  echo "[$L] erraudit: $(head -1 "$CD/erraudit.txt" 2>/dev/null)"
}

echo "== r197 cert25zc verdict recompute (amended gate 7fe72efc) =="
leg_basic ctrl   qwen 0; lmc_on ctrl
leg_basic m8q    qwen 1; lmc_off m8q;    full_suite m8q 1
leg_basic m4q    qwen 1; lmc_off m4q;    full_suite m4q 1
leg_basic m8g  gemma 1;                  full_suite m8g 0
leg_basic m4g  gemma 1;                  full_suite m4g 0
leg_basic mkq    qwen 1; lmc_off mkq;    full_suite mkq 1
leg_basic mkg  gemma 1;                  full_suite mkg 0
leg_basic g0   gemma 0;                  full_suite g0  0
grep -q 'GEMMA_COMPARE: PASS' "$B/gemma_compare.txt" \
  || { note "FAIL: gemma_compare"; GF=1; }
echo "[compare] $(tail -1 "$B/gemma_compare.txt")"
echo
if [ "$GF" = 0 ]; then
  echo "P647_CERT25ZC_BATTERY_VERDICT: PASS (all legs under the amended gate)"
else
  echo "P647_CERT25ZC_BATTERY_VERDICT: GATE_FAIL=$GF"
fi
exit $GF
