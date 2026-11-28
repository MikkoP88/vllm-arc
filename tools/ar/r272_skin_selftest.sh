#!/bin/bash
# r272_skin_selftest.sh -- run LOCALLY (git-bash). Executable proof of the
# r272 gemma skin law in tools/serve_arc_b70.sh: the shipped bytes of the
# template-default + family-resolution + template-case + parser-case
# fragments, sed-extracted from the file (never a copy -- drift in the
# serve script fails this selftest), executed under the skin postures:
#   T1 qwen-unset        -> the certified qwen defaults survive byte-identical
#                           (qwen template mount + qwen3/qwen3_coder flags)
#   T2 gemma-unset       -> the skin FULLY REMOVED: no template, no parser
#                           flags (the directive's "cannot be found" class --
#                           NEVER the qwen defaults)
#   T3 gemma+g3-template  -> the gemma template mounts, empty parsers omitted
#   T4 gemma+qwen-skin   -> BOOT-LOUD FATAL on every qwen mixing face
#                           (template value, __qwen_default__, both parsers)
#   T5 gemma+gemma4 parsers (the official-structure leg class) -> lawful pass
# The fragments are pure bash -- no docker is touched.
set -uo pipefail
S=$(cd "$(dirname "$0")/../.." && pwd)/tools/serve_arc_b70.sh
[ -f "$S" ] || { echo "R272_NO_SERVE_SCRIPT $S"; exit 1; }

# extract the four fragments by their unique line anchors (shipped bytes)
FAM=$(sed -n '/^# r272: the skin defaults resolve per model family$/,/^ARC_TEMPLATE=\${ARC_TEMPLATE-__qwen_default__}$/p' "$S")
TPL=$(sed -n '/^case "\$ARC_TEMPLATE" in$/,/^esac$/p' "$S")
PAR=$(sed -n '/^if \[ "\$ARC_FAMILY" = gemma \]; then$/,/^fi$/p' "$S")
[ -n "$FAM" ] || { echo "R272_EXTRACT_FAIL family fragment (the ARC_TEMPLATE default line moved)"; exit 1; }
[ -n "$TPL" ] || { echo "R272_EXTRACT_FAIL template fragment (the case line moved)"; exit 1; }
[ -n "$PAR" ] || { echo "R272_EXTRACT_FAIL parser fragment (the family if moved)"; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
: > "$TMP/chat_template_qwen38_high.jinja"    # T1's CT_SRC must exist
: > "$TMP/chat_template_gemma4.jinja"         # T3's ARC_TEMPLATE must exist

run_case() {  # env assignments in "$@"; CT_SRC from the caller's env
  local OUT
  OUT=$(env "$@" CT_SRC="${CT_SRC:-/nonexistent_qwen_tmpl.jinja}" \
    bash -c "$FAM
CT_MOUNT=''; CT_FLAG=''
$TPL
$PAR
echo \"\$ARC_FAMILY|\$CT_FLAG|\$RP_FLAG|\$TP_FLAG\"" 2>&1) || true
  if echo "$OUT" | grep -q "FATAL"; then
    echo "FATAL|$(echo "$OUT" | grep FATAL | head -1 | cut -c1-90)"
  else
    echo "$OUT" | tail -1
  fi
}

PASS=0; FAIL=0
chk() {  # $1 = name $2 = expected $3 = got
  if [ "$2" = "$3" ]; then
    echo "PASS $1"; PASS=$((PASS+1))
  else
    echo "FAIL $1: expected [$2] got [$3]"; FAIL=$((FAIL+1))
  fi
}

# T1: qwen, all knobs unset -> the certified qwen skin byte-identical
CT_SRC="$TMP/chat_template_qwen38_high.jinja"
chk T1_qwen_unset \
  "qwen|--chat-template /root/chat_template_qwen38_high.jinja|--reasoning-parser qwen3|--tool-call-parser qwen3_coder --enable-auto-tool-choice" \
  "$(run_case ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8)"

# T2: gemma, all knobs unset -> the skin FULLY REMOVED (never qwen)
CT_SRC=/nonexistent_qwen_tmpl.jinja
chk T2_gemma_unset "gemma|||" \
  "$(run_case ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b)"

# T3: gemma + the lawful g3 template + empty parsers -> template mounts,
# parser flags omitted
chk T3_gemma_g3 "gemma|--chat-template /root/chat_template_gemma4.jinja||" \
  "$(run_case ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
    ARC_TEMPLATE=$TMP/chat_template_gemma4.jinja ARC_RPARSER= ARC_TPARSER=)"

# T4a: gemma + an explicit qwen template value -> boot-loud FATAL
GOT=$(run_case ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
  ARC_TEMPLATE=/root/build/v127_stage/chat_template_qwen38_high.jinja)
echo "$GOT" | grep -q "FATAL|.*qwen chat template" \
  && { echo "PASS T4a_gemma_qwen_template_refused"; PASS=$((PASS+1)); } \
  || { echo "FAIL T4a_gemma_qwen_template_refused: got [$GOT]"; FAIL=$((FAIL+1)); }

# T4b: gemma + ARC_TEMPLATE=__qwen_default__ explicitly -> boot-loud FATAL
GOT=$(run_case ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
  ARC_TEMPLATE=__qwen_default__)
echo "$GOT" | grep -q "FATAL|.*ARC_TEMPLATE=__qwen_default__" \
  && { echo "PASS T4b_gemma_qwen_default_refused"; PASS=$((PASS+1)); } \
  || { echo "FAIL T4b_gemma_qwen_default_refused: got [$GOT]"; FAIL=$((FAIL+1)); }

# T4c: gemma + qwen reasoning parser -> boot-loud FATAL
GOT=$(run_case ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_RPARSER=qwen3)
echo "$GOT" | grep -q "FATAL|.*qwen reasoning parser" \
  && { echo "PASS T4c_gemma_qwen_rparser_refused"; PASS=$((PASS+1)); } \
  || { echo "FAIL T4c_gemma_qwen_rparser_refused: got [$GOT]"; FAIL=$((FAIL+1)); }

# T4d: gemma + qwen tool parser -> boot-loud FATAL
GOT=$(run_case ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_TPARSER=qwen3_coder)
echo "$GOT" | grep -q "FATAL|.*qwen tool parser" \
  && { echo "PASS T4d_gemma_qwen_tparser_refused"; PASS=$((PASS+1)); } \
  || { echo "FAIL T4d_gemma_qwen_tparser_refused: got [$GOT]"; FAIL=$((FAIL+1)); }

# T5: gemma + gemma-native parsers, no template -> the official-structure
# leg class passes lawfully (template removed, gemma parser flags set)
chk T5_gemma_gemma4_parsers "gemma||--reasoning-parser gemma4|--tool-call-parser gemma4 --enable-auto-tool-choice" \
  "$(run_case ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
    ARC_RPARSER=gemma4 ARC_TPARSER=gemma4)"

echo "R272_SKIN_SELFTEST pass=$PASS fail=$FAIL"
[ "$FAIL" = 0 ] || exit 1
exit 0
