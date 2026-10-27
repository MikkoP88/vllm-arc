#!/bin/bash
# wheel_guard.sh — md5 pin verification before any pinned wheel install.
# D-chain governance port (plan §6 law: no external dependency is installed
# into a vllm-arc image without md5 verification at image-build time).
#
# Usage: wheel_guard.sh <pins.md5> <wheels-dir>
# pins.md5 format: "<md5>  <filename>" lines, comments (#) ignored.
set -euo pipefail

pins=$1
dir=$2
fail=0

while read -r sum file; do
  if [ ! -f "$dir/$file" ]; then
    echo "WHEEL_GUARD MISSING $file"
    fail=1
    continue
  fi
  got=$(md5sum "$dir/$file" | cut -d' ' -f1)
  if [ "$got" != "$sum" ]; then
    echo "WHEEL_GUARD MISMATCH $file (want $sum got $got)"
    fail=1
  fi
done < <(grep -vE '^[[:space:]]*(#|$)' "$pins")

if [ "$fail" != 0 ]; then
  echo "WHEEL_GUARD FAILED"
  exit 1
fi
echo "WHEEL_GUARD OK ($(grep -cvE '^[[:space:]]*(#|$)' "$pins") pins verified)"
