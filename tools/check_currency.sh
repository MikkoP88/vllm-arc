#!/bin/bash
# check_currency.sh — main-currency gate (plan §11 rule 10).
#
# "All main information is always up to date based on the latest certified
# image" (2026-10-31). Fails when any current-pointer on main
# names a certified tag older than the newest rel/v0.30.0-b70-* tag, or
# when one of the anchor points is missing. Run (with leak_sweep.sh)
# before every push of main.
#
# Anchor points a certification close-out MUST repoint (this file is the
# checklist): README certified-release headline + releases bullet,
# CONFIGURATION-arc image-table head, MIGRATION-arc identities table,
# PERFORMANCE-arc "Current certified image" section, envs.md chain
# currency note.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

latest=$(git tag -l 'rel/v0.30.0-b70-*' | sort -V | tail -1)
img="vllm-arc:${latest#rel/}"   # rel/v0.30.0-b70-cert4 -> vllm-arc:v0.30.0-b70-cert4
fail=0

ck() {  # ck <file> <fixed-string> <what>
  grep -qF "$2" "$1" || { echo "HIT [currency] $1: missing $3"; fail=1; }
}

ck README.md            "Certified release: **\`$latest\`**"  "certified-release headline"
ck README.md            "current: \`$latest\`"                 "releases bullet"
# Row-value anchors (2026-11-11, P6.38): an anywhere-in-file grep let a stale
# table row pass while the cert string appeared elsewhere in the same doc
# (MIGRATION S1 image row said cert22 for two certs). Anchor the actual
# current-pointer ROWS, not file-wide presence.
ck docs/CONFIGURATION-arc.md "| Image | \`$img\`"                "image-table row"
ck docs/MIGRATION-arc.md     "| image | \`$img\`"               "identities image row"
ck docs/MIGRATION-arc.md     "ARC_IMAGE=$img"                    "S2 boot block image"
ck docs/PERFORMANCE-arc.md   "## Current certified image"      "current-record heading"
ck docs/PERFORMANCE-arc.md   "**Current release = $latest**"  "current-release statement"
ck docs/envs.md              "current certified image \`$img\`" "env chain currency note"

# No doc may present an older tag AS THE certified release:
while IFS= read -r line; do
  f=${line%%:*}
  case "$line" in
    *"Certified release: **\`$latest\`"*) ;;  # the one true mention
    *"Certified release:"*)
      echo "HIT [currency] ${line%%:*}: older tag presented as THE certified release:"
      echo "    $line"
      fail=1 ;;
  esac
done < <(grep -rn "Certified release:" README.md docs/*.md || true)

if [ "$fail" -ne 0 ]; then
  echo "check_currency: STALE — latest certified tag is $latest ($img). Repoint the anchors above."
  exit 1
fi
echo "check_currency: clean ($latest / $img)."
