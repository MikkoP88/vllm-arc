#!/bin/bash
# leak_sweep.sh — publishing-hygiene gate (plan §11 rule 9).
#
# Scans the TRACKED tree (tracked == pushable) for material that must
# never reach a remote: real host addresses, credentials, private
# identities. Exit 0 = clean; exit 1 = hits (each printed). Run before
# every push/merge; new exceptions go in tools/leak_sweep.allow with a
# reason (path ERE + reason per line).
#
# Scope notes:
# - Private/RFC1918 subnets + the rig subnet are matched with NUMERIC
#   final octets only, so the in-tree redaction token 10.20.3.REDACTED
#   and prose "10.20.3.<host>" do not self-trigger.
# - Generic public IPv4 is NOT matched: dotted-quad version strings
#   (0.1.14.1, 5.0.0.93) are indistinguishable from it by shape. If a
#   public address ever needs checking, grep it explicitly.
# - Upstream placeholder credentials (api_key="EMPTY", ${{ secrets.* }})
#   contain no values and are not secrets.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 2

ALLOW=tools/leak_sweep.allow
fail=0
total=0

sweep() {  # sweep <name> <ERE>
  local name="$1" pat="$2" hit allowed
  while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    local path="${hit%%:*}"
    allowed=""
    if [ -f "$ALLOW" ]; then
      while IFS=$'\t' read -r are pat_re; do
        case "$are" in ''|'#'*) continue ;; esac
        if [[ "$path" =~ $are ]]; then allowed="$pat_re"; break; fi
      done < "$ALLOW"
    fi
    if [ -n "$allowed" ]; then
      echo "ALLOWED [$name] $hit   # $allowed"
    else
      echo "HIT     [$name] $hit"
      fail=1
      total=$((total + 1))
    fi
  done < <(git grep -nI -E -- "$pat")
}

sweep rig-subnet      '10\.20\.[0-9]{1,3}\.[0-9]{1,3}'
sweep rfc1918        '(^|[^0-9.])(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})([^0-9.]|$)'
sweep root-at-host    'root@[0-9]{1,3}\.'
sweep private-key     'BEGIN [A-Z0-9 ]*PRIVATE KEY'
sweep sshpass         'sshpass'
sweep password-assign '(PASSWORD|PASSWD)=[^ "$]+'
sweep identity        'MikkoPes|kotiverko'

if [ "$fail" -ne 0 ]; then
  echo "leak_sweep: $total hit(s) — scrub or allowlist-with-reason; do not push."
  exit 1
fi
echo "leak_sweep: clean."
