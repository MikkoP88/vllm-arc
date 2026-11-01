#!/usr/bin/env bash
# p73 local logic gate — bash/awk MIRROR of _lmc_slice_store_window's
# window arithmetic (no local python on the dev box; the rig boot's
# py_compile remains the syntax gate, as in r48i). The mirror re-implements
# the helper's index math verbatim (same branch order) and drives it over
# the scenarios that matter for P6.13; the assertions encode the
# pre-registered invariants: <=budget tokens per non-final window, every
# non-final edge chunk-aligned, monotone cursor, exact final window,
# equivalence to the un-sliced call when the span fits one window.
#
# Adapter source of truth:
# vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py::_lmc_slice_store_window
set -u

# mirror(mask_len, f_first_true, offset_cursor, budget, chunk, n)
# prints "start end done" for the NEXT window, mirroring the helper.
mirror() {
  local n=$1 f=$2 offset=$3 budget=$4 chunk=$5
  local start end
  if (( n == 0 )); then echo "-1 -1 1"; return; fi
  if (( f >= n )); then echo "-1 -1 1"; return; fi
  if (( offset > f )); then start=$offset; else start=$f; fi
  if (( start >= n )); then echo "-1 -1 1"; return; fi
  local span=$(( budget > chunk ? budget : chunk ))
  end=$(( start + span )); if (( end > n )); then end=$n; fi
  if (( end < n )); then
    local aligned=$(( end / chunk * chunk ))
    local floor2=$(( start + chunk )); if (( floor2 > n )); then floor2=$n; fi
    if (( aligned < floor2 )); then aligned=$floor2; fi
    end=$aligned
  fi
  local done=0; if (( end >= n )); then done=1; fi
  echo "$start $end $done"
}

fails=0
chk() { # chk name expected actual
  if [[ "$2" == "$3" ]]; then echo "PASS $1"; else
    echo "FAIL $1: expected [$2] got [$3]"; fails=$((fails+1)); fi
}

# S1 — 64k deferred final: f=0 offset=0 n=63232 chunk=1664 S=4096.
# Windows must be <=4096 tokens, chunk-aligned except the last, cover
# [0,63232) exactly once, no gaps/overlaps.
s1_windows=""; cur=0; f=0; n=63232; S=4096; C=1664
while :; do
  read -r st en dn <<<"$(mirror $n $f $cur $S $C)"
  (( st < 0 )) && break
  s1_windows="$s1_windows $st:$en"
  (( en <= cur )) && { echo "FAIL S1 no-progress at $cur"; fails=$((fails+1)); break; }
  cur=$en; (( dn )) && break
done
# coverage + alignment + budget checks
cov=0; prev=0; ok=1
for w in $s1_windows; do
  st=${w%%:*}; en=${w##*:}
  (( st == prev )) || ok=0
  tok=$(( en - st ))
  if (( en < n )); then
    (( tok <= S )) || ok=0
    (( en % C == 0 )) || ok=0
  fi
  prev=$en; cov=$(( cov + tok ))
done
chk "S1 windows tile [0,63232) contiguously" "1" "$ok"
chk "S1 total tokens stored" "$n" "$cov"
s1_count=$(wc -w <<<"$s1_windows")
# aligned-window law: window size = floor(S/C)*C = 4096//1664*1664 =
# 3328 tok; 63232/3328 = 19.0 exactly => 19 windows.
chk "S1 window count == 19 (aligned 3328-tok windows)" "19" "$s1_count"
echo "S1 windows:$s1_windows"

# S2 — span fits one window (n=3000, f=0, S=4096): single window
# [0,3000), done=1 — byte-equivalent to the un-sliced call shape.
read -r st en dn <<<"$(mirror 3000 0 0 4096 1664)"
chk "S2 single window" "0 3000 1" "$st $en $dn"

# S3 — chunk-row pending with leading skip (f=8320 aligned, offset=8320,
# n=9984): one window, offset==f, mask tail semantics preserved.
read -r st en dn <<<"$(mirror 9984 8320 8320 4096 1664)"
chk "S3 skip-leading row" "8320 9984 1" "$st $en $dn"

# S4 — remainder cursor drives next window (64k slice 2): cursor=3328.
read -r st en dn <<<"$(mirror 63232 0 3328 4096 1664)"
chk "S4 cursor-resume" "3328 6656 0" "$st $en $dn"

# S5 — budget smaller than chunk: progress guaranteed (one chunk).
read -r st en dn <<<"$(mirror 63232 0 0 512 1664)"
chk "S5 budget<chunk => chunk window" "0 1664 0" "$st $en $dn"

# S6 — nothing storeable (mask all-False => f>=n representation).
read -r st en dn <<<"$(mirror 63232 63232 0 4096 1664)"
chk "S6 nothing storeable" "-1 -1 1" "$st $en $dn"

# S7 — 32k face (n=31616, S=4096, C=1664): tiling invariant.
cur=0; prev=0; cov=0; ok=1
while :; do
  read -r st en dn <<<"$(mirror 31616 0 $cur 4096 1664)"
  (( st < 0 )) && break
  (( st == prev )) || ok=0
  tok=$(( en - st ))
  if (( en < 31616 )); then (( tok <= 4096 )) || ok=0; (( en % 1664 == 0 )) || ok=0; fi
  prev=$en; cov=$(( cov + tok )); cur=$en; (( dn )) && break
done
chk "S7 32k tiling" "1" "$ok"
chk "S7 32k coverage" "31616" "$cov"

echo
if (( fails == 0 )); then echo "MIRROR GATE: ALL PASS"; else
  echo "MIRROR GATE: $fails FAIL"; exit 1; fi
