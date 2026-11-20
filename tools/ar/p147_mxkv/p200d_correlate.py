# p200d_correlate.py -- r200d A-stage garbage attribution, offline.
# Inputs (written by p200c_vlane_debug.py into /out):
#   mxkv_debug.bin   kernel dump (A: 4 tiles x 16 rows x 128-pitch @ 0)
#   mxkv_cache_slot.bin  host copy of the 16 cache slots
# For each staged k-row that mismatches, locate the got-bytes as a
# contiguous window in the cache. A row-permutation shows got == some
# other row's V payload; a wrong-row-stride shows got == cache bytes at
# v_start + k*S for a constant S != slot_pitch; an OOB/zero read shows
# no match at all. Reports per-row: exact window offsets (if any) and
# best partial match.
# Usage: python p200d_correlate.py <fmt>   (mxfp8 | mxk8v4)

from __future__ import annotations

import sys
from pathlib import Path

FMT = sys.argv[1] if len(sys.argv) > 1 else "mxfp8"
D = 128
NG = D // 32
K_BYTES = D
V_BYTES = D if FMT == "mxfp8" else D // 2
V_OFF = K_BYTES + NG          # V payload offset inside the slot
ROWS = 16
PITCH_A = 128                 # A dump row pitch


def main() -> int:
    cache = Path("mxkv_cache_slot.bin").read_bytes()
    blob = Path("mxkv_debug.bin").read_bytes()
    A = blob[0:8192]
    slot = len(cache) // ROWS
    print(f"fmt={FMT} slot_bytes={slot} v_off={V_OFF} v_bytes={V_BYTES}")

    def want(k: int, j: int) -> int:
        return cache[k * slot + V_OFF + j]

    n_bad = 0
    for k in range(ROWS):
        got = A[k * PITCH_A : k * PITCH_A + V_BYTES]
        wnt = bytes(want(k, j) for j in range(V_BYTES))
        if got == wnt:
            print(f"row {k:2d}: CLEAN")
            continue
        n_bad += 1
        # exact contiguous match anywhere in the cache
        hits = []
        start = 0
        while True:
            i = cache.find(got, start)
            if i < 0:
                break
            hits.append(i)
            start = i + 1
        # best single-row alignment: compare against every other row's
        # V payload and also raw slot rows at other offsets
        best = (-1, None)
        for kk in range(ROWS):
            for off in range(0, slot - V_BYTES + 1, 1):
                cand = cache[kk * slot + off : kk * slot + off + V_BYTES]
                eq = sum(a == b for a, b in zip(got, cand))
                if eq > best[0]:
                    best = (eq, (kk, off))
        # constant-stride hypothesis: v_start + k*S
        stride_hit = None
        for s in range(slot - V_BYTES + 1, slot + 65):
            base = V_OFF + k * s
            if base + V_BYTES <= len(cache):
                cand = cache[base : base + V_BYTES]
                if cand == got:
                    stride_hit = s
                    break
        print(
            f"row {k:2d}: BAD exact_hits={hits[:4]} "
            f"best_partial={best[0]}/{V_BYTES}@row{best[1][0]}+off{best[1][1]} "
            f"stride_match={stride_hit}"
        )
    print(f"bad rows: {n_bad}/{ROWS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
