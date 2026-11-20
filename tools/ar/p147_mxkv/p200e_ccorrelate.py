# p200e_ccorrelate.py -- r200f C-stage (dequant_vslm gather) attribution,
# offline, synth-v2 caches. A-stage is clean (HK=2 pitch law), so any C
# mismatch is the form-(ii) walk. v2 cache law (see p200c_vlane_debug.py):
# payload byte injective per byte-position j, scale code 96 + k*ng + g
# injective per (row, group), both heads identical. Every dumped float
# therefore decodes DIRECTLY to its source -- byte -> j (mxfp8: j IS the
# element column) or nibble-magnitude-class -> n (mxk8v4: the nibble value,
# with k from the scale) -- with no search and no modular-collision
# ambiguity (the v1 mod-125 alphabet fabricated a fake non-affine
# "coordinate law"; the v1 per-head scale codes fabricated a global 2x
# through the two-WG dump race).
# Usage: python p200e_ccorrelate.py <fmt>   (mxfp8 | mxk8v4)
# Reads mxkv_cache_slot.bin + mxkv_debug.bin from CWD.

from __future__ import annotations

import struct
import sys
from pathlib import Path

FMT = sys.argv[1] if len(sys.argv) > 1 else "mxfp8"
D = 128
NG = D // 32
V_IS_E4M3 = FMT in ("mxfp8", "mxk4v8")
K_BYTES = D
V_BYTES = D if V_IS_E4M3 else D // 2
V_OFF = K_BYTES + NG
V_SCALE_OFF = V_OFF + V_BYTES
ROWS = 16
VTILES = 4
VPB = V_BYTES // VTILES  # bytes per VV step: 32 (e4m3) / 16 (e2m1)

E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1(n: int) -> float:
    return -E2M1_MAG[n & 7] if n & 8 else E2M1_MAG[n & 7]


def e4m3(b: int) -> float:
    s = -1.0 if b & 0x80 else 1.0
    e = (b >> 3) & 0xF
    m = b & 0x7
    if e == 0:
        return s * m * 2**-9  # fn subnormals
    if e == 0xF and m == 0x7:
        return float("nan")
    return s * (1 + m / 8.0) * 2 ** (e - 7)


def e8m0(b: int) -> float:
    return 2.0 ** (b - 127)


def bf16(x: float) -> float:
    # round-to-nearest-even to bfloat16 precision, returned as f32
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def synth_byte(j: int) -> int:
    x = 2 * j + 1
    if x == 127:
        x = 126
    elif x == 255:
        x = 254
    return x


def main() -> int:
    cache = Path("mxkv_cache_slot.bin").read_bytes()
    blob = Path("mxkv_debug.bin").read_bytes()
    C = blob[8256:]
    slot = len(cache) // ROWS

    # --- source-side value tables (synth-v2 laws) -------------------
    # mxfp8 : value -> (j, k, g); element column v == j
    # mxk8v4: value -> (n, k, g); nibble value n (byte = hi<<4 | lo,
    #         j recovered only after the v-parity law is applied)
    inv = {}
    dup = 0
    for k in range(ROWS):
        for g in range(NG):
            sc = e8m0(96 + k * NG + g)
            if V_IS_E4M3:
                for j in range(V_BYTES):
                    val = bf16(e4m3(synth_byte(j)) * sc)
                    if val in inv:
                        dup += 1
                    inv[val] = (j, k, g)
            else:
                for n in range(16):
                    val = bf16(e2m1(n) * sc)
                    if val in inv:
                        dup += 1
                    inv[val] = (n, k, g)
    print(f"fmt={FMT} slot={slot} inverse entries={len(inv)} dups={dup}")
    if dup:
        print("SYNTH_NOT_INJECTIVE -- fix the alphabet before reading the map")
        return 1

    # --- decode the dump: (vv, wi, dv) -> observed (k, v/n) ---------
    def decode(vv: int, wi: int, dv: int):
        off = (((vv) * 16 + wi) * 32 + dv) * 4
        return struct.unpack("<f", C[off : off + 4])[0]

    unknown = 0
    obs = {}  # (wi, dv) -> (k, j) or (k, n); VV-independent coordinate law
    for vv in range(VTILES):
        for wi in range(16):
            for dv in range(32):
                g = decode(vv, wi, dv)
                if g not in inv:
                    unknown += 1
                    continue
                j, k, gd = inv[g]
                # scale group cross-check: the law says the element's
                # group is (vv*32 + v)/32; observed (k, g) must agree
                obs[(vv, wi, dv)] = (k, j, gd)

    print(f"undecodable dump slots: {unknown}/{VTILES*16*32}")

    # --- compare against the assumed form-(ii) law ------------------
    # v = t1 + 8*v0 + 16*v2 ; k = t0 + 2*v1  (mode 0 = v, mode 1 = k)
    bad = 0
    shown = 0
    for (vv, wi, dv), (k_obs, j_obs, g_obs) in sorted(obs.items()):
        t0, t1 = wi & 1, wi >> 1
        v0, v1, v2 = dv & 1, (dv >> 1) & 7, (dv >> 4) & 1
        v_law = t1 + 8 * v0 + 16 * v2
        k_law = t0 + 2 * v1
        if V_IS_E4M3:
            ok = (k_obs == k_law) and (j_obs == (vv * VPB + v_law) % V_BYTES)
        else:
            # predicted nibble value at column v_law
            pj = (vv * VPB + (v_law >> 1)) % V_BYTES
            byte = synth_byte(pj)
            n_pred = (byte >> 4) if (v_law & 1) else (byte & 0xF)
            ok = (k_obs == k_law) and (j_obs == n_pred)
        if not ok:
            bad += 1
            if shown < 16:
                print(f"  ({vv},{wi},{dv}): observed (k={k_obs}, "
                      f"{'j' if V_IS_E4M3 else 'n'}={j_obs}, g={g_obs}) "
                      f"law (k={k_law}, v={v_law})")
                shown += 1
    print(f"law mismatches (decoded slots): {bad}/{len(obs)}")

    # --- observed map table (vv=0) for eyeball/derivation ------------
    print("--- observed (wi, dv) -> (k, "
          + ("j)" if V_IS_E4M3 else "nibble)")
          + ", vv=0 ---")
    for wi in range(4):
        row = []
        for dv in range(32):
            e = obs.get((0, wi, dv))
            row.append(f"({e[0]},{e[1]})" if e else "?")
        print(f"wi={wi}: " + " ".join(row))
    print("--- assumed law: v = (wi>>1) + 8*(dv&1) + 16*((dv>>4)&1); "
          "k = (wi&1) + 2*((dv>>1)&7) ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
