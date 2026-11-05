#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P6.24 offline fp8-wire allreduce numerics calibration (2026-11-05).

Bit-exact python twin of the oneCCL patch converters (v3 header:
deps/libccl/.../allreduce_large_sycl_ring_fp8.hpp) plus an end-to-end
simulation of the exact wire algorithm under the exact ar_ceiling.py
randn-guard metric. Two questions answered without a GPU window:

1. Converter correctness: v3 fast/rare paths vs an exhaustive RTNE
   reference over the complete e4m3 grid (subnormals included).
2. Numerics-bar calibration: what rel-rms does the CORRECT algorithm
   produce on the bench construction? r97v2 measured 0.02285 on the rig;
   this sim gives 0.0235 -> the algorithm was already correct and the
   original 0.02 bar sat below the true e4m3 noise floor. Re-registered
   bar: 0.035 (5.7x below the >=0.2 corruption regime).

Also demonstrates the stock cross-check: fp16-exact allreduce reproduces
the rig's stock rel-rms 0.00022 to three digits (numpy RNG, not torch's --
distribution-level equivalence is sufficient for the metric).

Run: python fp8_noise_sim.py   (numpy required; py 3.8+)
"""
import numpy as np

E4M3_MAX = 448.0


def enc_e4m3(v):
    """v3 fp8ar_float_to_e4m3: branchless fast path + rint rare path."""
    x = np.float32(v).view(np.uint32)
    sign = (x >> np.uint32(24)) & np.uint32(0x80)
    ae = (x & np.uint32(0x7fffffff)).astype(np.uint64)
    rare = ae < (121 << 23)  # |v| < 2^-6: zero / subnormal / tiny
    y = ae - np.uint64(120 << 23)  # re-bias exponent field 127 -> 7
    z = y + np.uint64(0x7FFFF) + ((y >> np.uint64(20)) & np.uint64(1))  # RTNE, drop 20
    code = (z >> np.uint64(20)) & np.uint64(0x7F)
    code = np.where(code >= 0x7F, np.uint64(0x7E), code)  # saturate 448
    fast = sign.astype(np.uint64) | code
    scaled = np.abs(np.float32(v)) * np.float32(512.0)  # subnormal domain: q in 0..8
    q = np.rint(scaled).astype(np.uint64)  # RTNE
    q = np.minimum(q, np.uint64(8))  # 8 == min-normal code
    slow = sign.astype(np.uint64) | q
    return np.where(rare, slow, fast).astype(np.uint32)


def dec_e4m3(q):
    """v3 fp8ar_e4m3_to_float: bit-assembled normal + subnormal select."""
    q = np.asarray(q, dtype=np.uint32)
    e = (q >> np.uint32(3)) & np.uint32(0xF)
    m = q & np.uint32(7)
    nb = (((e + np.uint32(120)) << np.uint32(23)) | (m << np.uint32(20))).view(np.float32)
    sub = m.astype(np.float32) * np.float32(0.001953125)
    mag = np.where(e == 0, sub, nb)
    return np.where((q & np.uint32(0x80)) != 0, -mag, mag).astype(np.float32)


def _grid():
    """Complete monotone e4m3 magnitude grid from the decoder."""
    vals = [float(dec_e4m3(np.uint32(c))) for c in range(0x7F)]
    return vals


GRID = _grid()


def ref_code(av):
    """Exhaustive RTNE reference: nearest grid code, ties to even code."""
    best, bd = 0, abs(GRID[0] - av)
    for i in range(1, 0x7F):
        d = abs(GRID[i] - av)
        if d < bd or (d == bd and (i & 1) == 0 and (best & 1) == 1):
            best, bd = i, d
    return best


def check_converter(n=300000, seed=42):
    rng = np.random.default_rng(seed)
    tests = np.concatenate([
        rng.uniform(-460, 460, n).astype(np.float32),
        rng.uniform(-0.02, 0.02, n // 2).astype(np.float32),
        rng.uniform(440, 465, n // 10).astype(np.float32),
        np.float32([0.0, -0.0, 2**-9, 2**-6, 2**-6 * 1.0001, 0.015625,
                    0.01563, 1.0, -1.0, 448.0, 449.0, 464.0, 1e-8]),
    ])
    got = enc_e4m3(tests) & np.uint32(0x7F)
    bad = 0
    for i, v in enumerate(tests):
        if abs(float(v)) > 464.0:
            if got[i] != 0x7E:
                bad += 1
            continue
        if got[i] != ref_code(abs(float(v))):
            bad += 1
    print(f"converter: {bad} mismatches / {len(tests)} values")
    return bad == 0


def simulate(n=38_338_560, seed=1000):
    """End-to-end wire algorithm on the bench construction.

    Mirrors ar_ceiling.py randn guard: per-rank CPU-seeded randn*0.5 cast
    to fp16 (the wire dtype), fp32 reference, halves exchanged by the
    s=(rank+1)%2 law, amax per outgoing half, res = 2*max(s0, s1), one
    e4m3 round trip per output element on both ranks.
    Metric: rel = mean|out-ref| / ref.std()  (the rig's discriminator).
    """
    rng = np.random.default_rng(seed)
    a0 = (rng.standard_normal(n) * 0.5).astype(np.float32)
    a1 = (rng.standard_normal(n) * 0.5).astype(np.float32)
    h0 = a0.astype(np.float16).astype(np.float32)
    h1 = a1.astype(np.float16).astype(np.float32)
    rf = a0 + a1
    half = n // 2
    amax0 = np.abs(h0[half:]).max()  # rank0 outgoing half (s=1)
    amax1 = np.abs(h1[:half]).max()  # rank1 outgoing half (s=0)
    s0, s1 = amax0 / E4M3_MAX, amax1 / E4M3_MAX
    res = np.float32(2.0) * max(s0, s1)
    out = np.empty(n, np.float32)
    # rank0 computes half-0: in1 = h0[:half], in2 = quant(h1[:half]*mul1)
    sum_ = h0[:half] + dec_e4m3(enc_e4m3(h1[:half] * np.float32(E4M3_MAX / amax1))) * s1
    q = enc_e4m3(sum_ / res)
    out[:half] = dec_e4m3(q) * res
    # rank1 computes half-1
    sum_ = h1[half:] + dec_e4m3(enc_e4m3(h0[half:] * np.float32(E4M3_MAX / amax0))) * s0
    q = enc_e4m3(sum_ / res)
    out[half:] = dec_e4m3(q) * res
    cos = float(np.dot(out, rf)) / (float(np.linalg.norm(out)) * np.linalg.norm(rf))
    rel = float(np.abs(out - rf).mean() / rf.std())
    print(f"fp8 sim : cos {cos:.6f}  rel {rel:.6f}   (rig r97v2: cos 1.013* "
          f"rel 0.02285; rig cos carries a ~1.0145 metric artifact present "
          f"in the stock leg too)")
    return cos, rel


def stock_crosscheck(n=38_338_560, seed=1000):
    """fp16-exact allreduce under the same metric (rig stock: 0.00022)."""
    rng = np.random.default_rng(seed)
    a0 = (rng.standard_normal(n) * 0.5).astype(np.float32)
    a1 = (rng.standard_normal(n) * 0.5).astype(np.float32)
    af = (a0.astype(np.float16).astype(np.float32)
          + a1.astype(np.float16).astype(np.float32)).astype(np.float16).astype(np.float32)
    rf = a0 + a1
    rel = float(np.abs(af - rf).mean() / rf.std())
    print(f"stock sim: rel {rel:.6f}   (rig stock leg: 0.00022)")
    return rel


if __name__ == "__main__":
    ok = check_converter()
    cos, rel = simulate()
    stock_crosscheck()
    assert ok, "CONVERTER MISMATCH"
    assert rel < 0.035, "SIM ABOVE RE-REGISTERED BAR"
    print("SIM VERDICT: converters bit-exact; e4m3 noise floor consistent "
          "with rig; bar 0.035 calibrated")
