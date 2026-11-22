# p200c_vlane_debug.py -- r200c V-lane stage attribution validator.
# Runs ONE single-workgroup mxk8v4 decode (e4m3 K + e2m1 V -- isolates
# the V lane) with ARC_MXKV_DEBUG=1, reads the kernel's debug dump
# (/tmp/mxkv_debug.bin; layout in mx_paged_decode.hpp args comment) and
# byte-diffs each captured stage against host reconstruction from the
# same cache:
#   A  staged SLM image      -> convicts/acquits the form-(i) bwidth
#                               copy + tv-decoded scatter
#   B  anchors               -> the identity-slot-0 anchor law
#   C  gathered tArV values  -> the dequant_vslm form-(ii) walk
# Verdict names the FIRST diverging stage; all-ok points downstream
# (softmax/PV gemm), which the faces already cover.
# Shape law: 1 seq of 16 tokens, HK=1/HQ=1, 1 block -> exactly one
# workgroup, one k-tile (dump tile 0). Exit 0 = a stage was convicted
# or all matched (JSON verdict to argv[1]).

from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import os

import torch

_MXKV_DIR = (
    Path(__file__).resolve().parents[3] / "vllm_gaudi" / "xpu" / "patches" / "mx_kv"
)


def _load_mxkv():
    spec = importlib.util.spec_from_file_location(
        "mx_kv_lane", _MXKV_DIR / "__init__.py", submodule_search_locations=[str(_MXKV_DIR)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mxkv = _load_mxkv()
mx_store = mxkv.mx_store
slot_bytes = mxkv.slot_bytes
encode_mxfp8 = mxkv.encode_mxfp8
encode_mxfp4 = mxkv.encode_mxfp4

from vllm_xpu_kernels.mx_attn_interface import mx_paged_decode  # noqa: E402

D = 128
# HK=2 (qgroup 1): the V byte-surface row pitch is HK*slot — the 2D
# block copy op encodes pitch in 16-byte units, so ODD HK (slot 264/200
# -> pitch 8 mod 16) drifts +8 bytes/row (r200d correlator law: staged
# row k == cache[v_off + (slot+8)*k]). HK=2 keeps every format's pitch
# 16-aligned -- the faces/production geometry -- while staying one WG.
HK = 2
HQ = 2
BLOCK = 16
DEV = "xpu"
# P200C_FMT selects the V dtype lane: mxk8v4 (e2m1 V) or mxfp8 (e4m3 V).
# Running BOTH on one wheel splits copy-geometry defects from e2m1-only
# source-surface defects (r200d follow-up).
FMT = os.environ.get("P200C_FMT", "mxk8v4")
V_IS_E4M3 = FMT in ("mxfp8", "mxk4v8")

# e2m1 nibble -> float (quantizer law: elem 2j low, 2j+1 high).
# Nibble = s(1) e(2) m(1): magnitudes {0,.5,1,1.5,2,3,4,6}, sign bit 8.
E2M1_MAG = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1(n: int) -> float:
    return -E2M1_MAG[n & 7] if n & 8 else E2M1_MAG[n & 7]


def e4m3(b: int) -> float:
    t = torch.tensor([b], dtype=torch.uint8)
    return float(t.view(torch.float8_e4m3fn).float())


def e8m0_scale(b: int) -> float:
    return 2.0 ** (b - 127)


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/out/p200c_debug.json"
    verdict = {"format": FMT, "stages": {}}

    # --- build the cache: 16 tokens, one block, one head -------------
    # P200C_SYNTH=1 fills the slot bytes directly with a unique-value
    # law (payload byte = f(k, j), scale code = 96 + k*NG + g) so every
    # dequantized (k, v) value is globally unique -- the C-stage walk's
    # coordinate law then falls out of the dump exactly instead of
    # drowning in e2m1 magnitude collisions.
    T = BLOCK
    synth = os.environ.get("P200C_SYNTH", "0") == "1"
    verdict["synth"] = synth
    gen = torch.Generator().manual_seed(1234)
    ng = D // 32
    k_bytes = D if FMT in ("mxfp8", "mxk8v4") else D // 2
    v_bytes = D if FMT in ("mxfp8",) else D // 2
    k_width = k_bytes + ng
    v_scale_off = k_width + v_bytes
    if synth:
        sb = slot_bytes(D, FMT)
        host = torch.zeros(BLOCK, HK, sb, dtype=torch.uint8)
        for k in range(BLOCK):
            for h in range(HK):
                # r200f synth-v2 attribution law. v1 folded k into the
                # payload byte modulo 125 -- two contamination modes
                # surfaced in the r200e inversion: (a) byte(k, j) ==
                # byte(k-dk, j-3*dk) collisions made the (k, v)
                # attribution ambiguous (the fake "non-affine
                # coordinate law"), and (b) per-head scale codes (+1 on
                # h=1) turned the two-WG debug-dump race (HQ=2 -> 2
                # workgroups, both dumping) into a global 2x scale
                # shift. v2: payload byte injective per BYTE POSITION j
                # only (odd alphabet; j=63 -> 126 and j=127 -> 254,
                # the unique even-mantissa bytes 1.75*2^0 / 1.75*2^24,
                # since 0x7F/0xFF are e4m3 NaN), scale code 96 + k*ng +
                # g injective per (row, group) with no modular wrap,
                # and BOTH HEADS IDENTICAL so the race is benign.
                for j in range(v_bytes):
                    x = 2 * j + 1
                    if x == 127:
                        x = 126
                    elif x == 255:
                        x = 254
                    host[k, h, k_width + j] = x
                for g in range(ng):
                    host[k, h, v_scale_off + g] = 96 + k * ng + g
        cache = host.to(DEV).reshape(1, BLOCK, HK, sb).contiguous()
        q = torch.randn(1, HQ, D, generator=gen).to(torch.bfloat16).to(DEV)
        torch.xpu.synchronize()
    else:
        xk = torch.randn(T, HK, D, generator=gen).to(DEV)
        xv = torch.randn(T, HK, D, generator=gen).to(DEV)
        q = torch.randn(1, HQ, D, generator=gen).to(torch.bfloat16).to(DEV)
        cache = torch.zeros(1, BLOCK, HK, slot_bytes(D, FMT), dtype=torch.uint8, device=DEV)
        tb = torch.zeros(T, dtype=torch.int32, device=DEV)   # token -> block 0
        tp = torch.arange(T, dtype=torch.int32, device=DEV)  # token -> pos
        mx_store(xk, xv, cache, tb, tp)
        torch.xpu.synchronize()

    cache_host = cache[0, :, 0, :].cpu()  # [T, slot_bytes]
    sb = cache_host.shape[1]
    ng = D // 32
    k_bytes = D if FMT in ("mxfp8", "mxk8v4") else D // 2
    v_bytes = D if FMT in ("mxfp8",) else D // 2
    k_width = k_bytes + ng          # V payload starts here in the slot
    v_scale_off = k_width + v_bytes

    # --- single-WG decode with the debug lane on ---------------------
    os.environ["ARC_MXKV_DEBUG"] = "1"
    cu_q = torch.arange(2, dtype=torch.int32, device=DEV)
    sl = torch.tensor([T], dtype=torch.int32, device=DEV)
    bt = torch.zeros(1, 1, dtype=torch.int32, device=DEV)
    out = mx_paged_decode(
        q, cache, FMT, cu_q, sl, bt,
        max_seqlen_k=T, softmax_scale=1.0 / math.sqrt(D),
        num_splits_kv=0, host_kv_lens=None,
    )
    torch.xpu.synchronize()
    del out

    blob = open("/tmp/mxkv_debug.bin", "rb").read()
    # raw artifacts for offline cross-correlation (r200d follow-up):
    # match garbage rows against the full slot to convict permutation vs
    # wrong-offset vs OOB reads without re-running the kernel.
    Path("/out/mxkv_debug.bin").write_bytes(blob)
    Path("/out/mxkv_cache_slot.bin").write_bytes(cache_host.numpy().tobytes())
    A = blob[0:8192]
    B = struct.unpack("<8q", blob[8192:8256])
    C = blob[8256:]

    # --- stage A: staged SLM image vs the cache's V payload ----------
    # mxk8v4: kVBytesPerStep = 16 (32 elems / 2), VTiles = 4 -> pitch 64
    vtiles = 4
    kbps = v_bytes // vtiles          # 16 for mxk8v4
    pitch = v_bytes                    # 64
    # r203 permuted V frame: within each VV-step byte window the kernel
    # scatters logical col c to phys (c & mask)*4 + (c >> sh) (mask/sh
    # = 7/3 e4m3, 3/2 e2m1) so each (k-row, WI) quad is one u32;
    # invert before comparing to the payload bytes.
    mult = 8 if V_IS_E4M3 else 4
    a_bad = []
    for k in range(BLOCK):
        for j in range(pitch):
            w, p = divmod(j, kbps)
            c = w * kbps + (p >> 2) + mult * (p & 3)
            got = A[k * 128 + j]
            want = int(cache_host[k, k_width + c])
            if got != want:
                a_bad.append((k, j, got, want))
    verdict["stages"]["A_staged_image"] = {
        "mismatches": len(a_bad),
        "first": a_bad[:8],
    }

    # --- stage B: anchors --------------------------------------------
    verdict["stages"]["B_anchors"] = {
        "v_krow_base": B[0], "expected": 0,
        "v_byte_base": B[1], "expected_byte": 0,
        "ok": B[0] == 0 and B[1] == 0,
    }

    # --- stage C: gathered tArV vs host form-(ii) reconstruction -----
    def bf16(x: float) -> float:
        return float(torch.tensor(x).to(torch.bfloat16).float())

    c_bad = []
    for vv in range(vtiles):
        for wi in range(16):
            for dv in range(32):
                t0, t1 = wi & 1, wi >> 1
                v0, v1, v2 = dv & 1, (dv >> 1) & 7, (dv >> 4) & 1
                v = t1 + 8 * v0 + 16 * v2
                k = t0 + 2 * v1
                grp = (vv * 32 + v) // 32
                sc = e8m0_scale(int(cache_host[k, v_scale_off + grp]))
                if V_IS_E4M3:
                    byte = int(cache_host[k, k_width + vv * kbps + v])
                    want = bf16(e4m3(byte) * sc)
                else:
                    byte = int(cache_host[k, k_width + vv * kbps + (v >> 1)])
                    nib = (byte >> 4) if (v & 1) else (byte & 0xF)
                    want = bf16(e2m1(nib) * sc)
                off = (((0 * vtiles + vv) * 16 + wi) * 32 + dv) * 4
                got = struct.unpack("<f", C[off : off + 4])[0]
                if got != want:
                    c_bad.append((vv, wi, dv, k, v, got, want))
    verdict["stages"]["C_gather"] = {"mismatches": len(c_bad), "first": c_bad[:8]}

    # --- verdict: first diverging stage ------------------------------
    a_n = verdict["stages"]["A_staged_image"]["mismatches"]
    b_ok = verdict["stages"]["B_anchors"]["ok"]
    c_n = verdict["stages"]["C_gather"]["mismatches"]
    if a_n:
        verdict["verdict"] = "A_BAD: staging (bwidth copy + scatter) misplaces bytes"
    elif not b_ok:
        verdict["verdict"] = "B_BAD: anchor law broken on the V coordinate view"
    elif c_n:
        verdict["verdict"] = "C_BAD: dequant_vslm form-(ii) gather decode wrong"
    else:
        verdict["verdict"] = "ALL_STAGES_OK: corruption is downstream (softmax/PV)"
    Path(out_path).write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict["verdict"]))
    print(f"A mismatches: {a_n}  B ok: {b_ok}  C mismatches: {c_n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
